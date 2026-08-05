// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use crate::ListingTableUrl;
use crate::file_compression_type::FileCompressionType;
use crate::file_groups::FileGroup;
use crate::sink::DataSink;
use crate::write::demux::{DemuxedStreamReceiver, start_demuxer_task};

use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::{DataFusionError, Result, exec_err, not_impl_err};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::dml::InsertOp;

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, future, stream};
use object_store::{Error as ObjectStoreError, ObjectStore, ObjectStoreExt};

/// Determines how `FileSink` output paths are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileOutputMode {
    /// Infer output mode from the output URL (for example, by extension / trailing `/`).
    #[default]
    Automatic,
    /// Write to a single output file at the exact output path.
    SingleFile,
    /// Write to a directory under the output path with generated filenames.
    Directory,
}

impl FileOutputMode {
    /// Resolve this mode into a `single_file_output` boolean for the demuxer.
    pub fn single_file_output(self, base_output_path: &ListingTableUrl) -> bool {
        match self {
            Self::Automatic => {
                !base_output_path.is_collection()
                    && base_output_path.file_extension().is_some()
            }
            Self::SingleFile => true,
            Self::Directory => false,
        }
    }
}

impl From<Option<bool>> for FileOutputMode {
    fn from(value: Option<bool>) -> Self {
        match value {
            None => Self::Automatic,
            Some(true) => Self::SingleFile,
            Some(false) => Self::Directory,
        }
    }
}

impl From<FileOutputMode> for Option<bool> {
    fn from(value: FileOutputMode) -> Self {
        match value {
            FileOutputMode::Automatic => None,
            FileOutputMode::SingleFile => Some(true),
            FileOutputMode::Directory => Some(false),
        }
    }
}

/// General behaviors for files that do `DataSink` operations
#[async_trait]
pub trait FileSink: DataSink {
    /// Retrieves the file sink configuration.
    fn config(&self) -> &FileSinkConfig;

    /// Spawns writer tasks and joins them to perform file writing operations.
    /// Is a critical part of `FileSink` trait, since it's the very last step for `write_all`.
    ///
    /// This function handles the process of writing data to files by:
    /// 1. Spawning tasks for writing data to individual files.
    /// 2. Coordinating the tasks using a demuxer to distribute data among files.
    /// 3. Collecting results using `tokio::join`, ensuring that all tasks complete successfully.
    /// 4. Applying shared insert-operation checks and post-success overwrite cleanup.
    ///
    /// # Parameters
    /// - `context`: The execution context (`TaskContext`) that provides resources
    ///   like memory management and runtime environment.
    /// - `demux_task`: A spawned task that handles demuxing, responsible for splitting
    ///   an input [`SendableRecordBatchStream`] into dynamically determined partitions.
    ///   See `start_demuxer_task()`
    /// - `file_stream_rx`: A receiver that yields streams of record batches and their
    ///   corresponding file paths for writing. See `start_demuxer_task()`
    /// - `object_store`: A handle to the object store where the files are written.
    ///
    /// # Returns
    /// - `Result<u64>`: Returns the total number of rows written across all files.
    async fn spawn_writer_tasks_and_join(
        &self,
        context: &Arc<TaskContext>,
        demux_task: SpawnedTask<Result<()>>,
        file_stream_rx: DemuxedStreamReceiver,
        object_store: Arc<dyn ObjectStore>,
    ) -> Result<u64>;

    /// File sink implementation of the [`DataSink::write_all`] method.
    async fn write_all(
        &self,
        data: SendableRecordBatchStream,
        context: &Arc<TaskContext>,
    ) -> Result<u64> {
        let config = self.config();
        let object_store = context
            .runtime_env()
            .object_store(&config.object_store_url)?;
        let single_file_output = config.table_partition_cols.is_empty()
            && config
                .file_output_mode
                .single_file_output(&config.table_paths[0]);

        match config.insert_op {
            InsertOp::Replace => {
                return not_impl_err!(
                    "Replace is not supported because raw files have no row key"
                );
            }
            InsertOp::Append if single_file_output => {
                // This preflight is best-effort: the buffered multipart writer does not
                // expose conditional creation, so concurrent writers can still race.
                match object_store.head(config.table_paths[0].prefix()).await {
                    Ok(_) => {
                        return exec_err!(
                            "Cannot append to existing file '{}'; exact-file append is not supported",
                            config.original_url
                        );
                    }
                    Err(ObjectStoreError::NotFound { .. }) => {}
                    Err(e) => return Err(e.into()),
                }
            }
            InsertOp::Append | InsertOp::Overwrite => {}
        }

        let overwrite_paths = if config.insert_op == InsertOp::Overwrite
            && !single_file_output
        {
            object_store
                .list(Some(config.table_paths[0].prefix()))
                .try_filter(|meta| {
                    let matches_extension = match &config.overwrite_file_extension {
                        Some(extension) => meta.location.as_ref().ends_with(extension),
                        None => {
                            matches_file_extension(&meta.location, &config.file_extension)
                        }
                    };
                    future::ready(
                        config.table_paths[0].contains(&meta.location, false)
                            && matches_extension,
                    )
                })
                .map_ok(|meta| meta.location)
                .try_collect::<Vec<_>>()
                .await?
        } else {
            vec![]
        };

        let (demux_task, file_stream_rx) = start_demuxer_task(config, data, context);
        let write_result = self
            .spawn_writer_tasks_and_join(
                context,
                demux_task,
                file_stream_rx,
                Arc::clone(&object_store),
            )
            .await;

        let cleanup_result = if write_result.is_ok() && !overwrite_paths.is_empty() {
            let locations =
                stream::iter(overwrite_paths.into_iter().map(Ok::<_, ObjectStoreError>))
                    .boxed();
            object_store
                .delete_stream(locations)
                .map(|result| match result {
                    Err(ObjectStoreError::NotFound { .. }) => Ok(None),
                    result => result.map(Some).map_err(DataFusionError::from),
                })
                .try_for_each(|_| future::ready(Ok(())))
                .await
        } else {
            Ok(())
        };

        if let Some(cache) = context.runtime_env().cache_manager.get_list_files_cache() {
            let path = config.table_paths[0].prefix();
            for key in cache
                .list_entries()
                .into_keys()
                .filter(|key| key.path == *path)
            {
                cache.remove(&key);
            }
        }

        let row_count = write_result?;
        cleanup_result?;
        Ok(row_count)
    }
}

fn strip_compression_extension(value: &str) -> &str {
    FileCompressionType::COMPRESSED
        .iter()
        .find_map(|compression| value.strip_suffix(compression.extension()))
        .unwrap_or(value)
}

fn matches_file_extension(path: &object_store::path::Path, extension: &str) -> bool {
    strip_compression_extension(path.filename().unwrap_or_default())
        .strip_suffix(strip_compression_extension(extension))
        .is_some_and(|prefix| prefix.ends_with('.'))
}

/// The base configurations to provide when creating a physical plan for
/// writing to any given file format.
#[derive(Debug, Clone)]
pub struct FileSinkConfig {
    /// The unresolved URL specified by the user
    pub original_url: String,
    /// Object store URL, used to get an ObjectStore instance
    pub object_store_url: ObjectStoreUrl,
    /// A collection of files organized into groups.
    /// Each FileGroup contains one or more PartitionedFile objects.
    pub file_group: FileGroup,
    /// Vector of partition paths
    pub table_paths: Vec<ListingTableUrl>,
    /// The schema of the output file
    pub output_schema: SchemaRef,
    /// A vector of column names and their corresponding data types,
    /// representing the partitioning columns for the file
    pub table_partition_cols: Vec<(String, DataType)>,
    /// Controls how new data should be written to the file, determining whether
    /// to append to, overwrite, or replace records in existing files.
    pub insert_op: InsertOp,
    /// Controls whether partition columns are kept for the file
    pub keep_partition_by_columns: bool,
    /// File extension without a dot(.)
    pub file_extension: String,
    /// Optional existing-file suffix used for overwrite cleanup. `None` matches the
    /// output format and its compression variants; an empty suffix matches all files.
    pub overwrite_file_extension: Option<String>,
    /// Determines how the output path is interpreted.
    pub file_output_mode: FileOutputMode,
}

impl FileSinkConfig {
    /// Get output schema
    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PartitionedFile;
    use arrow::datatypes::Schema;
    use bytes::Bytes;
    use datafusion_common::exec_err;
    use datafusion_execution::cache::TableScopedPath;
    use datafusion_execution::cache::cache_manager::CachedFileList;
    use datafusion_execution::runtime_env::RuntimeEnv;
    use datafusion_physical_plan::stream::EmptyRecordBatchStream;
    use datafusion_physical_plan::{DisplayAs, DisplayFormatType};
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };
    use std::fmt;
    use std::sync::Mutex;
    use url::Url;

    #[derive(Debug)]
    struct TestSink {
        config: FileSinkConfig,
        writer_error: bool,
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl DisplayAs for TestSink {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(f, "TestSink")
        }
    }

    #[async_trait]
    impl FileSink for TestSink {
        fn config(&self) -> &FileSinkConfig {
            &self.config
        }

        async fn spawn_writer_tasks_and_join(
            &self,
            _context: &Arc<TaskContext>,
            demux_task: SpawnedTask<Result<()>>,
            mut file_stream_rx: DemuxedStreamReceiver,
            object_store: Arc<dyn ObjectStore>,
        ) -> Result<u64> {
            if self.writer_error {
                return exec_err!("writer failed");
            }

            while let Some((_path, mut receiver)) = file_stream_rx.recv().await {
                while receiver.recv().await.is_some() {}
            }
            demux_task
                .join_unwind()
                .await
                .map_err(|e| DataFusionError::ExecutionJoin(Box::new(e)))??;
            let output_path = if self
                .config
                .file_output_mode
                .single_file_output(&self.config.table_paths[0])
            {
                self.config.table_paths[0].prefix().clone()
            } else {
                self.config.table_paths[0].prefix().clone().join("new.csv")
            };
            object_store
                .put(&output_path, Bytes::from_static(b"new").into())
                .await?;
            self.events.lock().unwrap().push("writer_finished");
            Ok(1)
        }
    }

    #[async_trait]
    impl DataSink for TestSink {
        fn schema(&self) -> &SchemaRef {
            self.config.output_schema()
        }

        async fn write_all(
            &self,
            data: SendableRecordBatchStream,
            context: &Arc<TaskContext>,
        ) -> Result<u64> {
            FileSink::write_all(self, data, context).await
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum DeleteBehavior {
        Pass,
        Fail,
        NotFound,
    }

    #[derive(Debug)]
    struct TestObjectStore {
        inner: Arc<InMemory>,
        events: Arc<Mutex<Vec<&'static str>>>,
        delete_behavior: DeleteBehavior,
    }

    impl fmt::Display for TestObjectStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(self.inner.as_ref(), f)
        }
    }

    #[async_trait]
    impl ObjectStore for TestObjectStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            self.inner.get_opts(location, options).await
        }

        #[expect(clippy::result_large_err)]
        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            match self.delete_behavior {
                DeleteBehavior::Pass => self.inner.delete_stream(locations),
                delete_behavior => {
                    let events = Arc::clone(&self.events);
                    locations
                        .map(move |location| {
                            location.and_then(|location| {
                                events.lock().unwrap().push("delete_called");
                                Err(match delete_behavior {
                                    DeleteBehavior::Fail => ObjectStoreError::Generic {
                                        store: "test",
                                        source: std::io::Error::other("delete failed")
                                            .into(),
                                    },
                                    DeleteBehavior::NotFound => {
                                        ObjectStoreError::NotFound {
                                            path: location.to_string(),
                                            source: std::io::Error::other(
                                                "already deleted",
                                            )
                                            .into(),
                                        }
                                    }
                                    DeleteBehavior::Pass => unreachable!(),
                                })
                            })
                        })
                        .boxed()
                }
            }
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    async fn test_context(
        old_path: &str,
        delete_behavior: DeleteBehavior,
    ) -> (
        Arc<TaskContext>,
        Arc<InMemory>,
        Arc<Mutex<Vec<&'static str>>>,
    ) {
        let inner = Arc::new(InMemory::new());
        inner
            .put(&Path::from(old_path), Bytes::from_static(b"old").into())
            .await
            .unwrap();
        let events = Arc::new(Mutex::new(vec![]));
        let store = Arc::new(TestObjectStore {
            inner: Arc::clone(&inner),
            events: Arc::clone(&events),
            delete_behavior,
        });
        let runtime = Arc::new(RuntimeEnv::default());
        runtime.register_object_store(
            &Url::parse("memory://test").unwrap(),
            store as Arc<dyn ObjectStore>,
        );
        (
            Arc::new(TaskContext::default().with_runtime(runtime)),
            inner,
            events,
        )
    }

    fn test_sink(
        target: &str,
        old_path: &str,
        insert_op: InsertOp,
        file_output_mode: FileOutputMode,
        writer_error: bool,
        events: Arc<Mutex<Vec<&'static str>>>,
    ) -> TestSink {
        TestSink {
            config: FileSinkConfig {
                original_url: target.to_string(),
                object_store_url: ObjectStoreUrl::parse("memory://test").unwrap(),
                file_group: FileGroup::new(vec![PartitionedFile::new(old_path, 3)]),
                table_paths: vec![ListingTableUrl::parse(target).unwrap()],
                output_schema: Arc::new(Schema::empty()),
                table_partition_cols: vec![],
                insert_op,
                keep_partition_by_columns: false,
                file_extension: "csv".to_string(),
                overwrite_file_extension: None,
                file_output_mode,
            },
            writer_error,
            events,
        }
    }

    fn empty_stream() -> SendableRecordBatchStream {
        Box::pin(EmptyRecordBatchStream::new(Arc::new(Schema::empty())))
    }

    #[test]
    fn file_extension_matches_compression_variants() {
        assert!(matches_file_extension(&Path::from("data.csv"), "csv"));
        assert!(matches_file_extension(&Path::from("data.csv.gz"), "csv"));
        assert!(matches_file_extension(&Path::from("data.csv.gz"), "csv.gz"));
        assert!(matches_file_extension(&Path::from("data.part.csv"), "csv"));
        assert!(matches_file_extension(
            &Path::from("part=x/data.json.zst"),
            "json"
        ));
        assert!(!matches_file_extension(&Path::from("data.notcsv"), "csv"));
        assert!(!matches_file_extension(&Path::from("data.csv.txt"), "csv"));
    }

    #[tokio::test]
    async fn file_sink_operation_failure_ordering() {
        let (context, store, events) =
            test_context("output/old.csv", DeleteBehavior::Fail).await;
        let sink = test_sink(
            "memory://test/output/",
            "output/old.csv",
            InsertOp::Overwrite,
            FileOutputMode::Directory,
            true,
            Arc::clone(&events),
        );
        let cache_key = TableScopedPath {
            table: None,
            path: Path::from("output"),
        };
        let cache = context
            .runtime_env()
            .cache_manager
            .get_list_files_cache()
            .unwrap();
        cache.put(
            &cache_key,
            CachedFileList::new(vec![
                store.head(&Path::from("output/old.csv")).await.unwrap(),
            ]),
        );
        let err = FileSink::write_all(&sink, empty_stream(), &context)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("writer failed"));
        assert!(events.lock().unwrap().is_empty());
        assert!(!cache.contains_key(&cache_key));
        assert!(store.head(&Path::from("output/old.csv")).await.is_ok());

        let (context, store, events) =
            test_context("output/old.csv", DeleteBehavior::Fail).await;
        let sink = test_sink(
            "memory://test/output/",
            "output/old.csv",
            InsertOp::Overwrite,
            FileOutputMode::Directory,
            false,
            Arc::clone(&events),
        );
        let err = FileSink::write_all(&sink, empty_stream(), &context)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("delete failed"));
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &["writer_finished", "delete_called"]
        );
        assert!(store.head(&Path::from("output/old.csv")).await.is_ok());
        assert!(store.head(&Path::from("output/new.csv")).await.is_ok());

        let (context, store, events) =
            test_context("output.csv", DeleteBehavior::Fail).await;
        let sink = test_sink(
            "memory://test/output.csv",
            "output.csv",
            InsertOp::Overwrite,
            FileOutputMode::SingleFile,
            false,
            Arc::clone(&events),
        );
        assert_eq!(
            FileSink::write_all(&sink, empty_stream(), &context)
                .await
                .unwrap(),
            1
        );
        assert_eq!(events.lock().unwrap().as_slice(), &["writer_finished"]);
        assert_eq!(
            store
                .get(&Path::from("output.csv"))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            Bytes::from_static(b"new")
        );
    }

    #[tokio::test]
    async fn directory_overwrite_uses_execution_snapshot_and_ignores_not_found() {
        let (context, store, events) =
            test_context("output/current.csv", DeleteBehavior::Pass).await;
        let mut sink = test_sink(
            "memory://test/output/",
            "output/stale.csv",
            InsertOp::Overwrite,
            FileOutputMode::Directory,
            false,
            Arc::clone(&events),
        );
        sink.config.overwrite_file_extension = Some(String::new());
        store
            .put(
                &Path::from("output/current.bin"),
                Bytes::from_static(b"old").into(),
            )
            .await
            .unwrap();
        assert_eq!(
            FileSink::write_all(&sink, empty_stream(), &context)
                .await
                .unwrap(),
            1
        );
        assert!(store.head(&Path::from("output/current.csv")).await.is_err());
        assert!(store.head(&Path::from("output/current.bin")).await.is_err());
        assert!(store.head(&Path::from("output/new.csv")).await.is_ok());

        let (context, store, events) =
            test_context("output/old.csv", DeleteBehavior::NotFound).await;
        let sink = test_sink(
            "memory://test/output/",
            "output/old.csv",
            InsertOp::Overwrite,
            FileOutputMode::Directory,
            false,
            Arc::clone(&events),
        );
        assert_eq!(
            FileSink::write_all(&sink, empty_stream(), &context)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &["writer_finished", "delete_called"]
        );
        assert!(store.head(&Path::from("output/new.csv")).await.is_ok());
    }

    #[tokio::test]
    async fn exact_file_append_fails_before_writer() {
        let (context, store, events) =
            test_context("output.csv", DeleteBehavior::Fail).await;
        let sink = test_sink(
            "memory://test/output.csv",
            "output.csv",
            InsertOp::Append,
            FileOutputMode::SingleFile,
            false,
            Arc::clone(&events),
        );
        let err = FileSink::write_all(&sink, empty_stream(), &context)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("exact-file append is not supported")
        );
        assert!(events.lock().unwrap().is_empty());
        assert!(store.head(&Path::from("output.csv")).await.is_ok());
    }
}
