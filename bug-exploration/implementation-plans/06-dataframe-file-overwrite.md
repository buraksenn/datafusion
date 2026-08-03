# Implementation plan: DataFrame file overwrite semantics

Source issue: [#4986](https://github.com/apache/datafusion/issues/4986)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

## Objective

Make `DataFrame::write_csv`, `DataFrame::write_json`, and `DataFrame::write_parquet` honor `DataFrameWriteOptions::with_insert_operation(InsertOp::Overwrite)` for both exact-file and directory outputs.

At the same time, make the existing `Append` default honest. An exact-file append cannot safely concatenate CSV/JSON/Parquet output, so it must not silently replace an existing object.

## Current behavior and root cause

The public option already exists, but the direct file APIs reject it before planning:

- `datafusion/core/src/dataframe/mod.rs`, `write_csv` and `write_json`, return `NotImplemented` unless `insert_op == Append`.
- `datafusion/core/src/dataframe/parquet.rs`, `write_parquet`, has the same guard.

Even if those guards are removed, the operation is lost and rejected downstream:

1. The three methods build `LogicalPlan::Copy` nodes.
2. `CopyTo` has no insert-operation field.
3. `datafusion/core/src/physical_planner.rs` hard-codes `FileSinkConfig.insert_op` to `InsertOp::Append`.
4. CSV, JSON, and Parquet `FileFormat::create_writer_physical_plan` implementations reject every non-append operation.
5. `FileSink::write_all` never consults `FileSinkConfig.insert_op` and never removes old files.

The current default is also inconsistent by output mode. Two observed `COPY` writes to the same exact CSV path, first with value `1` and then value `2`, left a single file containing only `2`; the nominal append silently overwrote. Repeating the writes to a directory produced two randomly named files containing `1` and `2`, which is genuine dataset append behavior.

The common sink configuration already carries most required information:

- `FileSinkConfig.insert_op` carries `Append`, `Overwrite`, or `Replace`.
- `FileSinkConfig.file_group` can snapshot the old dataset files.
- `FileOutputMode` distinguishes exact-file and directory output.
- `ListingTableUrl::list_all_files` can enumerate existing objects for a format.

## Target contract

| Operation   | Exact-file output                                                        | Directory / partitioned output                                    |
| ----------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `Append`    | Create when absent; error when the object already exists.                | Keep old files and add newly generated files.                     |
| `Overwrite` | Replace the exact object after a successful writer close.                | Write new files, then delete the pre-write snapshot of old files. |
| `Replace`   | Continue to return a clear unsupported error; raw files have no row key. | Continue to return a clear unsupported error for the same reason. |

This contract deliberately does not define byte append. Concatenating bytes is invalid for Parquet and is unsafe for CSV headers, compression streams, and JSON framing.

Directory overwrite is best-effort rather than transactional: object stores do not expose one portable atomic directory swap. Old files should be deleted only after all new writers and the demux task succeed, so a serialization/upload failure leaves the previous dataset intact. A cleanup failure after successful writes can expose old and new files together and must be returned to the caller.

## Files to change

| File / area                                                                                     | Planned change                                                                                            |
| ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `datafusion/core/src/dataframe/mod.rs`                                                          | Remove CSV/JSON early guards and propagate the selected operation into `CopyTo`.                          |
| `datafusion/core/src/dataframe/parquet.rs`                                                      | Remove the Parquet early guard and use the same propagation path.                                         |
| `datafusion/expr/src/logical_plan/dml.rs`                                                       | Add a typed `InsertOp` to `CopyTo`, defaulting existing constructors to `Append`.                         |
| `datafusion/expr/src/logical_plan/builder.rs`                                                   | Add an operation-aware `copy_to` builder path without breaking the existing append-default helper.        |
| `datafusion/core/src/physical_planner.rs`                                                       | Forward the operation, snapshot old format files for overwrite, and populate `FileSinkConfig.file_group`. |
| `datafusion/proto-models/proto/datafusion.proto` and `datafusion/proto/src/logical_plan/mod.rs` | Round-trip the new `CopyTo` operation, using protobuf default `Append` for older payloads.                |
| `datafusion/datasource/src/file_sink_config.rs`                                                 | Centralize append/overwrite/replace execution and post-success cleanup.                                   |
| `datafusion/datasource-csv/src/file_format.rs`                                                  | Permit `Append` and `Overwrite`; retain a `Replace` error.                                                |
| `datafusion/datasource-json/src/file_format.rs`                                                 | Permit `Append` and `Overwrite`; retain a `Replace` error.                                                |
| `datafusion/datasource-parquet/src/file_format.rs`                                              | Permit `Append` and `Overwrite`; retain a `Replace` error.                                                |
| Focused DataFrame, datasource, and proto tests                                                  | Cover the complete operation × output-mode × format matrix.                                               |

Do not enable Arrow IPC overwrite in this change; no DataFrame `write_arrow` API is part of #4986.

## Detailed implementation sequence

### 1. Lock down the public behavior with failing tests

Add a shared test helper that writes two one-row DataFrames with different values. Run it for CSV, newline-delimited JSON, and Parquet.

Required exact-file cases:

- First `Append` creates a readable file.
- Second `Append` to the same path errors and preserves the first value.
- `Overwrite` replaces the first value with the second.
- `Overwrite` of a missing path creates it.
- Empty-input overwrite creates the format's valid empty exact file, matching current single-file guarantees.

Required directory cases:

- Two appends leave both values readable through the directory.
- Overwrite removes every old file captured before the write and leaves only the new value.
- Empty-input overwrite removes old format files and leaves an empty dataset.
- Hive-partitioned overwrite removes stale partitions as well as partitions present in the new input, because `InsertOp::Overwrite` means whole-table replacement, not dynamic-partition overwrite.

Use `TempDir` and verify through DataFusion readers, not only filesystem entry counts.

### 2. Carry `InsertOp` as typed logical-plan state

Add `insert_op: InsertOp` to `CopyTo` and include it in debug output, equality/order/hash semantics where the corresponding operation can affect plan behavior.

Preserve the existing `CopyTo::new` and `LogicalPlanBuilder::copy_to` append default for source compatibility. Add an explicitly named constructor/builder path that accepts `InsertOp`; the DataFrame write methods use it.

Do not encode the operation as a magic format-option string. It is sink behavior, already has a typed enum, and must not leak into `FileFormatFactory::create`.

### 3. Preserve the operation through protobuf

Add an `InsertOp` field to `CopyToNode` using a new field number. Reuse the existing protobuf `InsertOp` enum.

Conversion requirements:

- Missing/default protobuf value decodes as `Append` for backward compatibility.
- All three variants round-trip, even though execution still rejects `Replace`.
- Add a logical-plan round-trip test that distinguishes otherwise-identical append and overwrite plans.

### 4. Snapshot old files during physical planning

Forward `CopyTo.insert_op` instead of constructing every `FileSinkConfig` with `Append`.

For directory overwrite, obtain the target object store and call `ListingTableUrl::list_all_files` with the resolved output extension (including compression suffix when applicable). Convert the resulting metadata to `PartitionedFile`s and store them in `FileSinkConfig.file_group` before execution starts.

The snapshot must be taken before new random output names are generated. Cleanup can then delete exactly the old objects without racing against the current write's objects.

For exact-file output, no post-write deletion list is needed: the completed object-store put replaces that object. Still retain enough target information for the append existence check.

### 5. Enforce the operation once in `FileSink`

Keep format serializers unaware of deletion policy. In the common `FileSink::write_all` flow:

1. Resolve the object store.
2. Reject `Replace` before starting any writer.
3. For exact-file `Append`, check that the target does not exist and fail before consuming input if it does.
4. Run demuxing and all format writers.
5. Only after both sides succeed, delete the old `file_group` snapshot for directory `Overwrite` using `ObjectStore::delete_stream`.
6. Return the row count only after cleanup succeeds.

Document that the exact-file append preflight is not a cross-client transaction. `object_store::buffered::BufWriter` does not expose conditional create for its multipart path, so concurrent writers can still race; fixing that requires a separate object-store writer API change.

### 6. Remove format-local overwrite denials

CSV, JSON, and Parquet factories should accept `Append` and `Overwrite`, construct their existing sinks unchanged, and reject only `Replace` with a message that explains raw file sinks have no replacement key.

Deletion must not be duplicated in each format. All three use `FileSink::write_all`, so one common implementation keeps semantics aligned.

### 7. Cover failure ordering

Use an in-memory or fault-injecting object store to verify:

- A serialization/upload failure does not delete the old file snapshot.
- A deletion failure is returned after the successful write.
- Exact-file overwrite does not run post-write deletion and therefore does not delete the newly replaced object.
- The old-file list is filtered by the sink's extension, so unrelated sibling objects are retained.

## Invariants and non-goals

- `Overwrite` replaces the whole target dataset, not only partitions touched by the input.
- `Replace` remains unsupported for raw files.
- No byte-level append is introduced.
- Writer format options, sorting, partition columns, compression, and single-file selection remain unchanged.
- Do not claim atomic directory replacement across object stores.
- Do not delete old data before successful new-data production.
- Do not broaden this patch into output-manifest or transaction support.

## Risks and mitigations

| Risk                                          | Mitigation                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------- |
| A failed overwrite destroys the old dataset   | Delete only the pre-write snapshot after every writer succeeds.           |
| Exact-file cleanup deletes the new object     | Never post-delete in `SingleFile` mode.                                   |
| New files are mistaken for old files          | Snapshot before execution; generated names are not in that snapshot.      |
| Unrelated files under the directory disappear | List/filter by the resolved format extension and test sibling retention.  |
| Append continues to overwrite exact files     | Add an explicit existence error and preservation test.                    |
| Logical-plan serialization changes behavior   | Add append/overwrite proto round trips with append as the wire default.   |
| Concurrent exact-file writers race            | Document the limitation; conditional multipart creation is separate work. |

## Focused validation commands

```bash
cargo test -p datafusion dataframe:: --lib
cargo test -p datafusion-datasource --lib file_sink
cargo test -p datafusion-datasource-csv --lib
cargo test -p datafusion-datasource-json --lib
cargo test -p datafusion-datasource-parquet --lib
cargo test -p datafusion-proto --test roundtrip_logical_plan copy
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- All three DataFrame file APIs accept and execute `InsertOp::Overwrite`.
- Exact-file append no longer silently replaces existing data.
- Directory append retains old files; directory overwrite removes the pre-write old-file snapshot.
- Empty and hive-partitioned overwrites follow the same whole-dataset contract.
- Failed new writes preserve old files.
- `Replace` fails before writing.
- Logical and protobuf plans preserve the operation.
- CSV, JSON, and Parquet share one cleanup implementation.
