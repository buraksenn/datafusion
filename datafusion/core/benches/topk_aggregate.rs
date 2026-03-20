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

mod data_utils;

use arrow::array::Int64Builder;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use criterion::{Criterion, criterion_group, criterion_main};
use data_utils::make_data;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::SessionContext;
use datafusion::{datasource::MemTable, error::Result};
use datafusion_execution::config::SessionConfig;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::Runtime;

const LIMIT: usize = 10;

/// Create deterministic data for DISTINCT benchmarks with predictable trace_ids
/// This ensures consistent results across benchmark runs
fn make_distinct_data(
    partition_cnt: i32,
    sample_cnt: i32,
) -> Result<(Arc<Schema>, Vec<Vec<RecordBatch>>)> {
    let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
    let total_samples = partition_cnt as usize * sample_cnt as usize;
    let mut ids = Vec::new();
    for i in 0..total_samples {
        ids.push(i as i64);
    }
    ids.shuffle(&mut rng);

    let mut global_idx = 0;
    let schema = test_distinct_schema();
    let mut partitions = vec![];
    for _ in 0..partition_cnt {
        let mut id_builder = Int64Builder::new();

        for _ in 0..sample_cnt {
            let id = ids[global_idx];
            id_builder.append_value(id);
            global_idx += 1;
        }

        let id_col = Arc::new(id_builder.finish());
        let batch = RecordBatch::try_new(schema.clone(), vec![id_col])?;
        partitions.push(vec![batch]);
    }

    Ok((schema, partitions))
}

/// Returns a Schema for distinct benchmarks with i64 trace_id
fn test_distinct_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
}

async fn create_context(
    partition_cnt: i32,
    sample_cnt: i32,
    asc: bool,
    use_topk: bool,
    use_view: bool,
) -> Result<SessionContext> {
    let (schema, parts) = make_data(partition_cnt, sample_cnt, asc, use_view).unwrap();
    let mem_table = Arc::new(MemTable::try_new(schema, parts).unwrap());

    // Create the DataFrame
    let mut cfg = SessionConfig::new();
    let opts = cfg.options_mut();
    opts.optimizer.enable_topk_aggregation = use_topk;
    let ctx = SessionContext::new_with_config(cfg);
    let _ = ctx.register_table("traces", mem_table)?;

    Ok(ctx)
}

async fn create_context_distinct(
    partition_cnt: i32,
    sample_cnt: i32,
    use_topk: bool,
) -> Result<SessionContext> {
    // Use deterministic data generation for DISTINCT queries to ensure consistent results
    let (schema, parts) = make_distinct_data(partition_cnt, sample_cnt).unwrap();
    let mem_table = Arc::new(MemTable::try_new(schema, parts).unwrap());

    // Create the DataFrame
    let mut cfg = SessionConfig::new();
    let opts = cfg.options_mut();
    opts.optimizer.enable_topk_aggregation = use_topk;
    let ctx = SessionContext::new_with_config(cfg);
    let _ = ctx.register_table("traces", mem_table)?;

    Ok(ctx)
}

fn run(rt: &Runtime, ctx: SessionContext, limit: usize, use_topk: bool, asc: bool) {
    black_box(rt.block_on(async { aggregate(ctx, limit, use_topk, asc).await })).unwrap();
}

fn run_string(rt: &Runtime, ctx: SessionContext, limit: usize, use_topk: bool) -> String {
    black_box(rt.block_on(async { aggregate_string(ctx, limit, use_topk).await }))
        .unwrap()
}

fn run_distinct(
    rt: &Runtime,
    ctx: SessionContext,
    limit: usize,
    use_topk: bool,
    asc: bool,
) {
    black_box(rt.block_on(async { aggregate_distinct(ctx, limit, use_topk, asc).await }))
        .unwrap();
}

async fn aggregate(
    ctx: SessionContext,
    limit: usize,
    use_topk: bool,
    asc: bool,
) -> Result<()> {
    let sql = format!(
        "select max(timestamp_ms) from traces group by trace_id order by max(timestamp_ms) desc limit {limit};"
    );
    let df = ctx.sql(sql.as_str()).await?;
    let plan = df.create_physical_plan().await?;
    let actual_phys_plan = displayable(plan.as_ref()).indent(true).to_string();
    assert_eq!(
        actual_phys_plan.contains(&format!("lim=[{limit}]")),
        use_topk
    );

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.len(), 1);
    let batch = batches.first().unwrap();
    assert_eq!(batch.num_rows(), LIMIT);

    let actual = format!("{}", pretty_format_batches(&batches)?).to_lowercase();
    let expected_asc = r#"
+--------------------------+
| max(traces.timestamp_ms) |
+--------------------------+
| 16909009999999           |
| 16909009999998           |
| 16909009999997           |
| 16909009999996           |
| 16909009999995           |
| 16909009999994           |
| 16909009999993           |
| 16909009999992           |
| 16909009999991           |
| 16909009999990           |
+--------------------------+
        "#
    .trim();
    if asc {
        assert_eq!(actual.trim(), expected_asc);
    }

    Ok(())
}

/// Benchmark for string aggregate functions with topk optimization.
/// This tests grouping by a numeric column (timestamp_ms) and aggregating
/// a string column (trace_id) with Utf8 or Utf8View data types.
async fn aggregate_string(
    ctx: SessionContext,
    limit: usize,
    use_topk: bool,
) -> Result<String> {
    let sql = format!(
        "select max(trace_id) from traces group by timestamp_ms order by max(trace_id) desc limit {limit};"
    );
    let df = ctx.sql(sql.as_str()).await?;
    let plan = df.create_physical_plan().await?;
    let actual_phys_plan = displayable(plan.as_ref()).indent(true).to_string();
    assert_eq!(
        actual_phys_plan.contains(&format!("lim=[{limit}]")),
        use_topk
    );

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.len(), 1);
    let batch = batches.first().unwrap();
    assert_eq!(batch.num_rows(), LIMIT);

    Ok(format!("{}", pretty_format_batches(&batches)?))
}

async fn aggregate_distinct(
    ctx: SessionContext,
    limit: usize,
    use_topk: bool,
    asc: bool,
) -> Result<()> {
    let order_direction = if asc { "asc" } else { "desc" };
    let sql = format!(
        "select id from traces group by id order by id {order_direction} limit {limit};"
    );
    let df = ctx.sql(sql.as_str()).await?;
    let plan = df.create_physical_plan().await?;
    let actual_phys_plan = displayable(plan.as_ref()).indent(true).to_string();
    assert_eq!(
        actual_phys_plan.contains(&format!("lim=[{limit}]")),
        use_topk
    );
    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.len(), 1);
    let batch = batches.first().unwrap();
    assert_eq!(batch.num_rows(), LIMIT);

    let actual = format!("{}", pretty_format_batches(&batches)?).to_lowercase();

    let expected_asc = r#"
+----+
| id |
+----+
| 0  |
| 1  |
| 2  |
| 3  |
| 4  |
| 5  |
| 6  |
| 7  |
| 8  |
| 9  |
+----+
"#
    .trim();

    let expected_desc = r#"
+---------+
| id      |
+---------+
| 9999999 |
| 9999998 |
| 9999997 |
| 9999996 |
| 9999995 |
| 9999994 |
| 9999993 |
| 9999992 |
| 9999991 |
| 9999990 |
+---------+
"#
    .trim();

    // Verify exact results match expected values
    if asc {
        assert_eq!(
            actual.trim(),
            expected_asc,
            "Ascending DISTINCT results do not match expected values"
        );
    } else {
        assert_eq!(
            actual.trim(),
            expected_desc,
            "Descending DISTINCT results do not match expected values"
        );
    }

    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let limit = LIMIT;
    let partitions = 10;
    let samples = 1_000_000;
    let total_rows = partitions * samples;

    // Numeric aggregate benchmarks
    // (asc, use_topk, use_view)
    let numeric_cases: &[(bool, bool, bool, &str)] = &[
        (false, false, false, "aggregate {rows} time-series rows"),
        (true, false, false, "aggregate {rows} worst-case rows"),
        (
            false,
            true,
            false,
            "top k={limit} aggregate {rows} time-series rows",
        ),
        (
            true,
            true,
            false,
            "top k={limit} aggregate {rows} worst-case rows",
        ),
        (
            false,
            true,
            true,
            "top k={limit} aggregate {rows} time-series rows [Utf8View]",
        ),
        (
            true,
            true,
            true,
            "top k={limit} aggregate {rows} worst-case rows [Utf8View]",
        ),
    ];
    for &(asc, use_topk, use_view, name_tpl) in numeric_cases {
        let name = name_tpl
            .replace("{rows}", &total_rows.to_string())
            .replace("{limit}", &limit.to_string());
        let ctx = rt
            .block_on(create_context(partitions, samples, asc, use_topk, use_view))
            .unwrap();
        c.bench_function(&name, |b| {
            b.iter(|| run(&rt, ctx.clone(), limit, use_topk, asc))
        });
    }

    for asc in [false, true] {
        for use_topk in [false, true] {
            let ctx_utf8 = rt
                .block_on(create_context(partitions, samples, asc, use_topk, false))
                .unwrap();
            let ctx_view = rt
                .block_on(create_context(partitions, samples, asc, use_topk, true))
                .unwrap();
            let result_utf8 = run_string(&rt, ctx_utf8, limit, use_topk);
            let result_view = run_string(&rt, ctx_view, limit, use_topk);
            assert_eq!(
                result_utf8, result_view,
                "Utf8 vs Utf8View mismatch for asc={asc}, use_topk={use_topk}"
            );
        }
    }

    // String aggregate benchmarks
    // (asc, use_topk, use_view)
    let string_cases: &[(bool, bool, bool)] = &[
        (false, false, false),
        (true, false, false),
        (false, false, true),
        (true, false, true),
        (false, true, false),
        (true, true, false),
        (false, true, true),
        (true, true, true),
    ];
    for &(asc, use_topk, use_view) in string_cases {
        let scenario = if asc { "worst-case" } else { "time-series" };
        let type_label = if use_view { "Utf8View" } else { "Utf8" };
        let name = if use_topk {
            format!(
                "top k={limit} string aggregate {total_rows} {scenario} rows [{type_label}]"
            )
        } else {
            format!("string aggregate {total_rows} {scenario} rows [{type_label}]")
        };
        let ctx = rt
            .block_on(create_context(partitions, samples, asc, use_topk, use_view))
            .unwrap();
        c.bench_function(&name, |b| {
            b.iter(|| run_string(&rt, ctx.clone(), limit, use_topk))
        });
    }

    // DISTINCT benchmarks
    for use_topk in [false, true] {
        let ctx = rt.block_on(async {
            create_context_distinct(partitions, samples, use_topk)
                .await
                .unwrap()
        });
        let topk_label = if use_topk { "TopK" } else { "no TopK" };
        for asc in [false, true] {
            let dir = if asc { "asc" } else { "desc" };
            let name = format!("distinct {total_rows} rows {dir} [{topk_label}]");
            c.bench_function(&name, |b| {
                b.iter(|| run_distinct(&rt, ctx.clone(), limit, use_topk, asc))
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
