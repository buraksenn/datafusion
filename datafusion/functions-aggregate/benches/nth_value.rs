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

use arrow::array::{ArrayRef, Int64Array};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use datafusion_common::ScalarValue;
use datafusion_expr::function::AccumulatorArgs;
use datafusion_expr::{Accumulator, AggregateUDFImpl, EmitTo};
use datafusion_functions_aggregate::nth_value::NthValueAgg;
use datafusion_physical_expr::expressions::{Literal, col};
use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;

fn seedable_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

struct BenchData {
    values: ArrayRef,
    orderings: ArrayRef,
    group_indices: Vec<usize>,
}

fn generate_bench_data(num_rows: usize, num_groups: usize) -> BenchData {
    let mut rng = seedable_rng();
    let values: Vec<Option<i64>> = (0..num_rows).map(|_| Some(rng.random())).collect();
    let orderings: Vec<i64> = (0..num_rows).map(|_| rng.random()).collect();
    let group_indices: Vec<usize> = (0..num_rows)
        .map(|_| rng.random_range(0..num_groups))
        .collect();

    BenchData {
        values: Arc::new(Int64Array::from(values)),
        orderings: Arc::new(Int64Array::from(orderings)),
        group_indices,
    }
}

fn make_accumulator_args<'a>(
    schema: &'a Schema,
    exprs: &'a [Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>],
    expr_fields: &'a [Arc<Field>],
    order_bys: &'a [PhysicalSortExpr],
) -> AccumulatorArgs<'a> {
    AccumulatorArgs {
        return_field: Arc::new(Field::new("nth_value", DataType::Int64, true)),
        schema,
        expr_fields,
        ignore_nulls: false,
        order_bys,
        is_reversed: false,
        name: "NTH_VALUE(a)",
        is_distinct: false,
        exprs,
    }
}

fn bench_update_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("nth_value_update_batch");

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, false),
    ]));

    let expr_a = col("a", &schema).unwrap();
    let n_literal = Arc::new(Literal::new(ScalarValue::Int64(Some(2))));
    let exprs: Vec<
        Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>,
    > = vec![Arc::clone(&expr_a), n_literal];
    let expr_fields = vec![
        expr_a.return_field(&schema).unwrap(),
        Arc::new(Field::new("n", DataType::Int64, false)),
    ];

    let order_bys = [PhysicalSortExpr {
        expr: col("b", &schema).unwrap(),
        options: SortOptions::default(),
    }];

    let args = make_accumulator_args(&schema, &exprs, &expr_fields, &order_bys);
    let nth_value_fn = NthValueAgg::new();

    for &(num_rows, num_groups) in
        &[(8192, 100), (8192, 1000), (65536, 100), (65536, 1000)]
    {
        let data = generate_bench_data(num_rows, num_groups);
        let values_and_orderings: Vec<ArrayRef> =
            vec![Arc::clone(&data.values), Arc::clone(&data.orderings)];

        let label = format!("rows={num_rows}/groups={num_groups}");
        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(&values_and_orderings, &data.group_indices, num_groups),
            |b, (vals, group_indices, total_groups)| {
                b.iter_batched(
                    || {
                        nth_value_fn
                            .create_groups_accumulator(args.clone())
                            .unwrap()
                    },
                    |mut acc| {
                        acc.update_batch(vals, group_indices, None, *total_groups)
                            .unwrap();
                        acc
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_update_and_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("nth_value_update_and_evaluate");

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, false),
    ]));

    let nth_value_fn = NthValueAgg::new();

    for &n_val in &[1i64, 2, 10] {
        let expr_a = col("a", &schema).unwrap();
        let n_literal = Arc::new(Literal::new(ScalarValue::Int64(Some(n_val))));
        let exprs: Vec<
            Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>,
        > = vec![Arc::clone(&expr_a), n_literal];
        let expr_fields = vec![
            expr_a.return_field(&schema).unwrap(),
            Arc::new(Field::new("n", DataType::Int64, false)),
        ];

        let order_bys = [PhysicalSortExpr {
            expr: col("b", &schema).unwrap(),
            options: SortOptions::default(),
        }];

        let args = make_accumulator_args(&schema, &exprs, &expr_fields, &order_bys);

        for &(num_rows, num_groups) in &[(8192, 100), (65536, 1000)] {
            let data = generate_bench_data(num_rows, num_groups);
            let values_and_orderings: Vec<ArrayRef> =
                vec![Arc::clone(&data.values), Arc::clone(&data.orderings)];

            let label = format!("n={n_val}/rows={num_rows}/groups={num_groups}");
            group.bench_with_input(
                BenchmarkId::from_parameter(&label),
                &(&values_and_orderings, &data.group_indices, num_groups),
                |b, (vals, group_indices, total_groups)| {
                    b.iter_batched(
                        || {
                            nth_value_fn
                                .create_groups_accumulator(args.clone())
                                .unwrap()
                        },
                        |mut acc| {
                            acc.update_batch(vals, group_indices, None, *total_groups)
                                .unwrap();
                            acc.evaluate(EmitTo::All).unwrap()
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

fn bench_multi_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("nth_value_multi_batch");

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, false),
    ]));

    let expr_a = col("a", &schema).unwrap();
    let n_literal = Arc::new(Literal::new(ScalarValue::Int64(Some(2))));
    let exprs: Vec<
        Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>,
    > = vec![Arc::clone(&expr_a), n_literal];
    let expr_fields = vec![
        expr_a.return_field(&schema).unwrap(),
        Arc::new(Field::new("n", DataType::Int64, false)),
    ];

    let order_bys = [PhysicalSortExpr {
        expr: col("b", &schema).unwrap(),
        options: SortOptions::default(),
    }];

    let args = make_accumulator_args(&schema, &exprs, &expr_fields, &order_bys);
    let nth_value_fn = NthValueAgg::new();

    let num_groups = 1000;
    let batch_size = 8192;
    let num_batches = 8;

    let batches: Vec<BenchData> = (0..num_batches)
        .map(|_| generate_bench_data(batch_size, num_groups))
        .collect();

    group.bench_function(
        format!("{num_batches}x{batch_size} rows/{num_groups} groups"),
        |b| {
            b.iter_batched(
                || {
                    nth_value_fn
                        .create_groups_accumulator(args.clone())
                        .unwrap()
                },
                |mut acc| {
                    for batch in &batches {
                        let vals: Vec<ArrayRef> =
                            vec![Arc::clone(&batch.values), Arc::clone(&batch.orderings)];
                        acc.update_batch(&vals, &batch.group_indices, None, num_groups)
                            .unwrap();
                    }
                    acc.evaluate(EmitTo::All).unwrap()
                },
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn bench_state_and_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("nth_value_state_and_merge");

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, false),
    ]));

    let expr_a = col("a", &schema).unwrap();
    let n_literal = Arc::new(Literal::new(ScalarValue::Int64(Some(2))));
    let exprs: Vec<
        Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>,
    > = vec![Arc::clone(&expr_a), n_literal];
    let expr_fields = vec![
        expr_a.return_field(&schema).unwrap(),
        Arc::new(Field::new("n", DataType::Int64, false)),
    ];

    let order_bys = [PhysicalSortExpr {
        expr: col("b", &schema).unwrap(),
        options: SortOptions::default(),
    }];

    let args = make_accumulator_args(&schema, &exprs, &expr_fields, &order_bys);
    let nth_value_fn = NthValueAgg::new();

    let num_groups = 500;
    let data = generate_bench_data(8192, num_groups);
    let values_and_orderings: Vec<ArrayRef> =
        vec![Arc::clone(&data.values), Arc::clone(&data.orderings)];

    let mut partial = nth_value_fn
        .create_groups_accumulator(args.clone())
        .unwrap();
    partial
        .update_batch(&values_and_orderings, &data.group_indices, None, num_groups)
        .unwrap();
    let state = partial.state(EmitTo::All).unwrap();
    let merge_indices: Vec<usize> = (0..num_groups).collect();

    group.bench_function("state_serialize/500 groups", |b| {
        b.iter_batched(
            || {
                let mut acc = nth_value_fn
                    .create_groups_accumulator(args.clone())
                    .unwrap();
                acc.update_batch(
                    &values_and_orderings,
                    &data.group_indices,
                    None,
                    num_groups,
                )
                .unwrap();
                acc
            },
            |mut acc| acc.state(EmitTo::All).unwrap(),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("merge_batch/500 groups", |b| {
        b.iter_batched(
            || {
                nth_value_fn
                    .create_groups_accumulator(args.clone())
                    .unwrap()
            },
            |mut acc| {
                acc.merge_batch(&state, &merge_indices, None, num_groups)
                    .unwrap();
                acc
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

struct PerGroupData {
    cols: Vec<Vec<ArrayRef>>,
}

fn partition_by_group(data: &BenchData, num_groups: usize) -> PerGroupData {
    let vals = data.values.as_any().downcast_ref::<Int64Array>().unwrap();
    let ords = data
        .orderings
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();

    let mut group_vals: Vec<Vec<i64>> = vec![Vec::new(); num_groups];
    let mut group_ords: Vec<Vec<i64>> = vec![Vec::new(); num_groups];

    for (idx, &gidx) in data.group_indices.iter().enumerate() {
        group_vals[gidx].push(vals.value(idx));
        group_ords[gidx].push(ords.value(idx));
    }

    let cols: Vec<Vec<ArrayRef>> = group_vals
        .into_iter()
        .zip(group_ords)
        .map(|(v, o)| {
            let n_col = vec![2i64; v.len()];
            vec![
                Arc::new(Int64Array::from(v)) as ArrayRef,
                Arc::new(Int64Array::from(n_col)) as ArrayRef,
                Arc::new(Int64Array::from(o)) as ArrayRef,
            ]
        })
        .collect();

    PerGroupData { cols }
}

fn bench_row_vs_groups(c: &mut Criterion) {
    let mut group = c.benchmark_group("nth_value_row_vs_groups");

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, false),
    ]));

    let expr_a = col("a", &schema).unwrap();
    let n_literal = Arc::new(Literal::new(ScalarValue::Int64(Some(2))));
    let exprs: Vec<
        Arc<dyn datafusion_physical_expr_common::physical_expr::PhysicalExpr>,
    > = vec![Arc::clone(&expr_a), n_literal];
    let expr_fields = vec![
        expr_a.return_field(&schema).unwrap(),
        Arc::new(Field::new("n", DataType::Int64, false)),
    ];

    let order_bys = [PhysicalSortExpr {
        expr: col("b", &schema).unwrap(),
        options: SortOptions::default(),
    }];

    let args = make_accumulator_args(&schema, &exprs, &expr_fields, &order_bys);
    let nth_value_fn = NthValueAgg::new();

    for &(num_rows, num_groups) in &[(8192, 100), (8192, 1000)] {
        let data = generate_bench_data(num_rows, num_groups);
        let per_group = partition_by_group(&data, num_groups);
        let values_and_orderings: Vec<ArrayRef> =
            vec![Arc::clone(&data.values), Arc::clone(&data.orderings)];

        let label = format!("rows={num_rows}/groups={num_groups}");

        group.bench_with_input(
            BenchmarkId::new("groups_accumulator", &label),
            &(&values_and_orderings, &data.group_indices, num_groups),
            |b, (vals, group_indices, total_groups)| {
                b.iter_batched(
                    || {
                        nth_value_fn
                            .create_groups_accumulator(args.clone())
                            .unwrap()
                    },
                    |mut acc| {
                        acc.update_batch(vals, group_indices, None, *total_groups)
                            .unwrap();
                        acc.evaluate(EmitTo::All).unwrap()
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("row_accumulator", &label),
            &per_group,
            |b, per_group| {
                b.iter_batched(
                    || {
                        (0..num_groups)
                            .map(|_| nth_value_fn.accumulator(args.clone()).unwrap())
                            .collect::<Vec<Box<dyn Accumulator>>>()
                    },
                    |mut accumulators| {
                        for (gidx, cols) in per_group.cols.iter().enumerate() {
                            accumulators[gidx].update_batch(cols).unwrap();
                        }
                        let results: Vec<_> = accumulators
                            .iter_mut()
                            .map(|acc| acc.evaluate().unwrap())
                            .collect();
                        results
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_update_batch,
    bench_update_and_evaluate,
    bench_multi_batch,
    bench_state_and_merge,
    bench_row_vs_groups,
);
criterion_main!(benches);
