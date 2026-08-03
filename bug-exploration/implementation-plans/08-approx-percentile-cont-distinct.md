# Implementation plan: `APPROX_PERCENTILE_CONT(DISTINCT)`

Source issue: [#2406](https://github.com/apache/datafusion/issues/2406)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

## Objective

Make `approx_percentile_cont(DISTINCT value, percentile [, centroids])` compute a t-digest over the set of unique non-NULL input values rather than silently processing duplicates.

Both inline syntax and `WITHIN GROUP` syntax must preserve descending-order percentile reversal and the optional centroid limit.

## Current behavior and root cause

`ApproxPercentileCont::accumulator` always creates `ApproxPercentileAccumulator`. Neither it nor `create_accumulator` reads `AccumulatorArgs.is_distinct`.

The normal accumulator sorts every batch and merges every value into a `TDigest`, so duplicate rows retain their full weight. The logical expression still displays `DISTINCT`, producing a silent wrong result when the single-distinct optimizer does not rewrite the query.

Observed reproduction:

```sql
WITH t(x) AS (VALUES (1), (1), (1), (1), (9))
SELECT
  approx_percentile_cont(x, 0.5) AS plain,
  approx_percentile_cont(DISTINCT x, 0.5) AS actual_distinct,
  (SELECT approx_percentile_cont(x, 0.5)
   FROM (SELECT DISTINCT x FROM t)) AS expected_distinct
FROM t;
```

Observed: `plain = 1`, `actual_distinct = 1`, `expected_distinct = 5`.

A t-digest partial state cannot be used for DISTINCT. Once duplicates have been compressed into centroids, a final stage cannot tell whether equal values came from different partitions. Distinct partial state must retain raw unique values.

## Files to change

| File                                                           | Planned change                                                                               |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `datafusion/functions-aggregate/src/approx_percentile_cont.rs` | Add typed distinct accumulators, distinct state fields, and shared argument/config parsing.  |
| `datafusion/functions-aggregate-common/src/utils.rs`           | Reuse or minimally extend `GenericDistinctBuffer` if a common dispatch helper is worthwhile. |
| `datafusion/sqllogictest/test_files/aggregate.slt`             | Add mixed, grouped, ordering, NULL, and multi-partition regressions.                         |

Keep the separate small-dataset t-digest interpolation concern in #21528 out of this patch. DISTINCT must use the same t-digest estimator as the non-distinct function after deduplication.

## Detailed implementation sequence

### 1. Add failing semantic regressions

Protect the five-row reproduction above and compare against an explicit `SELECT DISTINCT` subquery.

Also cover:

- `Float16`, `Float32`, and `Float64` dispatch where practical.
- NULLs and all-NULL input.
- Duplicate values split across partitions.
- Grouped aggregation.
- Optional centroid count.
- Ascending and descending `WITHIN GROUP` order.
- Inline alternate syntax.
- Multiple/mixed aggregates to bypass `SingleDistinctToGroupBy`.

Do not assert that an approximate percentile must be an observed value; assert equivalence between the distinct call and the same estimator fed explicitly deduplicated values.

### 2. Separate call configuration from accumulator storage

Refactor `create_accumulator` so percentile validation, descending reversal, return type, and optional max-size parsing produce a small internal configuration independent of the chosen state representation.

The existing non-distinct path then constructs `ApproxPercentileAccumulator` from that configuration unchanged. The distinct path constructs a typed set accumulator from the same configuration.

This avoids duplicating:

- percentile literal validation;
- `1.0 - percentile` handling for descending order;
- centroid-limit validation;
- return-type selection.

### 3. Add a raw-value distinct accumulator

Implement `DistinctApproxPercentileAccumulator<T>` for each accepted Arrow float type using `GenericDistinctBuffer<T>`.

Behavior:

- `update_batch` inserts non-NULL values into the set.
- `state` emits one `List<T>` containing raw unique values.
- `merge_batch` unions list states, removing cross-partition duplicates.
- `evaluate` converts the unique set to `f64`, sorts it, creates a fresh `TDigest` with the configured max size, and estimates the configured percentile.
- `size` includes the set, return type, and configuration.

Use Arrow/DataFusion float equality and hashing consistently with other distinct numeric aggregates, including NaN and signed zero.

Do not maintain both a set and an incrementally updated digest: that doubles memory and the digest cannot be trusted until global deduplication is complete.

### 4. Branch intermediate state fields

For plain calls, retain the six t-digest state fields (`max_size`, `sum`, `count`, `max`, `min`, `centroids`).

For distinct calls, expose a single `List<input float type>` state matching the typed distinct buffer.

The state field must use the coerced input type, including Float16/Float32 rather than always advertising Float64.

### 5. Keep optimizer and ordering contracts intact

The UDAF continues to advertise `supports_within_group_clause() == true`.

Descending order affects the requested quantile, not set ordering. Deduplicate values first, then evaluate the same adjusted percentile. The physical ORDER BY requirement may sort input, but correctness must not depend on batch or partition order.

No groups accumulator should be advertised for the distinct path unless a true per-group distinct implementation exists.

### 6. Add direct merge tests

Create partial accumulators with overlapping values such as `{1, 2}` and `{2, 9}`. Merge their raw states and compare the final estimate with a single accumulator fed `{1, 2, 9}`.

Test both default and custom centroid limits and at least one descending-order configuration.

## Invariants and non-goals

- DISTINCT removes duplicate coerced values globally, across batches and partitions.
- NULL does not become a distinct percentile value.
- The percentile and centroid arguments remain literals and are not part of the per-row set.
- The approximation algorithm and interpolation semantics remain unchanged after deduplication.
- Do not solve weighted percentile or `approx_median` in this file; each has a separate implementation plan.
- Do not add Decimal support.

## Risks and mitigations

| Risk                                                    | Mitigation                                                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| TDigest state loses cross-partition identity            | Use raw list state for distinct calls and build the digest only after union.                            |
| Plain and distinct calls parse options differently      | Produce one shared immutable call configuration.                                                        |
| State type mismatches Float16/Float32                   | Dispatch state fields and accumulators from the coerced input type.                                     |
| Hash iteration changes approximate output               | Sort unique values before building the digest.                                                          |
| Existing small-data behavior gets conflated with #21528 | Compare against explicit deduplication using the same estimator, not a different percentile definition. |
| Optimizer rewrite masks the defect                      | Use mixed/multiple aggregates in SQL regressions.                                                       |

## Focused validation commands

```bash
cargo test -p datafusion-functions-aggregate approx_percentile_cont --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- The five-row reproduction returns the same value as explicit deduplication.
- Duplicate values crossing partial states are counted once.
- Plain calls keep the existing t-digest state and results.
- NULL, type dispatch, custom centroid count, grouping, and descending order are covered.
- The distinct intermediate state contains raw unique values, not compressed centroids.
