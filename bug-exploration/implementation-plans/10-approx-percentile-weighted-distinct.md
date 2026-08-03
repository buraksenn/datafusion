# Implementation plan: weighted `APPROX_PERCENTILE_CONT(DISTINCT)`

Source issue: [#2406](https://github.com/apache/datafusion/issues/2406)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

## Objective

Support DISTINCT for `approx_percentile_cont_with_weight` by eliminating duplicate `(value, weight)` input pairs before constructing t-digest centroids.

Both supported SQL forms must behave consistently:

```sql
approx_percentile_cont_with_weight(value, weight, percentile [, centroids])
approx_percentile_cont_with_weight(weight, percentile [, centroids])
  WITHIN GROUP (ORDER BY value)
```

## Current behavior and root cause

`ApproxPercentileContWithWeight::accumulator` explicitly returns:

```text
approx_percentile_cont_with_weight(DISTINCT) aggregations are not available
```

The normal accumulator filters rows where either value or weight is NULL, converts both arrays to `f64`, creates one t-digest centroid per pair, and merges those digests. Its serialized state is only the resulting t-digest.

A digest cannot implement global DISTINCT: equal pairs compressed independently in two partial partitions cannot be recognized and removed at the final stage. The distinct path must serialize raw unique pairs.

## DISTINCT key semantics

SQL DISTINCT applies to the aggregate's row-varying argument tuple. For this function the key is `(value, weight)`:

- Two identical `(value, weight)` rows collapse to one centroid.
- Equal values with different weights remain separate pairs.
- Percentile and centroid-limit arguments are validated literals and are identical for every row, so storing them in each distinct key adds no information.
- A row with NULL value or NULL weight is excluded before key insertion, matching current weighted behavior.

This contract must be explicit in tests and documentation; deduplicating only `value` would silently discard legitimate distinct weights.

## Files to change

| File                                                                          | Planned change                                                                                        |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `datafusion/functions-aggregate-common/src/aggregate/distinct_pairs.rs` (new) | Provide a reusable Float64-pair distinct buffer with raw list state and optional multiplicity counts. |
| `datafusion/functions-aggregate-common/src/aggregate.rs` / `lib.rs`           | Export the helper crate-privately/publicly as appropriate for aggregate implementations.              |
| `datafusion/functions-aggregate/src/approx_percentile_cont_with_weight.rs`    | Add distinct state fields, accumulator dispatch, evaluation, and tests.                               |
| `datafusion/sqllogictest/test_files/aggregate.slt`                            | Add syntax, grouping, NULL, and cross-partition regressions.                                          |

The pair-buffer utility is also useful for covariance, correlation, and regression plans. Keep it storage-focused; do not build a highly generic aggregate adapter.

## Detailed implementation sequence

### 1. Specify expected pair behavior with failing tests

Use data where each interpretation differs. For example:

```text
(value, weight)
(1, 1)
(1, 1)   -- exact duplicate: remove
(1, 3)   -- same value, different weight: keep
(9, 1)
```

Compare the distinct function with the plain function over `SELECT DISTINCT value, weight`.

Cover both inline and `WITHIN GROUP` syntax, and include a mixed aggregate so the physical distinct path is exercised.

### 2. Add a reusable distinct pair buffer

Implement a buffer keyed by `(Hashable<f64>, Hashable<f64>)` using DataFusion's `RandomState`.

Required operations:

- Insert aligned non-NULL pairs from two `Float64Array`s.
- Serialize the unique keys as two aligned `List<Float64>` states.
- Merge aligned list states and deduplicate across partitions.
- Materialize two arrays in a deterministic order for final evaluation.
- Report owned memory.
- Optionally maintain per-key multiplicity for `retract_batch`; serialized partial state still needs only one copy of each key.

Reject mismatched array/list lengths as an internal error rather than truncating through `zip`.

Sort materialized pairs with `f64::total_cmp` before feeding numerical accumulators so hash iteration order does not create run-to-run floating differences.

### 3. Branch weighted state fields

Non-distinct calls keep the six t-digest fields delegated from `ApproxPercentileCont`.

Distinct calls return two aligned list fields, named for distinct values and distinct weights. Their element type is the coerced type actually consumed by the accumulator (`Float64` today).

State field ordering must match `DistinctPairBuffer::state()` exactly.

### 4. Add the distinct weighted accumulator

Create `DistinctApproxPercentileWithWeightAccumulator` containing:

- the pair buffer;
- percentile after descending-order adjustment;
- t-digest max size;
- return type.

`update_batch` filters NULL pairs and inserts unique `(value, weight)` keys. `merge_batch` unions raw states. `evaluate` materializes the unique pairs, creates centroids with their retained weights, merges them into a fresh digest, and estimates the percentile.

Do not forward `is_distinct: true` into the existing plain `ApproxPercentileAccumulator`; it does not inspect the flag.

### 5. Preserve validation and ordering

Refactor enough configuration parsing to share:

- percentile literal checks;
- centroid-limit checks;
- descending-order percentile reversal;
- return type.

Keep current validation of weight types and current handling of invalid/negative weights unchanged unless an existing test proves it must participate in this feature.

### 6. Test partial and grouped execution

Create two partial states that both contain `(1, 1)` and unique additional pairs. After merge, assert the shared pair contributes one centroid.

Also cover:

- grouped aggregation;
- NULL value and NULL weight independently;
- equal value/different weight;
- equal weight/different value;
- custom centroid count;
- all-NULL pairs;
- deterministic state/evaluation under different batch orderings.

## Invariants and non-goals

- DISTINCT key is the coerced `(value, weight)` pair.
- NULL in either key component excludes the row.
- Raw pair state is required until global merge completes.
- Non-distinct weighted t-digest behavior remains unchanged.
- Do not change weight validation semantics in this patch.
- Do not conflate this with unweighted distinct percentile or #21528.
- Do not create a generic wrapper that obscures each aggregate's state schema.

## Risks and mitigations

| Risk                                                  | Mitigation                                                               |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| Deduplicating only values drops distinct weights      | Key and test the full `(value, weight)` pair.                            |
| Two list states become misaligned                     | Build both from one ordered key iteration and validate lengths on merge. |
| Digest state cannot remove cross-partition duplicates | Serialize raw pair lists for distinct calls.                             |
| Hash order changes approximate output                 | Sort pairs deterministically before centroid construction.               |
| Plain path regresses during refactor                  | Keep its accumulator/state unchanged and retain controls.                |

## Focused validation commands

```bash
cargo test -p datafusion-functions-aggregate-common distinct_pairs --lib
cargo test -p datafusion-functions-aggregate approx_percentile_cont_with_weight --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- Weighted distinct calls no longer return the current unsupported error.
- Results match the same weighted estimator over explicitly deduplicated `(value, weight)` pairs.
- Equal values with different weights remain distinct.
- Cross-partition duplicates collapse globally.
- NULL, grouping, syntax, and custom-centroid cases are covered.
- Non-distinct weighted behavior and state remain unchanged.
