# Implementation plan: `COVAR_SAMP(DISTINCT)` and `COVAR_POP(DISTINCT)`

Source issue: [#2406](https://github.com/apache/datafusion/issues/2406)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

## Objective

Make sample and population covariance apply DISTINCT to the full non-NULL `(y, x)` pair before computing their statistics.

The result must equal the corresponding covariance over `SELECT DISTINCT y, x` for grouped, partitioned, and window-capable execution.

## Current behavior and root cause

Both `CovarianceSample::accumulator` and `CovariancePopulation::accumulator` ignore their `AccumulatorArgs` parameter and always create `CovarianceAccumulator`.

The logical expression preserves and displays `DISTINCT`, but every duplicate pair is processed. Observed:

```sql
WITH t(y, x) AS (
  VALUES (1.0, 1.0), (1.0, 1.0), (9.0, 3.0)
)
SELECT
  covar_samp(DISTINCT y, x) AS actual_distinct,
  (SELECT covar_samp(y, x) FROM (SELECT DISTINCT y, x FROM t)) AS expected_distinct
FROM t;
```

Observed: `5.333333333333334` versus expected `8.0`.

The current partial state (`count`, two means, covariance constant) has already incorporated multiplicity. It cannot be merged into a globally distinct result. A distinct branch must retain raw unique pairs until final aggregation.

## Files to change

| File                                                                                    | Planned change                                                                                    |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `datafusion/functions-aggregate-common/src/aggregate/distinct_pairs.rs` (new or shared) | Store, merge, materialize, and optionally retract unique Float64 pairs.                           |
| `datafusion/functions-aggregate/src/covariance.rs`                                      | Add distinct state fields, sample/population accumulator dispatch, wrapper evaluation, and tests. |
| `datafusion/sqllogictest/test_files/aggregate.slt`                                      | Add semantic, grouped, NULL, and cross-partition regressions.                                     |

If the weighted-percentile plan lands first, reuse its `DistinctPairBuffer`; otherwise introduce the helper here and make later plans consume it.

## DISTINCT key and NULL semantics

The key is the ordered pair `(y, x)` after existing coercion to `Float64`.

- `(1, 2)` and `(2, 1)` are different.
- A duplicate is removed only when both values compare equal under DataFusion's float distinct semantics.
- If either component is NULL, the row is excluded before insertion, matching plain covariance.
- NaN and signed-zero behavior must match other DataFusion distinct float aggregates through the shared `Hashable` wrapper.

## Detailed implementation sequence

### 1. Add failing end-to-end tests

Protect the reproduction above for `covar_samp` and add `covar_pop` on the same input.

Add:

- Mixed plain/distinct calls.
- Two distinct calls over different pairs.
- Grouped input.
- Duplicate pairs crossing partition/batch boundaries.
- NULL in `y`, NULL in `x`, and both NULL.
- Empty and one-distinct-pair cases for sample/population edge semantics.

Expected distinct results should come from explicit tuple deduplication, not hand-maintained formulas alone.

### 2. Implement or reuse `DistinctPairBuffer`

Use a hash key of `(Hashable<f64>, Hashable<f64>)` and retain raw unique pairs.

Expose only storage operations:

- `update_batch(y, x)` with aligned-length validation and NULL filtering;
- `state()` as two aligned `List<Float64>` scalars;
- `merge_batch()` that unions every pair from partial lists;
- `values()` / `to_arrays()` in deterministic `total_cmp` order;
- `size()`;
- optional multiplicity-aware retract support.

Do not make the helper know covariance formulas or return types.

### 3. Add a distinct covariance wrapper

`DistinctCovarianceAccumulator` owns a pair buffer and `StatsType`.

- Update and merge delegate to the buffer.
- State returns the two raw pair lists.
- Evaluate materializes the unique pair arrays, feeds one fresh `CovarianceAccumulator` configured for sample or population, and returns its value.
- Size includes both wrapper and buffer.

Reusing the stable existing covariance algorithm avoids a second statistical implementation.

If a specialized retract path is added, use per-pair multiplicity: remove a key only when the last equal row leaves the frame. Otherwise leave `supports_retract_batch` false and let the window engine recompute; never remove a key on the first duplicate retraction.

### 4. Branch UDAF state and accumulator selection

For both covariance variants:

- Plain calls retain four scalar state fields and `CovarianceAccumulator`.
- Distinct calls expose two list state fields and create `DistinctCovarianceAccumulator` with the matching `StatsType`.

No distinct groups accumulator exists, so do not advertise one.

### 5. Add direct state-merge tests

Build partial sets `{(1,1), (2,2)}` and `{(1,1), (9,3)}`. Merge them and assert:

- state contains three pairs;
- final sample/population values match a single accumulator fed those three pairs;
- merging in the opposite order produces the same result within floating tolerance.

Also assert a mismatched pair-list state returns an internal error.

## Invariants and non-goals

- DISTINCT applies to the whole ordered argument tuple.
- NULL pairs are excluded before deduplication.
- Raw pair state, not covariance summaries, crosses partial stages.
- The existing Welford-style covariance implementation remains the only formula implementation.
- Do not add Decimal inputs or change coercion.
- Do not add a distinct groups accumulator unless it is genuinely vectorized and tested.

## Risks and mitigations

| Risk                                                 | Mitigation                                                   |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| Distinct is accidentally applied per column          | Store and hash one tuple key; test crossed pairs.            |
| Partial covariance summaries retain duplicate weight | Use raw aligned list state and evaluate after global union.  |
| Hash iteration changes floating results              | Materialize pairs in deterministic total order.              |
| Window retraction removes a still-live duplicate     | Use multiplicity counts or advertise no retract support.     |
| Pair state arrays become misaligned                  | Construct from one key iteration and reject unequal lengths. |

## Focused validation commands

```bash
cargo test -p datafusion-functions-aggregate-common distinct_pairs --lib
cargo test -p datafusion-functions-aggregate covariance --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- Both covariance variants match explicit tuple deduplication.
- Duplicate pairs crossing partial states are counted once.
- NULL and sample/population edge cases are protected.
- Plain covariance keeps its four-field state and current behavior.
- Distinct state contains two aligned raw-value lists and has deterministic evaluation.
