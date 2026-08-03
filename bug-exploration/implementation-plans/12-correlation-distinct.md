# Implementation plan: `CORR(DISTINCT)`

Source issue: [#2406](https://github.com/apache/datafusion/issues/2406)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

Dependency: reuse the pair-state contract from [11-covariance-distinct.md](11-covariance-distinct.md) rather than introducing another tuple set.

## Objective

Make `corr(DISTINCT y, x)` compute correlation over unique non-NULL `(y, x)` pairs, including correct partial-state merging and groups-accumulator selection.

## Current behavior and root cause

`Correlation::accumulator` ignores `AccumulatorArgs.is_distinct` and always constructs `CorrelationAccumulator`. `groups_accumulator_supported` also unconditionally returns `true`, so grouped distinct calls can select `CorrelationGroupsAccumulator`, which has no deduplication state.

The DISTINCT marker is therefore silently ignored in both scalar-accumulator and grouped execution paths.

Observed comparison:

```sql
WITH t(y, x) AS (
  VALUES (0.0, 0.0), (0.0, 0.0), (1.0, 2.0), (4.0, 3.0)
)
SELECT
  corr(DISTINCT y, x) AS actual_distinct,
  (SELECT corr(y, x) FROM (SELECT DISTINCT y, x FROM t)) AS expected_distinct
FROM t;
```

Observed: `0.9097992698698112`; expected: `0.8910421112136307`.

The current six scalar state fields already encode duplicate weight in covariance and two standard deviations. They cannot be globally deduplicated during final merge.

## Files to change

| File                                                                    | Planned change                                                                       |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `datafusion/functions-aggregate-common/src/aggregate/distinct_pairs.rs` | Reuse the shared pair buffer and raw aligned list state.                             |
| `datafusion/functions-aggregate/src/correlation.rs`                     | Add distinct state/accumulator branches, gate the groups accumulator, and add tests. |
| `datafusion/sqllogictest/test_files/aggregate.slt`                      | Add mixed, grouped, NULL, and partition-merge regressions.                           |

## Detailed implementation sequence

### 1. Add failing SQL coverage

Protect the reproduction above and add:

- Plain and distinct correlation in one projection.
- Multiple distinct pair expressions.
- Grouped input, which specifically catches the current unconditional groups-accumulator path.
- Duplicate pairs split across partitions.
- NULL in either argument.
- Fewer than two unique pairs.
- Zero variance after deduplication.
- NaN behavior consistent with plain correlation over explicit deduplication.

Use floating tolerance or stable SQLLogicTest formatting as appropriate.

### 2. Add a distinct correlation wrapper

`DistinctCorrelationAccumulator` owns `DistinctPairBuffer`.

- `update_batch` inserts aligned non-NULL pairs.
- `state` returns two aligned raw `List<Float64>` values.
- `merge_batch` unions partial pair sets.
- `evaluate` materializes deterministic pair arrays, feeds a fresh existing `CorrelationAccumulator`, and returns its result.
- `size` includes the pair buffer.

Do not reimplement covariance, variance, or correlation formulas. The existing `CorrelationAccumulator` already coordinates `CovarianceAccumulator` and two `StddevAccumulator`s and contains DataFusion's NaN/zero-variance behavior.

### 3. Branch state fields

Plain calls retain:

1. count;
2. mean1;
3. M2 for argument 1;
4. mean2;
5. M2 for argument 2;
6. covariance constant.

Distinct calls expose only the two aligned raw-pair lists required by `DistinctPairBuffer`.

Extract/reuse a helper for the two field definitions so covariance, correlation, regression, and weighted percentile agree on ordering and nullability.

### 4. Correct groups-accumulator selection

Change:

```text
groups_accumulator_supported(_) -> true
```

into a check that returns false when `args.is_distinct`.

Keep `CorrelationGroupsAccumulator` unchanged for plain calls. A future vectorized distinct groups accumulator can be added independently; routing distinct calls through it now would remain wrong.

Add a direct selection test so a grouped SQL result cannot pass only because the logical optimizer rewrites the plan.

### 5. Handle window retraction conservatively

The plain correlation accumulator supports `retract_batch`. For a distinct window, a pair must remain live until its final duplicate leaves the frame.

Either:

- use multiplicity counts in `DistinctPairBuffer` and implement correct retract; or
- leave the distinct wrapper without retract support and let the window engine recompute frames.

Never delegate a distinct retraction directly to plain correlation, and never erase a pair on the first duplicate removal.

### 6. Add partial-state tests

Merge partial sets with an overlapping pair and compare against one distinct buffer fed the union. Verify:

- the merged state cardinality;
- final correlation;
- opposite merge order;
- mismatched list-state lengths error;
- memory accounting grows with unique pairs, not input rows.

## Invariants and non-goals

- DISTINCT applies to `(y, x)` as one ordered tuple.
- Rows with either argument NULL are ignored.
- Raw unique pairs cross partial stages.
- Plain `CorrelationGroupsAccumulator` remains enabled and unchanged.
- Do not add a distinct groups accumulator as a shortcut unless it truly deduplicates per group and across final merge.
- Do not change correlation formulas, coercion, or Decimal support.

## Risks and mitigations

| Risk                                       | Mitigation                                                                        |
| ------------------------------------------ | --------------------------------------------------------------------------------- |
| Grouped calls still bypass distinct logic  | Gate `groups_accumulator_supported` on `!is_distinct` and test grouped execution. |
| Formula code is duplicated and drifts      | Materialize the unique pairs into the existing `CorrelationAccumulator`.          |
| Partial summaries cannot remove duplicates | Use raw pair-list state.                                                          |
| Floating output changes with hash order    | Sort materialized pairs deterministically.                                        |
| Sliding windows remove live duplicates     | Use multiplicities or disable retract optimization.                               |

## Focused validation commands

```bash
cargo test -p datafusion-functions-aggregate-common distinct_pairs --lib
cargo test -p datafusion-functions-aggregate correlation --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- `corr(DISTINCT y, x)` matches correlation over explicit tuple deduplication.
- Grouped distinct calls no longer select the plain groups accumulator.
- Cross-partition duplicates collapse once.
- NULL, degenerate, and NaN cases follow existing correlation semantics after deduplication.
- Plain scalar and groups paths remain unchanged.
