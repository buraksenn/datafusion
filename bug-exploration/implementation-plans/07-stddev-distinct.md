# Implementation plan: `STDDEV(DISTINCT)` and `STDDEV_POP(DISTINCT)`

Source issue: [#2406](https://github.com/apache/datafusion/issues/2406)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

## Objective

Support duplicate-eliminating standard deviation for both public variants and aliases:

- `stddev(DISTINCT x)` / `stddev_samp(DISTINCT x)`
- `stddev_pop(DISTINCT x)`

The result must equal applying the corresponding non-distinct function to `SELECT DISTINCT x`, including NULL and one-value edge cases.

## Current behavior and root cause

Both UDAFs explicitly reject `AccumulatorArgs.is_distinct` in `datafusion/functions-aggregate/src/stddev.rs`.

The sample branch also returns the misleading text `STDDEV_POP(DISTINCT) aggregations are not available`. A query containing plain and distinct forms reaches the physical accumulator and fails, so the single-distinct logical rewrite cannot provide general support.

This is an implementation gap, not a missing statistical primitive. `datafusion/functions-aggregate/src/variance.rs` already implements `DistinctVarianceAccumulator` for sample and population variance using `GenericDistinctBuffer<Float64Type>`. `StddevAccumulator` already defines standard deviation as the square root of the corresponding variance.

## Reproduction

```sql
SELECT
  stddev(x) AS plain,
  stddev(DISTINCT x) AS deduplicated
FROM (VALUES (1.0), (1.0), (3.0)) AS t(x);
```

Observed: `NotImplemented("STDDEV_POP(DISTINCT) aggregations are not available")`.

Expected sample standard deviation over `{1, 3}`: `sqrt(2)`. Population standard deviation over the same set: `1`.

## Files to change

| File                                               | Planned change                                                                                           |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `datafusion/functions-aggregate/src/stddev.rs`     | Add a distinct standard-deviation wrapper, branch state fields and accumulator creation, and unit tests. |
| `datafusion/functions-aggregate/src/variance.rs`   | Expose/reuse the distinct variance state contract without duplicating its set implementation.            |
| `datafusion/sqllogictest/test_files/aggregate.slt` | Add mixed, grouped, NULL, and sample/population regressions.                                             |

No public signature or return-type change is required.

## Detailed implementation sequence

### 1. Add failing SQL regressions

Use queries containing both distinct and non-distinct aggregates so `SingleDistinctToGroupBy` cannot make the test pass without exercising the distinct accumulator.

Cover:

- `{1, 1, 3}` for visibly different plain and distinct results.
- `stddev`, alias `stddev_samp`, and `stddev_pop`.
- NULLs, which are excluded before duplicate tracking.
- Empty input.
- One distinct non-NULL value: sample returns NULL; population returns `0`.
- Grouped input with duplicates crossing record batches/partitions.
- At least two distinct aggregates in one query.

Use tolerant real-number expectations where SQLLogicTest formatting requires it.

### 2. Reuse distinct variance state

Add `DistinctStddevAccumulator` containing `DistinctVarianceAccumulator` configured with `StatsType::Sample` or `StatsType::Population`.

Delegate:

- `update_batch`
- `merge_batch`
- `state`
- `size`

In `evaluate`, call the inner variance accumulator and return:

- NULL unchanged.
- `sqrt(value)` for a non-NULL `Float64`.
- An internal error for any impossible scalar type.

Do not create a second `HashSet` implementation. Distinct variance already has the required float hashing, NULL handling, state union, and edge-case rules.

### 3. Make state fields match the accumulator

For non-distinct calls, retain the current three fields: count, mean, and M2.

For distinct calls, return the same single `List<Float64>` state used by `DistinctVarianceAccumulator`. Prefer extracting a small crate-private helper from `variance.rs` rather than copying field construction three times.

State schema and `Accumulator::state()` must agree in partial, final, and final-partitioned aggregation modes.

### 4. Select the correct accumulator

In both `Stddev` and `StddevPop`:

- `is_distinct == false` keeps `StddevAccumulator`.
- `is_distinct == true` creates `DistinctStddevAccumulator` with the matching `StatsType`.
- Keep `groups_accumulator_supported` false for distinct calls; the existing groups accumulator does not deduplicate.

Remove both rejection branches and their incorrect shared message.

### 5. Add focused accumulator tests

Construct two partial distinct accumulators with overlapping sets, merge their list states, and assert the duplicate crossing the partition boundary is counted once.

Protect:

- NaN and signed-zero behavior inherited from `Hashable<Float64>`.
- NULL-only input.
- Sample/population one-value behavior.
- `size()` includes the distinct buffer.

A sliding-window optimization is not required for initial support. The distinct variance accumulator does not advertise retract support; the window engine may recompute frames. Do not claim a specialized retract path until one is implemented and tested.

## Invariants and non-goals

- DISTINCT applies after input coercion to `Float64`, matching the existing signature.
- NULLs are ignored, as in plain standard deviation.
- Partial states contain raw unique values, not pre-aggregated variance; otherwise duplicates across partitions cannot be removed.
- Keep non-distinct groups-accumulator performance unchanged.
- Do not add Decimal support; that is tracked separately.
- Do not change variance semantics while reusing its accumulator.

## Risks and mitigations

| Risk                                             | Mitigation                                                                          |
| ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Distinct values duplicated across partial states | Merge raw list state through `GenericDistinctBuffer`, then evaluate once.           |
| State schema still advertises count/mean/M2      | Branch `state_fields` on `is_distinct` and add a partial/final test.                |
| Sample and population variants get swapped       | Parameterize the wrapper with `StatsType` and test both on the same data.           |
| Single-distinct optimizer hides the accumulator  | Every end-to-end regression mixes aggregates or uses multiple distinct expressions. |
| Duplicate set logic diverges from variance       | Wrap `DistinctVarianceAccumulator`; do not copy it.                                 |

## Focused validation commands

```bash
cargo test -p datafusion-functions-aggregate stddev --lib
cargo test -p datafusion-functions-aggregate variance --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- Sample, alias, and population distinct calls execute without the current error.
- Results equal the corresponding aggregate over an explicitly deduplicated input.
- NULL, empty, and one-value edge cases are correct.
- Overlapping partial states deduplicate globally.
- Non-distinct state, groups-accumulator selection, and results are unchanged.
