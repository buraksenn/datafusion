# Implementation plan: `APPROX_MEDIAN(DISTINCT)`

Source issue: [#2406](https://github.com/apache/datafusion/issues/2406)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

Dependency: implement the raw-value distinct percentile state described in [08-approx-percentile-cont-distinct.md](08-approx-percentile-cont-distinct.md) first, or land both changes together.

## Objective

Support `approx_median(DISTINCT x)` as the approximate 50th percentile of the unique non-NULL input values.

It must work when mixed with other aggregates, grouped, or merged across partitions—not only when `SingleDistinctToGroupBy` happens to rewrite a lone distinct aggregate.

## Current behavior and root cause

`ApproxMedian::accumulator` explicitly rejects `AccumulatorArgs.is_distinct`:

```text
APPROX_MEDIAN(DISTINCT) aggregations are not available
```

A lone distinct call can appear to work because the optimizer rewrites it to a group-by over unique values. The existing SQLLogicTest demonstrates the gap:

- `approx_median(DISTINCT col_i8)` alone returns a result.
- `approx_median(col_i8), approx_median(DISTINCT col_i8)` errors because the physical distinct accumulator is required.

`ApproxMedian` is otherwise a thin fixed-percentile wrapper around `ApproxPercentileAccumulator`. Its state fields are the same six t-digest fields, so merely deleting the rejection would silently retain duplicates.

## Files to change

| File                                                           | Planned change                                                                                 |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `datafusion/functions-aggregate/src/approx_median.rs`          | Select plain or raw-value distinct percentile state at percentile `0.5`.                       |
| `datafusion/functions-aggregate/src/approx_percentile_cont.rs` | Expose a crate-private distinct accumulator/config constructor reusable by the median wrapper. |
| `datafusion/sqllogictest/test_files/aggregate.slt`             | Convert the existing mixed-query error into result coverage and add grouped/merge cases.       |

No signature, alias, or return-type change is required.

## Detailed implementation sequence

### 1. Turn the existing failure into a semantic regression

Replace the current expected-error case with a result assertion for:

```sql
SELECT
  approx_median(col_i8),
  approx_median(DISTINCT col_i8)
FROM median_table;
```

Keep the existing lone-distinct control, but do not treat it as sufficient evidence because the logical optimizer can satisfy it without the UDAF's distinct branch.

Add a small explicit comparison:

```sql
WITH t(x) AS (VALUES (1.0), (1.0), (1.0), (9.0))
SELECT
  approx_median(DISTINCT x),
  (SELECT approx_median(x) FROM (SELECT DISTINCT x FROM t))
FROM t;
```

The two values must match under the same t-digest implementation.

### 2. Share the percentile call configuration

Expose a crate-private constructor from the approximate-percentile implementation that accepts:

- fixed percentile (`0.5` here);
- coerced return/input type;
- optional max size (default for median);
- distinct flag.

It should return the appropriate boxed accumulator without requiring a synthetic percentile physical expression.

Avoid copying t-digest or distinct-buffer logic into `approx_median.rs`. `approx_median` is documented as an alias of approximate percentile at `0.5`; its implementation should remain an adapter.

### 3. Branch median intermediate state

For non-distinct median, retain the current six t-digest state fields.

For distinct median, return the raw `List<input float type>` state expected by the distinct approximate-percentile accumulator. Reuse a shared state-field helper so the percentile and median UDAFs cannot drift.

This state is required to remove duplicates that occur in different partial partitions.

### 4. Remove the explicit denial

Replace the early `not_impl_err!` branch with accumulator dispatch:

- non-distinct → existing `ApproxPercentileAccumulator` at `0.5`;
- distinct → shared `DistinctApproxPercentileAccumulator<T>` at `0.5`.

Keep the `DataType::Null` no-op path coherent with state fields. All-NULL distinct input must return typed NULL rather than create an invalid list state.

### 5. Add merge and edge-case coverage

Cover:

- duplicates split across partial accumulators;
- grouped input;
- all NULL and mixed NULL/non-NULL input;
- one unique value;
- at least two supported numeric input widths before coercion;
- a query with two different distinct aggregate arguments;
- a window call if current approximate-median window planning accepts it, verifying correctness without promising a retract optimization.

The expected value should be produced by the same approximate median over explicit deduplication. Do not hard-code a different exact-median definition.

## Invariants and non-goals

- `approx_median(DISTINCT x)` is exactly the distinct approximate percentile at `0.5`.
- NULLs are excluded before deduplication.
- Cross-partition duplicates must be removed before building the final digest.
- Keep t-digest interpolation behavior unchanged; #21528 tracks that separate question.
- Do not add a second distinct-buffer implementation.
- Do not add Decimal support.

## Risks and mitigations

| Risk                                                     | Mitigation                                                                                |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Deleting the rejection silently ignores DISTINCT         | Branch both accumulator and state fields to the shared raw-value distinct implementation. |
| Median and percentile implementations diverge            | Reuse a crate-private constructor/config and state helper.                                |
| Existing lone-distinct test passes without new code      | Preserve a mixed query and direct partial-state merge test.                               |
| Distinct state disagrees for `DataType::Null`            | Test all-NULL input through partial and final aggregation.                                |
| Separate percentile semantics issue changes expectations | Compare to explicit deduplication using `approx_median`, not exact `median`.              |

## Focused validation commands

```bash
cargo test -p datafusion-functions-aggregate approx_median --lib
cargo test -p datafusion-functions-aggregate approx_percentile_cont --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- The existing mixed plain/distinct query returns results instead of the current error.
- Distinct median equals approximate median over explicitly deduplicated input.
- Raw distinct state merges overlapping partition values correctly.
- NULL, grouping, and supported type cases are covered.
- Non-distinct approximate median retains its current state and behavior.
