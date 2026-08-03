# Implementation plan: DISTINCT linear-regression aggregates

Source issue: [#2406](https://github.com/apache/datafusion/issues/2406)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

Dependency: reuse the raw pair state from [11-covariance-distinct.md](11-covariance-distinct.md).

## Objective

Support DISTINCT for every aggregate implemented by `Regr`:

- `regr_count`
- `regr_slope`
- `regr_intercept`
- `regr_r2`
- `regr_avgx`
- `regr_avgy`
- `regr_sxx`
- `regr_syy`
- `regr_sxy`

Each function must compute its existing statistic after globally deduplicating non-NULL `(y, x)` pairs.

## Current behavior and root cause

All nine names share `Regr::accumulator`, which ignores `AccumulatorArgs.is_distinct` and always creates `RegrAccumulator`. `state_fields` always returns the six online-statistic fields.

Duplicates therefore retain their weight while the logical plan and output name still claim DISTINCT.

Observed:

```sql
WITH t(y, x) AS (
  VALUES (0.0, 0.0), (0.0, 0.0), (1.0, 2.0), (4.0, 3.0)
)
SELECT
  regr_count(DISTINCT y, x) AS actual_count,
  (SELECT regr_count(y, x) FROM (SELECT DISTINCT y, x FROM t)) AS expected_count,
  regr_slope(DISTINCT y, x) AS actual_slope,
  (SELECT regr_slope(y, x) FROM (SELECT DISTINCT y, x FROM t)) AS expected_slope
FROM t;
```

Observed: count `4` versus `3`; slope `1.148148148148148` versus `1.2142857142857146`.

Because all results derive from the same six statistics, this is one shared implementation slice—not nine independent accumulator implementations.

## Files to change

| File                                                                    | Planned change                                                                        |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `datafusion/functions-aggregate-common/src/aggregate/distinct_pairs.rs` | Reuse raw unique `(y, x)` storage and merge behavior.                                 |
| `datafusion/functions-aggregate/src/regr.rs`                            | Branch state/accumulator creation, add a distinct wrapper, and test every `RegrType`. |
| `datafusion/sqllogictest/test_files/aggregate.slt`                      | Add visible count/slope reproduction and a complete result matrix.                    |

## Detailed implementation sequence

### 1. Add a table-driven semantic matrix

Create one input containing:

- an exact duplicate pair;
- at least three unique non-collinear pairs;
- NULL in `y` and NULL in `x`;
- enough variation for all statistics to be non-degenerate.

For every `RegrType`, compare `regr_*(DISTINCT y, x)` with the same plain function over `SELECT DISTINCT y, x`.

At SQL level, at minimum expose `regr_count` and one floating statistic in a mixed query; direct Rust table-driven tests can cover all nine variants without an unwieldy SQL record.

### 2. Reuse `DistinctPairBuffer`

Use the same ordered tuple, float equality, NULL filtering, aligned state lists, deterministic materialization, merge behavior, and optional multiplicity tracking as covariance and correlation.

Do not deduplicate `x` and `y` independently. Crossed pairs prove why:

```text
(y, x) = (1, 2), (1, 3), (2, 2)
```

All three tuples are distinct even though each individual column repeats.

### 3. Add `DistinctRegrAccumulator`

The wrapper stores:

- `DistinctPairBuffer`;
- the selected `RegrType`.

It delegates update/state/merge/size to the buffer. At evaluation it materializes the unique pair arrays, feeds them to a fresh `RegrAccumulator::try_new(&regr_type)`, and returns that accumulator's existing result.

This preserves one implementation for:

- online means, M2, and covariance;
- degenerate-result NULL rules;
- `regr_count`'s non-NULL default `0`;
- every derived regression formula.

### 4. Branch UDAF state fields and creation

In the shared `Regr` implementation:

- Plain calls keep the six scalar state fields and `RegrAccumulator`.
- Distinct calls use two aligned `List<Float64>` fields and `DistinctRegrAccumulator`.

Because `Regr` is parameterized by `RegrType`, this one branch enables all nine registered UDAFs and aliases consistently.

### 5. Protect edge semantics

For explicit deduplicated input, verify:

- `regr_count` returns `0` on no valid pair and remains non-nullable.
- Other variants return their current NULL on empty/insufficient/zero-variance input.
- One unique pair.
- Horizontal and vertical lines, including current `regr_r2` behavior.
- NaN handling.
- Duplicate pairs crossing partial states.

The DISTINCT patch must not opportunistically change regression edge-case policy.

### 6. Handle sliding frames safely

`RegrAccumulator` supports retraction. The distinct wrapper may advertise it only if the pair buffer tracks multiplicity and removes a key after its final duplicate leaves.

Otherwise leave retract support false and rely on frame recomputation. Correctness is required; matching plain-path performance is not.

### 7. Add state-merge tests

For every `RegrType`, merge two overlapping pair sets and compare with one accumulator fed their unique union. This table-driven test catches a state schema or wrapper dispatch error once for all names.

Also verify deterministic results when partial merge order changes.

## Invariants and non-goals

- One implementation slice covers all nine functions because they share `Regr` and `RegrAccumulator`.
- DISTINCT key is the ordered `(y, x)` pair after Float64 coercion.
- Rows with either NULL are ignored.
- Raw pair lists, not six statistical summaries, cross distinct partial stages.
- Do not change regression formulas or edge-case decisions.
- Do not add Decimal support or a groups accumulator.

## Risks and mitigations

| Risk                                        | Mitigation                                                       |
| ------------------------------------------- | ---------------------------------------------------------------- |
| Only named functions tested get support     | Branch in shared `Regr` and table-test every `RegrType`.         |
| Pair columns deduplicated independently     | Use one tuple key and crossed-pair tests.                        |
| `regr_count` nullability/default regresses  | Reuse `RegrAccumulator` evaluation and add empty-input coverage. |
| Partial summaries preserve duplicate weight | Serialize raw unique pairs.                                      |
| Hash order changes floating statistics      | Sort pairs before feeding the online accumulator.                |

## Focused validation commands

```bash
cargo test -p datafusion-functions-aggregate regr --lib
cargo test -p datafusion-functions-aggregate-common distinct_pairs --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- All nine regression aggregate names match their plain counterpart over explicit tuple deduplication.
- The observed count and slope mismatches are corrected.
- Cross-partition duplicates collapse globally.
- Empty, NULL, degenerate, and NaN behavior remains consistent with existing regression semantics.
- Plain regression keeps its current six-field state and execution path.
