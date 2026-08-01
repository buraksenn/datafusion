# `regr_r2` returns NULL for a horizontal line instead of 1.0

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

`regr_r2(y, x)` returns NULL whenever the dependent variable has zero variance. The SQL regression-aggregate specification treats a horizontal line with varying `x` as a perfect fit and returns 1.0; PostgreSQL implements that behavior explicitly.

Likely source fix: 3–7 lines. No API change. Low risk; the vertical-line and insufficient-row cases remain NULL.

## Cause

`datafusion/functions-aggregate/src/regr.rs`, `RegrAccumulator::evaluate`, uses one combined NULL condition:

```rust
self.count <= 1 || var_pop_x == 0.0 || var_pop_y == 0.0
```

For a horizontal line, `var_pop_x > 0` and `var_pop_y == 0`. That case needs a separate return value of 1.0 rather than sharing the vertical-line NULL path.

## Reproduction

```sql
SELECT regr_r2(y, x)
FROM (VALUES
  (1.0, 1.0),
  (1.0, 2.0)
) AS t(y, x);
```

Observed: NULL.

Expected: `1.0`.

PostgreSQL's `float8_regr_r2` implementation documents the cases directly: vertical line (`Sxx == 0`) returns NULL; horizontal line (`Syy == 0`) returns 1.0.

## Possible fix

Handle the cases in order:

1. insufficient rows or `var_pop_x == 0` → NULL;
2. `var_pop_y == 0` → `Float64(Some(1.0))`;
3. otherwise compute the existing ratio.

## Regression test

Add horizontal, vertical, one-row, and ordinary sloped-line cases to the `regr_r2` tests in `regr.rs` and the aggregate SQLLogicTests.

## Novelty check

The current open bug list and targeted `regr_r2` horizontal/constant-Y searches found no matching issue or PR.

## Reference behavior

- PostgreSQL source: <https://github.com/postgres/postgres/blob/master/src/backend/utils/adt/float.c#L3845-L3880>
