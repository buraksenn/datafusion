# Formatted `to_date` rounds pre-epoch datetimes toward 1970

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

The formatted `to_date` path converts parsed milliseconds to whole days using truncating signed division. A datetime during 1969-12-31 therefore becomes day zero—1970-01-01—instead of day minus one.

Likely source fix: one line. No API change. Very low risk.

## Cause

`datafusion/functions/src/datetime/to_date.rs`, `ToDateFunc::to_date`, uses:

```rust
n / (24 * 60 * 60 * 1_000)
```

For `1969-12-31 12:00:00`, `n` is `-43_200_000`. Rust integer division truncates toward zero:

```text
-43_200_000 / 86_400_000 = 0
```

A Date32 day index needs floor/Euclidean division, which produces `-1`.

## Reproduction

```sql
SELECT to_date(
  '1969-12-31 12:00:00',
  '%Y-%m-%d %H:%M:%S'
);
```

Observed: `1970-01-01`.

Expected: `1969-12-31`.

The same result occurs in the array path; positive timestamps and exact pre-epoch midnights are not sufficient to expose it.

## Possible fix

Use Euclidean division:

```rust
n.div_euclid(24 * 60 * 60 * 1_000)
```

Positive timestamps and exact day multiples remain unchanged. Only negative timestamps with a nonzero time component move to the correct previous day.

## Regression test

Add formatted scalar and multi-row array cases to `to_date.rs` and `datafusion/sqllogictest/test_files/datetime/dates.slt`. Cover one second before epoch, noon on 1969-12-31, exact midnight, and a positive control.

## Novelty check

The current open bug list and targeted searches for formatted `to_date`, pre-epoch datetimes, and 1969-12-31 found no matching issue or PR. Existing pre-epoch tests use date-only input and do not exercise fractional negative days.
