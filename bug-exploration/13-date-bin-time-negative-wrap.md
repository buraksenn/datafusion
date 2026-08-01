# `date_bin` can create an invalid negative `TIME`

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

For a time-of-day earlier than its bin origin, `date_bin` can compute a boundary on the previous day. The helper uses signed remainder rather than Euclidean remainder, leaving the result negative and invalid for Arrow `TIME`.

Likely source fix: one line. No API change. Very low risk.

## Cause

`datafusion/functions/src/datetime/date_bin.rs`, `date_bin_time_value`, says the modulo keeps the value within one day but implements:

```rust
(binned % NANOSECONDS_IN_DAY) / scale
```

Rust's `%` keeps the sign of the left operand. A negative `binned` value therefore remains negative instead of wrapping into the previous day's time-of-day.

## Reproduction

```sql
SELECT date_bin(
  INTERVAL '3 hours',
  TIME '01:00:00',
  TIME '02:00:00'
);
```

Observed:

```text
Cast error: Failed to convert -3600000000000 to temporal for Time64(ns)
```

Expected: `23:00:00`. With three-hour boundaries anchored at 02:00, the boundary immediately preceding 01:00 is 23:00 on the cyclic time-of-day domain.

## Possible fix

Replace signed remainder with Euclidean remainder:

```rust
binned.rem_euclid(NANOSECONDS_IN_DAY) / scale
```

Nonnegative values are unchanged, while negative values are normalized into `[0, NANOSECONDS_IN_DAY)` as the existing comment intends.

## Regression test

Add source-before-origin cases for scalar and column inputs to the `DATE_BIN(... TIME ...)` SQLLogicTests and a focused helper test in `date_bin.rs`.

## Novelty check

The current open bug list and targeted searches found related extreme-value `date_bin` overflow issues, but none for ordinary TIME wraparound with a later origin. Existing tests use midnight or otherwise nonnegative origins.
