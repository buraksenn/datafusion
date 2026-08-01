# Timestamp-to-time casts are incorrectly treated as order-preserving

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

Casting an ordered timestamp to a time-of-day wraps at midnight, so it is not globally monotonic. DataFusion currently marks every temporal-to-temporal cast as order-preserving and can remove a required sort, producing incorrectly ordered results.

Likely source fix: 3–6 lines. No API change. Low risk; only timestamp-to-time casts lose an invalid optimization.

## Cause

`datafusion/physical-expr/src/expressions/cast.rs`, `is_order_preserving_cast_family`, contains this blanket rule:

```rust
source_type.is_temporal() && target_type.is_temporal()
```

`cast_expr_properties` then copies the timestamp's `SortProperties` to `CAST(ts AS TIME)`. Arrow's timestamp-to-`Time32`/`Time64` kernels extract the local time of day, which wraps from `23:59...` to `00:00...` at each day boundary.

## Reproduction

Create `/tmp/time-cast.csv`:

```text
ts
2024-01-01T23:00:00
2024-01-02T00:00:00
```

Then run:

```sql
CREATE EXTERNAL TABLE time_ordered (ts TIMESTAMP NOT NULL)
STORED AS CSV
WITH ORDER (ts ASC)
LOCATION '/tmp/time-cast.csv'
OPTIONS ('format.has_header' 'true');

EXPLAIN
SELECT CAST(ts AS TIME) AS t
FROM time_ordered
ORDER BY t ASC;

SELECT CAST(ts AS TIME) AS t
FROM time_ordered
ORDER BY t ASC;
```

Observed physical plan: no `SortExec` below `SortPreservingMergeExec`.

Observed result:

```text
23:00:00
00:00:00
```

Expected ascending time-of-day order:

```text
00:00:00
23:00:00
```

## Possible fix

Exclude these pairs from the blanket temporal family:

```text
Timestamp(_, _) -> Time32(_)
Timestamp(_, _) -> Time64(_)
```

A small deny-list is sufficient for this bug. A longer-term cleanup could replace the broad temporal rule with an explicit allow-list of genuinely monotonic casts.

## Regression test

Add the two-day ordered-source case to `datafusion/sqllogictest/test_files/monotonic_projection_test.slt`. Assert the correct output and that `SortExec` remains. A focused `cast_expr_properties` unit test should also assert `Unordered` for timestamp-to-time casts.

## Novelty check

The current open bug list and targeted searches for timestamp-to-time ordering, temporal cast monotonicity, and the cast-property helper found no matching issue or PR. Existing monotonic projection tests cover numeric precision loss, not daily time-of-day wraparound.
