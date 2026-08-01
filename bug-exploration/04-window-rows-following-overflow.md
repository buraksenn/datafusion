# Maximum `ROWS ... FOLLOWING` offset panics window execution

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

A valid `ROWS` window-frame offset of `u64::MAX` overflows index arithmetic and panics DataFusion instead of clamping the frame at the partition boundary.

Likely source fix: 2–8 lines. No API change. Very low risk because saturating/clamped arithmetic is already the intended frame behavior.

## Cause

`datafusion/expr/src/window_state.rs`, `WindowFrameContext::calculate_range_rows`, uses unchecked addition for finite following bounds:

```rust
std::cmp::min(idx + n as usize, length)      // frame start
std::cmp::min(idx + n as usize + 1, length)  // frame end
```

`min(..., length)` cannot clamp the result because the addition overflows first.

## Reproduction

```sql
SELECT sum(x) OVER (
  ORDER BY x
  ROWS BETWEEN CURRENT ROW
           AND 18446744073709551615 FOLLOWING
) AS s
FROM (VALUES (1), (2)) AS t(x);
```

Observed:

```text
thread 'main' panicked at datafusion/expr/src/window_state.rs:235:
attempt to add with overflow
```

Expected:

```text
3
2
```

The huge following bound simply reaches the end of the two-row partition.

## Possible fix

Use saturating additions before clamping to `length`, for both the start and end bound paths. Convert `u64` to `usize` with a checked/saturating conversion so the behavior is also correct on narrower targets.

Conceptually:

```rust
idx.saturating_add(offset).min(length)
idx.saturating_add(offset).saturating_add(1).min(length)
```

## Regression test

Add the exact SQL case to the window SQLLogicTests and focused unit cases for both `FOLLOWING` start and end bounds in `window_state.rs`. Include `u64::MAX` at `idx = 0` and at a nonzero index.

## Novelty check

No open issue or PR covers this exact `ROWS`/`u64::MAX` path. Merged PR #22140 fixed related positive/negative window-frame overflows but its coverage did not reach the current unchecked additions for the maximum unsigned offset.
