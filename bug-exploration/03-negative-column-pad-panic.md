# Dynamic negative lengths panic `lpad` and `rpad`

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

`lpad` and `rpad` correctly turn a negative scalar length into an empty string, but the equivalent value supplied from a column panics the process with `capacity overflow`.

Likely source fix: 1–5 lines in a shared helper. No API change. Very low risk.

## Cause

`datafusion/functions/src/unicode/common.rs`, `pad_data_capacity`, estimates output capacity before either padding loop validates and clamps each target length:

```rust
.fold(0, |acc, len| acc.saturating_add(len as usize))
```

Casting a negative `i64` to `usize` produces a huge value. `GenericStringBuilder::with_capacity` tries to reserve that capacity and panics before `lpad_impl` / `rpad_impl` reaches its existing `target_len < 0 => 0` logic.

The same helper is used by both functions, so this is one shared defect.

## Reproduction

```sql
SELECT lpad('x', n), rpad('x', n)
FROM (VALUES (-1::BIGINT)) AS t(n);
```

Observed:

```text
thread 'main' panicked at .../alloc/src/raw_vec/mod.rs:
capacity overflow
```

Control using scalar lengths:

```sql
SELECT lpad('x', -1), rpad('x', -1);
```

returns two empty strings, which is also the expected result for the column form.

## Possible fix

Make `pad_data_capacity` add only valid positive target lengths. Invalid negative lengths should contribute zero to the estimate. Lengths above the function's `i32::MAX` limit should also not trigger a huge pre-allocation before the existing execution error is returned.

For example, filter to `0 < len <= i32::MAX` before converting to `usize`, or validate the length array before constructing the builder.

## Regression test

Add array/column cases to the existing tests in both:

- `datafusion/functions/src/unicode/lpad.rs`
- `datafusion/functions/src/unicode/rpad.rs`

Include `-1`, `0`, NULL, and a value above `i32::MAX`; the first two must return empty strings, NULL must remain NULL, and the oversized value must return a normal DataFusion error rather than allocate or panic.

## Novelty check

No current open bug or PR matched `pad_data_capacity` or negative column-driven pad lengths. Historical PR #3829 addressed negative literal behavior; open PR #23223 concerns general fallible builders and does not cover this signed-to-unsigned estimate path.
