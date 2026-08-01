# Extreme positive `array_slice` stride overflows after the final element

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

An extreme positive stride panics even when the requested slice contains only one valid endpoint. The loop emits that endpoint and then performs an unnecessary overflowing addition.

Likely source fix: 5–9 lines. No API change. Low risk.

## Cause

`datafusion/functions-nested/src/extract.rs`, `compute_slice_plan`, builds arbitrary indices with unchecked arithmetic:

```rust
while index <= to {
    indices.push(index);
    index += stride;
}
```

For a regular List, offsets use `i32`. With `from = to = 1` and `stride = i32::MAX`, the correct index is pushed, then `1 + i32::MAX` overflows before the loop can terminate.

The negative-direction branch has the same unchecked `index += stride` shape.

## Reproduction

```sql
SELECT array_slice([1, 2], 2, 2, 2147483647);
```

Observed:

```text
thread 'main' panicked at datafusion/functions-nested/src/extract.rs:
attempt to add with overflow
```

Expected: `[2]`. The stride is irrelevant after the sole endpoint has been emitted.

## Possible fix

Advance with checked arithmetic and terminate the index sequence when addition overflows. Overflow cannot produce another representable index inside the already normalized bounded interval. Alternatively, stop before incrementing when the current index equals the endpoint.

Apply the correction to both stride directions and to the generic List/LargeList paths.

## Regression test

Add ordinary List and LargeList cases with their maximum positive strides to `datafusion/sqllogictest/test_files/array/array_slice.slt`, plus a maximum-magnitude negative stride case.

## Novelty check

The current open bug/PR lists and targeted stride-overflow searches found no match. Closed issue #10425 fixed a direction-mismatch loop, not arithmetic overflow after a valid endpoint.
