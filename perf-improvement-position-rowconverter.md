# Performance Improvement: `position.rs` — RowConverter for Array Elements

## Problem

`generic_position()` and `general_positions()` (the slow paths for `array_position`/`array_positions` when the element argument is an array) called `compare_element_to_list()` per row. This:

1. Created a `Scalar` wrapper from `element_array.slice(row_index, 1)` per row
2. Called `arrow_ord::cmp::not_distinct()` which allocated a `BooleanArray` per row
3. Scanned the BooleanArray for matches

Note: The common scalar-element case already had an optimized fast path (`array_position_scalar`). This optimization targets the array-element slow path.

## Solution

Replaced per-row `compare_element_to_list` calls with a single `RowConverter` pass:

1. Convert all list values to row format once via `converter.convert_columns()`
2. Convert the element array to row format once
3. Compare rows directly using byte equality (`value_rows.row(idx) == target`)

This eliminates N `BooleanArray` allocations and N `arrow_ord::cmp` kernel calls, replacing them with 2 `RowConverter::convert_columns` calls + direct byte comparisons.

## Files Changed

- `datafusion/functions-nested/src/position.rs` — Rewrote `generic_position` and `general_positions`

## Complexity: Low-Medium | Impact: Medium

Follows the same RowConverter pattern already established in `set_ops.rs` for `array_distinct`, `array_union`, and `array_intersect`.

## Verification

```bash
cargo test -p datafusion-functions-nested -- position
cargo test -p datafusion-sqllogictest -- array
```
