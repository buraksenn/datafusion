# Performance Improvement: `map.rs` — Avoid Decompose + Recompose

## Problem

`make_map_array_internal()` in `map.rs` called `list_to_arrays()` which decomposed a ListArray into N separate `ArrayRef`s (one per row), and then `build_map_array()` called `compute::concat()` to flatten them back into a single array.

This was a decompose → recompose round-trip:
```
ListArray.values() (flat) → list_to_arrays() → Vec<ArrayRef> (N arrays) → concat() → flat array
```

## Solution

Refactored `make_map_array_internal` to work directly with the ListArray's `.values()` and `.offsets()`, skipping the decomposition entirely:

- **No-nulls path**: Directly slices the flat values from the ListArray using the first and last offsets
- **Nulls path**: Builds index arrays for non-null rows and uses a single `compute::take()` call to extract the needed values

This eliminates N array allocations from `list_to_arrays()` and the `compute::concat()` call.

## Files Changed

- `datafusion/functions-nested/src/map.rs` — Rewrote `make_map_array_internal` to use offset-based access

## Complexity: Low | Impact: Low-Medium

Most impactful when there are many rows with small maps. Eliminates N array allocations + a concat call per batch.

## Verification

```bash
cargo test -p datafusion-functions-nested -- map
cargo test -p datafusion-sqllogictest -- map
```
