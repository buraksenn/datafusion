# Performance Improvement: `remove.rs` + `replace.rs` — RowConverter Comparisons

## Problem

Both `general_remove()` (`remove.rs`) and `general_replace()` (`replace.rs`) called `compare_element_to_list()` per row to find which elements match. This created a `BooleanArray` per row via `arrow_ord::cmp::not_distinct` (or `distinct` for remove's inverted logic).

Both functions already used `MutableArrayData` for efficient output construction, so the bottleneck was the per-row comparison, not the output building.

## Solution

Replaced per-row `compare_element_to_list` calls with a single `RowConverter` pass in both functions:

1. Convert all list values to row format once
2. Convert the element/from array to row format once
3. Compare rows directly using byte equality in the inner loop

The `MutableArrayData`-based output construction remains unchanged — only the comparison logic is optimized.

After this change, `compare_element_to_list` in `utils.rs` had no remaining callers and was removed.

## Files Changed

- `datafusion/functions-nested/src/remove.rs` — Rewrote `general_remove` comparison logic
- `datafusion/functions-nested/src/replace.rs` — Rewrote `general_replace` comparison logic
- `datafusion/functions-nested/src/utils.rs` — Removed unused `compare_element_to_list`

## Caveats

- `RowConverter` handles nested types (List, LargeList) natively, which `compare_element_to_list` handled with special-case code
- `RowConverter` row equality correctly handles NULL=NULL semantics (matching `not_distinct` behavior)
- Already validated by `set_ops.rs` which relies on `RowConverter` for the same type of comparisons

## Complexity: Medium | Impact: Medium

Eliminates N `BooleanArray` allocations and N `arrow_ord::cmp` kernel calls per batch.

## Verification

```bash
cargo test -p datafusion-functions-nested -- remove
cargo test -p datafusion-functions-nested -- replace
cargo test -p datafusion-sqllogictest -- array
```
