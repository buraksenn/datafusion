# `to_char` ignores row-varying formats when the value is scalar

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

When the datetime argument is scalar and the format argument is a column, `to_char` computes every formatted row and then returns only row zero as a scalar. DataFusion broadcasts that first string across the batch, ignoring all later format strings.

Likely source fix: 1–6 lines. No API change. Low risk; any array argument should produce row-wise array output.

## Cause

`datafusion/functions/src/datetime/to_char.rs`, `to_char_array`, correctly calls `ColumnarValue::values_to_arrays` and builds one result per format row. It then decides the output shape using only `args[0]`:

```rust
match args[0] {
    ColumnarValue::Scalar(_) => Scalar(result.value(0)),
    ColumnarValue::Array(_) => Array(result),
}
```

This ignores that `args[1]` is an array—the reason `to_char_array` was called—and discards `result[1..]`.

## Reproduction

```sql
SELECT
  f,
  to_char(TIMESTAMP '2024-01-02 03:04:05', f) AS formatted
FROM (VALUES ('%Y'), ('%m'), ('%d')) AS t(f);
```

Observed:

```text
%Y  2024
%m  2024
%d  2024
```

Expected:

```text
%Y  2024
%m  01
%d  02
```

The scalar-format controls return `2024`, `01`, and `02`, confirming that formatting itself is correct.

## Possible fix

Return `ColumnarValue::Array(Arc::new(result))` unconditionally from `to_char_array`. The dispatcher calls this function only when the format argument is already an array, so collapsing its output to a scalar is never valid.

## Regression test

- Expand `scalar_array_data` in `datafusion/functions/src/datetime/to_char.rs` to use at least three different formats.
- Add the SQL query to the datetime SQLLogicTests beside existing `to_char` dynamic-format cases.

## Novelty check

The current open bug list and searches for `to_char` format columns and scalar/array broadcasting found no matching issue. Draft performance work around formatter caching explicitly preserves current behavior and does not fix this shape loss.
