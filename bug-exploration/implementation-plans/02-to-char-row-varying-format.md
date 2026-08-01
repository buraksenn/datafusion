# Implementation plan: `to_char` with a scalar value and row-varying formats

Source report: [12-to-char-row-varying-format.md](../12-to-char-row-varying-format.md)

Status: planning only. This document does not implement the fix.

## Objective

Ensure `to_char(scalar_datetime, format_column)` returns one independently formatted value per format row instead of returning the first result as a scalar and broadcasting it across the batch.

Target behavior:

```sql
SELECT f, to_char(TIMESTAMP '2024-01-02 03:04:05', f)
FROM (VALUES ('%Y'), ('%m'), ('%d')) AS t(f);
```

must return `2024`, `01`, and `02` respectively.

## Current behavior and root cause

`ToCharFunc::invoke_with_args` dispatches to `to_char_array` whenever the format argument is a `ColumnarValue::Array`.

`to_char_array` correctly:

1. Broadcasts scalar arguments using `ColumnarValue::values_to_arrays`.
2. Iterates over every row of the format array.
3. Builds a complete `StringArray` result.

It then discards that shape information by inspecting only `args[0]`. If the datetime argument was scalar, it extracts `result[0]` and returns `ColumnarValue::Scalar`, even though the format argument is an array. The executor then broadcasts row zero across the batch.

The bug is therefore in output shape selection, not datetime formatting.

## Files to change

| File                                                         | Planned change                                                                         |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `datafusion/functions/src/datetime/to_char.rs`               | Update scalar-value/array-format tests and make `to_char_array` return its full array. |
| `datafusion/sqllogictest/test_files/datetime/timestamps.slt` | Add the three-format SQL regression.                                                   |

No public API or SQL signature change is required.

## Detailed implementation sequence

### 1. Convert the existing unit test to the correct shape contract

The `scalar_array_data` fixtures in `test_to_char` currently all use one-element format arrays and assert that the result is a scalar. That expectation codifies the bug.

Refactor this section so scalar datetime plus array format expects `ColumnarValue::Array`, even when the format array has length one.

Recommended test structure:

- Represent the expected value as a `StringArray`, not a single `String`.
- Assert output data type, length, values, and validity.
- Retain the current one-element cases across Date32, Date64, time, timestamp, and duration types to prevent type regressions.
- Add one multi-row timestamp fixture with formats `%Y`, `%m`, `%d` and expected values `2024`, `01`, `02`.
- Add a NULL format row in a focused case and assert the corresponding result row is NULL.

Confirm the multi-row fixture fails before the source change.

### 2. Add the end-to-end SQL test

Place the SQL reproduction near the existing `to_char` table/format-column cases in `datetime/timestamps.slt`.

Assert row order and values explicitly:

```text
%Y 2024
%m 01
%d 02
```

The test must keep the datetime expression scalar and the format expression columnar. Existing tests where both arguments are columns do not exercise this path.

### 3. Preserve the array result from `to_char_array`

Change only the final return-shape decision in `to_char_array`:

- Return the completed `StringArray` as `ColumnarValue::Array` for every call to this function.
- Do not change `to_char_scalar`, which is still the correct path for a scalar format string.

This is justified by the dispatcher invariant: `to_char_array` is called only when the format argument is an array. Therefore at least one argument is columnar, and the output is row-wise.

### 4. Verify adjacent behavior

Run and inspect cases for:

- Scalar datetime + scalar format → scalar result, unchanged.
- Array datetime + scalar format → array result, unchanged.
- Array datetime + array format → array result, unchanged.
- Scalar datetime + array format → array result, fixed.
- NULL datetime or NULL format row → NULL at that row.
- Date32 retry through Date64 for time specifiers → unchanged.
- Timezone-bearing timestamps → metadata and formatted value unchanged.

## Invariants and non-goals

- Any array argument implies row-wise output.
- `number_rows` must match the format array length in the mixed scalar/array path.
- Keep the existing `ArrayFormatter` and Date32 retry logic intact.
- Do not combine this correctness fix with formatter caching or performance work from other PRs.
- Do not change supported format syntax or PostgreSQL compatibility policy.

## Risks and mitigations

| Risk                                                         | Mitigation                                                                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Downstream code expected a scalar for a one-row format array | The standard ColumnarValue contract follows argument shape; update the existing unit expectation explicitly. |
| Date32 retry path changes accidentally                       | Retain the current one-row Date32 datetime-format fixtures and assert array output.                          |
| NULL validity is lost during the shape correction            | Add a format array containing NULL and compare the output bitmap.                                            |
| The SQL test is constant-folded                              | Keep the format values in a VALUES-derived column.                                                           |

## Focused validation commands

```bash
cargo test -p datafusion-functions test_to_char --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- timestamps
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- Existing scalar/array unit fixtures expect and receive arrays.
- The three-format SQL query returns `2024`, `01`, `02`.
- Scalar/scalar behavior remains scalar.
- NULL, Date32 retry, timezone, and all supported temporal input tests remain green.
- No formatter caching or unrelated refactor is included.
