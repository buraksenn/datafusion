# Empty-needle shortcut erases NULL rows in `array_has_all` and `array_has_any`

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

When every non-null needle is empty and the underlying child buffer has no values, `array_has_all`/`array_has_any` take a batch-wide shortcut. The shortcut produces true/false values without attaching the input validity bitmap, converting typed NULL needle rows into non-null results.

Likely source fix: 2–4 lines. No API change. Low risk; non-null empty-array identities remain unchanged.

## Cause

`datafusion/functions-nested/src/array_has.rs`, `array_has_all_and_any_dispatch`, contains:

```rust
if needle.values().is_empty() {
    let buffer = ...;
    Ok(Arc::new(BooleanArray::from(buffer)))
}
```

The normal paths construct a `BooleanArray` with combined haystack/needle nulls. This shortcut drops them entirely.

## Reproduction

```sql
SELECT
  array_has_all(h, n) AS all_result,
  array_has_any(h, n) AS any_result
FROM (VALUES
  ([1], NULL::BIGINT[]),
  ([1], []::BIGINT[])
) AS t(h, n);
```

Observed:

```text
true  false
true  false
```

Expected:

```text
NULL  NULL
true  false
```

A NULL list is unknown; a non-null empty needle uses the vacuous all/any identities.

## Possible fix

Build the shortcut result with `BooleanArray::new(buffer, NullBuffer::union(haystack.nulls(), needle.nulls()))`, matching the normal kernels.

## Regression test

Add the exact two-row typed List-column case to `datafusion/sqllogictest/test_files/array/array_has.slt`. It must be column-driven because scalar fast paths differ.

## Novelty check

The current open bug/PR lists and searches for empty/null `array_has_all` and `array_has_any` found no matching report. Nearby tests cover empty needles and NULLs separately but not a globally empty child buffer containing both row states.
