# `array_to_string` ignores `null_string` for `List(Null)`

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

`array_to_string` replaces NULL elements correctly when the list has a typed child array, but writes nothing when the child data type itself is `Null`. An all-NULL untyped list therefore loses every requested replacement token.

Likely source fix: 8–12 lines. No API change. Low risk.

## Cause

`datafusion/functions-nested/src/string.rs`, `compute_array_to_string`, routes typed leaves through `write_leaf_to_string`, which applies `null_string` per element. Its `DataType::Null` arm is simply:

```rust
Null => Ok(()),
```

That ignores both `arr.len()` and the supplied replacement. `make_array(NULL, NULL)` produces a `List(Null)` with a two-element `NullArray` child.

## Reproduction

```sql
SELECT array_to_string(
  make_array(NULL, NULL),
  ',',
  'X'
) AS s;
```

Observed: the empty string.

Expected: `X,X`.

Control:

```sql
SELECT array_to_string(
  make_array(CAST(NULL AS INT), CAST(NULL AS INT)),
  ',',
  'X'
);
```

returns `X,X`, proving the discrepancy is specific to the Null-typed leaf.

## Possible fix

In the `DataType::Null` arm, emit `null_string` once per element with the same delimiter and `first` handling as `write_leaf_to_string`. Preserve the current no-op when no replacement string is supplied.

## Regression test

Add both untyped and typed all-NULL lists to `datafusion/sqllogictest/test_files/array/array_to_string.slt`, with and without `null_string`.

## Novelty check

The current open bug/PR lists and targeted `List(Null)`/`null_string` searches found no match. Existing tests cover mixed and typed NULL lists, not an all-NULL `NullArray` leaf.
