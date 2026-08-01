# Empty scalar subquery overclaims non-nullability and fails at execution

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

A scalar subquery can return NULL when it produces zero rows, even if its selected field is declared `NOT NULL`. DataFusion copies the inner field's nullability into the scalar-subquery expression, so Arrow later rejects the correctly produced NULL against an incorrectly non-nullable output schema.

Likely source fix: 2–6 lines. No API change. Low risk; the fix is conservative metadata.

## Cause

`datafusion/expr/src/expr_schema.rs` has two inconsistent scalar-subquery schema paths:

- `ExprSchemable::nullable` returns `subquery.schema().field(0).is_nullable()`.
- `ExprSchemable::to_field` clones the inner field unchanged.

Both ignore the SQL rule that a zero-row scalar subquery evaluates to a typed NULL. The physical scalar-subquery implementation follows that rule, creating a runtime/schema mismatch.

## Reproduction

```sql
CREATE TABLE sq(x INT NOT NULL);

DESCRIBE SELECT (SELECT x FROM sq) AS v;
SELECT (SELECT x FROM sq) AS v;
```

Observed `DESCRIBE` output:

```text
v  Int32  NO
```

Observed query error:

```text
Arrow error: Invalid argument error:
Column 'v' is declared as non-nullable but contains null values
```

Expected: `v` is nullable and the query returns one row containing NULL.

A second optimizer-sensitive reproduction is:

```sql
SELECT (SELECT 1 WHERE FALSE) % 1;
```

It must remain NULL; non-nullability metadata can permit simplifications that incorrectly replace it with zero.

## Possible fix

Make scalar-subquery output conservative in both schema paths:

- `nullable()` returns `true`.
- `to_field()` clones the selected field with `with_nullable(true)`.

A future refinement could retain non-nullability only for subplans proven to produce exactly one row, but that is not needed for this correction.

## Regression test

Add the empty-table SQL scenario to `datafusion/sqllogictest/test_files/subquery.slt` and focused nullability assertions in `datafusion/expr/src/expr_schema.rs`.

## Novelty check

The current open bug list and targeted scalar-subquery nullability/error searches found no matching report. Open issue #23428 concerns `InSubquery` nullability, a different expression kind and execution path.
