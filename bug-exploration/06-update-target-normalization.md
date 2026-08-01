# Unquoted `UPDATE` targets bypass identifier normalization

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

With DataFusion's default identifier normalization, unquoted `ID` resolves to column `id` in ordinary expressions but not in an `UPDATE ... SET` target. The valid statement is rejected as if identifiers were case-sensitive only in this position.

Likely source fix: 2–4 lines. No API change. Very low risk; it applies the planner's existing identifier policy consistently.

## Cause

`datafusion/sql/src/statement.rs`, `SqlToRel::update_to_plan`, takes the assignment target's raw `Ident.value` and uses it for schema lookup and the assignment map:

```rust
table_schema.field_with_unqualified_name(&col_name.value)?;
Ok((col_name.value.clone(), assign.value.clone()))
```

Unlike normal column references and INSERT targets, this path never calls the configured `IdentNormalizer`.

## Reproduction

```sql
CREATE TABLE u(id INT);
INSERT INTO u VALUES (1);
UPDATE u SET ID = 2;
SELECT * FROM u;
```

Observed:

```text
Schema error: No field named "ID". Did you mean 'u.id'?
```

Control:

```sql
UPDATE u SET id = 2;
```

succeeds. Ordinary `SELECT ID FROM u` also resolves under the default normalization.

Expected: the unquoted uppercase target updates `id` to 2. A quoted target such as `"ID"` must remain case-sensitive.

## Possible fix

Normalize the target `Ident` once with `self.ident_normalizer` and use the normalized value for both validation and the assignment-map key. Preserve quote style so quoted identifiers and sessions with `enable_ident_normalization = false` retain their existing exact-case behavior.

## Regression test

Add cases to `datafusion/sqllogictest/test_files/dml_update.slt` for:

- unquoted uppercase target under default normalization;
- quoted mixed-case target;
- normalization disabled.

## Novelty check

The current open bug list and targeted searches for `update_to_plan`, assignment-target normalization, and UPDATE target case sensitivity found no matching issue or PR. Nearby UPDATE tests use lowercase targets only.
