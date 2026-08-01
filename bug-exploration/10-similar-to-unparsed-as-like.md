# SQL unparser changes `SIMILAR TO` into `LIKE`

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

The logical plan retains a `SIMILAR TO` predicate, but the SQL unparser emits it as `LIKE`. These operators accept different pattern languages, so plan-to-SQL roundtripping can change query results.

Likely source fix: 1–3 lines. No API change. Low risk; sqlparser already has the target AST variant.

## Cause

`datafusion/sql/src/unparser/expr.rs`, `Unparser::expr_to_sql_inner`, matches `Expr::SimilarTo` but constructs `ast::Expr::Like`:

```rust
Expr::SimilarTo(...) => Ok(ast::Expr::Like { ... })
```

The existing `expr_to_sql_ok` unit case currently codifies the defect by expecting `a LIKE 'foo' ESCAPE 'o'` for an `Expr::SimilarTo` input.

## Reproduction

The semantic difference is visible directly:

```sql
SELECT
  '123' SIMILAR TO '[0-9]+' AS similar_result,
  '123' LIKE '[0-9]+' AS like_result;
```

Observed:

```text
similar_result  like_result
true            false
```

Now parse and unparse the first expression/query with `expr_to_sql` or `plan_to_sql`. Current source and the existing passing unit case show that the regenerated SQL uses `LIKE`, changing `true` to `false` when executed.

Focused verification command:

```text
cargo test -p datafusion-sql expr_to_sql_ok --lib
```

passes with the current incorrect `LIKE` expectation.

## Possible fix

Construct `sqlparser::ast::Expr::SimilarTo` in the `Expr::SimilarTo` arm, preserving `negated`, expression, pattern, and escape character. Remove the LIKE-only `any` field from this path.

## Regression test

- Change the current `expr_to_sql_ok` expectation to `a SIMILAR TO 'foo' ESCAPE 'o'`.
- Add a full `plan_to_sql` roundtrip case using `'123' SIMILAR TO '[0-9]+'` so logical-plan equality and result semantics are both protected.

## Novelty check

The current open bug list and targeted unparser/`SIMILAR TO` searches found no matching issue or open PR. Open issue #22263 concerns execution semantics for `%` inside `SIMILAR TO`; it does not cover the unparser substituting a different operator.
