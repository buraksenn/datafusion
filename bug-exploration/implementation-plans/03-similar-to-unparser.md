# Implementation plan: preserve `SIMILAR TO` in SQL unparsing

Source report: [10-similar-to-unparsed-as-like.md](../10-similar-to-unparsed-as-like.md)

Status: planning only. This document does not implement the fix.

## Objective

Make expression and plan unparsing emit `SIMILAR TO` for `Expr::SimilarTo` rather than changing the operator to `LIKE`.

The generated SQL must preserve negation, operands, and the optional escape character.

## Current behavior and root cause

In `datafusion/sql/src/unparser/expr.rs`, `Unparser::expr_to_sql_inner` matches DataFusion's `Expr::SimilarTo(Like { ... })` but constructs `sqlparser::ast::Expr::Like`.

This changes semantics because `LIKE` and `SIMILAR TO` use different pattern languages. The existing `expr_to_sql_ok` test currently expects the incorrect `LIKE` string and therefore locks in the defect.

The installed `sqlparser` version already exposes the required AST node:

```text
sqlparser::ast::Expr::SimilarTo {
    negated,
    expr,
    pattern,
    escape_char,
}
```

No parser dependency or public API change is needed.

## Files to change

| File                                        | Planned change                                                 |
| ------------------------------------------- | -------------------------------------------------------------- |
| `datafusion/sql/src/unparser/expr.rs`       | Correct the AST variant and update/add expression-level tests. |
| `datafusion/sql/tests/cases/plan_to_sql.rs` | Add a full logical-plan-to-SQL roundtrip regression.           |

## Detailed implementation sequence

### 1. Correct the expression-level expected output first

Update the current `expr_to_sql_ok` fixture whose input is `Expr::SimilarTo`:

- Expected text becomes `a SIMILAR TO 'foo' ESCAPE 'o'`.
- The old implementation should fail this expectation before the source change.

Add a second focused fixture for `negated: true` so `NOT SIMILAR TO` is protected.

Ensure the tests cover:

- Expression conversion.
- Pattern conversion.
- Escape-character conversion.
- Negation.

A case-insensitive field exists in DataFusion's shared `Like` struct but is not part of SQL `SIMILAR TO`; do not invent ILIKE-like behavior for this arm.

### 2. Add a plan roundtrip case

Add a query to `roundtrip_statement` in `datafusion/sql/tests/cases/plan_to_sql.rs` using a pattern whose semantics visibly differ from LIKE, for example:

```sql
SELECT '123' SIMILAR TO '[0-9]+';
```

This pattern avoids relying on the separate open `%`-wildcard compatibility work for `SIMILAR TO`.

The existing helper should:

1. Parse SQL to a logical plan.
2. Unparse the plan.
3. Parse the generated SQL again.
4. Assert logical-plan equality.

Optionally add a snapshot asserting that the generated string contains `SIMILAR TO`, so a future normalization cannot hide another operator substitution behind plan equality.

### 3. Construct the correct sqlparser AST node

In the `Expr::SimilarTo` arm:

- Keep recursive conversion of `expr` and `pattern`.
- Keep the current conversion of the optional escape character to `ValueWithSpan`.
- Preserve `negated`.
- Construct `ast::Expr::SimilarTo`, not `ast::Expr::Like`.
- Remove the LIKE-only `any: false` field from this path.

Do not modify the adjacent `Expr::Like`/`ILike` arms.

### 4. Verify dialect behavior

Run the default/generic unparser tests. If a dialect does not support `SIMILAR TO`, it is safer for that dialect to reject or explicitly override the expression than to silently emit a different predicate. Do not add a semantic fallback to LIKE in this patch.

## Invariants and non-goals

- Unparsing must preserve the logical operator exactly.
- Preserve NOT and ESCAPE syntax.
- Do not change runtime evaluation of `SIMILAR TO`.
- Do not address `%` wildcard semantics tracked elsewhere.
- Do not refactor unrelated LIKE/ILIKE/RLIKE handling.
- Do not add dialect-specific rewrites unless an existing dialect hook requires a minimal adjustment.

## Risks and mitigations

| Risk                                                      | Mitigation                                                                              |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| A test currently relies on the incorrect LIKE string      | Update only fixtures whose input is `Expr::SimilarTo`; leave true LIKE cases unchanged. |
| The full roundtrip passes without checking emitted syntax | Add a direct string assertion or snapshot in addition to plan equality.                 |
| Escape syntax changes                                     | Retain the existing `SingleQuotedString(...).into()` conversion and test it.            |
| Separate SimilarTo execution bugs confuse the regression  | Use `[0-9]+`, not `%`, in this patch's end-to-end case.                                 |

## Focused validation commands

```bash
cargo test -p datafusion-sql expr_to_sql_ok --lib
cargo test -p datafusion-sql --test sql_integration roundtrip_statement
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- Direct expression conversion emits `SIMILAR TO` and `NOT SIMILAR TO` correctly.
- ESCAPE is preserved.
- The full plan roundtrip retains `Expr::SimilarTo`.
- LIKE and ILIKE tests are unchanged and still pass.
- No runtime evaluator or unrelated unparser behavior changes.
