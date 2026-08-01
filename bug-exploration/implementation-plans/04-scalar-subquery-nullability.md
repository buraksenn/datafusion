# Implementation plan: scalar-subquery nullability for zero-row results

Source report: [05-empty-scalar-subquery-nullability.md](../05-empty-scalar-subquery-nullability.md)

Status: planning only. This document does not implement the fix.

## Objective

Model every scalar-subquery expression as nullable unless DataFusion introduces a separate proof that the subquery always returns exactly one row.

This must be consistent across logical expression schema inference and physical expression planning, so an empty scalar subquery over a `NOT NULL` column produces a typed NULL without an Arrow schema error or an invalid optimizer simplification.

## SQL contract

A scalar subquery has three row-count outcomes:

- Zero rows → one typed NULL value.
- One row → that row's value.
- More than one row → execution error.

The zero-row case makes the scalar expression nullable independently of the selected field's declared nullability.

## Current behavior and root cause

Three paths copy the inner field's nullability and therefore disagree with execution semantics:

1. `datafusion/expr/src/expr_schema.rs`, `ExprSchemable::nullable` for `Expr::ScalarSubquery`.
2. The same file's `ExprSchemable::to_field` branch, which clones the inner field unchanged.
3. `datafusion/physical-expr/src/planner.rs`, which passes `schema.field(0).is_nullable()` into `ScalarSubqueryExpr::new`.

`ScalarSubqueryExpr::return_field` then exposes that physical nullable flag. Meanwhile `ScalarSubqueryExec` correctly stores a typed NULL when the subquery returns zero rows, creating the runtime/schema contradiction.

## Files to change

| File                                              | Planned change                                                                                                   |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `datafusion/expr/src/expr_schema.rs`              | Add logical schema tests; make scalar-subquery nullability conservative in `nullable` and `to_field`.            |
| `datafusion/physical-expr/src/planner.rs`         | Ensure planned `ScalarSubqueryExpr` is nullable and add focused planner coverage if practical.                   |
| `datafusion/physical-expr/src/scalar_subquery.rs` | Add/adjust a return-field test only if needed to protect the physical contract; no evaluator change is expected. |
| `datafusion/sqllogictest/test_files/subquery.slt` | Add an empty `NOT NULL` scalar-subquery execution/schema/optimizer regression.                                   |

## Detailed implementation sequence

### 1. Add logical expression-schema tests

In the `expr_schema.rs` test module, construct a scalar subquery whose single projected field is non-nullable. A minimal plan can project a non-null literal or scan a schema with `Field::new(..., nullable = false)`.

Build `Expr::ScalarSubquery(Subquery { ... })` and assert:

- `expr.nullable(input_schema)` is `true`.
- `expr.to_field(input_schema)` has the original data type and metadata but `is_nullable() == true`.

The test should fail on both current branches before the source change.

### 2. Add physical-planning coverage

The physical planner independently reads the inner subquery schema. Protect that path so a future logical fix cannot leave physical metadata stale.

Preferred focused test:

1. Create the same one-column non-nullable subquery.
2. Register it in a `PhysicalPlanningContext` using `SubqueryIndex::new(0)` and `ScalarSubqueryResults::new(1)`.
3. Call `create_physical_expr` for the logical scalar-subquery expression.
4. Assert `return_field(...).is_nullable()` is true.

If constructing this test in `planner.rs` is disproportionately complex, add a direct `ScalarSubqueryExpr::return_field` test plus an end-to-end physical-plan assertion. The plan must still explicitly cover the planner's nullable argument, not only the expression type.

### 3. Add the SQLLogicTest regression

Extend the existing uncorrelated scalar-subquery row-count section in `subquery.slt` with a genuinely non-nullable empty source:

```sql
CREATE TABLE sq_nonnull(v INT NOT NULL);
```

Cover three observations:

- `DESCRIBE SELECT (SELECT v FROM sq_nonnull) AS v` reports nullable/YES.
- `SELECT (SELECT v FROM sq_nonnull)` returns NULL without an Arrow error.
- `SELECT (SELECT v FROM sq_nonnull) % 1` returns NULL and is not folded to zero.

The existing `sq_empty` fixture is insufficient because its column schema is nullable.

### 4. Apply consistent conservative nullability

Update all three metadata paths:

- Logical `nullable` returns true for `Expr::ScalarSubquery`.
- Logical `to_field` clones the inner field while setting nullable true, preserving data type, name, and metadata.
- Physical planning passes nullable true into `ScalarSubqueryExpr`.

Do not alter the subquery's own output schema; only the scalar expression adds the zero-row NULL possibility.

### 5. Verify optimizer and execution behavior

Confirm that:

- The `% 1` expression is no longer simplified to a non-NULL zero.
- Empty scalar subqueries execute to typed NULL.
- One-row scalar subqueries still return their value.
- Multi-row scalar subqueries still error.
- Aggregate scalar subqueries such as `SELECT count(*)` still execute correctly, even though their outer scalar-expression metadata may now be conservatively nullable.

## Invariants and non-goals

- Correctness takes priority over preserving non-nullability for guaranteed-one-row subplans.
- Do not add row-count proof machinery in this patch.
- Do not change `InSubquery`; it has separate nullability semantics and tracking work.
- Do not change multi-row error behavior or scalar-subquery execution scheduling.
- Preserve inner field type and metadata.

## Risks and mitigations

| Risk                                                                        | Mitigation                                                                           |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Conservative nullability disables an optimization                           | Accept this correctness-first tradeoff; a proven-one-row refinement can be separate. |
| Logical and physical metadata diverge                                       | Update and test all three paths in one patch.                                        |
| Existing empty-subquery tests pass accidentally because fields are nullable | Add an explicit `NOT NULL` table fixture.                                            |
| Metadata/name is lost by rebuilding the field                               | Clone the existing field and apply `with_nullable(true)`.                            |

## Focused validation commands

```bash
cargo test -p datafusion-expr expr_schema --lib
cargo test -p datafusion-physical-expr scalar_subquery --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- subquery
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- Logical `nullable` and `to_field` agree that scalar subqueries are nullable.
- Physical `ScalarSubqueryExpr::return_field` is nullable for a non-nullable inner field.
- The `NOT NULL` empty-subquery SQL returns NULL without an Arrow error.
- The optimizer-sensitive modulo query returns NULL.
- One-row and multi-row scalar-subquery contracts remain unchanged.
