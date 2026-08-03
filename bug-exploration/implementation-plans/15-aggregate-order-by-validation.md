# Implementation plan: reject unsupported aggregate argument `ORDER BY`

Source issue: [#9924](https://github.com/apache/datafusion/issues/9924)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: partially valid; planning only. This document covers the remaining ORDER BY gap and does not implement it.

## Validity assessment

The `IGNORE NULLS` / `RESPECT NULLS` half of #9924 is already fixed by [PR #18441](https://github.com/apache/datafusion/pull/18441):

- `AggregateUDFImpl::supports_null_handling_clause()` defaults to `false`.
- The SQL planner rejects the clause unless a UDAF opts in.
- `array_agg`, `first_value`, and `last_value` opt in because they consume the setting.

Current verification:

```sql
SELECT count(*) IGNORE NULLS
FROM (VALUES (1), (NULL), (2));
```

returns the planning error:

```text
[IGNORE | RESPECT] NULLS are not permitted for count
```

No further null-clause change is needed for this issue.

The ORDER BY half remains valid. The SQL planner converts an inline aggregate `ORDER BY` for every UDAF without asking whether that UDAF implements ordered semantics.

Current examples:

- `avg(column1 ORDER BY column2)` succeeds and returns `2.0`.
- Its physical plan contains `SortExec` because `Avg` inherits the default `HardRequirement`, even though the accumulator does not use ordering.
- `sum(column1 ORDER BY column2)` also succeeds, but `Sum` declares `AggregateOrderSensitivity::Insensitive`, so no `SortExec` is inserted and the ordering is intentionally discarded.

Thus behavior is inconsistent and unsupported syntax can either waste a sort or be silently ignored. `AggregateOrderSensitivity` is an execution/optimization property; it is not an explicit declaration that SQL argument ORDER BY is implemented.

## Objective

Add an explicit aggregate-UDF capability for inline argument ordering and reject unsupported calls before execution.

Target behavior:

```sql
SELECT avg(x ORDER BY y) FROM t;
```

fails with a planning error such as:

```text
ORDER BY in aggregate arguments is not supported for avg
```

while actual ordered aggregates continue to work:

- `array_agg(x ORDER BY y)`
- `string_agg(x, ',' ORDER BY y)`
- aggregate `first_value(x ORDER BY y)`
- aggregate `last_value(x ORDER BY y)`
- aggregate `nth_value(x, 2 ORDER BY y)`

Ordered-set `WITHIN GROUP` support remains controlled separately by `supports_within_group_clause()`.

## API decision

Add:

```rust
fn supports_order_by_clause(&self) -> bool {
    false
}
```

to `AggregateUDFImpl`, plus the forwarding accessor on `AggregateUDF`.

Use the explicit name `supports_order_by_clause`, not `supports_ordering`:

- It identifies SQL syntax rather than physical input properties.
- It does not overlap semantically with `order_sensitivity()`.
- It is distinct from `supports_within_group_clause()`.

Defaulting to false follows the null-handling fix: a UDAF must opt in to syntax whose semantics it consumes.

## Files to change

| File / area                                                            | Planned change                                                                                                                             |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `datafusion/expr/src/udaf.rs`                                          | Add the trait method/accessor and forward through aliased/reversed wrappers.                                                               |
| `datafusion/sql/src/expr/function.rs`                                  | Reject normal aggregate argument ORDER BY unless the UDAF opts in.                                                                         |
| `datafusion/physical-expr/src/aggregate.rs`                            | Add a defensive builder check for programmatic/non-SQL expressions.                                                                        |
| `datafusion/functions-aggregate/src/array_agg.rs`                      | Opt in.                                                                                                                                    |
| `datafusion/functions-aggregate/src/string_agg.rs`                     | Opt in.                                                                                                                                    |
| `datafusion/functions-aggregate/src/first_last.rs`                     | Opt in for both functions.                                                                                                                 |
| `datafusion/functions-aggregate/src/nth_value.rs`                      | Opt in.                                                                                                                                    |
| Ordered-set aggregate implementations                                  | Continue using `supports_within_group_clause`; no duplicate opt-in is needed unless they also expose normal inline ordering independently. |
| `datafusion/ffi/src/udaf/mod.rs`                                       | Carry the capability across the stable FFI aggregate bridge.                                                                               |
| SQLLogicTests, SQL integration tests, and user-defined aggregate tests | Add rejection, opt-in, physical-invariant, and FFI coverage.                                                                               |
| `docs/source/library-user-guide/upgrading/`                            | Document the default-false behavior change for custom UDAFs.                                                                               |

## Detailed implementation sequence

### 1. Add failing SQL tests

Add negative cases for representative order-insensitive or ordering-unaware aggregates:

- `avg(x ORDER BY y)`
- `count(x ORDER BY y)` and `count(* ORDER BY y)` if parser syntax permits it
- `sum(x ORDER BY y)`
- `min(x ORDER BY y)`
- `median(x ORDER BY y)`
- a custom UDAF that does not override the new method

Add positive controls for every built-in opt-in listed above.

Keep ordered-set controls:

- `percentile_cont(...) WITHIN GROUP (ORDER BY x)` remains valid.
- Existing alternate inline syntax that the planner deliberately normalizes into WITHIN GROUP remains valid.
- `WITHIN GROUP` on a non-ordered-set UDAF still uses its existing error.

### 2. Add the trait capability and all forwarding paths

Implement the default-false method on `AggregateUDFImpl` and a public `AggregateUDF::supports_order_by_clause()` accessor.

Forward through every wrapper that delegates aggregate behavior, especially `AliasedAggregateUDFImpl` and reversed/decorated implementations. A wrapper must not erase an inner UDAF's opt-in.

Update `SimpleAggregateUDF` only if its API gains an explicit way to opt in; do not silently opt every simple UDAF into ordering.

### 3. Validate normal SQL aggregate ordering

In `sql_function_to_expr`, resolve the aggregate metadata first, then distinguish:

- normal inline aggregate ORDER BY;
- inline syntax intentionally normalized for a UDAF with `supports_within_group_clause()`;
- explicit WITHIN GROUP.

For the normal path, if `order_by` is non-empty and `supports_order_by_clause()` is false, return a planning error naming the function before creating sort expressions.

Do not infer support from `AggregateOrderSensitivity`:

- `Insensitive` means ordering cannot affect the result, not that the syntax should be accepted.
- The default `HardRequirement` does not prove the accumulator reads ordering fields.
- `SoftRequirement`/`Beneficial` describe execution choices after a supported ordered call exists.

### 4. Add a defensive physical-builder invariant

SQL is not the only source of `Expr::AggregateFunction`; the DataFrame API and custom expression planners can attach `.order_by(...)` directly.

In `AggregateExprBuilder::build`, reject non-empty physical `order_bys` unless either:

- `supports_order_by_clause()` is true; or
- `supports_within_group_clause()` is true for an ordered-set expression normalized by the logical planner.

This closes the original failure mode for non-SQL callers and prevents a custom UDAF from silently receiving and ignoring sort expressions.

Return a normal planning/not-implemented error, not an internal assertion, because programmatic callers can construct this input.

### 5. Opt in only real consumers

Audit built-ins and return true only where accumulator/state logic uses `AccumulatorArgs.order_bys` to define the result:

- `ArrayAgg`
- `StringAgg`
- `FirstValue`
- `LastValue`
- `NthValueAgg`

Do not opt in commutative/idempotent functions merely because ordering does not change their answer. The purpose is to reject unsupported caller intent and avoid needless sorting.

Ordered-set percentile functions retain their existing capability and planner normalization. Avoid requiring two flags for the same WITHIN GROUP path.

### 6. Preserve the capability over FFI

Add a boolean or callback field to `FFI_AggregateUDF`, populate it from the provider-side UDAF, clone it, and return it from `ForeignAggregateUDF::supports_order_by_clause`.

Add an FFI round-trip test with an ordered custom aggregate. Without this, a UDAF may work in-process but be rejected after crossing a dynamic-library boundary.

Treat the stable-struct change according to the repository's FFI versioning/upgrade policy.

### 7. Add custom-UDAF contract tests

Adapt the user-defined aggregate tests from closed PR #9953 or current equivalents:

- A default custom UDAF rejects SQL and DataFrame `.order_by` calls.
- An opt-in UDAF receives the expected physical ordering expressions and executes correctly.
- Aliasing/reversal preserves the capability.
- An ordered-set UDAF continues through its separate capability.

A plan-shape assertion should prove unsupported `avg ORDER BY` no longer creates the currently observed `SortExec`.

## Invariants and non-goals

- The already-fixed null-treatment path is unchanged.
- Inline argument ORDER BY and WITHIN GROUP remain separate capabilities.
- `AggregateOrderSensitivity` retains its current optimizer meaning.
- Plain calls such as `avg(x)` are unchanged.
- Window `OVER (ORDER BY ...)` is not aggregate argument ORDER BY and must remain valid under window-function rules.
- Do not add an optimizer that strips accepted ORDER BY from unsupported functions; reject the syntax instead, as #9924 requests.
- Do not redesign all aggregate ordering internals.

## Risks and mitigations

| Risk                                                                    | Mitigation                                                                                          |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Valid ordered-set syntax is rejected                                    | Validate only the normal inline path; accept existing `supports_within_group_clause` normalization. |
| DataFrame callers still bypass SQL validation                           | Add the defensive `AggregateExprBuilder` check.                                                     |
| Wrapper or FFI erases opt-in                                            | Forward through all decorators and add round-trip tests.                                            |
| Order-insensitive functions lose previously accepted but useless syntax | Document the behavior change; it prevents silent intent loss and unnecessary sorts.                 |
| Window ORDER BY is confused with argument ORDER BY                      | Test `OVER (ORDER BY ...)` separately and inspect the parser branches.                              |
| Existing custom UDAFs rely on ordering                                  | Default false with an upgrade note and a one-method opt-in.                                         |

## Focused validation commands

```bash
cargo test -p datafusion-sql --lib function
cargo test -p datafusion-physical-expr aggregate --lib
cargo test -p datafusion --test core_integration user_defined_aggregate
cargo test -p datafusion-ffi udaf --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
cargo test -p datafusion-sqllogictest --test sqllogictests -- array_agg
cargo test -p datafusion-sqllogictest --test sqllogictests -- group_by
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- `count IGNORE NULLS` remains rejected by the already-landed capability.
- `avg(x ORDER BY y)` and other non-opt-in aggregates fail during planning.
- No `SortExec` is created for a rejected aggregate.
- All true ordered aggregates and ordered-set functions remain valid.
- SQL, DataFrame/programmatic, wrapper, and FFI paths enforce/preserve the same capability.
- Plain aggregate calls and window `OVER ORDER BY` behavior are unchanged.
