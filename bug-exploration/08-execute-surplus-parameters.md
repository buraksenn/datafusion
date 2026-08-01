# Untyped prepared statements ignore surplus `EXECUTE` arguments

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

A prepared statement with an inferred/unknown parameter type accepts extra arguments and silently ignores them. Typed prepared statements correctly validate arity, so behavior depends accidentally on whether `PreparedPlan.fields` is populated.

Likely source fix: 5–10 lines. No API change. Low risk; only currently invalid calls with unused arguments become errors.

## Cause

`datafusion/core/src/execution/context/mod.rs`, `SessionContext::execute_prepared`, places parameter-count validation inside:

```rust
if !prepared.fields.is_empty() {
    if params.len() != prepared.fields.len() { ... }
    // casts
}
```

For `PREPARE p AS SELECT $1`, planning cannot infer a concrete field type, so the stored `fields` vector is empty. `replace_params_with_values` consumes the referenced first value and does not reject unused list entries.

## Reproduction

```sql
PREPARE p AS SELECT $1;
EXECUTE p(10, 20);
```

Observed:

```text
$1
10
```

The value `20` is silently discarded.

Expected: an error such as:

```text
Prepared statement 'p' expects 1 parameter, but 2 were provided
```

## Possible fix

Validate arity unconditionally before the typed-cast branch. Derive the count from the prepared logical plan's parameter names (for example, `get_parameter_names()?.len()`), or store the parameter count in `PreparedPlan` when PREPARE is processed. Keep type casts conditional on known fields.

## Regression test

Add cases beside existing PREPARE tests in `datafusion/sqllogictest/test_files/prepare.slt`:

- correct arity with unknown type;
- one missing argument;
- one surplus argument;
- repeated use of the same placeholder index, if supported by the count helper.

## Novelty check

The current open bug list and targeted issue/PR searches for surplus EXECUTE arguments found no match. Open issues #22506 and #24042 concern placeholder coercion and structural matching, not parameter-count validation.
