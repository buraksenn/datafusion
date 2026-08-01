# Runtime `SET`/`RESET` silently deletes prepared statements

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

Changing a `datafusion.runtime.*` setting rebuilds `SessionState` and silently clears every prepared statement. Session-local SQL state should survive a runtime configuration update.

Likely source fix: 4–8 lines. No public API change is required. Low risk; the newly built runtime environment remains the same, while unrelated session state is preserved.

## Cause

`datafusion/core/src/execution/context/mod.rs` implements `set_runtime_variable` and `reset_runtime_variable` by replacing the entire state:

```rust
*state = SessionStateBuilder::from(state.clone())
    .with_runtime_env(...)
    .build();
```

`SessionState` owns `prepared_plans`, but `SessionStateBuilder` has no corresponding field. `SessionStateBuilder::build` in `datafusion/core/src/execution/session_state.rs` therefore initializes:

```rust
prepared_plans: HashMap::new(),
```

The rebuild also changes other identity-like state such as the generated session ID.

## Reproduction

```sql
PREPARE p(INT) AS SELECT $1 + 1;
SET datafusion.runtime.memory_limit = '10M';
EXECUTE p(41);
```

Observed:

```text
Execution error: Prepared statement 'p' does not exist
```

Expected: one row containing `42`.

The equivalent problem applies to `RESET datafusion.runtime...` because it uses the same rebuild pattern.

## Possible fix

Avoid rebuilding `SessionState` merely to replace its runtime environment. Add a crate-private `SessionState` setter/replacement method for `runtime_env` and use it in both runtime SET and RESET paths.

This is safer than teaching the general-purpose builder to clone prepared plans, because arbitrary builder use may intentionally create a fresh state; this path specifically wants to mutate one existing session.

## Regression test

In `datafusion/core/tests/sql/runtime_config.rs`:

1. prepare a statement;
2. SET a runtime option and execute it;
3. RESET the option and execute it again;
4. optionally assert the session ID is unchanged.

## Novelty check

The current open bug list and searches for prepared-plan loss through `SessionStateBuilder`, runtime memory settings, and SET/RESET found no matching issue or PR. Open issue #23697 concerns nondeterministic function-registry roundtripping, not prepared statements being erased.
