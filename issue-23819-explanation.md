# Issue 23819: nullable `UNIQUE` must not remove a `GROUP BY` column

## Branch and base

- Branch: `fix-23819-nullable-unique-group-by`
- Base: `7269d13e63aa6b67d5db08655134d9c6f9c71ce8` (`upstream/main` when the branch was created)
- Issue: <https://github.com/apache/datafusion/issues/23819>
- Draft source examined: `origin/draft-null-issue` at `5d7c9379ad1a9e7225de2b8cfbcf73ecf7118cbe`
- The implementation is independent of issue 23818. It does not contain the 23818 sort-key pruning change.

## What was taken from `draft-null-issue`

The draft branch is primarily an attempted fix for issue 23819, because it changes `get_required_group_by_exprs_indices` and adds issue-23819 GROUP BY tests. However, its single commit also changes DISTINCT elimination, aggregate dependency semantics, implicit grouping, UNNEST dependency propagation, several existing results, and tests for other bugs.

Only the issue-relevant idea was transferred: GROUP BY pruning must not consume a nullable dependency when its determinant can actually contain NULL.

The following draft changes were intentionally not transferred because they belong to other bugs or alter unrelated plans/results:

- `ReplaceDistinctWithAggregate` changes for issue 23634;
- aggregate-wide dependency/nullability changes;
- implicit-selection behavior for issue 23820;
- UNNEST dependency removal and its test;
- unrelated existing `group_by.slt` baseline changes; and
- the draft's broad additional tests.

The draft branch itself remains unchanged.

## Problem

A SQL `UNIQUE` constraint permits more than one row whose key is `NULL`. For this table:

```sql
CREATE TABLE t_uniq (x INT UNIQUE, y INT) AS VALUES
  (NULL, 2),
  (NULL, 1),
  (1, 3);
```

`x` determines `y` for non-NULL key values, but not for the two NULL-keyed rows.

The projection optimizer reduces an aggregate's grouping expressions with `get_required_group_by_exprs_indices`. Before this fix, that helper saw the functional dependency `x -> y`, ignored that it came from a nullable `UNIQUE` key, and simplified:

```sql
GROUP BY x, y
```

to:

```sql
GROUP BY x
```

That merges the two rows `(NULL, 2)` and `(NULL, 1)` into one group. The query returns two rows instead of three.

The bad optimized logical plan was:

```text
Aggregate: groupBy=[[t_uniq.x]], aggr=[[]]
  TableScan: t_uniq projection=[x]
```

The required plan is:

```text
Projection: t_uniq.x
  Aggregate: groupBy=[[t_uniq.x, t_uniq.y]], aggr=[[]]
    TableScan: t_uniq projection=[x, y]
```

## Why the one-line guard is not sufficient by itself

DataFusion's SQL planner historically added every column reachable through a functional dependency as a synthetic GROUP BY expression, even when that column was not referenced by SELECT, HAVING, QUALIFY, ORDER BY, or DISTINCT ON. The projection optimizer then removed unused synthetic columns.

For example, while planning:

```sql
SELECT x FROM t_uniq GROUP BY x;
```

the builder temporarily produced `GROUP BY x, y`; projection pruning later removed unused `y`, restoring `GROUP BY x`.

If GROUP BY pruning simply stops using the nullable dependency, that synthetic `y` survives. This changes existing plans and exposes separate known bugs, including issues 23634 and 23820. The initial one-line fix was tested and did exactly that, so it was not accepted as contained.

The final fix therefore addresses both sides of the internal dependency without changing the behavior of those other queries.

## Fix

### 1. Reject unreliable nullable dependencies when pruning GROUP BY

`get_required_group_by_exprs_indices` now skips a dependency when both are true:

- the dependency is nullable; and
- at least one determinant/source field is actually nullable.

This is the unsafe nullable-`UNIQUE` case: equal NULL determinant values can occur on multiple rows while target values differ.

The dependency remains usable when:

- it is non-nullable, as for a primary key; or
- its determinant fields are declared `NOT NULL`, as for `UNIQUE NOT NULL`.

### 2. Add only required synthetic GROUP BY columns in SQL planning

`datafusion/sql/src/select.rs` now computes the synthetic dependent columns that are actually needed:

1. Build the initial set of explicit grouping expressions plus aggregate expressions.
2. Rebase SELECT, HAVING, QUALIFY, ORDER BY, and DISTINCT ON expressions against those aggregate outputs.
3. Collect input columns that remain referenced outside aggregate expressions.
4. Add only referenced columns that are functionally determined by the explicit grouping keys.

Consequences:

- `SELECT x FROM t_uniq GROUP BY x` no longer synthesizes unused `y`; its existing plan/result remain unchanged without relying on unsafe pruning.
- `SELECT x, y FROM t_uniq GROUP BY x` still adds referenced `y`, preserving the current behavior and baseline for separately tracked issue 23820.
- Explicit `SELECT x FROM t_uniq GROUP BY x, y` keeps explicit `y`, which fixes issue 23819.
- Aggregate arguments are rebased before column collection, so a column used only inside `SUM(...)`, `COUNT(...)`, and similar functions is not incorrectly added as a grouping key.
- Invalid-query diagnostics retain their previous context. If any referenced input column is not functionally determined, all dependency targets are added as before, and the existing planner error text remains unchanged.

The small `aggregate_projection_exprs` helper centralizes grouping-set flattening used by both the preliminary and final aggregate rewrites.

## Affected files

- `datafusion/common/src/functional_dependencies.rs`
  - Prevents GROUP BY pruning from using nullable dependencies whose determinants can contain NULL.
- `datafusion/sql/src/select.rs`
  - Adds only dependency targets actually referenced outside aggregates, preserving existing plans and other known-bug behavior after the pruning guard becomes conservative.
- `datafusion/sqllogictest/test_files/functional_dependencies.slt`
  - Changes the issue-23819 case from its documented bad result/plan to the correct result/plan.
  - Adds a `UNIQUE NOT NULL` no-regression control.
  - Leaves all other functional-dependency sections unchanged.
- `issue-23819-explanation.md`
  - This temporary explanation.

No manifests, dependencies, lockfiles, generated files, public APIs, CI files, benchmarks, unrelated snapshots, or unrelated expected plans/results are changed.

## Test coverage

The canonical regression is in section 3.2 of `functional_dependencies.slt`.

It verifies both observable contracts:

1. `SELECT x FROM t_uniq GROUP BY x, y` returns three rows: `1`, `NULL`, `NULL`.
2. The optimized logical plan retains both `x` and `y` as grouping keys.

The same file verifies containment:

- Section 3.1: a `PRIMARY KEY` still allows `y` to be pruned.
- Section 3.3: `UNIQUE NOT NULL` still allows `y` to be pruned.
- Section 1.4: DISTINCT over `GROUP BY x` keeps its existing correct result and plan.
- Section 2.3: ORDER BY over a grouped result keeps its existing plan.
- Section 3.4: selecting explicit `y` keeps both grouping fields as before.
- Section 4.2: the behavior and plan tracked by separate issue 23820 remain unchanged.
- Sections 5.1 and 5.2: outer-join NULL-padding result and plan remain unchanged.
- Existing planner-error assertions in `group_by.slt` keep their exact messages.
- The existing `group_by.slt` file passes unchanged and is not part of the branch diff.

## Regression proof: red before the fix, green after the fix

The branch's updated `functional_dependencies.slt` was run against the exact pre-fix base commit `7269d13e63aa6b67d5db08655134d9c6f9c71ce8`.

It failed only the issue-specific result and plan assertions:

```text
SELECT x FROM t_uniq GROUP BY x, y;
expected: 1, NULL, NULL
actual:   1, NULL

EXPLAIN ...
expected: Projection -> Aggregate groupBy=[x, y] -> TableScan [x, y]
actual:   Aggregate groupBy=[x] -> TableScan [x]
```

At branch HEAD, the exact targeted command exits 0:

```bash
cargo test -p datafusion-sqllogictest --test sqllogictests -- functional_dependencies.slt
```

A broader focused run also exits 0:

```bash
cargo test -p datafusion-sqllogictest --test sqllogictests -- functional_dependencies.slt group_by.slt
```

The SQL planner unit/integration tests also exit 0:

```bash
cargo test -p datafusion-sql
```

## Verification commands and results

These commands exit 0 on this branch:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p datafusion-sqllogictest --test sqllogictests -- functional_dependencies.slt
```

The repository-mandated extended local suite also exits 0:

```bash
RUST_BACKTRACE=1 cargo test --profile ci \
  --exclude datafusion-examples \
  --exclude datafusion-benchmarks \
  --exclude datafusion-cli \
  --workspace --lib --tests --bins \
  --features avro,json,backtrace,extended_tests,recursive_protection,parquet_encryption
```

The containment check exits 0:

```bash
git diff --check 7269d13e63aa6b67d5db08655134d9c6f9c71ce8...HEAD
```

### Why `cargo test --workspace --all-features` is not the full-suite gate

That exact command was attempted. It enables the test-only `force_hash_collisions` feature and aborts in the unrelated existing test `dataframe::test_grouping_with_alias` at `datafusion/physical-expr-common/src/binary_map.rs:436`.

The same isolated test was reproduced on the clean pre-fix base commit with the same SIGABRT, proving that failure is not introduced by either issue branch. Fixing or suppressing it would require an unrelated change, which is outside this branch's boundary. The supported full local suite from `AGENTS.md`, shown above, is the full-suite evidence for this change.

## Containment conclusion

Only the nullable-`UNIQUE` explicit GROUP BY case changes behavior. Primary keys, non-null UNIQUE keys, implicit dependency-derived grouping, DISTINCT behavior, ORDER BY behavior, the separately tracked issue-23820 query, outer-join behavior, and existing planner diagnostics retain their previous results and plans.
