# Implementation plan: `NTH_VALUE(DISTINCT)` aggregate support

Source issue: [#2406](https://github.com/apache/datafusion/issues/2406)

Validated on revision `07e9281e848509770c23ff6e9cba589c907dba7e`.

Status: valid; planning only. This document does not implement the change.

## Objective

Make the aggregate form of `nth_value(DISTINCT value, n [ORDER BY ...])` select the nth unique value rather than the nth input row.

The implementation must preserve ordering across partial partitions, support positive and negative `n`/reversed planning, and avoid unbounded state beyond what is needed for the requested rank.

## Current behavior and root cause

`NthValueAgg::accumulator` reads `n`, reversal, and `order_bys`, but never reads `AccumulatorArgs.is_distinct`. Both `TrivialNthValueAccumulator` and `NthValueAccumulator` append duplicate values directly.

Observed:

```sql
WITH t(x) AS (VALUES (1), (1), (9))
SELECT
  nth_value(DISTINCT x, 2 ORDER BY x) AS actual_distinct,
  (SELECT nth_value(x, 2 ORDER BY x)
   FROM (SELECT DISTINCT x FROM t)) AS expected_distinct
FROM t;
```

Observed: `1`; expected: `9`.

Unlike covariance-style aggregates, nth-value state already stores raw candidate values and, when ordered, corresponding ordering rows. The fix should extend that state machine rather than introduce a separate all-input set accumulator.

## DISTINCT and ordering contract

- DISTINCT applies to the aggregate argument tuple. `n` is required to be a constant, so in practice uniqueness is determined by `value`.
- For a DISTINCT aggregate with ORDER BY, every ordering expression must appear in the aggregate argument list. Match the validation already used by `array_agg(DISTINCT ... ORDER BY ...)`.
- `nth_value(DISTINCT x, 2 ORDER BY x)` is valid.
- `nth_value(DISTINCT x, 2 ORDER BY unrelated_col)` must error rather than choose an arbitrary representative ordering for duplicate `x` values.
- NULL is a value unless a separately supported null-treatment contract says otherwise; preserve current nth-value NULL behavior.

## Files to change

| File                                               | Planned change                                                                                    |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `datafusion/functions-aggregate/src/nth_value.rs`  | Validate DISTINCT ordering, track seen candidates, deduplicate update/merge paths, and add tests. |
| `datafusion/sqllogictest/test_files/aggregate.slt` | Add ordered, grouped, NULL, reverse, and cross-partition regressions.                             |

No new common tuple buffer is required for the current two-argument signature.

## Detailed implementation sequence

### 1. Add failing SQL regressions

Protect the three-row reproduction and add:

- Positive `n` with ascending and descending order.
- Negative `n` or the planner's reversed-UDAF path.
- Duplicates split across partitions.
- Grouped input.
- NULL duplicates and mixed NULL/non-NULL input.
- Too few unique values returns typed NULL.
- Invalid DISTINCT/ORDER BY expression combinations.
- No-ORDER-BY form as a nondeterministic-but-distinct cardinality control.

Use a mixed aggregate or multiple distinct expressions so the test reaches the nth-value accumulator.

### 2. Validate DISTINCT ordering at accumulator construction

When `is_distinct` and `order_bys` is non-empty, require each physical sort expression to equal one of the physical aggregate argument expressions.

Reuse the error text:

```text
In an aggregate with DISTINCT, ORDER BY expressions must appear in argument list
```

For nth value, `value` and the constant `n` are the argument expressions. The useful normal case orders by `value`.

Do not silently retain the first ordering row for duplicate values ordered by an unrelated expression; that result is not well-defined.

### 3. Add distinct candidate tracking

Pass an `is_distinct` flag into both trivial and ordered accumulator constructors. For distinct instances maintain a `HashSet<ScalarValue, RandomState>` (or an equivalent row representation) for candidate values currently retained.

During `append_new_data`:

1. Read the candidate value.
2. Skip it if already seen.
3. Otherwise append the value and its ordering row together.
4. Stop only after collecting the required number of unique candidates for positive `n`.

For negative/from-end selection, retain the last `abs(n)` unique candidates and remove evicted values from the membership set so a later occurrence can become the retained representative when appropriate.

Keep membership and `VecDeque` state synchronized through every drain/replacement.

### 4. Deduplicate partial-state merging

Each partial distinct accumulator needs to retain at most `abs(n)` distinct candidates from its side. This is sufficient: a value outside a partition's first/last `N` unique values cannot enter the global first/last `N` under the same ordering.

For ordered merging:

1. Use the existing `merge_ordered_arrays` to merge candidate/value rows by sort order.
2. Walk the merged sequence in the selected direction.
3. Keep the first `abs(n)` unique values and matching ordering rows.
4. Rebuild the membership set from the retained values.

For the trivial path, apply the same uniqueness and bound while merging list states in their existing arbitrary partition order.

Do not serialize the hash set separately; the retained value list is the source of truth and is sufficient to rebuild it.

### 5. Keep state fields stable

The current list of candidate values plus optional list-of-struct ordering rows can represent both plain and distinct nth-value states. No state-field shape change is required.

Add tests that state/merge preserve value-ordering alignment after duplicates are removed and candidates are drained.

### 6. Account for memory and reversal

Include the membership set in `size()` for distinct accumulators.

`reverse_expr` returns the same UDAF with `is_reversed` communicated through `AccumulatorArgs`; ensure distinct behavior is identical after `n` sign reversal and ordering reversal.

The current accumulators do not implement retract support. Do not add a partially correct distinct retract path; window frames can recompute until multiplicity-aware retention is designed.

## Invariants and non-goals

- Candidate values and their ordering rows are inserted, merged, and evicted atomically.
- Retained distinct state is bounded by `abs(n)` candidates per partial accumulator.
- Cross-partition duplicates are removed during final merge.
- Invalid DISTINCT ordering is rejected.
- Preserve current plain nth-value behavior and state format.
- Do not change the separate window-function implementation in `datafusion/functions-window`.
- Do not add null-treatment syntax support as part of DISTINCT.

## Risks and mitigations

| Risk                                                         | Mitigation                                                                                |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Membership set and deques diverge                            | Centralize append/evict helpers and assert synchronized lengths/state in tests.           |
| Partial pruning drops a possible global candidate            | Retain first/last `N` unique candidates per partition and document the ordering argument. |
| Duplicate value has unrelated ordering rows                  | Enforce ORDER BY expressions appear in aggregate arguments.                               |
| Reverse/negative `n` keeps the wrong occurrence              | Test positive, negative, ascending, descending, and reversed-UDAF paths.                  |
| Complex `ScalarValue` hashing is missed in memory accounting | Use existing ScalarValue hash/size helpers and test non-primitive values if supported.    |

## Focused validation commands

```bash
cargo test -p datafusion-functions-aggregate nth_value --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- aggregate
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- The reproduction returns `9`, matching explicit deduplication.
- Positive, negative/reversed, grouped, NULL, and multi-partition cases are correct.
- Invalid distinct ordering errors before execution.
- Partial state remains bounded by the requested distinct rank.
- Plain nth-value behavior and state schema remain unchanged.
