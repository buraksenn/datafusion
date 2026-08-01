# Implementation plan: timestamp-to-time casts must not preserve ordering

Source report: [02-timestamp-to-time-ordering.md](../02-timestamp-to-time-ordering.md)

Status: planning only. This document does not implement the fix.

## Objective

Prevent DataFusion from claiming that `CAST(timestamp AS time)` preserves global ordering. The cast discards the date and wraps at midnight, so a timestamp-ordered input is not necessarily ordered by the resulting time of day.

An explicit `ORDER BY CAST(ts AS TIME)` must retain a real `SortExec` when the input ordering is only `ts`.

## Current behavior and root cause

`datafusion/physical-expr/src/expressions/cast.rs`, `is_order_preserving_cast_family`, currently treats every temporal-to-temporal cast as order-preserving:

```text
source_type.is_temporal() && target_type.is_temporal()
```

`cast_expr_properties` therefore copies the child's ordered `SortProperties` to timestamp-to-Time32/Time64 casts. `EnsureRequirements` can then remove the requested sort.

The defect is narrower than the whole temporal family: timestamp-to-time specifically discards the day component. This plan intentionally avoids a broad redesign of temporal monotonicity.

## Files to change

| File                                                               | Planned change                                                                                               |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `datafusion/physical-expr/src/expressions/cast.rs`                 | Add table-driven property tests and exclude Timestamp-to-Time32/Time64 from order-preserving temporal casts. |
| `datafusion/sqllogictest/test_files/monotonic_projection_test.slt` | Add ordered-source plan and result regressions spanning midnight.                                            |

No public API or cast execution behavior changes.

## Detailed implementation sequence

### 1. Add direct cast-family/property tests

In the existing `cast.rs` test module, add a focused table-driven test for `is_order_preserving_cast_family`.

Required negative cases:

- `Timestamp(Second, None)` → `Time32(Second)`.
- `Timestamp(Millisecond, None)` → `Time32(Millisecond)`.
- `Timestamp(Microsecond, None)` → `Time64(Microsecond)`.
- `Timestamp(Nanosecond, None)` → `Time64(Nanosecond)`.
- At least one timezone-bearing timestamp → Time64.

Retain positive controls so the guard is not accidentally too broad:

- Timestamp precision widening remains order-preserving.
- Time32 → Time64 remains order-preserving.
- Identity temporal casts remain order-preserving.
- Date32 → Date64 behavior remains unchanged.

Also test `cast_expr_properties`, not only the boolean helper, for at least one timestamp-to-time pair. Construct an ordered child `ExprProperties` with a timestamp interval and assert the result is `SortProperties::Unordered`. This verifies that no other path, such as `check_bigger_cast`, restores the invalid ordering.

### 2. Add an end-to-end ordered-source fixture

Follow the existing scratch-Parquet pattern in `monotonic_projection_test.slt`:

1. `COPY` two rows ordered by timestamp to a scratch Parquet file:
   - `2024-01-01 23:00:00`
   - `2024-01-02 00:00:00`
2. Register the file as an external Parquet table with `WITH ORDER (ts)`.
3. Run `EXPLAIN` for:

```sql
SELECT CAST(ts AS TIME) AS t
FROM timestamp_ordered
ORDER BY t;
```

4. Assert the physical plan contains `SortExec` between the scan/projection and final merge as appropriate.
5. Execute the query and assert `00:00:00` precedes `23:00:00`.

Using a generated scratch file avoids adding a permanent CSV fixture and keeps declared ordering truthful.

### 3. Narrow the temporal-family predicate

Add a small explicit exclusion before or inside the temporal family condition:

- Source matches `DataType::Timestamp(_, _)`.
- Target matches `DataType::Time32(_) | DataType::Time64(_)`.
- This pair returns false for order preservation.

Keep `source_type.eq(target_type)` and existing numeric/widening behavior intact. Do not modify the actual cast kernel.

### 4. Verify sort-property details

Confirm after the change:

- Timestamp-to-time result is unordered regardless of timestamp unit/timezone.
- `strictly_order_preserving` is not claimed.
- Timestamp-to-timestamp and time-to-time casts retain their previous properties.
- The end-to-end query contains a real sort and returns correct order.
- A query ordering by the original timestamp remains optimized as before.

## Invariants and non-goals

- The source's declared timestamp ordering remains valid; only the projected time ordering is removed.
- Do not disable every temporal cast optimization.
- Do not change timezone display or cast values.
- Do not modify numeric cast-ordering rules.
- Do not introduce range proofs limited to a single day in this patch; that is a separate optimization.

## Risks and mitigations

| Risk                                                       | Mitigation                                                                     |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Overly broad guard removes valid temporal ordering         | Match only Timestamp → Time32/Time64 and retain positive-control tests.        |
| Unit helper test passes but optimizer still drops the sort | Add plan-shape and result SQLLogicTests.                                       |
| Test source is not actually ordered                        | Generate the file with `COPY (... ORDER BY ts)` and declare the same ordering. |
| Timezone-bearing timestamps follow a different path        | Include at least one timezone unit in the table-driven unit test.              |

## Focused validation commands

```bash
cargo test -p datafusion-physical-expr expressions::cast --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- monotonic_projection_test
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- Every Timestamp-to-Time32/Time64 pair is reported unordered.
- Other temporal-family controls retain current behavior.
- The midnight-crossing query retains `SortExec` and returns `00:00:00`, then `23:00:00`.
- No cast values, public APIs, or unrelated ordering rules change.
