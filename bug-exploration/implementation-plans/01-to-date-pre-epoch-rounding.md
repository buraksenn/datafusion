# Implementation plan: formatted `to_date` pre-epoch rounding

Source report: [14-to-date-pre-epoch-rounding.md](../14-to-date-pre-epoch-rounding.md)

Status: planning only. This document does not implement the fix.

## Objective

Make formatted `to_date` convert negative, sub-day timestamps to the correct `Date32` day by using floor/Euclidean day arithmetic instead of truncating toward zero.

The target behavior is:

```sql
SELECT to_date(
  '1969-12-31 12:00:00',
  '%Y-%m-%d %H:%M:%S'
);
```

returning `1969-12-31`, not `1970-01-01`.

## Current behavior and root cause

The formatted path in `datafusion/functions/src/datetime/to_date.rs`, `ToDateFunc::to_date`, performs these steps:

1. Parse the string with `string_to_timestamp_millis_formatted`.
2. Convert milliseconds to days using ordinary signed division.
3. Convert the resulting `i64` day index to `i32` for `Date32`.

The current quotient is:

```text
n / (24 * 60 * 60 * 1_000)
```

Rust integer division truncates toward zero. Negative values between `-86_399_999` and `-1` milliseconds therefore become day zero rather than day minus one.

The one-argument `to_date` parser and integer-input paths do not use this arithmetic and are out of scope.

## Files to change

| File                                                    | Planned change                                                                                                          |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `datafusion/functions/src/datetime/to_date.rs`          | Add focused scalar/array regressions and replace truncating day division with Euclidean division in the formatted path. |
| `datafusion/sqllogictest/test_files/datetime/dates.slt` | Add an end-to-end pre-epoch formatted-datetime regression.                                                              |

No public API, signature, or return-type change is required.

## Detailed implementation sequence

### 1. Add a failing focused unit test

Add a new test near `test_to_date_with_format`, rather than broadening its date-only `TestCase` structure. The existing helper computes expected values by parsing a date-only string, so forcing datetime inputs into it would make that test harder to understand.

Create a focused test such as `test_to_date_formatted_pre_epoch_subday` using the existing `invoke_to_date_with_args` helper.

Cover both execution shapes:

- Scalar input plus scalar format.
- String array plus scalar or matching format array.

Use explicit expected `Date32` day indices so the rounding contract is visible:

| Input                     | Expected day index | Expected date |
| ------------------------- | -----------------: | ------------- |
| `1969-12-31 00:00:00.000` |                 -1 | 1969-12-31    |
| `1969-12-31 12:00:00.000` |                 -1 | 1969-12-31    |
| `1969-12-31 23:59:59.999` |                 -1 | 1969-12-31    |
| `1970-01-01 00:00:00.000` |                  0 | 1970-01-01    |
| `1970-01-01 12:00:00.000` |                  0 | 1970-01-01    |

At minimum, assert the noon-before-epoch scalar and a two-row array spanning the epoch. Confirm the test fails before changing the arithmetic.

### 2. Add the SQL regression

In `datetime/dates.slt`, place the case near the existing `to_date` formatting tests:

```sql
SELECT to_date(
  '1969-12-31 12:00:00',
  '%Y-%m-%d %H:%M:%S'
);
```

Expected output: `1969-12-31`.

Optionally add one exact-midnight control in the same record. Avoid timezone-offset cases in this patch; they exercise parsing/timezone policy beyond the identified rounding defect.

### 3. Change only the day quotient

Replace the formatted path's ordinary division with `i64::div_euclid` using the same milliseconds-per-day divisor.

Keep these behaviors unchanged:

- Parser selection and fallback across multiple format strings.
- Error propagation from `string_to_timestamp_millis_formatted`.
- Checked `i64` to `i32` conversion.
- One-argument and numeric-input `to_date` paths.

A named local or module constant for milliseconds per day is optional. Do not refactor the shared parsing helpers as part of this fix.

### 4. Re-run the regression matrix

Verify that:

- Negative sub-day values floor to the previous day.
- Exact negative-day multiples remain unchanged.
- Epoch and positive values remain unchanged.
- Scalar and array inputs agree.
- The output type remains `Date32` and NULL inputs remain NULL.

## Invariants and non-goals

- `Date32` represents whole days since 1970-01-01; conversion must use floor semantics for negative milliseconds.
- Do not change Arrow's general Date64-to-Date32 cast behavior in this patch.
- Do not change timezone interpretation or accepted format syntax.
- Do not add new public configuration or API.
- Do not broaden the patch into other datetime functions unless a shared helper is already used by this exact path.

## Risks and mitigations

| Risk                                            | Mitigation                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| Changing positive-date behavior                 | `div_euclid` equals ordinary division for nonnegative inputs; retain positive controls. |
| Accidentally changing exact pre-epoch midnights | Add an exact `-86_400_000 ms` equivalent test.                                          |
| Testing only scalar constant folding            | Include an array/column path test.                                                      |
| Hiding an `i32` overflow                        | Keep the existing checked conversion and its error behavior.                            |

## Focused validation commands

```bash
cargo test -p datafusion-functions test_to_date --lib
cargo test -p datafusion-sqllogictest --test sqllogictests -- dates
```

Before submission, also run the repository-required formatting, linting, and applicable workspace tests.

## Definition of done

- The new unit test fails on the old arithmetic and passes with the change.
- The SQL reproduction returns `1969-12-31`.
- Scalar and array cases agree around the epoch boundary.
- Existing formatted and unformatted `to_date` tests still pass.
- The patch changes only the formatted milliseconds-to-days conversion and its tests.
