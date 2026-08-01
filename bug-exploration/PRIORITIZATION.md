# DataFusion bug-fix prioritization

This document ranks the 19 reports that remained after the existing-issue/PR audit. Report 11 is intentionally absent because inverse log/power simplification was already discussed in PR reviews.

## Recommendation

Start with **report 14**. It is a one-line arithmetic correction, returns a provably wrong date today, has a deterministic SQL reproduction, and has minimal regression risk.

Recommended opening sequence:

1. Report 14 — formatted `to_date` pre-epoch rounding
2. Report 12 — `to_char` row-varying format collapse
3. Report 10 — `SIMILAR TO` unparsed as `LIKE`
4. Report 05 — empty scalar-subquery nullability
5. Report 02 — timestamp-to-time ordering

If the goal is the absolute smallest low-risk patch regardless of user impact, report 09 is also easy, but it fixes metadata rather than query correctness.

## Ranking method

Each report receives two scores from 1 to 5:

- **Importance**
  - 5: wrong query results in core/common SQL paths or generated SQL
  - 4: panic, state loss, valid-query failure, or wrong result in a narrower feature
  - 3: extreme edge case, validation gap, or narrowly used semantic defect
  - 2: metadata-only defect
- **Ease**
  - 5: approximately 1–4 source lines, one local branch, focused tests
  - 4: approximately 5–8 source lines or two closely related paths
  - 3: approximately 8–15 source lines, caller propagation, or broader test setup
  - 2: approximately 15–25 source lines with a semantic interaction matrix

The combined score is:

```text
priority score = 2 × importance + ease
```

Importance is deliberately weighted twice as heavily as ease. Ties are broken by expected user reach, severity, fix conservatism, and regression-test simplicity. Source-line estimates exclude tests.

## Ranked list

| Rank | Report                                                                                    | Importance | Ease | Score | Reasoning                                                                                                                                                                                             |
| ---: | ----------------------------------------------------------------------------------------- | :--------: | :--: | :---: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    1 | [14 — formatted `to_date` pre-epoch rounding](14-to-date-pre-epoch-rounding.md)           |     5      |  5   |  15   | Returns the wrong calendar date. The fix is a one-line switch to `div_euclid`; positive dates and exact day boundaries remain unchanged.                                                              |
|    2 | [12 — `to_char` row-varying format collapse](12-to-char-row-varying-format.md)            |     5      |  5   |  15   | Produces wrong values for every row after the first. The array-format dispatcher already proves the output must be an array, so the fix is local and easy to defend.                                  |
|    3 | [10 — `SIMILAR TO` unparsed as `LIKE`](10-similar-to-unparsed-as-like.md)                 |     5      |  5   |  15   | Generated SQL can change query results or pushed-down semantics. `sqlparser` already has the correct AST variant; the source change is one expression arm plus tests.                                 |
|    4 | [05 — empty scalar-subquery nullability](05-empty-scalar-subquery-nullability.md)         |     5      |  4   |  14   | A core SQL semantic mismatch causes both execution errors and optimizer wrong results. The correction is conservative but must be applied consistently in two schema methods.                         |
|    5 | [02 — timestamp-to-time ordering](02-timestamp-to-time-ordering.md)                       |     5      |  4   |  14   | Can violate an explicit `ORDER BY`. A small deny-list for timestamp-to-time casts is safe; the main effort is an ordered-source regression fixture.                                                   |
|    6 | [01 — binary-log ordering](01-binary-log-base-ordering.md)                                |     5      |  3   |  13   | Also violates `ORDER BY`, but binary-log monotonicity has more domain combinations. The safest patch disables the optimization unless ranges prove it, which needs careful property tests.            |
|    7 | [03 — negative column pad panic](03-negative-column-pad-panic.md)                         |     4      |  5   |  13   | User data can crash the process. One shared capacity helper fixes both `lpad` and `rpad`; tests should also ensure oversized lengths error before allocation.                                         |
|    8 | [13 — negative `date_bin` TIME](13-date-bin-time-negative-wrap.md)                        |     4      |  5   |  13   | A valid query yields an invalid Arrow TIME/error. `rem_euclid` is a one-line fix with unchanged behavior for nonnegative intermediates.                                                               |
|    9 | [17 — empty-needle array NULL validity](17-array-has-empty-needle-null-validity.md)       |     4      |  5   |  13   | Returns non-NULL booleans for NULL inputs. Attaching the same combined validity buffer used by normal kernels is a small, low-risk correction.                                                        |
|   10 | [19 — `regr_r2` horizontal line](19-regr-r2-horizontal-line.md)                           |     4      |  4   |  12   | Returns the wrong SQL-standard/PostgreSQL result. The branch ordering is straightforward, but numerical aggregate edge cases deserve a focused test matrix.                                           |
|   11 | [07 — runtime SET drops prepared statements](07-runtime-set-drops-prepared-statements.md) |     4      |  4   |  12   | Silently destroys session state. The likely fix is a crate-private runtime-environment replacement method plus SET/RESET lifecycle tests.                                                             |
|   12 | [15 — CSV `null_regex` ignored during reads](15-csv-null-regex-execution.md)              |     4      |  3   |  11   | An explicitly configured option is ignored and can make valid scans fail. Compiling and propagating the regex may require making the reader builder fallible across callers.                          |
|   13 | [06 — UPDATE target normalization](06-update-target-normalization.md)                     |     3      |  5   |  11   | A valid DML statement is rejected inconsistently. The source fix is tiny; quoted identifiers, normalization-disabled sessions, and map keys need explicit tests.                                      |
|   14 | [08 — surplus EXECUTE parameters](08-execute-surplus-parameters.md)                       |     3      |  4   |  10   | Silently ignores caller input. The main design detail is counting logical parameters correctly when types are unknown or placeholder indices repeat.                                                  |
|   15 | [04 — maximum ROWS FOLLOWING overflow](04-window-rows-following-overflow.md)              |     3      |  4   |  10   | Panics rather than clamping, but requires an extreme `u64::MAX` frame. Checked/saturating arithmetic is local; cover both bounds and narrower targets.                                                |
|   16 | [16 — `array_slice` stride overflow](16-array-slice-stride-overflow.md)                   |     3      |  4   |  10   | Panics on an extreme stride after emitting the valid endpoint. Generic checked arithmetic must cover positive/negative directions and List/LargeList offsets.                                         |
|   17 | [18 — `array_to_string` Null leaf](18-array-to-string-null-leaf.md)                       |     3      |  3   |   9   | Produces a wrong string for an uncommon `List(Null)` representation. The fix needs delimiter/first-element handling rather than a single assignment.                                                  |
|   18 | [09 — information-schema numeric metadata](09-information-schema-numeric-metadata.md)     |     2      |  5   |   9   | Probably the easiest patch: one constant and one match arm. It ranks lower because query execution is correct and only metadata consumers are affected.                                               |
|   19 | [20 — wildcard `ILIKE` ignored](20-wildcard-ilike-ignored.md)                             |     3      |  2   |   8   | The feature silently returns the wrong projection, but implementing matching correctly requires `%`/`_`, escaping, Unicode, qualified wildcards, EXCLUDE/REPLACE interactions, and no-match behavior. |

## Suggested work batches

### Batch A: high-value, minimal patches

Reports 14, 12, 10, 05, and 02. These combine wrong-result severity with small, conservative changes and direct regression tests.

### Batch B: contained function and state fixes

Reports 01, 03, 13, 17, 19, and 07. These are still good standalone contributions, with either narrower affected inputs or modestly broader test requirements.

### Batch C: useful but lower-return or broader changes

Reports 15, 06, 08, 04, 16, 18, 09, and 20. They remain valid bugs, but are lower priority because of narrower reach, extreme inputs, metadata-only impact, or larger semantic/test surfaces.
