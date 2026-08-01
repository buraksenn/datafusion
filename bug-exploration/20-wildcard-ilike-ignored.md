# SELECT-list wildcard `ILIKE` is parsed but ignored

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

DataFusion accepts Snowflake-style `SELECT * ILIKE 'pattern'`, stores the option in the logical wildcard expression, and then expands every column without consulting the pattern. The query silently returns a different projection than requested.

Likely source fix: 15–20 lines. No API change. Low risk when pattern matching reuses DataFusion/Arrow's existing ILIKE semantics.

## Cause

`datafusion/sql/src/select.rs`, `plan_wildcard_options`, preserves `options.opt_ilike` in `WildcardOptions.ilike`.

`datafusion/expr/src/utils.rs`, both `expand_wildcard` and `expand_qualified_wildcard`, destructure only EXCLUDE/EXCEPT fields. Neither reads `ilike`; repository usage otherwise only formats/displays the stored option.

## Reproduction

```sql
SELECT * ILIKE '%1'
FROM (VALUES (1, 2));
```

Observed:

```text
column1  column2
1        2
```

Expected:

```text
column1
1
```

Only `column1` matches the case-insensitive `%1` pattern.

## Possible fix

Filter wildcard candidate field names by `WildcardOptions.ilike.pattern` before applying exclusions and replacements. Reuse the existing ILIKE kernel/semantics rather than hand-implementing `%`, `_`, escaping, Unicode, and case folding. Apply the same logic to qualified wildcards.

## Regression test

Add cases to `datafusion/sqllogictest/test_files/wildcard.slt` for:

- exact and `%`/`_` patterns;
- qualified wildcards;
- escaped wildcard characters;
- interaction with EXCLUDE and REPLACE;
- no matches.

## Novelty check

The current open bug list and targeted searches for Snowflake wildcard ILIKE and `WildcardOptions.ilike` found no matching issue or PR. Existing tests only verify display formatting, not expansion.
