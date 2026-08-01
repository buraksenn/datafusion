# CSV `format.null_regex` is used for inference but ignored during reads

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

CSV schema inference applies `format.null_regex`, but the execution reader drops the option. A token inferred as NULL is later parsed as an ordinary value, causing either a parse error or a literal string instead of SQL NULL.

Likely source fix: 8–15 lines. No API change. Low risk; behavior changes only when the option is explicitly configured.

## Cause

`datafusion/datasource-csv/src/file_format.rs`, `CsvFormat::infer_schema_from_stream`, compiles `options.null_regex` and calls Arrow's `Format::with_null_regex`.

The execution path eventually reaches `datafusion/datasource-csv/src/source.rs`, `CsvSource::builder`. That builder wires delimiter, header, quote, truncation, terminator, projection, escape, and comment—but never calls `ReaderBuilder::with_null_regex`.

## Reproduction

Create `/tmp/null-regex.csv`:

```text
v
1
NULL
2
```

Run:

```sql
CREATE EXTERNAL TABLE null_regex_t(v BIGINT)
STORED AS CSV
LOCATION '/tmp/null-regex.csv'
OPTIONS (
  'format.has_header' 'true',
  'format.null_regex' '^NULL$|^$'
);

SELECT v FROM null_regex_t;
```

Observed: parsing fails on `NULL` as `Int64`.

With `v VARCHAR`, the middle row is the literal string `NULL` and `v IS NULL` is false.

Expected rows: `1`, SQL NULL, `2`.

## Possible fix

Apply the configured regex in `CsvSource::builder` via Arrow's `ReaderBuilder::with_null_regex`. Prefer making the builder fallible and propagating invalid regex configuration as a normal error rather than adding another `expect`.

## Regression test

Extend the existing `infer_schema_with_null_regex` test to collect the data, not just inspect the inferred schema. Cover numeric and string schemas plus an invalid regex.

## Novelty check

No current issue or PR matched `null_regex` execution. Merged PR #13228 introduced the option and schema-inference coverage but did not wire it into `CsvSource` decoding.
