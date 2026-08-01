# `information_schema.columns` misreports Float64 and Decimal256 metadata

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

`information_schema.columns` reports a Float64 column as having 24 binary significant digits (the Float32 value) and reports no numeric precision or scale for a Decimal256 column.

Likely source fix: 2–4 lines. No API change. Low risk; this changes metadata only.

## Cause

`datafusion/catalog/src/information_schema.rs`, `InformationSchemaColumnsBuilder::add_column`, contains:

```rust
Float32 => (Some(24), Some(2), None),
Float64 => (Some(24), Some(2), None),
Decimal128(precision, scale) => ...,
_ => (None, None, None),
```

Float64 has 53 binary significant digits, not 24. `Decimal256` carries the same declared precision and scale information as `Decimal128` but falls through to the nonnumeric default.

## Reproduction

```sql
SET datafusion.catalog.information_schema = true;
CREATE TABLE meta_t(f DOUBLE, d DECIMAL(40, 2));

SELECT
  column_name,
  numeric_precision,
  numeric_precision_radix,
  numeric_scale
FROM information_schema.columns
WHERE table_name = 'meta_t'
ORDER BY column_name;
```

Observed:

```text
d  NULL  NULL  NULL
f  24    2     NULL
```

Expected:

```text
d  40    10    2
f  53    2     NULL
```

`DECIMAL(40, 2)` uses Arrow's Decimal256 representation, exposing the missing match arm.

## Possible fix

- Change Float64 numeric precision from 24 to 53.
- Handle `Decimal128(precision, scale) | Decimal256(precision, scale)` in the same decimal arm.

## Regression test

Update/add cases in `datafusion/sqllogictest/test_files/information_schema_columns.slt` for Float16, Float32, Float64, Decimal128, and Decimal256. The Float64 fixture currently records the incorrect value and must be updated.

## Novelty check

The current open bug list and searches for Decimal256 information-schema metadata and Float64 numeric precision found no matching issue or PR. Existing information-schema tickets found during review concern catalog scope or character lengths, not these numeric fields.
