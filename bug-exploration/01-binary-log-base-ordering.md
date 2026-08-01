# `log(base, value)` can falsely satisfy an `ORDER BY`

Validated on revision `455a3add52d051a20df9960a726ee9acb98528a3`.

## Summary

`log(base, value)` is reported as preserving the ordering of `value` whenever `base` is a scalar. That is false for bases between zero and one, where the logarithm is decreasing. DataFusion can consequently remove the required sort and return rows in the wrong order.

Likely source fix: 3–12 lines. No API change. Low risk; the conservative fix only retains sorts that are currently removed without proof.

## Cause

`datafusion/functions/src/math/log.rs`, `LogFunc::output_ordering`, matches an ordered value with any singleton base and returns the value's ordering unchanged:

```rust
(
    first @ (SortProperties::Ordered(_) | SortProperties::Singleton),
    SortProperties::Singleton,
) => Ok(first),
```

The base's value/range is not inspected. For $0 < b < 1$, $\log_b(x)$ reverses order.

## Reproduction

Create `/tmp/log-order.csv`:

```text
x
1
2
4
```

Then run:

```sql
CREATE EXTERNAL TABLE log_ordered (x DOUBLE)
STORED AS CSV
WITH ORDER (x ASC)
LOCATION '/tmp/log-order.csv'
OPTIONS ('format.has_header' 'true');

EXPLAIN
SELECT x, log(0.5, x) AS y
FROM log_ordered
ORDER BY y ASC;

SELECT x, log(0.5, x) AS y
FROM log_ordered
ORDER BY y ASC;
```

Observed physical plan: `SortPreservingMergeExec` over the projection, with no `SortExec` that actually orders `y`.

Observed result:

```text
x   y
1  -0
2  -1
4  -2
```

That output is descending by `y`, despite `ORDER BY y ASC`.

Expected:

```text
x   y
4  -2
2  -1
1  -0
```

## Possible fix

The smallest safe correction is to return `SortProperties::Unordered` for the two-argument overload unless the base and value ranges prove a valid monotonic direction. Unary `log(value)`, whose base is fixed at 10, can keep its existing positive-domain optimization.

Do not merely reverse every two-argument result: monotonicity also depends on the base/value domains when both arguments vary.

## Regression test

- Add a range/order unit case to `test_log_output_ordering` in `datafusion/functions/src/math/log.rs`.
- Add the ordered CSV scenario to `datafusion/sqllogictest/test_files/monotonic_projection_test.slt` or `order.slt` and assert both the result and the retained `SortExec`.

## Novelty check

The current top 100 open `bug` issues and targeted searches for binary-log ordering and `log(0.5, ...)` had no matching report. Existing logarithm issues found during the search concern argument wiring, precision, or NULL placement rather than monotonicity for a base below one.
