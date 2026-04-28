# `regexpsplittoarray.rs` — line-by-line walkthrough

> File: `datafusion/functions/src/regex/regexpsplittoarray.rs`
>
> Purpose: implements the `regexp_split_to_array(str, pattern[, flags])` scalar UDF. PostgreSQL-compatible; splits a string by a regex and returns the parts as a `List<Utf8|LargeUtf8|Utf8View>`.

This document walks every block of the file, explains what it does, how it does it, why it's written that way, and shows where equivalent patterns exist elsewhere in the DataFusion codebase. Each section ends with a short **Verification** note confirming the code is correct or flagging anything observed.

---

## Table of contents

1. [License header](#1-license-header)
2. [Imports](#2-imports)
3. [`#[user_doc]` attribute](#3-user_doc-attribute)
4. [`RegexpSplitToArrayFunc` struct](#4-regexpsplittoarrayfunc-struct)
5. [`Default` impl](#5-default-impl)
6. [`RegexpSplitToArrayFunc::new` — `Signature` construction](#6-new--signature-construction)
7. [`ScalarUDFImpl` impl](#7-scalarudfimpl-impl)
   - [`name`](#71-name)
   - [`signature`](#72-signature)
   - [`return_type`](#73-return_type)
   - [`invoke_with_args`](#74-invoke_with_args)
   - [`documentation`](#75-documentation)
8. [`regexp_split_to_array_func` — arg validation & routing](#8-regexp_split_to_array_func)
9. [`regexp_split_to_array` — type-specialized dispatch](#9-regexp_split_to_array-dispatch)
10. [`StringListBuilder` trait & impls](#10-stringlistbuilder-trait)
11. [`split_and_append`](#11-split_and_append)
12. [`split_chars_and_append`](#12-split_chars_and_append)
13. [`regexp_split_to_array_inner` — core loop](#13-regexp_split_to_array_inner)
14. [Tests module](#14-tests-module)

---

## 1. License header

```rust
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
...
```

**What**: Standard Apache 2.0 license preamble required for every Apache-project source file.

**Why**: Every `.rs` file in DataFusion must begin with this exact boilerplate. Enforced by `./dev/rust_lint.sh` via `rat` or similar license-header checks.

**Cross-reference**: Identical block at the top of `regexpmatch.rs`, `regexpcount.rs`, and every file under `datafusion/functions/src/`.

**Verification**: ✅ Matches the canonical header; required.

---

## 2. Imports

```rust
use arrow::array::{
    Array, ArrayBuilder, ArrayRef, AsArray, GenericStringBuilder, ListBuilder,
    OffsetSizeTrait, StringArrayType, StringViewBuilder,
};
use arrow::datatypes::DataType;
use arrow::datatypes::DataType::{LargeUtf8, Utf8, Utf8View};
use arrow::datatypes::Field;
use arrow::error::ArrowError;
use datafusion_common::{Result, ScalarValue, exec_err, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarUDFImpl, Signature, TypeSignature::Exact,
    Volatility,
};
use datafusion_macros::user_doc;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;

use crate::regex::{compile_and_cache_regex, compile_regex};
```

### From `arrow::array`

| Import | Role |
|---|---|
| `Array` | Base trait for all Arrow arrays. Used behind `&dyn Array` for runtime polymorphism. |
| `ArrayBuilder` | Parent trait of every `*Builder`. `StringListBuilder` below extends it. |
| `ArrayRef` | Alias for `Arc<dyn Array>`. The standard "pass arrays around cheaply" type. |
| `AsArray` | Extension trait giving ergonomic downcasts: `array.as_string::<i32>()`, `array.as_string_view()`. |
| `GenericStringBuilder` | Builder for `StringArray` / `LargeStringArray` (parameterized by offset type). |
| `ListBuilder` | Builder for `ListArray`. Wraps a child builder for the element values. |
| `OffsetSizeTrait` | Marker trait for `i32` / `i64` (the two valid list/string offset types). |
| `StringArrayType` | Unified trait covering `&StringArray`, `&LargeStringArray`, `&StringViewArray` — lets the inner function be generic over which string layout we got. |
| `StringViewBuilder` | Builder for `StringViewArray` (the "Utf8View" physical representation). |

### From `arrow::datatypes`

- `DataType` — the enum describing column types (`Utf8`, `Int64`, `List(Field)`, etc.)
- `DataType::{LargeUtf8, Utf8, Utf8View}` — re-imported as unqualified names so the signature list stays compact.
- `Field` — schema entry for a column / list element. Needed for `DataType::List(Field)` construction.
- `ArrowError` — Arrow's error type. The low-level splitting function returns `Result<_, ArrowError>`.

### From `datafusion_common`

- `Result` — DataFusion's alias for `Result<T, DataFusionError>`.
- `ScalarValue` — the single-row scalar enum (string, int, list, …).
- `exec_err!` / `internal_err!` — macros that construct `DataFusionError::Execution` / `::Internal`. Prefer `exec_err!` for bad user input; `internal_err!` signals a bug in DataFusion itself.

### From `datafusion_expr`

- `ColumnarValue` — the argument wrapper passed to UDFs: either `Scalar(ScalarValue)` or `Array(ArrayRef)`.
- `Documentation` — type returned by `ScalarUDFImpl::documentation()`.
- `ScalarUDFImpl` — the trait every UDF must implement.
- `Signature` / `TypeSignature::Exact` / `Volatility` — types used to declare what argument combinations the function accepts.

### From `datafusion_macros`

- `user_doc` — procedural macro that generates the `fn doc(&self) -> Option<&Documentation>` method from a structured attribute block (see §3).

### Third-party / std

- `regex::Regex` — the Rust regex engine.
- `HashMap` — used for the regex-compilation cache inside the inner loop.
- `Arc` — for wrapping shared list fields and returned arrays.

### Project-internal

- `crate::regex::{compile_and_cache_regex, compile_regex}` — shared helpers in `regex/mod.rs` that compile a regex with optional inline `(?flags)`. Accept `allow_global: bool` to either strip or reject the `g` flag.

**Cross-reference**: `regexpmatch.rs` and `regexpcount.rs` pull roughly the same set; the Arrow trait objects (`Array`, `AsArray`, `ArrayRef`) are the DataFusion UDF lingua franca.

**Verification**: ✅ All imports are used. No dead imports.

---

## 3. `#[user_doc]` attribute

```rust
#[user_doc(
    doc_section(label = "Regular Expression Functions"),
    description = "Splits a string by a [regular expression] ...",
    syntax_example = "regexp_split_to_array(str, regexp[, flags])",
    sql_example = r#"```sql
> select regexp_split_to_array('hello world', '\s+');
...
```"#,
    standard_argument(name = "str", prefix = "String"),
    argument(name = "regexp", description = "..."),
    argument(name = "flags", description = "...")
)]
```

**What**: A procedural macro attribute that produces a `doc(&self) -> Option<&Documentation>` method on the struct it's attached to.

**How**:
- `doc_section(label = …)` groups the function in the auto-generated docs.
- `description` is shown at the top of the function's doc page.
- `syntax_example` renders the signature line.
- `sql_example` is a raw SQL example (with a fenced ```sql block).
- `standard_argument(name, prefix)` uses a canned description for common args like `str`.
- `argument(name, description)` for custom-described arguments.

The macro expands into a static `Documentation` struct; `documentation()` on the UDF returns a reference to it.

**Why**: DataFusion auto-builds its [user-guide docs page](https://github.com/apache/datafusion/blob/main/docs/source/user-guide/sql/scalar_functions.md) from these annotations. Keeping docs next to the code prevents drift.

**Cross-reference**: Every regex function uses this: `regexpmatch.rs:32-66`, `regexpcount.rs:36-62`, `regexpreplace.rs:51-90`.

**Verification**: ✅ Fields match the macro's expected keys. Table widths in `sql_example` are consistent (verified visually: the `----` dash line and the data rows are the same width).

**Note**: `A NULL value is treated as if no flags were supplied.` was added to the `flags` arg description to document the explicit NULL-flags behavior.

---

## 4. `RegexpSplitToArrayFunc` struct

```rust
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RegexpSplitToArrayFunc {
    signature: Signature,
}
```

**What**: The concrete UDF type. Holds a single field — the immutable `Signature` describing accepted argument shapes.

**How**: Derives:
- `Debug` — needed because `ScalarUDFImpl: Debug`.
- `PartialEq` / `Eq` — used for optimizer equality checks (e.g., deduplicating identical expressions).
- `Hash` — lets the optimizer put UDF expressions in hash-based containers.

**Why**: UDFs are stateless once constructed; all per-call state lives in `invoke_with_args`. The only durable state is the signature.

**Cross-reference**: Same shape in `RegexpMatchFunc` (`regexpmatch.rs:68-71`), `RegexpCountFunc` (`regexpcount.rs:64-67`), `RegexpReplaceFunc` (`regexpreplace.rs:92-95`).

**Verification**: ✅ Derives align with the `ScalarUDFImpl: Debug + DynEq + DynHash + Send + Sync + Any` trait bounds (auto-impls provide `DynEq`/`DynHash` via the blanket impl over `Eq + Hash`).

---

## 5. `Default` impl

```rust
impl Default for RegexpSplitToArrayFunc {
    fn default() -> Self {
        Self::new()
    }
}
```

**What**: Makes `RegexpSplitToArrayFunc::default()` equivalent to `::new()`.

**Why**: The `make_udf_function!` macro in `regex/mod.rs:33-41` registers the UDF using `Default::default()`. Every UDF in this crate follows the same pattern.

**Cross-reference**: Same 4 lines in every other `regexp*.rs` file.

**Verification**: ✅ Trivial; matches convention.

---

## 6. `new` — `Signature` construction

```rust
impl RegexpSplitToArrayFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    Exact(vec![Utf8View, Utf8View]),
                    Exact(vec![Utf8, Utf8]),
                    Exact(vec![LargeUtf8, LargeUtf8]),
                    Exact(vec![Utf8View, Utf8View, Utf8View]),
                    Exact(vec![Utf8, Utf8, Utf8]),
                    Exact(vec![LargeUtf8, LargeUtf8, LargeUtf8]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}
```

**What**: Declares the six valid argument type tuples.

**How**:
- `Signature::one_of(variants, volatility)` — accept any one of the listed type lists.
- `TypeSignature::Exact(vec![...])` — a fixed ordered list of types; the planner must coerce inputs to exactly this shape.
- `Volatility::Immutable` — output depends only on inputs (no RNG, no clock). Lets the optimizer constant-fold calls where all args are constants.

The variant order matters: **planner tries candidates top-to-bottom**. `Utf8View` is listed first because it's now the preferred string layout in DataFusion.

**Why same-type triples only**: The string-type branches in the inner dispatch (`regexp_split_to_array` at §9) require all inputs to share one of `Utf8 | LargeUtf8 | Utf8View`. If mixed types reach the dispatch they'd fall into the catch-all error.

**Cross-reference**:
- `regexpmatch.rs:79-99` — identical structure (same three types × with-or-without-flags).
- `regexpreplace.rs:101-113` — uses `Uniform(N, vec![Utf8View, LargeUtf8, Utf8])` instead (a compact form that accepts N args of *any* one of the listed types). The split function could in principle use that too, but `Exact` mirrors `regexp_match` so the pattern is familiar.
- `Volatility` values:
  - `Immutable` — pure function.
  - `Stable` — stable within a single query (e.g., `current_date`).
  - `Volatile` — can change per row (e.g., `random()`).

**Verification**: ✅ Six variants × `Immutable` is correct. Planner's preference order (`Utf8View` first) is idiomatic.

---

## 7. `ScalarUDFImpl` impl

This is the contract every UDF implements. The full trait definition lives at `datafusion/expr/src/udf.rs:524+`.

### 7.1 `name`

```rust
fn name(&self) -> &str { "regexp_split_to_array" }
```

**What**: Returns the SQL-visible function name.

**Why**: Used for error messages, EXPLAIN output, and function lookup.

**Verification**: ✅ String matches the SQL identifier and the `make_udf_function!` binding in `regex/mod.rs`.

### 7.2 `signature`

```rust
fn signature(&self) -> &Signature { &self.signature }
```

**What**: Gives the planner the type signature declared in `new()`.

**Why**: The planner uses this to decide which variant to coerce arguments into, or to reject the query.

**Verification**: ✅ One-liner; trivial.

### 7.3 `return_type`

```rust
fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
    Ok(match &arg_types[0] {
        DataType::Null => DataType::Null,
        other => DataType::List(Arc::new(Field::new_list_field(other.clone(), true))),
    })
}
```

**What**: Given the (already coerced) arg types, compute the return type.

**How**:
- If the first arg is `Null`, propagate `Null`.
- Otherwise, return `List<T>` where `T` is the first arg's type (so `Utf8 → List<Utf8>`, `Utf8View → List<Utf8View>`, etc.).
- `Field::new_list_field(dtype, nullable)` constructs the element `Field` for the list.
- `Arc::new(...)` — `DataType::List` holds the field as `Arc<Field>` so it can be cheaply cloned.
- `true` — the list's *elements* are nullable (matches the null-pattern-in-column SLT test, where the row itself is null).

**Why mirror the first arg's type**: The implementation always produces a list whose element type is the same string layout as the input. This way callers don't get an unexpected cast on the happy path.

**Cross-reference**:
- `regexpmatch.rs:110-115` uses the identical pattern (Null passthrough + `List<other>`).
- `regexpreplace.rs:117-148` is different — it *flattens* the output back to `Utf8`/`LargeUtf8`/`Utf8View` instead of wrapping in a list (because replace returns a string, not a list).

**Verification**: ✅ The signature only allows `Utf8 | LargeUtf8 | Utf8View` as the first arg, so the catch-all `other` is only those three types plus `Null` in practice. Since Null is handled explicitly, `other.clone()` is safe.

### 7.4 `invoke_with_args`

```rust
fn invoke_with_args(
    &self,
    args: datafusion_expr::ScalarFunctionArgs,
) -> Result<ColumnarValue> {
    let args = &args.args;

    let len = args
        .iter()
        .fold(Option::<usize>::None, |acc, arg| match arg {
            ColumnarValue::Scalar(_) => acc,
            ColumnarValue::Array(a) => Some(a.len()),
        });

    let is_scalar = len.is_none();
    let inferred_length = len.unwrap_or(1);
    let args = args
        .iter()
        .map(|arg| arg.to_array(inferred_length))
        .collect::<Result<Vec<_>>>()?;

    let result = regexp_split_to_array_func(&args);
    if is_scalar {
        let result = result.and_then(|arr| ScalarValue::try_from_array(&arr, 0));
        result.map(ColumnarValue::Scalar)
    } else {
        result.map(ColumnarValue::Array)
    }
}
```

**What**: The entry point the engine calls for each batch of rows.

**How** — line by line:

| Lines | Meaning |
|---|---|
| `let args = &args.args;` | `ScalarFunctionArgs` is a struct; `.args` is the `Vec<ColumnarValue>`. We keep it as a reference to avoid cloning. |
| `.fold(Option::<usize>::None, ...)` | Walk all arguments. For each `Array` arg, record its length. Scalars leave the accumulator unchanged. |
| `is_scalar = len.is_none()` | If *every* arg was `Scalar`, the result should be a `Scalar` too. |
| `inferred_length = len.unwrap_or(1)` | If there were array args they all have the same length (enforced upstream); pick that length. Otherwise 1 (one scalar row). |
| `args.iter().map(|arg| arg.to_array(inferred_length))` | Materialize every `ColumnarValue` into a concrete `ArrayRef`. `to_array(n)` on a `Scalar` produces an `n`-element array of repeated copies; on an `Array` it returns the inner array. |
| `.collect::<Result<Vec<_>>>()?` | Propagate any materialization error. |
| `regexp_split_to_array_func(&args)` | Do the real work. Always receives `&[ArrayRef]`. |
| `if is_scalar { ScalarValue::try_from_array(&arr, 0) }` | When the caller passed only scalars, extract the single row back into a `ScalarValue` so the optimizer can constant-fold. |

**Why two output shapes**: The optimizer distinguishes `Scalar(ScalarValue::List(...))` from `Array(ListArray)`; constant-folding requires scalar.

**Why `to_array(inferred_length)` up front**: It's the simplest path — after this point, every arg is a concrete array and the inner code doesn't have to branch on scalar-vs-array. The *downside* is that a scalar regex argument gets materialized into N copies, so the inner code's scalar detection (`regex_array.len() == 1`) only fires when every arg is a scalar (see the **Known limitation** note in §13).

**Cross-reference**:
- `regexpmatch.rs:117-141` — essentially identical.
- `regexpcount.rs:107-132` — same shape but later uses `Datum::get()` on the resulting arrays to recover the scalar/array distinction, which is what a performance-optimized variant of this function would do.
- `regexpreplace.rs:149-168` — keeps the `ColumnarValue` slice and dispatches off `is_scalar` at a lower level (the most sophisticated approach).

**Verification**: ✅ Correct. Matches `regexp_match` template. Performance caveat called out in §13.

### 7.5 `documentation`

```rust
fn documentation(&self) -> Option<&Documentation> { self.doc() }
```

**What**: Exposes the `#[user_doc]`-generated static documentation to the engine.

**How**: `self.doc()` is a method injected onto the struct by the `user_doc` macro.

**Cross-reference**: Same one-liner everywhere.

**Verification**: ✅ Correct.

---

## 8. `regexp_split_to_array_func`

```rust
fn regexp_split_to_array_func(args: &[ArrayRef]) -> Result<ArrayRef> {
    let args_len = args.len();
    if !(2..=3).contains(&args_len) {
        return exec_err!(
            "regexp_split_to_array was called with {args_len} arguments. It requires at least 2 and at most 3."
        );
    }

    let values = &args[0];
    match values.data_type() {
        Utf8 | LargeUtf8 | Utf8View => (),
        other => {
            return internal_err!(
                "Unsupported data type {other:?} for function regexp_split_to_array"
            );
        }
    }

    let flags_array = if args_len > 2 {
        Some(args[2].as_ref())
    } else {
        None
    };

    regexp_split_to_array(values, args[1].as_ref(), flags_array).map_err(|e| e.into())
}
```

**What**: Defensive argument-validation shim that routes into the type-specialized dispatch.

**How**:
- `(2..=3).contains(&args_len)` is Rust's range-contains idiom — readable "is it 2 or 3?".
- `values.data_type()` returns a `&DataType`; the `match` pattern matches on the unqualified names imported at the top.
- `args[1].as_ref()` — `ArrayRef` is `Arc<dyn Array>`; `.as_ref()` gives a `&dyn Array` which is what the next function wants.
- `.map_err(|e| e.into())` — converts `ArrowError` into `DataFusionError` (the `From<ArrowError>` impl lives in `datafusion-common`).

**Why `internal_err!` (not `exec_err!`) for type mismatch**: Signatures were already validated by the planner. If we reach this code with a non-string type, that's a bug in DataFusion — `internal_err` conveys that.

**Why `exec_err!` for arity**: The arity is a user-facing error (too few / too many args). `exec_err` maps to `DataFusionError::Execution`.

**Cross-reference**: `regexpmatch.rs:148-189`, `regexpcount.rs:139-164`. Same validation shape.

**Verification**: ✅ Both error categorizations are consistent with convention. Note: the arity check is strictly redundant because the `Signature` already restricts to 2 or 3, but it's defensive programming — a safety net if this function is called from Rust outside the UDF pipeline (e.g., tests).

---

## 9. `regexp_split_to_array` — type-specialized dispatch

```rust
fn regexp_split_to_array(
    values: &dyn Array,
    regex_array: &dyn Array,
    flags_array: Option<&dyn Array>,
) -> Result<ArrayRef, ArrowError> {
    match (values.data_type(), regex_array.data_type(), flags_array) {
        (Utf8, Utf8, None) => regexp_split_to_array_inner::<_, GenericStringBuilder<i32>>(
            &values.as_string::<i32>(),
            &regex_array.as_string::<i32>(),
            None,
        ),
        (Utf8, Utf8, Some(flags)) if *flags.data_type() == Utf8 => { ... }
        (LargeUtf8, LargeUtf8, None) => { ... }
        (LargeUtf8, LargeUtf8, Some(flags)) if *flags.data_type() == LargeUtf8 => { ... }
        (Utf8View, Utf8View, None) => { ... }
        (Utf8View, Utf8View, Some(flags)) if *flags.data_type() == Utf8View => { ... }
        _ => Err(ArrowError::ComputeError(
            "regexp_split_to_array() expected the input arrays to be of type Utf8, LargeUtf8, or Utf8View and the data types to match".to_string(),
        )),
    }
}
```

**What**: A big `match` that downcasts the trait-object arrays to concrete typed references, picks a builder offset type, then calls the generic `regexp_split_to_array_inner` monomorphization.

**How**:
- `values.as_string::<i32>()` returns `&StringArray` (offset `i32`).
- `values.as_string::<i64>()` returns `&LargeStringArray`.
- `values.as_string_view()` returns `&StringViewArray`.
- The builder type parameter (`GenericStringBuilder<i32>`, etc.) controls the output list's element type.

**Match guards**: `if *flags.data_type() == Utf8` is a guard pattern — the arm only fires when the flags array is also `Utf8`. This is belt-and-suspenders (the signature already enforces matching types across all string args), but makes the dispatch correct even when called with hand-rolled arguments in tests.

**Why compile-time monomorphization**: Calling a generic function with a specific type parameter makes `rustc` emit specialized machine code per type — no runtime dispatch inside the hot inner loop. The *outer* dispatch (this `match`) runs once per batch.

**Cross-reference**: This exact "match on all-three-types → call generic inner" pattern mirrors how `arrow-string`'s regex kernels handle the same three string layouts. `regexpmatch.rs` delegates to `arrow::compute::kernels::regexp::regexp_match` which has the same fan-out internally.

**Verification**: ✅ Each arm consumes the downcast result via `&`; the turbofish `::<_, Builder>` lets rustc infer the `S` (string-array type) while the builder stays explicit.

---

## 10. `StringListBuilder` trait & impls

```rust
trait StringListBuilder: ArrayBuilder {
    fn new_builder() -> Self;
    fn append_str_value(&mut self, val: &str);
}

impl<O: OffsetSizeTrait> StringListBuilder for GenericStringBuilder<O> {
    fn new_builder() -> Self { GenericStringBuilder::<O>::new() }
    fn append_str_value(&mut self, val: &str) { self.append_value(val); }
}

impl StringListBuilder for StringViewBuilder {
    fn new_builder() -> Self { StringViewBuilder::new() }
    fn append_str_value(&mut self, val: &str) { self.append_value(val); }
}
```

**What**: A tiny adapter trait that abstracts over the three string builders so the inner loop can be generic.

**How**:
- Super-trait `ArrayBuilder` — required for use inside `ListBuilder<B>`, which wants `B: ArrayBuilder`.
- `new_builder()` — constructor, since all three builders have the same-named `::new()` but no common trait method for it.
- `append_str_value(&mut self, val: &str)` — unified `append_value(&str)` call. `GenericStringBuilder` and `StringViewBuilder` both already have this method, but with inherent impls, not through a shared trait — so the adapter is necessary.

**Why two `impl` blocks**:
- `impl<O: OffsetSizeTrait> ... for GenericStringBuilder<O>` — covers `GenericStringBuilder<i32>` and `GenericStringBuilder<i64>` in one generic impl.
- `impl ... for StringViewBuilder` — `StringViewBuilder` is a different type; its internal storage model (variadic-size views) isn't offset-parameterized, so it gets its own impl.

**Cross-reference**:
- `datafusion/functions-nested/src/*` uses several similar ad-hoc trait bridges over Arrow builders.
- `arrow::array::StringArrayType` (imported above) is a similar unified trait on the *reader* side.

**Verification**: ✅ Both impls forward trivially to inherent methods. The `OffsetSizeTrait` bound is necessary to constrain `O` to `i32`/`i64`.

---

## 11. `split_and_append`

```rust
fn split_and_append<B: StringListBuilder>(
    list_builder: &mut ListBuilder<B>,
    value: &str,
    pattern: &Regex,
) {
    for part in pattern.split(value) {
        list_builder.values().append_str_value(part);
    }
    list_builder.append(true);
}
```

**What**: Splits `value` by `pattern`, pushing each piece as an element into the current list row, then closes the row as non-null.

**How**:
- `pattern.split(value)` returns an iterator of `&str` chunks between matches.
- `list_builder.values()` returns a `&mut B` — the *child* builder that holds the list's element values.
- `.append_str_value(part)` pushes one element.
- `list_builder.append(true)` finalizes the current list row, marking it non-null. The `true` is the "validity" bit: `true` means the row is present.

**Why the `append(true)` at the end**: `ListBuilder` is row-oriented — each call to `.append(...)` closes the row that's been accumulated in the child builder since the last close. Miss this and the offsets get misaligned and no row is ever emitted.

**Cross-reference**: Same pattern used in `array_agg` / `array_distinct` builders under `functions-nested`. The `values()` + `append(true)` dance is the standard Arrow `ListBuilder` idiom.

**Verification**: ✅ Correct list-builder usage. Edge cases:
- If `value` is `""` and `pattern` is non-empty and doesn't match, `split` yields `[""]` (one empty chunk) — row is `[""]`. Matches PostgreSQL.
- If `pattern` has zero-width matches (e.g., `\b`), `split` still yields non-overlapping chunks; no panics.

---

## 12. `split_chars_and_append`

```rust
// Special-case for an empty pattern so the result matches PostgreSQL's
// `regexp_split_to_array('abc', '') = {a,b,c}`. Rust's `Regex::new("").split("abc")`
// matches every zero-width boundary and yields `["", "a", "b", "c", ""]` — which
// would surface as extra leading/trailing empty strings.
fn split_chars_and_append<B: StringListBuilder>(
    list_builder: &mut ListBuilder<B>,
    value: &str,
) {
    let mut buf = [0u8; 4];
    for c in value.chars() {
        list_builder
            .values()
            .append_str_value(c.encode_utf8(&mut buf));
    }
    list_builder.append(true);
}
```

**What**: A specialization used when the regex pattern is empty — splits the input into individual Unicode code points.

**How**:
- `value.chars()` iterates `char` (Rust Unicode scalar values).
- `[0u8; 4]` — a stack buffer sized for the longest possible UTF-8 sequence (max 4 bytes).
- `c.encode_utf8(&mut buf)` writes the character's UTF-8 bytes into `buf` and returns a `&str` slice pointing at the written bytes. No heap allocation per character.
- `.append_str_value(...)` pushes that `&str` into the element builder.

**Why this exists at all**: As the comment explains, Rust's `regex::Regex::new("").split(s)` yields leading/trailing empty strings because the empty regex matches between every character *including* both ends. PostgreSQL returns `{a,b,c}` for `regexp_split_to_array('abc', '')` — three elements, no leading/trailing empty. To match PG, we bypass `Regex::split` entirely for the empty-pattern case.

**Why `encode_utf8` instead of `c.to_string()`**: The earlier version used `&c.to_string()` which heap-allocates a `String` per character. For a 1 MB input string that's 1 M+ allocations. `encode_utf8` reuses the same 4-byte stack buffer.

**Cross-reference**: `std::char::encode_utf8` is the canonical zero-alloc idiom. Used internally by `str::chars` reverse iteration and by JSON/URL encoders.

**Verification**: ✅ `encode_utf8` returns a `&str` with a lifetime tied to `buf`; the builder's `append_value` copies the bytes into its own storage before returning, so by the next iteration the buffer can be safely reused. No aliasing hazard.

---

## 13. `regexp_split_to_array_inner` — core loop

```rust
fn regexp_split_to_array_inner<'a, S, B>(
    values: &S,
    regex_array: &S,
    flags_array: Option<&S>,
) -> Result<ArrayRef, ArrowError>
where
    S: StringArrayType<'a>,
    B: StringListBuilder,
{
    ...
}
```

**What**: The real work. Decides, for each row, whether to split-by-regex or split-into-chars, using a cached compile when the regex varies per row.

**Generics**:
- `S: StringArrayType<'a>` — any string-array layout (`&StringArray`, `&LargeStringArray`, `&StringViewArray`).
- `B: StringListBuilder` — matching builder for the output list's elements.
- Lifetime `'a` — tied to the underlying string buffer from which `value(i)` borrows.

### Scalar detection (lines 294–308)

```rust
let is_regex_scalar = regex_array.len() == 1;
let is_flags_scalar = flags_array.is_none_or(|f| f.len() == 1);

let regex_is_null_scalar = is_regex_scalar && regex_array.is_null(0);

let regex_scalar = if is_regex_scalar && !regex_array.is_null(0) {
    Some(regex_array.value(0))
} else {
    None
};

let flags_scalar = match flags_array {
    Some(fa) if is_flags_scalar && !fa.is_null(0) => Some(fa.value(0)),
    _ => None,
};
```

**What**: Decides whether regex / flags vary per row, and caches the single scalar value if they don't.

**How**:
- `is_none_or` — `Option::is_none_or(|x| pred(x))` returns `true` if `None`, else runs the predicate. Here: "is there no flags arg, or is it scalar?" Introduced in Rust 1.82; replaces the older `map_or(true, ...)` idiom.
- `regex_is_null_scalar` — captures the specific edge case "regex is a single null scalar" so we can skip everything and emit all nulls.

**Why detect scalars**: if both regex and flags are scalar, we can compile the regex **once** before the loop instead of per-row. Big perf win on repeated-pattern queries.

**⚠️ Known limitation**: `invoke_with_args` above eagerly expands all scalar args to full arrays via `to_array(inferred_length)`. So `is_regex_scalar = regex_array.len() == 1` only fires when every arg was scalar (`inferred_length == 1`). In the common case `regexp_split_to_array(col, 'const')`, the `'const'` regex is broadcast to N rows, so `is_regex_scalar` is `false` even though semantically the regex is constant. The cache in the per-row branch rescues correctness (only one compile across all rows) but still pays O(n) cache lookups. See `regexpcount.rs:187-195` for how to preserve the scalar distinction via the `Datum` trait.

### All-null shortcut (lines 312–317)

```rust
if regex_is_null_scalar {
    for _ in 0..values.len() {
        list_builder.append(false);
    }
    return Ok(Arc::new(list_builder.finish()));
}
```

**What**: If the pattern is a single null scalar, every row's output is null.

**How**: `list_builder.append(false)` — the `false` marks the row as null. `values()` is never populated.

**Why**: Short-circuit for a common use pattern (NULL pattern in a SQL expression). Also matches the behavior of DataFusion's other regex functions (NULL pattern → NULL output).

**Verification**: ✅ Correct — `ListArray` handles null rows by setting the row's validity bit to 0 without adding any elements.

### The four-arm dispatch (lines 319–418)

```rust
match (is_regex_scalar, is_flags_scalar) {
    (true, true)   => { /* fast path: single compile */ }
    (true, false)  => { /* regex constant, flags per row */ }
    (false, true)  => { /* regex per row, flags constant */ }
    (false, false) => { /* everything per row */ }
}
```

**What**: Four distinct code paths, each optimized for its specific scalar-vs-array pattern.

**How, arm-by-arm**:

#### `(true, true)` — both scalar

```rust
let regex_str = regex_scalar.unwrap();
if regex_str.is_empty() {
    // empty pattern → char-by-char split
    for i in 0..values.len() { ... split_chars_and_append(...) }
} else {
    let pattern = compile_regex(regex_str, flags_scalar, true)?;
    for i in 0..values.len() { ... split_and_append(..., &pattern) }
}
```

- `.unwrap()` is safe because `regex_is_null_scalar` was caught earlier and returned.
- `compile_regex(..., allow_global=true)` — the `true` tells the shared helper to silently strip the `g` flag (PostgreSQL accepts it; splitting is inherently global).
- Compile **once**, split N times.

#### `(true, false)` — regex scalar, flags per row

```rust
let regex_str = regex_scalar.unwrap();
let flags_array = flags_array.unwrap();
if regex_str.is_empty() {
    // empty pattern → ignore flags entirely (flags mean nothing for char split)
    ...
} else {
    let mut regex_cache = HashMap::new();
    for i in 0..values.len() {
        ...
        let flags = if flags_array.is_null(i) { None } else { Some(flags_array.value(i)) };
        let pattern = compile_and_cache_regex(regex_str, flags, true, &mut regex_cache)?;
        split_and_append(&mut list_builder, values.value(i), pattern);
    }
}
```

- `HashMap` keyed by `(regex_str, flags)` — so identical per-row flag values compile once and hit the cache thereafter.
- NULL flag is treated as "no flags" (matches `regexp_match`'s convention; documented in the `user_doc` block).

#### `(false, true)` — regex per row, flags scalar

```rust
let mut regex_cache = HashMap::new();
for i in 0..values.len() {
    if values.is_null(i) || regex_array.is_null(i) {
        list_builder.append(false);
    } else if regex_array.value(i).is_empty() {
        split_chars_and_append(&mut list_builder, values.value(i));
    } else {
        let regex_str = regex_array.value(i);
        let pattern = compile_and_cache_regex(regex_str, flags_scalar, true, &mut regex_cache)?;
        split_and_append(&mut list_builder, values.value(i), pattern);
    }
}
```

- Per-row empty-pattern check is repeated here (unlike the scalar branch where it's hoisted outside the loop).
- Any row with a null value *or* null pattern yields a null list row.

#### `(false, false)` — both per-row

Same as `(false, true)` but also fetches flags per row and passes them to the cache.

**Why four arms instead of one general one**: Each arm avoids conditionals that would otherwise run per row. In the `(true, true)` arm, `compile_regex` is called *once* — not once per row with a cache-hit check. Tradeoff: more code duplication, but hot-loop efficiency is noticeable on large batches.

**Cross-reference**:
- `regexpcount.rs:300-545` uses an 8-arm dispatch (regex × start × flags, each scalar/array). Same philosophy: branch once, loop cleanly.
- `regexpinstr.rs` has a simpler one that always goes per-row with a cache.

### Final finalize (lines 420)

```rust
Ok(Arc::new(list_builder.finish()))
```

**What**: Calls `finish()` on the `ListBuilder` to produce a `ListArray`, wraps it in `Arc<dyn Array>` (which is `ArrayRef`), and returns `Ok`.

**Verification**: ✅ `ListBuilder::finish(&mut self) -> ListArray`; `ListArray: Array`, so `Arc::new(...)` coerces to `ArrayRef` via the unsized coercion rules.

---

## 14. Tests module

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, GenericStringArray, ListArray, StringViewArray};
    use arrow::datatypes::Field;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::ScalarFunctionArgs;

    fn invoke_with_scalars(args: &[ScalarValue]) -> Result<ColumnarValue> { ... }
    fn result_to_string_vec(result: &ColumnarValue) -> Vec<String> { ... }

    // 19 unit tests total
}
```

**What**: `#[cfg(test)]` — only compiled under `cargo test`. Contains helper functions and 19 test cases.

### Helper: `invoke_with_scalars`

```rust
fn invoke_with_scalars(args: &[ScalarValue]) -> Result<ColumnarValue> {
    let args_values = args.iter().map(|sv| ColumnarValue::Scalar(sv.clone())).collect();
    let arg_fields = args
        .iter()
        .enumerate()
        .map(|(idx, a)| Field::new(format!("arg_{idx}"), a.data_type(), true).into())
        .collect::<Vec<_>>();
    let return_type = match args[0].data_type() {
        LargeUtf8 => DataType::List(Arc::new(Field::new_list_field(LargeUtf8, true))),
        Utf8View => DataType::List(Arc::new(Field::new_list_field(Utf8View, true))),
        _ => DataType::List(Arc::new(Field::new_list_field(Utf8, true))),
    };
    RegexpSplitToArrayFunc::new().invoke_with_args(ScalarFunctionArgs {
        args: args_values,
        arg_fields,
        number_rows: 1,
        return_field: Field::new("f", return_type, true).into(),
        config_options: Arc::new(ConfigOptions::default()),
    })
}
```

**What**: Simulates the engine calling `invoke_with_args` with a list of scalar values.

**How**:
- Each `ScalarValue` is wrapped in `ColumnarValue::Scalar(...)`.
- `arg_fields` — schema metadata for each arg (name, type, nullable).
- `return_type` — duplicates the `return_type` logic inline (tests don't call `return_type` through the trait).
- `ConfigOptions::default()` — empty runtime config.

**Why this exists**: `invoke_with_args` expects a full `ScalarFunctionArgs` struct with schema + config — constructing that boilerplate in every test would be noisy. The helper centralizes it.

**Cross-reference**: `regexpmatch.rs` tests call the `regexp_match(&[ArrayRef])` helper directly rather than going through the UDF trait. This file's tests go through the trait because they want to verify the `ColumnarValue::Scalar` return shape (`try_from_array`).

**Verification**: ✅ `Field::new(...).into()` converts `Field` → `Arc<Field>` via `From`. The `Arc<ConfigOptions>` is required by the trait.

### Helper: `result_to_string_vec`

```rust
fn result_to_string_vec(result: &ColumnarValue) -> Vec<String> {
    match result {
        ColumnarValue::Scalar(ScalarValue::List(arr)) => {
            let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
            let values = list_arr.value(0);
            match values.data_type() {
                Utf8View => {
                    let str_arr = values.as_string_view();
                    (0..str_arr.len()).map(|i| str_arr.value(i).to_string()).collect()
                }
                _ => {
                    let str_arr = values.as_string::<i32>();
                    (0..str_arr.len()).map(|i| str_arr.value(i).to_string()).collect()
                }
            }
        }
        _ => panic!("Expected scalar list result"),
    }
}
```

**What**: Converts a `ColumnarValue::Scalar(List(...))` into `Vec<String>` for easy `assert_eq!`.

**How**:
- `ScalarValue::List(arr)` — `arr` is `Arc<GenericListArray<i32>>`.
- `arr.as_any().downcast_ref::<ListArray>()` — escape the `Arc<dyn Array>` wrapper.
- `list_arr.value(0)` — fetch row 0's inner array (an `ArrayRef`).
- Match on its data type to pick the right string-array accessor.

**Why**: Test assertions are cleaner as `assert_eq!(vec, vec!["a", "b"])` than as raw ListArray comparisons.

**Note**: Only handles the `Utf8View` and `i32`-offset (`Utf8`) cases. For `LargeUtf8`, the existing test (`test_largeutf8`) inlines its own downcast.

**Verification**: ✅ `as_any().downcast_ref::<ListArray>()` is safe given the SQL signature only produces `List<Utf8|LargeUtf8|Utf8View>`.

### Unit tests

| # | Test | What it verifies |
|---|---|---|
| 1 | `test_basic_split` | `'hello world'` + `\s+` → `[hello, world]` |
| 2 | `test_case_insensitive` | `'HeLLo'`, `'l'`, `'i'` flag → `[He, , o]` (three pieces) |
| 3 | `test_no_match` | Pattern with no match returns `[whole_string]` |
| 4 | `test_null_value` | NULL input → NULL list row |
| 5 | `test_null_pattern` | NULL pattern → NULL list row |
| 6 | `test_empty_pattern_splits_chars` | `'abc'` + `''` → `[a, b, c]` (the §12 behavior) |
| 7 | `test_pattern_at_boundaries` | `',a,,b,'` + `','` → `["", "a", "", "b", ""]` (no trimming) |
| 8 | `test_unicode` | Multi-byte chars preserved |
| 9 | `test_global_flag_ignored` | `'g'` flag accepted and stripped, result matches no-flag version |
| 10 | `test_largeutf8` | Works end-to-end for `LargeUtf8` layout |
| 11 | `test_utf8view` | Works end-to-end for `Utf8View` layout (also checks element type) |
| 12 | `test_array_input` | Column input with per-row patterns |
| 13 | `test_array_input_with_flags` | Column + per-row flags |
| 14 | `test_stringview_array` | `Utf8View` column in & out |
| 15 | `test_null_flags_treated_as_no_flags` | NULL flag → behaves as no-flags (doc'd contract) |
| 16 | `test_largeutf8_array_input` | `LargeUtf8` column in & out |
| 17 | `test_mixed_nulls_in_column` | Some rows null (input or pattern), others not |
| 18 | `test_invalid_regex_errors` | Malformed regex returns `Regular expression did not compile` error |
| 19 | `test_scalar_str_array_pattern` | Scalar string × array pattern — exercises the `(false, true)` arm |

**Verification**: ✅ All 19 pass locally. The mix covers every arm of the four-way dispatch plus every string layout.

---

## Summary of observations

| Area | Status |
|---|---|
| Compile | ✅ Builds clean |
| Clippy | ✅ `-D warnings` clean |
| Formatting | ✅ `cargo fmt --check` passes |
| Unit tests | ✅ 19/19 pass |
| SLT tests | ✅ All 6 regexp files pass |
| Docs table widths | ✅ Aligned in both `user_doc` and `scalar_functions.md` |
| NULL-flags contract | ✅ Documented + tested |
| Empty-pattern PG parity | ✅ Documented via §12 comment |
| Dead code | ✅ None (`unwrap_or("")` removed; duplicate compile helpers merged into shared `regex/mod.rs`) |
| Known perf gap | ⚠️ Scalar detection defeated by eager `to_array` in `invoke_with_args`; cache rescues correctness but not cache-lookup overhead. Follow-up PR candidate — use `Datum` trait like `regexpcount.rs:187-195`. |

## Reference map

- `datafusion/expr/src/udf.rs:524+` — the `ScalarUDFImpl` trait.
- `datafusion/functions/src/regex/mod.rs` — `compile_regex` / `compile_and_cache_regex` (shared helpers, `allow_global: bool`).
- `datafusion/functions/src/regex/regexpmatch.rs` — closest sibling; same signature/invoke shape.
- `datafusion/functions/src/regex/regexpcount.rs` — for the `Datum`-based scalar-preservation pattern.
- `datafusion/functions/src/regex/regexpreplace.rs` — more advanced `ColumnarValue`-native dispatch; reference for a future perf upgrade.
- `datafusion/sqllogictest/test_files/regexp/regexp_split_to_array.slt` — SQL-level tests.
- `docs/source/user-guide/sql/scalar_functions.md` — user-facing docs (auto-derivable section).
