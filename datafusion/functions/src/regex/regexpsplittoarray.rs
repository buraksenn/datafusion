// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

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

#[user_doc(
    doc_section(label = "Regular Expression Functions"),
    description = "Splits a string by a [regular expression](https://docs.rs/regex/latest/regex/#syntax) pattern and returns a text array of the splits.",
    syntax_example = "regexp_split_to_array(str, regexp[, flags])",
    sql_example = r#"```sql
> select regexp_split_to_array('hello world', '\s+');
+--------------------------------------------------------+
| regexp_split_to_array(Utf8("hello world"),Utf8("\s+")) |
+--------------------------------------------------------+
| [hello, world]                                         |
+--------------------------------------------------------+

> select regexp_split_to_array('HeLLo', 'l', 'i');
+----------------------------------------------------------+
| regexp_split_to_array(Utf8("HeLLo"),Utf8("l"),Utf8("i")) |
+----------------------------------------------------------+
| [He, , o]                                                |
+----------------------------------------------------------+
```"#,
    standard_argument(name = "str", prefix = "String"),
    argument(
        name = "regexp",
        description = "Regular expression to split by.
            Can be a constant, column, or function."
    ),
    argument(
        name = "flags",
        description = r#"Optional regular expression flags that control the behavior of the regular expression. A NULL value is treated as if no flags were supplied. The following flags are supported:
  - **g**: (global) Accepted but has no effect, as splitting is inherently global
  - **i**: case-insensitive: letters match both upper and lower case
  - **m**: multi-line mode: ^ and $ match begin/end of line
  - **s**: allow . to match \n
  - **R**: enables CRLF mode: when multi-line mode is enabled, \r\n is used
  - **U**: swap the meaning of x* and x*?"#
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RegexpSplitToArrayFunc {
    signature: Signature,
}

impl Default for RegexpSplitToArrayFunc {
    fn default() -> Self {
        Self::new()
    }
}

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

impl ScalarUDFImpl for RegexpSplitToArrayFunc {
    fn name(&self) -> &str {
        "regexp_split_to_array"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(match &arg_types[0] {
            DataType::Null => DataType::Null,
            other => DataType::List(Arc::new(Field::new_list_field(other.clone(), true))),
        })
    }

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

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

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
        (Utf8, Utf8, Some(flags)) if *flags.data_type() == Utf8 => {
            regexp_split_to_array_inner::<_, GenericStringBuilder<i32>>(
                &values.as_string::<i32>(),
                &regex_array.as_string::<i32>(),
                Some(&flags.as_string::<i32>()),
            )
        }
        (LargeUtf8, LargeUtf8, None) => {
            regexp_split_to_array_inner::<_, GenericStringBuilder<i64>>(
                &values.as_string::<i64>(),
                &regex_array.as_string::<i64>(),
                None,
            )
        }
        (LargeUtf8, LargeUtf8, Some(flags)) if *flags.data_type() == LargeUtf8 => {
            regexp_split_to_array_inner::<_, GenericStringBuilder<i64>>(
                &values.as_string::<i64>(),
                &regex_array.as_string::<i64>(),
                Some(&flags.as_string::<i64>()),
            )
        }
        (Utf8View, Utf8View, None) => {
            regexp_split_to_array_inner::<_, StringViewBuilder>(
                &values.as_string_view(),
                &regex_array.as_string_view(),
                None,
            )
        }
        (Utf8View, Utf8View, Some(flags)) if *flags.data_type() == Utf8View => {
            regexp_split_to_array_inner::<_, StringViewBuilder>(
                &values.as_string_view(),
                &regex_array.as_string_view(),
                Some(&flags.as_string_view()),
            )
        }
        _ => Err(ArrowError::ComputeError(
            "regexp_split_to_array() expected the input arrays to be of type Utf8, LargeUtf8, or Utf8View and the data types to match".to_string(),
        )),
    }
}

trait StringListBuilder: ArrayBuilder {
    fn new_builder() -> Self;
    fn append_str_value(&mut self, val: &str);
}

impl<O: OffsetSizeTrait> StringListBuilder for GenericStringBuilder<O> {
    fn new_builder() -> Self {
        GenericStringBuilder::<O>::new()
    }

    fn append_str_value(&mut self, val: &str) {
        self.append_value(val);
    }
}

impl StringListBuilder for StringViewBuilder {
    fn new_builder() -> Self {
        StringViewBuilder::new()
    }

    fn append_str_value(&mut self, val: &str) {
        self.append_value(val);
    }
}

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

fn regexp_split_to_array_inner<'a, S, B>(
    values: &S,
    regex_array: &S,
    flags_array: Option<&S>,
) -> Result<ArrayRef, ArrowError>
where
    S: StringArrayType<'a>,
    B: StringListBuilder,
{
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

    let mut list_builder = ListBuilder::new(B::new_builder());

    if regex_is_null_scalar {
        for _ in 0..values.len() {
            list_builder.append(false);
        }
        return Ok(Arc::new(list_builder.finish()));
    }

    match (is_regex_scalar, is_flags_scalar) {
        (true, true) => {
            let regex_str = regex_scalar.unwrap();
            if regex_str.is_empty() {
                for i in 0..values.len() {
                    if values.is_null(i) {
                        list_builder.append(false);
                    } else {
                        split_chars_and_append(&mut list_builder, values.value(i));
                    }
                }
            } else {
                let pattern = compile_regex(regex_str, flags_scalar, true)?;
                for i in 0..values.len() {
                    if values.is_null(i) {
                        list_builder.append(false);
                    } else {
                        split_and_append(&mut list_builder, values.value(i), &pattern);
                    }
                }
            }
        }
        (true, false) => {
            let regex_str = regex_scalar.unwrap();
            let flags_array = flags_array.unwrap();
            if regex_str.is_empty() {
                for i in 0..values.len() {
                    if values.is_null(i) {
                        list_builder.append(false);
                    } else {
                        split_chars_and_append(&mut list_builder, values.value(i));
                    }
                }
            } else {
                let mut regex_cache = HashMap::new();
                for i in 0..values.len() {
                    if values.is_null(i) {
                        list_builder.append(false);
                    } else {
                        let flags = if flags_array.is_null(i) {
                            None
                        } else {
                            Some(flags_array.value(i))
                        };
                        let pattern = compile_and_cache_regex(
                            regex_str,
                            flags,
                            true,
                            &mut regex_cache,
                        )?;
                        split_and_append(&mut list_builder, values.value(i), pattern);
                    }
                }
            }
        }
        (false, true) => {
            let mut regex_cache = HashMap::new();
            for i in 0..values.len() {
                if values.is_null(i) || regex_array.is_null(i) {
                    list_builder.append(false);
                } else if regex_array.value(i).is_empty() {
                    split_chars_and_append(&mut list_builder, values.value(i));
                } else {
                    let regex_str = regex_array.value(i);
                    let pattern = compile_and_cache_regex(
                        regex_str,
                        flags_scalar,
                        true,
                        &mut regex_cache,
                    )?;
                    split_and_append(&mut list_builder, values.value(i), pattern);
                }
            }
        }
        (false, false) => {
            let flags_array = flags_array.unwrap();
            let mut regex_cache = HashMap::new();
            for i in 0..values.len() {
                if values.is_null(i) || regex_array.is_null(i) {
                    list_builder.append(false);
                } else if regex_array.value(i).is_empty() {
                    split_chars_and_append(&mut list_builder, values.value(i));
                } else {
                    let regex_str = regex_array.value(i);
                    let flags = if flags_array.is_null(i) {
                        None
                    } else {
                        Some(flags_array.value(i))
                    };
                    let pattern = compile_and_cache_regex(
                        regex_str,
                        flags,
                        true,
                        &mut regex_cache,
                    )?;
                    split_and_append(&mut list_builder, values.value(i), pattern);
                }
            }
        }
    }

    Ok(Arc::new(list_builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, GenericStringArray, ListArray, StringViewArray};
    use arrow::datatypes::Field;
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::ScalarFunctionArgs;

    fn invoke_with_scalars(args: &[ScalarValue]) -> Result<ColumnarValue> {
        let args_values = args
            .iter()
            .map(|sv| ColumnarValue::Scalar(sv.clone()))
            .collect();

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

    fn result_to_string_vec(result: &ColumnarValue) -> Vec<String> {
        match result {
            ColumnarValue::Scalar(ScalarValue::List(arr)) => {
                let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
                let values = list_arr.value(0);
                match values.data_type() {
                    Utf8View => {
                        let str_arr = values.as_string_view();
                        (0..str_arr.len())
                            .map(|i| str_arr.value(i).to_string())
                            .collect()
                    }
                    _ => {
                        let str_arr = values.as_string::<i32>();
                        (0..str_arr.len())
                            .map(|i| str_arr.value(i).to_string())
                            .collect()
                    }
                }
            }
            _ => panic!("Expected scalar list result"),
        }
    }

    #[test]
    fn test_basic_split() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("hello world".to_string())),
            ScalarValue::Utf8(Some("\\s+".to_string())),
        ])
        .unwrap();
        assert_eq!(result_to_string_vec(&result), vec!["hello", "world"]);
    }

    #[test]
    fn test_case_insensitive() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("HeLLo".to_string())),
            ScalarValue::Utf8(Some("l".to_string())),
            ScalarValue::Utf8(Some("i".to_string())),
        ])
        .unwrap();
        assert_eq!(result_to_string_vec(&result), vec!["He", "", "o"]);
    }

    #[test]
    fn test_no_match() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("hello".to_string())),
            ScalarValue::Utf8(Some("xyz".to_string())),
        ])
        .unwrap();
        assert_eq!(result_to_string_vec(&result), vec!["hello"]);
    }

    #[test]
    fn test_null_value() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(None),
            ScalarValue::Utf8(Some("\\s+".to_string())),
        ])
        .unwrap();
        match result {
            ColumnarValue::Scalar(ScalarValue::List(arr)) => {
                let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
                assert!(list_arr.is_null(0));
            }
            _ => panic!("Expected scalar list result"),
        }
    }

    #[test]
    fn test_null_pattern() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("hello".to_string())),
            ScalarValue::Utf8(None),
        ])
        .unwrap();
        match result {
            ColumnarValue::Scalar(ScalarValue::List(arr)) => {
                let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
                assert!(list_arr.is_null(0));
            }
            _ => panic!("Expected scalar list result"),
        }
    }

    #[test]
    fn test_empty_pattern_splits_chars() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("abc".to_string())),
            ScalarValue::Utf8(Some("".to_string())),
        ])
        .unwrap();
        assert_eq!(result_to_string_vec(&result), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_pattern_at_boundaries() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some(",a,,b,".to_string())),
            ScalarValue::Utf8(Some(",".to_string())),
        ])
        .unwrap();
        assert_eq!(result_to_string_vec(&result), vec!["", "a", "", "b", ""]);
    }

    #[test]
    fn test_unicode() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("Düsseldorf Köln Москва".to_string())),
            ScalarValue::Utf8(Some("\\s+".to_string())),
        ])
        .unwrap();
        assert_eq!(
            result_to_string_vec(&result),
            vec!["Düsseldorf", "Köln", "Москва"]
        );
    }

    #[test]
    fn test_global_flag_ignored() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("hello world".to_string())),
            ScalarValue::Utf8(Some("\\s+".to_string())),
            ScalarValue::Utf8(Some("g".to_string())),
        ])
        .unwrap();
        assert_eq!(result_to_string_vec(&result), vec!["hello", "world"]);
    }

    #[test]
    fn test_largeutf8() {
        let result = invoke_with_scalars(&[
            ScalarValue::LargeUtf8(Some("hello world".to_string())),
            ScalarValue::LargeUtf8(Some("\\s+".to_string())),
        ])
        .unwrap();
        match result {
            ColumnarValue::Scalar(ScalarValue::List(arr)) => {
                let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
                let values = list_arr.value(0);
                let str_arr = values.as_string::<i64>();
                let parts: Vec<&str> =
                    (0..str_arr.len()).map(|i| str_arr.value(i)).collect();
                assert_eq!(parts, vec!["hello", "world"]);
            }
            _ => panic!("Expected scalar list result"),
        }
    }

    #[test]
    fn test_utf8view() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8View(Some("hello world".to_string())),
            ScalarValue::Utf8View(Some("\\s+".to_string())),
        ])
        .unwrap();
        match &result {
            ColumnarValue::Scalar(ScalarValue::List(arr)) => {
                let list_arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
                assert_eq!(*list_arr.value(0).data_type(), Utf8View);
            }
            _ => panic!("Expected scalar list result"),
        }
        assert_eq!(result_to_string_vec(&result), vec!["hello", "world"]);
    }

    #[test]
    fn test_array_input() {
        let values: GenericStringArray<i32> =
            vec!["hello world", "foo-bar-baz", "abc"].into();
        let patterns: GenericStringArray<i32> = vec!["\\s+", "-", "b"].into();

        let result =
            regexp_split_to_array_func(&[Arc::new(values), Arc::new(patterns)]).unwrap();

        let list_arr = result.as_any().downcast_ref::<ListArray>().unwrap();

        let row0 = list_arr.value(0);
        let arr0 = row0.as_string::<i32>();
        assert_eq!(
            (0..arr0.len()).map(|i| arr0.value(i)).collect::<Vec<_>>(),
            vec!["hello", "world"]
        );

        let row1 = list_arr.value(1);
        let arr1 = row1.as_string::<i32>();
        assert_eq!(
            (0..arr1.len()).map(|i| arr1.value(i)).collect::<Vec<_>>(),
            vec!["foo", "bar", "baz"]
        );

        let row2 = list_arr.value(2);
        let arr2 = row2.as_string::<i32>();
        assert_eq!(
            (0..arr2.len()).map(|i| arr2.value(i)).collect::<Vec<_>>(),
            vec!["a", "c"]
        );
    }

    #[test]
    fn test_array_input_with_flags() {
        let values: GenericStringArray<i32> = vec!["Hello World", "FOO"].into();
        let patterns: GenericStringArray<i32> = vec!["l", "o"].into();
        let flags: GenericStringArray<i32> = vec!["i", "i"].into();

        let result = regexp_split_to_array_func(&[
            Arc::new(values),
            Arc::new(patterns),
            Arc::new(flags),
        ])
        .unwrap();

        let list_arr = result.as_any().downcast_ref::<ListArray>().unwrap();

        let row0 = list_arr.value(0);
        let arr0 = row0.as_string::<i32>();
        assert_eq!(
            (0..arr0.len()).map(|i| arr0.value(i)).collect::<Vec<_>>(),
            vec!["He", "", "o Wor", "d"]
        );

        let row1 = list_arr.value(1);
        let arr1 = row1.as_string::<i32>();
        assert_eq!(
            (0..arr1.len()).map(|i| arr1.value(i)).collect::<Vec<_>>(),
            vec!["F", "", ""]
        );
    }

    #[test]
    fn test_stringview_array() {
        let values = StringViewArray::from(vec!["hello world", "foo bar"]);
        let patterns = StringViewArray::from(vec!["\\s+", "\\s+"]);

        let result =
            regexp_split_to_array_func(&[Arc::new(values), Arc::new(patterns)]).unwrap();

        let list_arr = result.as_any().downcast_ref::<ListArray>().unwrap();

        let row0 = list_arr.value(0);
        assert_eq!(*row0.data_type(), Utf8View);
        let arr0 = row0.as_string_view();
        assert_eq!(
            (0..arr0.len()).map(|i| arr0.value(i)).collect::<Vec<_>>(),
            vec!["hello", "world"]
        );
    }

    #[test]
    fn test_null_flags_treated_as_no_flags() {
        let result = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("HeLLo".to_string())),
            ScalarValue::Utf8(Some("l".to_string())),
            ScalarValue::Utf8(None),
        ])
        .unwrap();
        // Case-sensitive because NULL flags => no flags
        assert_eq!(result_to_string_vec(&result), vec!["HeLLo"]);
    }

    #[test]
    fn test_largeutf8_array_input() {
        let values: GenericStringArray<i64> =
            vec!["hello world", "foo-bar", "abc"].into();
        let patterns: GenericStringArray<i64> = vec!["\\s+", "-", "b"].into();

        let result =
            regexp_split_to_array_func(&[Arc::new(values), Arc::new(patterns)]).unwrap();
        let list_arr = result.as_any().downcast_ref::<ListArray>().unwrap();

        let row0 = list_arr.value(0);
        assert_eq!(*row0.data_type(), LargeUtf8);
        let arr0 = row0.as_string::<i64>();
        assert_eq!(
            (0..arr0.len()).map(|i| arr0.value(i)).collect::<Vec<_>>(),
            vec!["hello", "world"]
        );

        let row2 = list_arr.value(2);
        let arr2 = row2.as_string::<i64>();
        assert_eq!(
            (0..arr2.len()).map(|i| arr2.value(i)).collect::<Vec<_>>(),
            vec!["a", "c"]
        );
    }

    #[test]
    fn test_mixed_nulls_in_column() {
        let values: GenericStringArray<i32> =
            vec![Some("hello world"), None, Some("foo-bar"), Some("abc")].into();
        let patterns: GenericStringArray<i32> =
            vec![Some("\\s+"), Some(","), None, Some("b")].into();

        let result =
            regexp_split_to_array_func(&[Arc::new(values), Arc::new(patterns)]).unwrap();
        let list_arr = result.as_any().downcast_ref::<ListArray>().unwrap();

        assert!(!list_arr.is_null(0));
        assert!(list_arr.is_null(1));
        assert!(list_arr.is_null(2));
        assert!(!list_arr.is_null(3));

        let row3 = list_arr.value(3);
        let arr3 = row3.as_string::<i32>();
        assert_eq!(
            (0..arr3.len()).map(|i| arr3.value(i)).collect::<Vec<_>>(),
            vec!["a", "c"]
        );
    }

    #[test]
    fn test_invalid_regex_errors() {
        let err = invoke_with_scalars(&[
            ScalarValue::Utf8(Some("abc".to_string())),
            ScalarValue::Utf8(Some("(unclosed".to_string())),
        ])
        .expect_err("invalid regex should fail");
        assert!(
            err.to_string()
                .contains("Regular expression did not compile"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_scalar_str_array_pattern() {
        // exercises the (false, true) arm: array regex, scalar flags
        let values: GenericStringArray<i32> = vec!["a,b;c"; 3].into();
        let patterns: GenericStringArray<i32> = vec![",", ";", "b"].into();

        let result =
            regexp_split_to_array_func(&[Arc::new(values), Arc::new(patterns)]).unwrap();
        let list_arr = result.as_any().downcast_ref::<ListArray>().unwrap();

        let row0 = list_arr.value(0);
        let arr0 = row0.as_string::<i32>();
        assert_eq!(
            (0..arr0.len()).map(|i| arr0.value(i)).collect::<Vec<_>>(),
            vec!["a", "b;c"]
        );

        let row1 = list_arr.value(1);
        let arr1 = row1.as_string::<i32>();
        assert_eq!(
            (0..arr1.len()).map(|i| arr1.value(i)).collect::<Vec<_>>(),
            vec!["a,b", "c"]
        );

        let row2 = list_arr.value(2);
        let arr2 = row2.as_string::<i32>();
        assert_eq!(
            (0..arr2.len()).map(|i| arr2.value(i)).collect::<Vec<_>>(),
            vec!["a,", ";c"]
        );
    }
}
