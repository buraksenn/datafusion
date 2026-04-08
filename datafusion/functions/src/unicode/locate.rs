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

use std::sync::Arc;

use crate::utils::{make_scalar_function, utf8_to_int_type};
use arrow::array::{
    ArrayRef, ArrowPrimitiveType, AsArray, Int64Array, PrimitiveArray, StringArrayType,
};
use arrow::datatypes::{
    ArrowNativeType, DataType, Field, FieldRef, Int32Type, Int64Type,
};
use datafusion_common::cast::as_int64_array;
use datafusion_common::types::{
    NativeType, logical_int32, logical_int64, logical_string,
};
use datafusion_common::{Result, exec_err, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns the position of the first occurrence of a substring in a string, starting the search from an optional position. Positions begin at 1. If the substring is not found, returns 0.",
    syntax_example = "locate(substr, str[, pos])",
    sql_example = r#"```sql
> select locate('bar', 'foobarbar');
+----------------------------------------------+
| locate(Utf8("bar"),Utf8("foobarbar"))        |
+----------------------------------------------+
| 4                                            |
+----------------------------------------------+

> select locate('bar', 'foobarbar', 5);
+--------------------------------------------------+
| locate(Utf8("bar"),Utf8("foobarbar"),Int64(5))   |
+--------------------------------------------------+
| 7                                                |
+--------------------------------------------------+
```"#,
    argument(name = "substr", description = "Substring expression to search for."),
    argument(name = "str", description = "String expression to search within."),
    argument(
        name = "pos",
        description = "Optional starting position for the search (1-indexed). Defaults to 1."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct LocateFunc {
    signature: Signature,
}

impl Default for LocateFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl LocateFunc {
    pub fn new() -> Self {
        let string = Coercion::new_exact(TypeSignatureClass::Native(logical_string()));
        let int64 = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_int64()),
            vec![TypeSignatureClass::Native(logical_int32())],
            NativeType::Int64,
        );
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Coercible(vec![string.clone(), string.clone()]),
                    TypeSignature::Coercible(vec![
                        string.clone(),
                        string.clone(),
                        int64.clone(),
                    ]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for LocateFunc {
    fn name(&self) -> &str {
        "locate"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(
        &self,
        args: datafusion_expr::ReturnFieldArgs,
    ) -> Result<FieldRef> {
        // The str argument is at index 1 (locate(substr, str [, pos]))
        utf8_to_int_type(args.arg_fields[1].data_type(), "locate").map(|data_type| {
            Field::new(
                self.name(),
                data_type,
                args.arg_fields.iter().any(|x| x.is_nullable()),
            )
            .into()
        })
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(locate, vec![])(&args.args)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn locate(args: &[ArrayRef]) -> Result<ArrayRef> {
    macro_rules! dispatch_locate {
        ($haystack:expr, $result_type:ty, $args:expr) => {
            match $args[0].data_type() {
                DataType::Utf8 => {
                    if $args.len() == 3 {
                        locate_with_pos::<_, _, $result_type>(
                            $haystack,
                            $args[0].as_string::<i32>(),
                            as_int64_array(&$args[2])?,
                        )
                    } else {
                        locate_general::<_, _, $result_type>(
                            $haystack,
                            $args[0].as_string::<i32>(),
                        )
                    }
                }
                DataType::LargeUtf8 => {
                    if $args.len() == 3 {
                        locate_with_pos::<_, _, $result_type>(
                            $haystack,
                            $args[0].as_string::<i64>(),
                            as_int64_array(&$args[2])?,
                        )
                    } else {
                        locate_general::<_, _, $result_type>(
                            $haystack,
                            $args[0].as_string::<i64>(),
                        )
                    }
                }
                DataType::Utf8View => {
                    if $args.len() == 3 {
                        locate_with_pos::<_, _, $result_type>(
                            $haystack,
                            $args[0].as_string_view(),
                            as_int64_array(&$args[2])?,
                        )
                    } else {
                        locate_general::<_, _, $result_type>(
                            $haystack,
                            $args[0].as_string_view(),
                        )
                    }
                }
                other => exec_err!("Unsupported data type {other:?} for locate needle"),
            }
        };
    }

    // args[0] = substr (needle), args[1] = str (haystack), args[2] = optional pos
    match args[1].data_type() {
        DataType::Utf8 => dispatch_locate!(args[1].as_string::<i32>(), Int32Type, args),
        DataType::LargeUtf8 => {
            dispatch_locate!(args[1].as_string::<i64>(), Int64Type, args)
        }
        DataType::Utf8View => {
            dispatch_locate!(args[1].as_string_view(), Int32Type, args)
        }
        other => {
            exec_err!("Unsupported data type {other:?} for locate haystack")
        }
    }
}

/// 2-arg locate: equivalent to strpos with swapped arguments
fn locate_general<'a, V1, V2, T: ArrowPrimitiveType>(
    haystack_array: V1,
    needle_array: V2,
) -> Result<ArrayRef>
where
    V1: StringArrayType<'a, Item = &'a str> + Copy,
    V2: StringArrayType<'a, Item = &'a str> + Copy,
{
    let ascii_only = needle_array.is_ascii() && haystack_array.is_ascii();

    let result = haystack_array
        .iter()
        .zip(needle_array.iter())
        .map(|(haystack, needle)| match (haystack, needle) {
            (Some(haystack), Some(needle)) => {
                let pos = haystack.find(needle);
                match pos {
                    None => T::Native::from_usize(0),
                    Some(byte_offset) => {
                        if ascii_only {
                            T::Native::from_usize(byte_offset + 1)
                        } else {
                            T::Native::from_usize(
                                haystack[..byte_offset].chars().count() + 1,
                            )
                        }
                    }
                }
            }
            _ => None,
        })
        .collect::<PrimitiveArray<T>>();

    Ok(Arc::new(result) as ArrayRef)
}

/// 3-arg locate: find needle in haystack starting from a given position (1-indexed)
fn locate_with_pos<'a, V1, V2, T: ArrowPrimitiveType>(
    haystack_array: V1,
    needle_array: V2,
    pos_array: &Int64Array,
) -> Result<ArrayRef>
where
    V1: StringArrayType<'a, Item = &'a str> + Copy,
    V2: StringArrayType<'a, Item = &'a str> + Copy,
{
    let ascii_only = needle_array.is_ascii() && haystack_array.is_ascii();

    let result = haystack_array
        .iter()
        .zip(needle_array.iter())
        .zip(pos_array.iter())
        .map(|((haystack, needle), pos)| match (haystack, needle, pos) {
            (Some(haystack), Some(needle), Some(pos)) => {
                if pos < 1 {
                    return T::Native::from_usize(0);
                }
                let pos = pos as usize;

                // Convert 1-indexed char position to byte offset
                let byte_start = if ascii_only {
                    let idx = pos - 1;
                    if idx >= haystack.len() {
                        return T::Native::from_usize(0);
                    }
                    idx
                } else {
                    let mut char_count = 0;
                    let mut byte_idx = 0;
                    for (i, c) in haystack.char_indices() {
                        char_count += 1;
                        if char_count == pos {
                            byte_idx = i;
                            break;
                        }
                        // If we run past the end without reaching pos
                        if i + c.len_utf8() == haystack.len() && char_count < pos - 1 {
                            return T::Native::from_usize(0);
                        }
                    }
                    if char_count < pos {
                        return T::Native::from_usize(0);
                    }
                    byte_idx
                };

                let substring = &haystack[byte_start..];
                match substring.find(needle) {
                    None => T::Native::from_usize(0),
                    Some(byte_offset) => {
                        if ascii_only {
                            T::Native::from_usize(byte_start + byte_offset + 1)
                        } else {
                            let total_byte_offset = byte_start + byte_offset;
                            T::Native::from_usize(
                                haystack[..total_byte_offset].chars().count() + 1,
                            )
                        }
                    }
                }
            }
            _ => None,
        })
        .collect::<PrimitiveArray<T>>();

    Ok(Arc::new(result) as ArrayRef)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, Int32Array, Int64Array};
    use arrow::datatypes::DataType::{Int32, Int64};
    use datafusion_common::{Result, ScalarValue};
    use datafusion_expr::{ColumnarValue, ScalarUDFImpl};

    use crate::unicode::locate::LocateFunc;
    use crate::utils::test::test_function;

    #[test]
    fn test_locate_two_args() {
        // locate(substr, str) - note: reversed arg order vs strpos
        test_function!(
            LocateFunc::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("bar".to_owned()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("foobarbar".to_owned()))),
            ],
            Ok(Some(4)),
            i32,
            Int32,
            Int32Array
        );

        test_function!(
            LocateFunc::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("xyz".to_owned()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("foobarbar".to_owned()))),
            ],
            Ok(Some(0)),
            i32,
            Int32,
            Int32Array
        );

        // LargeUtf8 variant
        test_function!(
            LocateFunc::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some("bar".to_owned()))),
                ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(
                    "foobarbar".to_owned()
                ))),
            ],
            Ok(Some(4)),
            i64,
            Int64,
            Int64Array
        );
    }

    #[test]
    fn test_locate_three_args() {
        test_function!(
            LocateFunc::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("bar".to_owned()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("foobarbar".to_owned()))),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(5))),
            ],
            Ok(Some(7)),
            i32,
            Int32,
            Int32Array
        );

        // pos beyond string length
        test_function!(
            LocateFunc::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("bar".to_owned()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("foobarbar".to_owned()))),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(100))),
            ],
            Ok(Some(0)),
            i32,
            Int32,
            Int32Array
        );
    }

    #[test]
    fn test_locate_unicode() {
        test_function!(
            LocateFunc::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                    "\u{1F4CA}".to_owned()
                ))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                    "\u{0414}\u{0430}\u{0442}\u{0430}\u{0424}\u{0443}\u{0441}\u{0438}\u{043E}\u{043D}\u{6570}\u{636E}\u{878D}\u{5408}\u{1F4CA}\u{1F525}".to_owned()
                ))),
            ],
            Ok(Some(15)),
            i32,
            Int32,
            Int32Array
        );
    }
}
