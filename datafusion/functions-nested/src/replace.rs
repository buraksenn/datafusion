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

//! [`ScalarUDFImpl`] definitions for array_replace, array_replace_n and array_replace_all functions.

use arrow::array::{
    Array, ArrayRef, AsArray, Capacities, GenericListArray, MutableArrayData,
    NullBufferBuilder, OffsetSizeTrait, new_null_array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field};
use arrow::row::{RowConverter, SortField};
use datafusion_common::cast::as_int64_array;
use datafusion_common::utils::ListCoercion;
use datafusion_common::{Result, exec_err, utils::take_function_args};
use datafusion_expr::{
    ArrayFunctionArgument, ArrayFunctionSignature, ColumnarValue, Documentation,
    ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use datafusion_macros::user_doc;

use crate::utils::make_scalar_function;

use std::any::Any;
use std::sync::Arc;

// Create static instances of ScalarUDFs for each function
make_udf_expr_and_func!(ArrayReplace,
    array_replace,
    array from to,
    "replaces the first occurrence of the specified element with another specified element.",
    array_replace_udf
);
make_udf_expr_and_func!(ArrayReplaceN,
    array_replace_n,
    array from to max,
    "replaces the first `max` occurrences of the specified element with another specified element.",
    array_replace_n_udf
);
make_udf_expr_and_func!(ArrayReplaceAll,
    array_replace_all,
    array from to,
    "replaces all occurrences of the specified element with another specified element.",
    array_replace_all_udf
);

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Replaces the first occurrence of the specified element with another specified element.",
    syntax_example = "array_replace(array, from, to)",
    sql_example = r#"```sql
> select array_replace([1, 2, 2, 3, 2, 1, 4], 2, 5);
+--------------------------------------------------------+
| array_replace(List([1,2,2,3,2,1,4]),Int64(2),Int64(5)) |
+--------------------------------------------------------+
| [1, 5, 2, 3, 2, 1, 4]                                  |
+--------------------------------------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(name = "from", description = "Initial element."),
    argument(name = "to", description = "Final element.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ArrayReplace {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for ArrayReplace {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayReplace {
    pub fn new() -> Self {
        Self {
            signature: Signature {
                type_signature: TypeSignature::ArraySignature(
                    ArrayFunctionSignature::Array {
                        arguments: vec![
                            ArrayFunctionArgument::Array,
                            ArrayFunctionArgument::Element,
                            ArrayFunctionArgument::Element,
                        ],
                        array_coercion: Some(ListCoercion::FixedSizedListToList),
                    },
                ),
                volatility: Volatility::Immutable,
                parameter_names: None,
            },
            aliases: vec![String::from("list_replace")],
        }
    }
}

impl ScalarUDFImpl for ArrayReplace {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_replace"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        Ok(args[0].clone())
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        make_scalar_function(array_replace_inner)(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Replaces the first `max` occurrences of the specified element with another specified element.",
    syntax_example = "array_replace_n(array, from, to, max)",
    sql_example = r#"```sql
> select array_replace_n([1, 2, 2, 3, 2, 1, 4], 2, 5, 2);
+-------------------------------------------------------------------+
| array_replace_n(List([1,2,2,3,2,1,4]),Int64(2),Int64(5),Int64(2)) |
+-------------------------------------------------------------------+
| [1, 5, 5, 3, 2, 1, 4]                                             |
+-------------------------------------------------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(name = "from", description = "Initial element."),
    argument(name = "to", description = "Final element."),
    argument(name = "max", description = "Number of first occurrences to replace.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct ArrayReplaceN {
    signature: Signature,
    aliases: Vec<String>,
}

impl ArrayReplaceN {
    pub fn new() -> Self {
        Self {
            signature: Signature {
                type_signature: TypeSignature::ArraySignature(
                    ArrayFunctionSignature::Array {
                        arguments: vec![
                            ArrayFunctionArgument::Array,
                            ArrayFunctionArgument::Element,
                            ArrayFunctionArgument::Element,
                            ArrayFunctionArgument::Index,
                        ],
                        array_coercion: Some(ListCoercion::FixedSizedListToList),
                    },
                ),
                volatility: Volatility::Immutable,
                parameter_names: None,
            },
            aliases: vec![String::from("list_replace_n")],
        }
    }
}

impl ScalarUDFImpl for ArrayReplaceN {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_replace_n"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        Ok(args[0].clone())
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        make_scalar_function(array_replace_n_inner)(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Replaces all occurrences of the specified element with another specified element.",
    syntax_example = "array_replace_all(array, from, to)",
    sql_example = r#"```sql
> select array_replace_all([1, 2, 2, 3, 2, 1, 4], 2, 5);
+------------------------------------------------------------+
| array_replace_all(List([1,2,2,3,2,1,4]),Int64(2),Int64(5)) |
+------------------------------------------------------------+
| [1, 5, 5, 3, 5, 1, 4]                                      |
+------------------------------------------------------------+
```"#,
    argument(
        name = "array",
        description = "Array expression. Can be a constant, column, or function, and any combination of array operators."
    ),
    argument(name = "from", description = "Initial element."),
    argument(name = "to", description = "Final element.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct ArrayReplaceAll {
    signature: Signature,
    aliases: Vec<String>,
}

impl ArrayReplaceAll {
    pub fn new() -> Self {
        Self {
            signature: Signature {
                type_signature: TypeSignature::ArraySignature(
                    ArrayFunctionSignature::Array {
                        arguments: vec![
                            ArrayFunctionArgument::Array,
                            ArrayFunctionArgument::Element,
                            ArrayFunctionArgument::Element,
                        ],
                        array_coercion: Some(ListCoercion::FixedSizedListToList),
                    },
                ),
                volatility: Volatility::Immutable,
                parameter_names: None,
            },
            aliases: vec![String::from("list_replace_all")],
        }
    }
}

impl ScalarUDFImpl for ArrayReplaceAll {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "array_replace_all"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, args: &[DataType]) -> Result<DataType> {
        Ok(args[0].clone())
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        make_scalar_function(array_replace_all_inner)(&args.args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// For each element of `list_array[i]`, replaces up to `arr_n[i]`  occurrences
/// of `from_array[i]`, `to_array[i]`.
///
/// The type of each **element** in `list_array` must be the same as the type of
/// `from_array` and `to_array`. This function also handles nested arrays
/// (\[`ListArray`\] of \[`ListArray`\]s)
///
/// For example, when called to replace a list array (where each element is a
/// list of int32s, the second and third argument are int32 arrays, and the
/// fourth argument is the number of occurrences to replace
///
/// ```text
/// general_replace(
///   [1, 2, 3, 2], 2, 10, 1    ==> [1, 10, 3, 2]   (only the first 2 is replaced)
///   [4, 5, 6, 5], 5, 20, 2    ==> [4, 20, 6, 20]  (both 5s are replaced)
/// )
/// ```
fn general_replace<O: OffsetSizeTrait>(
    list_array: &GenericListArray<O>,
    from_array: &ArrayRef,
    to_array: &ArrayRef,
    arr_n: &[i64],
) -> Result<ArrayRef> {
    let converter = RowConverter::new(vec![SortField::new(list_array.value_type())])?;

    let first_offset = list_array.offsets()[0].as_usize();
    let last_offset = list_array.offsets()[list_array.len()].as_usize();
    let visible_values = list_array
        .values()
        .slice(first_offset, last_offset - first_offset);
    let value_rows = converter.convert_columns(&[visible_values])?;
    let from_rows = converter.convert_columns(&[Arc::clone(from_array)])?;

    let mut offsets: Vec<O> = vec![O::usize_as(0)];
    let values = list_array.values();
    let original_data = values.to_data();
    let to_data = to_array.to_data();
    let capacity = Capacities::Array(original_data.len());

    let mut mutable = MutableArrayData::with_capacities(
        vec![&original_data, &to_data],
        false,
        capacity,
    );

    let mut valid = NullBufferBuilder::new(list_array.len());

    for (row_index, offset_window) in list_array.offsets().windows(2).enumerate() {
        if list_array.is_null(row_index) {
            offsets.push(offsets[row_index]);
            valid.append_null();
            continue;
        }

        let start = offset_window[0];
        let end = offset_window[1];
        let row_start = start.to_usize().unwrap() - first_offset;
        let row_end = end.to_usize().unwrap() - first_offset;
        let row_len = row_end - row_start;
        let target = from_rows.row(row_index);

        let num_matches = (row_start..row_end)
            .filter(|&idx| value_rows.row(idx) == target)
            .count();

        if num_matches == 0 {
            mutable.extend(0, start.to_usize().unwrap(), end.to_usize().unwrap());
            offsets.push(offsets[row_index] + (end - start));
            valid.append_non_null();
            continue;
        }

        let n = arr_n[row_index];
        let mut counter = 0i64;

        for i in 0..row_len {
            let matches = value_rows.row(row_start + i) == target;
            if matches && counter < n {
                mutable.extend(1, row_index, row_index + 1);
                counter += 1;
                if counter == n {
                    let remaining_start = start.to_usize().unwrap() + i + 1;
                    mutable.extend(0, remaining_start, end.to_usize().unwrap());
                    break;
                }
            } else {
                mutable.extend(
                    0,
                    start.to_usize().unwrap() + i,
                    start.to_usize().unwrap() + i + 1,
                );
            }
        }

        offsets.push(offsets[row_index] + (end - start));
        valid.append_non_null();
    }

    let data = mutable.freeze();

    Ok(Arc::new(GenericListArray::<O>::try_new(
        Arc::new(Field::new_list_field(list_array.value_type(), true)),
        OffsetBuffer::<O>::new(offsets.into()),
        arrow::array::make_array(data),
        valid.finish(),
    )?))
}

fn array_replace_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [array, from, to] = take_function_args("array_replace", args)?;

    // replace at most one occurrence for each element
    let arr_n = vec![1; array.len()];
    match array.data_type() {
        DataType::List(_) => {
            let list_array = array.as_list::<i32>();
            general_replace::<i32>(list_array, from, to, &arr_n)
        }
        DataType::LargeList(_) => {
            let list_array = array.as_list::<i64>();
            general_replace::<i64>(list_array, from, to, &arr_n)
        }
        DataType::Null => Ok(new_null_array(array.data_type(), 1)),
        array_type => exec_err!("array_replace does not support type '{array_type}'."),
    }
}

fn array_replace_n_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [array, from, to, max] = take_function_args("array_replace_n", args)?;

    // replace the specified number of occurrences
    let arr_n = as_int64_array(max)?.values().to_vec();
    match array.data_type() {
        DataType::List(_) => {
            let list_array = array.as_list::<i32>();
            general_replace::<i32>(list_array, from, to, &arr_n)
        }
        DataType::LargeList(_) => {
            let list_array = array.as_list::<i64>();
            general_replace::<i64>(list_array, from, to, &arr_n)
        }
        DataType::Null => Ok(new_null_array(array.data_type(), 1)),
        array_type => {
            exec_err!("array_replace_n does not support type '{array_type}'.")
        }
    }
}

fn array_replace_all_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [array, from, to] = take_function_args("array_replace_all", args)?;

    // replace all occurrences (up to "i64::MAX")
    let arr_n = vec![i64::MAX; array.len()];
    match array.data_type() {
        DataType::List(_) => {
            let list_array = array.as_list::<i32>();
            general_replace::<i32>(list_array, from, to, &arr_n)
        }
        DataType::LargeList(_) => {
            let list_array = array.as_list::<i64>();
            general_replace::<i64>(list_array, from, to, &arr_n)
        }
        DataType::Null => Ok(new_null_array(array.data_type(), 1)),
        array_type => {
            exec_err!("array_replace_all does not support type '{array_type}'.")
        }
    }
}
