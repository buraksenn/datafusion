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

//! array function utils

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields};

use arrow::array::{Array, ArrayRef, GenericListArray, OffsetSizeTrait};
use arrow::buffer::OffsetBuffer;
use datafusion_common::cast::{
    as_fixed_size_list_array, as_large_list_array, as_list_array,
};
use datafusion_common::{Result, ScalarValue, internal_err, plan_err};

use datafusion_expr::ColumnarValue;
use itertools::Itertools as _;

pub(crate) fn check_datatypes(name: &str, args: &[&ArrayRef]) -> Result<()> {
    let data_type = args[0].data_type();
    if !args.iter().all(|arg| {
        arg.data_type().equals_datatype(data_type)
            || arg.data_type().equals_datatype(&DataType::Null)
    }) {
        let types = args.iter().map(|arg| arg.data_type()).collect::<Vec<_>>();
        return plan_err!(
            "{name} received incompatible types: {}",
            types.iter().join(", ")
        );
    }

    Ok(())
}

/// array function wrapper that differentiates between scalar (length 1) and array.
pub(crate) fn make_scalar_function<F>(
    inner: F,
) -> impl Fn(&[ColumnarValue]) -> Result<ColumnarValue>
where
    F: Fn(&[ArrayRef]) -> Result<ArrayRef>,
{
    move |args: &[ColumnarValue]| {
        // first, identify if any of the arguments is an Array. If yes, store its `len`,
        // as any scalar will need to be converted to an array of len `len`.
        let len = args
            .iter()
            .fold(Option::<usize>::None, |acc, arg| match arg {
                ColumnarValue::Scalar(_) => acc,
                ColumnarValue::Array(a) => Some(a.len()),
            });

        let is_scalar = len.is_none();

        let args = ColumnarValue::values_to_arrays(args)?;

        let result = (inner)(&args);

        if is_scalar {
            // If all inputs are scalar, keeps output as scalar
            let result = result.and_then(|arr| ScalarValue::try_from_array(&arr, 0));
            result.map(ColumnarValue::Scalar)
        } else {
            result.map(ColumnarValue::Array)
        }
    }
}

pub(crate) fn align_array_dimensions<O: OffsetSizeTrait>(
    args: Vec<ArrayRef>,
) -> Result<Vec<ArrayRef>> {
    let args_ndim = args
        .iter()
        .map(|arg| datafusion_common::utils::list_ndims(arg.data_type()))
        .collect::<Vec<_>>();
    let max_ndim = args_ndim.iter().max().unwrap_or(&0);

    // Align the dimensions of the arrays
    let aligned_args: Result<Vec<ArrayRef>> = args
        .into_iter()
        .zip(args_ndim.iter())
        .map(|(array, ndim)| {
            if ndim < max_ndim {
                let mut aligned_array = Arc::clone(&array);
                for _ in 0..(max_ndim - ndim) {
                    let data_type = aligned_array.data_type().to_owned();
                    let array_lengths = vec![1; aligned_array.len()];
                    let offsets = OffsetBuffer::<O>::from_lengths(array_lengths);

                    aligned_array = Arc::new(GenericListArray::<O>::try_new(
                        Arc::new(Field::new_list_field(data_type, true)),
                        offsets,
                        aligned_array,
                        None,
                    )?)
                }
                Ok(aligned_array)
            } else {
                Ok(Arc::clone(&array))
            }
        })
        .collect();

    aligned_args
}

/// Returns the length of each array dimension
pub(crate) fn compute_array_dims(
    arr: Option<ArrayRef>,
) -> Result<Option<Vec<Option<u64>>>> {
    let mut value = match arr {
        Some(arr) => arr,
        None => return Ok(None),
    };
    if value.is_empty() {
        return Ok(None);
    }
    let mut res = vec![Some(value.len() as u64)];

    loop {
        match value.data_type() {
            DataType::List(_) => {
                value = as_list_array(&value)?.value(0);
                res.push(Some(value.len() as u64));
            }
            DataType::LargeList(_) => {
                value = as_large_list_array(&value)?.value(0);
                res.push(Some(value.len() as u64));
            }
            DataType::FixedSizeList(..) => {
                value = as_fixed_size_list_array(&value)?.value(0);
                res.push(Some(value.len() as u64));
            }
            _ => return Ok(Some(res)),
        }
    }
}

pub(crate) fn get_map_entry_field(data_type: &DataType) -> Result<&Fields> {
    match data_type {
        DataType::Map(field, _) => {
            let field_data_type = field.data_type();
            match field_data_type {
                DataType::Struct(fields) => Ok(fields),
                _ => {
                    internal_err!("Expected a Struct type, got {}", field_data_type)
                }
            }
        }
        _ => internal_err!("Expected a Map type, got {data_type}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::ListArray;
    use arrow::datatypes::Int64Type;
    use datafusion_common::utils::SingleRowListArrayBuilder;

    /// Only test internal functions, array-related sql functions will be tested in sqllogictest `array.slt`
    #[test]
    fn test_align_array_dimensions() {
        let array1d_1: ArrayRef =
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(1), Some(2), Some(3)]),
                Some(vec![Some(4), Some(5)]),
            ]));
        let array1d_2: ArrayRef =
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(vec![
                Some(vec![Some(6), Some(7), Some(8)]),
            ]));

        let array2d_1: ArrayRef = Arc::new(
            SingleRowListArrayBuilder::new(Arc::clone(&array1d_1)).build_list_array(),
        );
        let array2d_2 = Arc::new(
            SingleRowListArrayBuilder::new(Arc::clone(&array1d_2)).build_list_array(),
        );

        let res = align_array_dimensions::<i32>(vec![
            array1d_1.to_owned(),
            array2d_2.to_owned(),
        ])
        .unwrap();

        let expected = as_list_array(&array2d_1).unwrap();
        let expected_dim = datafusion_common::utils::list_ndims(array2d_1.data_type());
        assert_ne!(as_list_array(&res[0]).unwrap(), expected);
        assert_eq!(
            datafusion_common::utils::list_ndims(res[0].data_type()),
            expected_dim
        );

        let array3d_1: ArrayRef =
            Arc::new(SingleRowListArrayBuilder::new(array2d_1).build_list_array());
        let array3d_2: ArrayRef =
            Arc::new(SingleRowListArrayBuilder::new(array2d_2).build_list_array());
        let res = align_array_dimensions::<i32>(vec![array1d_1, array3d_2]).unwrap();

        let expected = as_list_array(&array3d_1).unwrap();
        let expected_dim = datafusion_common::utils::list_ndims(array3d_1.data_type());
        assert_ne!(as_list_array(&res[0]).unwrap(), expected);
        assert_eq!(
            datafusion_common::utils::list_ndims(res[0].data_type()),
            expected_dim
        );
    }
}
