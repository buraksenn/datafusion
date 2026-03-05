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

//! Defines NTH_VALUE aggregate expression which may specify ordering requirement
//! that can evaluated at runtime during query execution

use std::any::Any;
use std::collections::VecDeque;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, PrimitiveArray,
    StructArray, new_empty_array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, FieldRef, Fields};

use datafusion_common::utils::{SingleRowListArrayBuilder, compare_rows, get_row_at_idx};
use datafusion_common::{
    Result, ScalarValue, assert_or_internal_err, exec_err, internal_err, not_impl_err,
};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, EmitTo, ExprFunctionExt,
    GroupsAccumulator, ReversedUDAF, Signature, SortExpr, Volatility, lit,
};
use datafusion_functions_aggregate_common::merge_arrays::merge_ordered_arrays;
use datafusion_functions_aggregate_common::utils::{get_sort_options, ordering_fields};
use datafusion_macros::user_doc;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};

create_func!(NthValueAgg, nth_value_udaf);

/// Returns the nth value in a group of values.
pub fn nth_value(
    expr: datafusion_expr::Expr,
    n: i64,
    order_by: Vec<SortExpr>,
) -> datafusion_expr::Expr {
    let args = vec![expr, lit(n)];
    if !order_by.is_empty() {
        nth_value_udaf()
            .call(args)
            .order_by(order_by)
            .build()
            .unwrap()
    } else {
        nth_value_udaf().call(args)
    }
}

#[user_doc(
    doc_section(label = "Statistical Functions"),
    description = "Returns the nth value in a group of values.",
    syntax_example = "nth_value(expression, n ORDER BY expression)",
    sql_example = r#"```sql
> SELECT dept_id, salary, NTH_VALUE(salary, 2) OVER (PARTITION BY dept_id ORDER BY salary ASC) AS second_salary_by_dept
  FROM employee;
+---------+--------+-------------------------+
| dept_id | salary | second_salary_by_dept   |
+---------+--------+-------------------------+
| 1       | 30000  | NULL                    |
| 1       | 40000  | 40000                   |
| 1       | 50000  | 40000                   |
| 2       | 35000  | NULL                    |
| 2       | 45000  | 45000                   |
+---------+--------+-------------------------+
```"#,
    argument(
        name = "expression",
        description = "The column or expression to retrieve the nth value from."
    ),
    argument(
        name = "n",
        description = "The position (nth) of the value to retrieve, based on the ordering."
    )
)]
/// Expression for a `NTH_VALUE(..., ... ORDER BY ...)` aggregation. In a multi
/// partition setting, partial aggregations are computed for every partition,
/// and then their results are merged.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct NthValueAgg {
    signature: Signature,
}

impl NthValueAgg {
    /// Create a new `NthValueAgg` aggregate function
    pub fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for NthValueAgg {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateUDFImpl for NthValueAgg {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "nth_value"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let n = match acc_args.exprs[1]
            .as_any()
            .downcast_ref::<Literal>()
            .map(|lit| lit.value())
        {
            Some(ScalarValue::Int64(Some(value))) => {
                if acc_args.is_reversed {
                    -*value
                } else {
                    *value
                }
            }
            _ => {
                return not_impl_err!(
                    "{} not supported for n: {}",
                    self.name(),
                    &acc_args.exprs[1]
                );
            }
        };

        let Some(ordering) = LexOrdering::new(acc_args.order_bys.to_vec()) else {
            return TrivialNthValueAccumulator::try_new(
                n,
                acc_args.return_field.data_type(),
            )
            .map(|acc| Box::new(acc) as _);
        };
        let ordering_dtypes = ordering
            .iter()
            .map(|e| e.expr.data_type(acc_args.schema))
            .collect::<Result<Vec<_>>>()?;

        let data_type = acc_args.expr_fields[0].data_type();
        NthValueAccumulator::try_new(n, data_type, &ordering_dtypes, ordering)
            .map(|acc| Box::new(acc) as _)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        let mut fields = vec![Field::new_list(
            format_state_name(self.name(), "nth_value"),
            // See COMMENTS.md to understand why nullable is set to true
            Field::new_list_field(args.input_fields[0].data_type().clone(), true),
            false,
        )];
        let orderings = args.ordering_fields.to_vec();
        if !orderings.is_empty() {
            fields.push(Field::new_list(
                format_state_name(self.name(), "nth_value_orderings"),
                Field::new_list_field(DataType::Struct(Fields::from(orderings)), true),
                false,
            ));
        }
        Ok(fields.into_iter().map(Arc::new).collect())
    }

    fn reverse_expr(&self) -> ReversedUDAF {
        ReversedUDAF::Reversed(nth_value_udaf())
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        use arrow::datatypes::DataType::*;
        !args.order_bys.is_empty()
            && matches!(
                args.return_field.data_type(),
                Int8 | Int16
                    | Int32
                    | Int64
                    | UInt8
                    | UInt16
                    | UInt32
                    | UInt64
                    | Float16
                    | Float32
                    | Float64
                    | Decimal32(_, _)
                    | Decimal64(_, _)
                    | Decimal128(_, _)
                    | Decimal256(_, _)
                    | Date32
                    | Date64
                    | Time32(_)
                    | Time64(_)
                    | Timestamp(_, _)
            )
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        use arrow::datatypes::*;

        let n = match args.exprs[1]
            .as_any()
            .downcast_ref::<Literal>()
            .map(|lit| lit.value())
        {
            Some(ScalarValue::Int64(Some(value))) => {
                if args.is_reversed {
                    -*value
                } else {
                    *value
                }
            }
            _ => {
                return not_impl_err!(
                    "{} not supported for n: {}",
                    self.name(),
                    &args.exprs[1]
                );
            }
        };

        fn create_accumulator<T: ArrowPrimitiveType + Send>(
            args: AccumulatorArgs,
            n: i64,
        ) -> Result<Box<dyn GroupsAccumulator>> {
            let Some(ordering) = LexOrdering::new(args.order_bys.to_vec()) else {
                return internal_err!("Groups accumulator must have an ordering.");
            };

            let ordering_dtypes = ordering
                .iter()
                .map(|e| e.expr.data_type(args.schema))
                .collect::<Result<Vec<_>>>()?;

            NthPrimitiveGroupsAccumulator::<T>::try_new(
                n,
                ordering,
                args.ignore_nulls,
                args.return_field.data_type(),
                &ordering_dtypes,
            )
            .map(|acc| Box::new(acc) as _)
        }

        match args.return_field.data_type() {
            DataType::Int8 => create_accumulator::<Int8Type>(args, n),
            DataType::Int16 => create_accumulator::<Int16Type>(args, n),
            DataType::Int32 => create_accumulator::<Int32Type>(args, n),
            DataType::Int64 => create_accumulator::<Int64Type>(args, n),
            DataType::UInt8 => create_accumulator::<UInt8Type>(args, n),
            DataType::UInt16 => create_accumulator::<UInt16Type>(args, n),
            DataType::UInt32 => create_accumulator::<UInt32Type>(args, n),
            DataType::UInt64 => create_accumulator::<UInt64Type>(args, n),
            DataType::Float16 => create_accumulator::<Float16Type>(args, n),
            DataType::Float32 => create_accumulator::<Float32Type>(args, n),
            DataType::Float64 => create_accumulator::<Float64Type>(args, n),
            DataType::Decimal32(_, _) => create_accumulator::<Decimal32Type>(args, n),
            DataType::Decimal64(_, _) => create_accumulator::<Decimal64Type>(args, n),
            DataType::Decimal128(_, _) => create_accumulator::<Decimal128Type>(args, n),
            DataType::Decimal256(_, _) => create_accumulator::<Decimal256Type>(args, n),
            DataType::Timestamp(TimeUnit::Second, _) => {
                create_accumulator::<TimestampSecondType>(args, n)
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                create_accumulator::<TimestampMillisecondType>(args, n)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                create_accumulator::<TimestampMicrosecondType>(args, n)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                create_accumulator::<TimestampNanosecondType>(args, n)
            }
            DataType::Date32 => create_accumulator::<Date32Type>(args, n),
            DataType::Date64 => create_accumulator::<Date64Type>(args, n),
            DataType::Time32(TimeUnit::Second) => {
                create_accumulator::<Time32SecondType>(args, n)
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                create_accumulator::<Time32MillisecondType>(args, n)
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                create_accumulator::<Time64MicrosecondType>(args, n)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                create_accumulator::<Time64NanosecondType>(args, n)
            }
            _ => internal_err!("Unsupported data type for NTH_VALUE groups accumulator"),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[derive(Debug)]
pub struct TrivialNthValueAccumulator {
    /// The `N` value.
    n: i64,
    /// Stores entries in the `NTH_VALUE` result.
    values: VecDeque<ScalarValue>,
    /// Data types of the value.
    datatype: DataType,
}

impl TrivialNthValueAccumulator {
    /// Create a new order-insensitive NTH_VALUE accumulator based on the given
    /// item data type.
    pub fn try_new(n: i64, datatype: &DataType) -> Result<Self> {
        // n cannot be 0
        assert_or_internal_err!(
            n != 0,
            "Nth value indices are 1 based. 0 is invalid index"
        );
        Ok(Self {
            n,
            values: VecDeque::new(),
            datatype: datatype.clone(),
        })
    }

    /// Updates state, with the `values`. Fetch contains missing number of entries for state to be complete
    /// None represents all of the new `values` need to be added to the state.
    fn append_new_data(
        &mut self,
        values: &[ArrayRef],
        fetch: Option<usize>,
    ) -> Result<()> {
        let n_row = values[0].len();
        let n_to_add = if let Some(fetch) = fetch {
            std::cmp::min(fetch, n_row)
        } else {
            n_row
        };
        for index in 0..n_to_add {
            let mut row = get_row_at_idx(values, index)?;
            self.values.push_back(row.swap_remove(0));
            // At index 1, we have n index argument, which is constant.
        }
        Ok(())
    }
}

impl Accumulator for TrivialNthValueAccumulator {
    /// Updates its state with the `values`. Assumes data in the `values` satisfies the required
    /// ordering for the accumulator (across consecutive batches, not just batch-wise).
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if !values.is_empty() {
            let n_required = self.n.unsigned_abs() as usize;
            let from_start = self.n > 0;
            if from_start {
                // direction is from start
                let n_remaining = n_required.saturating_sub(self.values.len());
                self.append_new_data(values, Some(n_remaining))?;
            } else {
                // direction is from end
                self.append_new_data(values, None)?;
                let start_offset = self.values.len().saturating_sub(n_required);
                if start_offset > 0 {
                    self.values.drain(0..start_offset);
                }
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if !states.is_empty() {
            // First entry in the state is the aggregation result.
            let n_required = self.n.unsigned_abs() as usize;
            let array_agg_res = ScalarValue::convert_array_to_scalar_vec(&states[0])?;
            for v in array_agg_res.into_iter().flatten() {
                self.values.extend(v);
                if self.values.len() > n_required {
                    // There is enough data collected, can stop merging:
                    break;
                }
            }
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut values_cloned = self.values.clone();
        let values_slice = values_cloned.make_contiguous();
        Ok(vec![ScalarValue::List(ScalarValue::new_list_nullable(
            values_slice,
            &self.datatype,
        ))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let n_required = self.n.unsigned_abs() as usize;
        let from_start = self.n > 0;
        let nth_value_idx = if from_start {
            // index is from start
            let forward_idx = n_required - 1;
            (forward_idx < self.values.len()).then_some(forward_idx)
        } else {
            // index is from end
            self.values.len().checked_sub(n_required)
        };
        if let Some(idx) = nth_value_idx {
            Ok(self.values[idx].clone())
        } else {
            ScalarValue::try_from(self.datatype.clone())
        }
    }

    fn size(&self) -> usize {
        size_of_val(self) + ScalarValue::size_of_vec_deque(&self.values)
            - size_of_val(&self.values)
            + size_of::<DataType>()
    }
}

#[derive(Debug)]
pub struct NthValueAccumulator {
    /// The `N` value.
    n: i64,
    /// Stores entries in the `NTH_VALUE` result.
    values: VecDeque<ScalarValue>,
    /// Stores values of ordering requirement expressions corresponding to each
    /// entry in `values`. This information is used when merging results from
    /// different partitions. For detailed information how merging is done, see
    /// [`merge_ordered_arrays`].
    ordering_values: VecDeque<Vec<ScalarValue>>,
    /// Stores datatypes of expressions inside values and ordering requirement
    /// expressions.
    datatypes: Vec<DataType>,
    /// Stores the ordering requirement of the `Accumulator`.
    ordering_req: LexOrdering,
}

impl NthValueAccumulator {
    /// Create a new order-sensitive NTH_VALUE accumulator based on the given
    /// item data type.
    pub fn try_new(
        n: i64,
        datatype: &DataType,
        ordering_dtypes: &[DataType],
        ordering_req: LexOrdering,
    ) -> Result<Self> {
        // n cannot be 0
        assert_or_internal_err!(
            n != 0,
            "Nth value indices are 1 based. 0 is invalid index"
        );
        let mut datatypes = vec![datatype.clone()];
        datatypes.extend(ordering_dtypes.iter().cloned());
        Ok(Self {
            n,
            values: VecDeque::new(),
            ordering_values: VecDeque::new(),
            datatypes,
            ordering_req,
        })
    }

    fn evaluate_orderings(&self) -> Result<ScalarValue> {
        let fields = ordering_fields(&self.ordering_req, &self.datatypes[1..]);

        let mut column_wise_ordering_values = vec![];
        let num_columns = fields.len();
        for i in 0..num_columns {
            let column_values = self
                .ordering_values
                .iter()
                .map(|x| x[i].clone())
                .collect::<Vec<_>>();
            let array = if column_values.is_empty() {
                new_empty_array(fields[i].data_type())
            } else {
                ScalarValue::iter_to_array(column_values.into_iter())?
            };
            column_wise_ordering_values.push(array);
        }

        let struct_field = Fields::from(fields);
        let ordering_array =
            StructArray::try_new(struct_field, column_wise_ordering_values, None)?;

        Ok(SingleRowListArrayBuilder::new(Arc::new(ordering_array)).build_list_scalar())
    }

    fn evaluate_values(&self) -> ScalarValue {
        let mut values_cloned = self.values.clone();
        let values_slice = values_cloned.make_contiguous();
        ScalarValue::List(ScalarValue::new_list_nullable(
            values_slice,
            &self.datatypes[0],
        ))
    }

    /// Updates state, with the `values`. Fetch contains missing number of entries for state to be complete
    /// None represents all of the new `values` need to be added to the state.
    fn append_new_data(
        &mut self,
        values: &[ArrayRef],
        fetch: Option<usize>,
    ) -> Result<()> {
        let n_row = values[0].len();
        let n_to_add = if let Some(fetch) = fetch {
            std::cmp::min(fetch, n_row)
        } else {
            n_row
        };
        for index in 0..n_to_add {
            let row = get_row_at_idx(values, index)?;
            self.values.push_back(row[0].clone());
            // At index 1, we have n index argument.
            // Ordering values cover starting from 2nd index to end
            self.ordering_values.push_back(row[2..].to_vec());
        }
        Ok(())
    }
}

impl Accumulator for NthValueAccumulator {
    /// Updates its state with the `values`. Assumes data in the `values` satisfies the required
    /// ordering for the accumulator (across consecutive batches, not just batch-wise).
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        let n_required = self.n.unsigned_abs() as usize;
        let from_start = self.n > 0;
        if from_start {
            // direction is from start
            let n_remaining = n_required.saturating_sub(self.values.len());
            self.append_new_data(values, Some(n_remaining))?;
        } else {
            // direction is from end
            self.append_new_data(values, None)?;
            let start_offset = self.values.len().saturating_sub(n_required);
            if start_offset > 0 {
                self.values.drain(0..start_offset);
                self.ordering_values.drain(0..start_offset);
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        // Second entry stores values received for ordering requirement columns
        // for each aggregation value inside NTH_VALUE list. For each `StructArray`
        // inside this list, we will receive an `Array` that stores values received
        // from its ordering requirement expression. This information is necessary
        // during merging.
        let Some(agg_orderings) = states[1].as_list_opt::<i32>() else {
            return exec_err!("Expects to receive a list array");
        };

        // Stores NTH_VALUE results coming from each partition
        let mut partition_values = vec![self.values.clone()];
        // First entry in the state is the aggregation result.
        let array_agg_res = ScalarValue::convert_array_to_scalar_vec(&states[0])?;
        for v in array_agg_res.into_iter().flatten() {
            partition_values.push(v.into());
        }
        // Stores ordering requirement expression results coming from each partition:
        let mut partition_ordering_values = vec![self.ordering_values.clone()];
        let orderings = ScalarValue::convert_array_to_scalar_vec(agg_orderings)?;
        // Extract value from struct to ordering_rows for each group/partition:
        for partition_ordering_rows in orderings.into_iter().flatten() {
            let ordering_values = partition_ordering_rows.into_iter().map(|ordering_row| {
                let ScalarValue::Struct(s_array) = ordering_row else {
                    return exec_err!(
                        "Expects to receive ScalarValue::Struct(Some(..), _) but got: {:?}",
                        ordering_row.data_type()
                    );
                };
                s_array
                    .columns()
                    .iter()
                    .map(|column| ScalarValue::try_from_array(column, 0))
                    .collect()
            }).collect::<Result<VecDeque<_>>>()?;
            partition_ordering_values.push(ordering_values);
        }

        let sort_options = self
            .ordering_req
            .iter()
            .map(|sort_expr| sort_expr.options)
            .collect::<Vec<_>>();
        let (new_values, new_orderings) = merge_ordered_arrays(
            &mut partition_values,
            &mut partition_ordering_values,
            &sort_options,
        )?;
        self.values = new_values.into();
        self.ordering_values = new_orderings.into();
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate_values(), self.evaluate_orderings()?])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let n_required = self.n.unsigned_abs() as usize;
        let from_start = self.n > 0;
        let nth_value_idx = if from_start {
            // index is from start
            let forward_idx = n_required - 1;
            (forward_idx < self.values.len()).then_some(forward_idx)
        } else {
            // index is from end
            self.values.len().checked_sub(n_required)
        };
        if let Some(idx) = nth_value_idx {
            Ok(self.values[idx].clone())
        } else {
            ScalarValue::try_from(self.datatypes[0].clone())
        }
    }

    fn size(&self) -> usize {
        let mut total = size_of_val(self) + ScalarValue::size_of_vec_deque(&self.values)
            - size_of_val(&self.values);

        // Add size of the `self.ordering_values`
        total += size_of::<Vec<ScalarValue>>() * self.ordering_values.capacity();
        for row in &self.ordering_values {
            total += ScalarValue::size_of_vec(row) - size_of_val(row);
        }

        // Add size of the `self.datatypes`
        total += size_of::<DataType>() * self.datatypes.capacity();
        for dtype in &self.datatypes {
            total += dtype.size() - size_of_val(dtype);
        }

        // Add size of the `self.ordering_req`
        total += size_of::<PhysicalSortExpr>() * self.ordering_req.capacity();
        // TODO: Calculate size of each `PhysicalSortExpr` more accurately.
        total
    }
}

struct NthPrimitiveGroupsAccumulator<T: ArrowPrimitiveType + Send> {
    n: i64,
    ordering_req: LexOrdering,
    ordering_dtypes: Vec<DataType>,
    sort_options: Vec<SortOptions>,
    ignore_nulls: bool,
    data_type: DataType,
    groups: Vec<Vec<NthGroupCandidate<T::Native>>>,
}

struct NthGroupCandidate<T> {
    value: T,
    is_null: bool,
    ordering: Vec<ScalarValue>,
}

impl<T> NthPrimitiveGroupsAccumulator<T>
where
    T: ArrowPrimitiveType + Send,
{
    fn try_new(
        n: i64,
        ordering_req: LexOrdering,
        ignore_nulls: bool,
        data_type: &DataType,
        ordering_dtypes: &[DataType],
    ) -> Result<Self> {
        if n == 0 {
            return internal_err!("Nth value indices are 1 based. 0 is invalid index");
        }
        let sort_options = get_sort_options(&ordering_req);
        Ok(Self {
            n,
            ordering_req,
            ordering_dtypes: ordering_dtypes.to_vec(),
            sort_options,
            ignore_nulls,
            data_type: data_type.clone(),
            groups: Vec::new(),
        })
    }

    fn resize(&mut self, total_num_groups: usize) {
        if self.groups.len() < total_num_groups {
            self.groups.resize_with(total_num_groups, Vec::new);
        }
    }

    fn n_required(&self) -> usize {
        self.n.unsigned_abs() as usize
    }

    fn keep_smallest_n(&self) -> bool {
        self.n > 0
    }

    fn insert_candidate(
        &mut self,
        group_idx: usize,
        value: T::Native,
        is_null: bool,
        ordering: Vec<ScalarValue>,
    ) {
        let keep_smallest_n = self.keep_smallest_n();
        let n_required = self.n_required();
        let sort_options = &self.sort_options;
        let group = &mut self.groups[group_idx];

        // Keep per-group candidates ordered by sort key and preserve batch order
        // for ties by inserting after existing equal elements.
        let insert_idx = group.partition_point(|candidate| {
            compare_rows(&candidate.ordering, &ordering, sort_options)
                .unwrap_or(std::cmp::Ordering::Equal)
                != std::cmp::Ordering::Greater
        });
        group.insert(
            insert_idx,
            NthGroupCandidate {
                value,
                is_null,
                ordering,
            },
        );

        if group.len() > n_required {
            if keep_smallest_n {
                let _ = group.pop();
            } else {
                let _ = group.remove(0);
            }
        }
    }

    fn pick_nth(&self, group_idx: usize) -> (T::Native, bool) {
        let n_required = self.n_required();
        let group = &self.groups[group_idx];

        if group.len() < n_required {
            (T::Native::default(), true)
        } else if self.keep_smallest_n() {
            let candidate = &group[n_required - 1];
            (candidate.value, candidate.is_null)
        } else {
            let candidate = &group[0];
            (candidate.value, candidate.is_null)
        }
    }
}

impl<T> GroupsAccumulator for NthPrimitiveGroupsAccumulator<T>
where
    T: ArrowPrimitiveType + Send,
{
    fn update_batch(
        &mut self,
        values_and_order_cols: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.resize(total_num_groups);
        let vals = values_and_order_cols[0].as_primitive::<T>();

        for (idx, &group_idx) in group_indices.iter().enumerate() {
            if let Some(filter) = opt_filter {
                if !filter.value(idx) {
                    continue;
                }
            }
            if self.ignore_nulls && vals.is_null(idx) {
                continue;
            }

            let ordering = get_row_at_idx(&values_and_order_cols[1..], idx)?;
            self.insert_candidate(
                group_idx,
                vals.value(idx),
                vals.is_null(idx),
                ordering,
            );
        }

        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let num_groups = match emit_to {
            EmitTo::All => self.groups.len(),
            EmitTo::First(n) => n,
        };

        let mut result = Vec::with_capacity(num_groups);
        let mut null_buf = Vec::with_capacity(num_groups);

        for group_idx in 0..num_groups {
            let (val, is_null) = self.pick_nth(group_idx);
            result.push(val);
            null_buf.push(!is_null);
        }

        if let EmitTo::First(n) = emit_to {
            self.groups.drain(0..n);
        }

        let null_buffer = NullBuffer::new(BooleanBuffer::from(null_buf));
        Ok(Arc::new(
            PrimitiveArray::<T>::new(result.into(), Some(null_buffer))
                .with_data_type(self.data_type.clone()),
        ))
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let num_groups = match emit_to {
            EmitTo::All => self.groups.len(),
            EmitTo::First(n) => n,
        };
        let order_fields = ordering_fields(&self.ordering_req, &self.ordering_dtypes);

        let mut value_scalars: Vec<ScalarValue> = Vec::with_capacity(num_groups);
        let mut ordering_scalars: Vec<ScalarValue> = Vec::with_capacity(num_groups);

        for group_idx in 0..num_groups {
            let group = &self.groups[group_idx];

            let scalars: Vec<ScalarValue> = group
                .iter()
                .map(|candidate| {
                    if candidate.is_null {
                        ScalarValue::try_from(self.data_type.clone())
                    } else {
                        ScalarValue::try_from_array(
                            &(Arc::new(
                                PrimitiveArray::<T>::new(
                                    vec![candidate.value].into(),
                                    None,
                                )
                                .with_data_type(self.data_type.clone()),
                            ) as ArrayRef),
                            0,
                        )
                    }
                })
                .collect::<Result<_>>()?;

            value_scalars.push(ScalarValue::List(ScalarValue::new_list_nullable(
                &scalars,
                &self.data_type,
            )));

            if order_fields.is_empty() {
                ordering_scalars.push(ScalarValue::List(ScalarValue::new_list_nullable(
                    &[],
                    &DataType::Null,
                )));
            } else {
                let struct_fields = Fields::from(order_fields.clone());
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(order_fields.len());
                for col_idx in 0..order_fields.len() {
                    let col_scalars: Vec<ScalarValue> = group
                        .iter()
                        .map(|candidate| candidate.ordering[col_idx].clone())
                        .collect();
                    if col_scalars.is_empty() {
                        columns.push(new_empty_array(order_fields[col_idx].data_type()));
                    } else {
                        columns.push(ScalarValue::iter_to_array(col_scalars)?);
                    }
                }
                let struct_array = StructArray::try_new(struct_fields, columns, None)?;
                ordering_scalars.push(
                    SingleRowListArrayBuilder::new(Arc::new(struct_array))
                        .build_list_scalar(),
                );
            }
        }

        if let EmitTo::First(n) = emit_to {
            self.groups.drain(0..n);
        }

        Ok(vec![
            ScalarValue::iter_to_array(value_scalars)?,
            ScalarValue::iter_to_array(ordering_scalars)?,
        ])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        _opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.resize(total_num_groups);

        let value_lists = values[0].as_list::<i32>();
        let ordering_lists = values[1].as_list::<i32>();

        for (idx, &group_idx) in group_indices.iter().enumerate() {
            let inner_vals = value_lists.value(idx);
            let inner_vals = inner_vals.as_primitive::<T>();

            let inner_orderings = ordering_lists.value(idx);
            let Some(inner_orderings) =
                inner_orderings.as_any().downcast_ref::<StructArray>()
            else {
                return exec_err!("Expected ordering state as StructArray");
            };

            for row_idx in 0..inner_vals.len() {
                let is_null = inner_vals.is_null(row_idx);
                if self.ignore_nulls && is_null {
                    continue;
                }

                let mut ordering_row = Vec::with_capacity(inner_orderings.num_columns());
                for col in inner_orderings.columns() {
                    ordering_row.push(ScalarValue::try_from_array(col, row_idx)?);
                }
                self.insert_candidate(
                    group_idx,
                    inner_vals.value(row_idx),
                    is_null,
                    ordering_row,
                );
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        let mut total = 0;
        total += self.ordering_dtypes.capacity() * size_of::<DataType>();
        for dtype in &self.ordering_dtypes {
            total += dtype.size() - size_of_val(dtype);
        }
        total += self.ordering_req.capacity() * size_of::<PhysicalSortExpr>();
        total += self.groups.capacity() * size_of::<Vec<NthGroupCandidate<T::Native>>>();
        for group in &self.groups {
            total += group.capacity() * size_of::<NthGroupCandidate<T::Native>>();
            for candidate in group {
                total += ScalarValue::size_of_vec(&candidate.ordering);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{Int64Type, Schema};
    use datafusion_physical_expr::expressions::col;

    #[test]
    fn test_nth_group_acc_first() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
            Field::new("c", DataType::Int64, true),
        ]));

        let sort_keys = [PhysicalSortExpr {
            expr: col("c", &schema).unwrap(),
            options: SortOptions::default(),
        }];

        let mut group_acc = NthPrimitiveGroupsAccumulator::<Int64Type>::try_new(
            1,
            sort_keys.into(),
            true,
            &DataType::Int64,
            &[DataType::Int64],
        )?;

        let val_with_orderings = {
            let vals = Arc::new(Int64Array::from(vec![
                Some(10),
                Some(20),
                Some(30),
                Some(40),
            ])) as Arc<dyn Array>;
            let orderings =
                Arc::new(Int64Array::from(vec![1, -9, 3, -6])) as Arc<dyn Array>;
            vec![vals, orderings]
        };

        group_acc.update_batch(&val_with_orderings, &[0, 0, 0, 0], None, 1)?;

        let result = group_acc.evaluate(EmitTo::All)?;
        let result_arr = result.as_primitive::<Int64Type>();
        assert_eq!(result_arr.value(0), 20);

        Ok(())
    }

    #[test]
    fn test_nth_group_acc_second() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("c", DataType::Int64, true),
        ]));

        let sort_keys = [PhysicalSortExpr {
            expr: col("c", &schema).unwrap(),
            options: SortOptions::default(),
        }];

        let mut group_acc = NthPrimitiveGroupsAccumulator::<Int64Type>::try_new(
            2,
            sort_keys.into(),
            false,
            &DataType::Int64,
            &[DataType::Int64],
        )?;

        let val_with_orderings = {
            let vals = Arc::new(Int64Array::from(vec![
                Some(10),
                Some(20),
                Some(30),
                Some(40),
            ])) as Arc<dyn Array>;
            let orderings =
                Arc::new(Int64Array::from(vec![1, -9, 3, -6])) as Arc<dyn Array>;
            vec![vals, orderings]
        };

        group_acc.update_batch(&val_with_orderings, &[0, 0, 0, 0], None, 1)?;

        let result = group_acc.evaluate(EmitTo::All)?;
        let result_arr = result.as_primitive::<Int64Type>();
        assert_eq!(result_arr.value(0), 40);

        Ok(())
    }

    #[test]
    fn test_nth_group_acc_exceeds_count() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("c", DataType::Int64, true),
        ]));

        let sort_keys = [PhysicalSortExpr {
            expr: col("c", &schema).unwrap(),
            options: SortOptions::default(),
        }];

        let mut group_acc = NthPrimitiveGroupsAccumulator::<Int64Type>::try_new(
            10,
            sort_keys.into(),
            true,
            &DataType::Int64,
            &[DataType::Int64],
        )?;

        let val_with_orderings = {
            let vals = Arc::new(Int64Array::from(vec![
                Some(10),
                Some(20),
                Some(30),
                Some(40),
            ])) as Arc<dyn Array>;
            let orderings =
                Arc::new(Int64Array::from(vec![1, -9, 3, -6])) as Arc<dyn Array>;
            vec![vals, orderings]
        };

        group_acc.update_batch(&val_with_orderings, &[0, 0, 0, 0], None, 1)?;

        let result = group_acc.evaluate(EmitTo::All)?;
        let result_arr = result.as_primitive::<Int64Type>();
        assert!(result_arr.is_null(0));

        Ok(())
    }

    #[test]
    fn test_nth_group_acc_multiple_groups() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("c", DataType::Int64, true),
        ]));

        let sort_keys = [PhysicalSortExpr {
            expr: col("c", &schema).unwrap(),
            options: SortOptions::default(),
        }];

        let mut group_acc = NthPrimitiveGroupsAccumulator::<Int64Type>::try_new(
            1,
            sort_keys.into(),
            false,
            &DataType::Int64,
            &[DataType::Int64],
        )?;

        let val_with_orderings = {
            let vals = Arc::new(Int64Array::from(vec![
                Some(10),
                Some(20),
                Some(30),
                Some(40),
            ])) as Arc<dyn Array>;
            let orderings =
                Arc::new(Int64Array::from(vec![1, -9, 3, -6])) as Arc<dyn Array>;
            vec![vals, orderings]
        };

        group_acc.update_batch(
            &val_with_orderings,
            &[0, 1, 2, 1],
            Some(&BooleanArray::from(vec![true, true, false, true])),
            3,
        )?;

        let result = group_acc.evaluate(EmitTo::All)?;
        let result_arr = result.as_primitive::<Int64Type>();
        assert_eq!(result_arr.value(0), 10);
        assert_eq!(result_arr.value(1), 20);
        assert!(result_arr.is_null(2));

        Ok(())
    }

    #[test]
    fn test_nth_group_acc_from_end() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("c", DataType::Int64, true),
        ]));

        let sort_keys = [PhysicalSortExpr {
            expr: col("c", &schema).unwrap(),
            options: SortOptions::default(),
        }];

        let mut group_acc = NthPrimitiveGroupsAccumulator::<Int64Type>::try_new(
            -1,
            sort_keys.into(),
            false,
            &DataType::Int64,
            &[DataType::Int64],
        )?;

        let val_with_orderings = {
            let vals = Arc::new(Int64Array::from(vec![
                Some(10),
                Some(20),
                Some(30),
                Some(40),
            ])) as Arc<dyn Array>;
            let orderings =
                Arc::new(Int64Array::from(vec![1, -9, 3, -6])) as Arc<dyn Array>;
            vec![vals, orderings]
        };

        group_acc.update_batch(&val_with_orderings, &[0, 0, 0, 0], None, 1)?;

        let result = group_acc.evaluate(EmitTo::All)?;
        let result_arr = result.as_primitive::<Int64Type>();
        assert_eq!(result_arr.value(0), 30);

        Ok(())
    }

    #[test]
    fn test_nth_group_acc_state_and_merge_with_empty_group() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("c", DataType::Int64, true),
        ]));

        let sort_keys = [PhysicalSortExpr {
            expr: col("c", &schema).unwrap(),
            options: SortOptions::default(),
        }];

        let mut partial_acc = NthPrimitiveGroupsAccumulator::<Int64Type>::try_new(
            1,
            sort_keys.clone().into(),
            false,
            &DataType::Int64,
            &[DataType::Int64],
        )?;

        let val_with_orderings = {
            let vals =
                Arc::new(Int64Array::from(vec![Some(20), Some(10)])) as Arc<dyn Array>;
            let orderings = Arc::new(Int64Array::from(vec![2, 1])) as Arc<dyn Array>;
            vec![vals, orderings]
        };

        // Keep group 1 empty to ensure state serialization remains schema-stable.
        partial_acc.update_batch(&val_with_orderings, &[0, 0], None, 2)?;

        let states = partial_acc.state(EmitTo::All)?;
        let DataType::List(list_item_field) = states[1].data_type() else {
            panic!("Expected list state for ordering values");
        };
        let DataType::Struct(struct_fields) = list_item_field.data_type() else {
            panic!("Expected struct state for ordering values");
        };
        assert_eq!(struct_fields.len(), 1);
        assert_eq!(struct_fields[0].data_type(), &DataType::Int64);

        let mut final_acc = NthPrimitiveGroupsAccumulator::<Int64Type>::try_new(
            1,
            sort_keys.into(),
            false,
            &DataType::Int64,
            &[DataType::Int64],
        )?;
        final_acc.merge_batch(&states, &[0, 1], None, 2)?;

        let result = final_acc.evaluate(EmitTo::All)?;
        let result_arr = result.as_primitive::<Int64Type>();
        assert_eq!(result_arr.value(0), 10);
        assert!(result_arr.is_null(1));

        Ok(())
    }
}
