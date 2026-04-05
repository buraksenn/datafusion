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

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{
    Column, DFSchema, Result, TableReference, internal_err, plan_datafusion_err, plan_err,
};
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::sort_properties::{ExprProperties, SortProperties};
use datafusion_expr::{
    ColumnarValue, CreateFunction, Expr, ExprSchemable, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use super::context::{FunctionFactory, RegisterFunction};
use crate::execution::session_state::SessionState;

/// Built-in [`FunctionFactory`] that handles `CREATE MACRO` statements
/// by creating scalar UDFs that expand macro bodies via [`ScalarUDFImpl::simplify`].
#[derive(Debug)]
pub struct MacroFunctionFactory;

#[async_trait::async_trait]
impl FunctionFactory for MacroFunctionFactory {
    async fn create(
        &self,
        _state: &SessionState,
        statement: CreateFunction,
    ) -> Result<RegisterFunction> {
        let wrapper = ScalarMacroWrapper::try_from(statement)?;
        Ok(RegisterFunction::Scalar(Arc::new(ScalarUDF::from(wrapper))))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ScalarMacroWrapper {
    name: String,
    body: Expr,
    arg_names: Vec<String>,
    arg_defaults: Vec<Option<Expr>>,
    signature: Signature,
}

impl TryFrom<CreateFunction> for ScalarMacroWrapper {
    type Error = datafusion_common::DataFusionError;

    fn try_from(def: CreateFunction) -> Result<Self> {
        let body = def
            .params
            .function_body
            .ok_or_else(|| plan_datafusion_err!("Macro body is required"))?;

        let (arg_names, arg_defaults) = match def.args {
            Some(args) => {
                let mut names = Vec::with_capacity(args.len());
                let mut defaults = Vec::with_capacity(args.len());
                for arg in args {
                    let name = arg
                        .name
                        .ok_or_else(|| {
                            plan_datafusion_err!("Macro arguments must be named")
                        })?
                        .value;
                    names.push(name);
                    defaults.push(arg.default_expr);
                }
                (names, defaults)
            }
            None => (vec![], vec![]),
        };

        let signature =
            Signature::variadic_any(def.params.behavior.unwrap_or(Volatility::Volatile));

        Ok(Self {
            name: def.name,
            body,
            arg_names,
            arg_defaults,
            signature,
        })
    }
}

impl ScalarUDFImpl for ScalarMacroWrapper {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Null)
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let mut qualified_fields: Vec<(Option<TableReference>, Arc<Field>)> =
            Vec::with_capacity(self.arg_names.len());
        for (i, name) in self.arg_names.iter().enumerate() {
            let data_type = if i < args.arg_fields.len() {
                args.arg_fields[i].data_type().clone()
            } else {
                DataType::Null
            };
            let nullable = i >= args.arg_fields.len() || args.arg_fields[i].is_nullable();
            qualified_fields.push((
                None,
                Arc::new(Field::new(name.clone(), data_type, nullable)),
            ));
        }
        let schema = DFSchema::new_with_metadata(qualified_fields, HashMap::new())?;
        let data_type = self.body.get_type(&schema)?;
        let nullable = self.body.nullable(&schema)?;
        Ok(Arc::new(Field::new(self.name(), data_type, nullable)))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("Macros should be simplified before execution")
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        _info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let required_count = self.arg_defaults.iter().take_while(|d| d.is_none()).count();
        let max_count = self.arg_names.len();

        if args.len() < required_count || args.len() > max_count {
            return plan_err!(
                "Macro '{}' expects {} to {} arguments, got {}",
                self.name,
                required_count,
                max_count,
                args.len()
            );
        }

        let replacement = self.body.clone().transform(|e| {
            if let Expr::Column(Column {
                relation: None,
                ref name,
                ..
            }) = e
                && let Some(pos) = self.arg_names.iter().position(|n| n == name)
            {
                let replacement_expr = if pos < args.len() {
                    args[pos].clone()
                } else if let Some(default) = &self.arg_defaults[pos] {
                    default.clone()
                } else {
                    return plan_err!(
                        "Missing argument '{}' for macro '{}'",
                        name,
                        self.name
                    );
                };
                return Ok(Transformed::yes(replacement_expr));
            }
            Ok(Transformed::no(e))
        })?;

        Ok(ExprSimplifyResult::Simplified(replacement.data))
    }

    fn output_ordering(&self, _input: &[ExprProperties]) -> Result<SortProperties> {
        Ok(SortProperties::Unordered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::SessionContext;

    #[tokio::test]
    async fn test_scalar_macro_basic() -> Result<()> {
        let ctx = SessionContext::new();
        ctx.sql("CREATE MACRO add_macro(a, b) AS a + b")
            .await?
            .show()
            .await?;

        let result = ctx.sql("SELECT add_macro(1, 2)").await?.collect().await?;
        assert_eq!(result[0].num_rows(), 1);
        let val = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(val, 3);
        Ok(())
    }

    #[tokio::test]
    async fn test_scalar_macro_with_defaults() -> Result<()> {
        let ctx = SessionContext::new();
        ctx.sql("CREATE MACRO add_default(a, b := 5) AS a + b")
            .await?
            .show()
            .await?;

        let result = ctx.sql("SELECT add_default(10)").await?.collect().await?;
        let val = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(val, 15);
        Ok(())
    }

    #[tokio::test]
    async fn test_scalar_macro_or_replace() -> Result<()> {
        let ctx = SessionContext::new();
        ctx.sql("CREATE MACRO my_add(a, b) AS a + b")
            .await?
            .show()
            .await?;
        ctx.sql("CREATE OR REPLACE MACRO my_add(a, b) AS a + b + 1")
            .await?
            .show()
            .await?;

        let result = ctx.sql("SELECT my_add(1, 2)").await?.collect().await?;
        let val = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(val, 4);
        Ok(())
    }

    #[tokio::test]
    async fn test_scalar_macro_drop() -> Result<()> {
        let ctx = SessionContext::new();
        ctx.sql("CREATE MACRO drop_me(a) AS a + 1")
            .await?
            .show()
            .await?;
        ctx.sql("DROP FUNCTION drop_me").await?.show().await?;
        let result = ctx.sql("SELECT drop_me(1)").await;
        assert!(result.is_err());
        Ok(())
    }
}
