//! Named tool registration, schema validation, and invocation dispatch.

use std::{collections::BTreeMap, fmt, sync::Arc};

use crate::{
    message::{ToolCallBlock, ToolResultBlock},
    model::ToolDefinition,
};

use super::{Tool, ToolContext, ToolError, ToolFuture, ToolResult};

/// A collection of named tools with compiled JSON Schema validators.
#[derive(Default)]
pub struct ToolRegistry {
    entries: BTreeMap<String, RegisteredTool>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Registers one owned tool.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] when the name or schema is invalid, or when the
    /// name is already registered.
    pub fn register<T>(&mut self, tool: T) -> ToolResult<()>
    where
        T: Tool + 'static,
    {
        self.register_shared(Arc::new(tool))
    }

    /// Registers one shared tool trait object.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] when the name or schema is invalid, or when the
    /// name is already registered.
    pub fn register_shared(&mut self, tool: Arc<dyn Tool>) -> ToolResult<()> {
        let definition = tool.definition();
        ToolDefinition::new(
            definition.name.clone(),
            definition.description.clone(),
            definition.input_schema.clone(),
        )
        .map_err(|error| ToolError::new(error.to_string()).with_code("invalid_tool_definition"))?;
        if self.entries.contains_key(&definition.name) {
            return Err(ToolError::new(format!(
                "tool `{}` is already registered",
                definition.name
            ))
            .with_code("duplicate_tool"));
        }
        let validator = jsonschema::validator_for(&definition.input_schema).map_err(|error| {
            ToolError::new(format!("invalid JSON Schema: {error}")).with_code("invalid_tool_schema")
        })?;
        self.entries
            .insert(definition.name.clone(), RegisteredTool { tool, validator });
        Ok(())
    }

    /// Removes and returns a registered tool.
    pub fn remove(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        self.entries.remove(name).map(|entry| entry.tool)
    }

    /// Returns a shared handle to a registered tool.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.entries.get(name).map(|entry| Arc::clone(&entry.tool))
    }

    /// Returns whether a name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether no tools are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns model-visible definitions in stable name order.
    #[must_use]
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.entries
            .values()
            .map(|entry| entry.tool.definition().clone())
            .collect()
    }

    /// Validates and dispatches one model-produced tool call.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] when the tool is unknown, the arguments are not
    /// valid JSON or do not match its schema, or execution fails.
    #[must_use]
    pub fn invoke<'a>(
        &'a self,
        call: &ToolCallBlock,
        context: ToolContext,
    ) -> ToolFuture<'a, ToolResultBlock> {
        let Some(entry) = self.entries.get(call.name()) else {
            let error = ToolError::new(format!("tool `{}` is not registered", call.name()))
                .with_code("unknown_tool");
            return Box::pin(async move { Err(error) });
        };
        let input = match call.parsed_input() {
            Ok(input) => input,
            Err(error) => {
                let error = ToolError::new(error.to_string()).with_code("invalid_tool_input");
                return Box::pin(async move { Err(error) });
            }
        };
        if let Err(error) = entry.validator.validate(&input) {
            let location = error.instance_path();
            let message = if location.as_str().is_empty() {
                format!("tool input does not match its JSON Schema: {error}")
            } else {
                format!("tool input at `{location}` does not match its JSON Schema: {error}")
            };
            let error = ToolError::new(message).with_code("tool_schema_mismatch");
            return Box::pin(async move { Err(error) });
        }
        entry.tool.invoke(call, context)
    }
}

impl fmt::Debug for ToolRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolRegistry")
            .field("tool_names", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}

struct RegisteredTool {
    tool: Arc<dyn Tool>,
    validator: jsonschema::Validator,
}
