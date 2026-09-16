use super::{McpClient, error};
use agentscope::{Tool, ToolContext, ToolDefinition, ToolFuture, ToolResult, ToolResultOutput};
use serde_json::{Value, json};
use std::sync::Arc;

/// A discovered tool tied to one live connection. `ToolContext` metadata is never
/// sent to the server. MCP defines no universal idempotency guarantee; application
/// approval and external reconciliation remain necessary for side effects.
#[derive(Clone)]
pub struct McpTool {
    client: McpClient,
    remote_name: String,
    definition: ToolDefinition,
    input_validator: Arc<jsonschema::Validator>,
    output_validator: Option<Arc<jsonschema::Validator>>,
}

pub(crate) fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl McpTool {
    pub(crate) fn from_wire(client: McpClient, namespace: &str, value: &Value) -> ToolResult<Self> {
        let remote = value
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| valid_name(n))
            .ok_or_else(|| error("mcp_name", "MCP tool name is not model-portable"))?;
        let name = format!("{namespace}__{remote}");
        if !valid_name(&name) {
            return Err(error(
                "mcp_name",
                "namespaced MCP tool name exceeds 64 bytes",
            ));
        }
        let schema = value
            .get("inputSchema")
            .filter(|s| s.is_object() && s.get("type").and_then(Value::as_str) == Some("object"))
            .ok_or_else(|| error("mcp_schema", "MCP tool requires an object input schema"))?;
        let description = match value.get("description") {
            Some(Value::String(text)) => text.clone(),
            None => String::new(),
            _ => return Err(error("mcp_schema", "invalid MCP tool description")),
        };
        let input_validator = Arc::new(
            jsonschema::validator_for(schema)
                .map_err(|_| error("mcp_schema", "invalid MCP input schema"))?,
        );
        let output_validator = value
            .get("outputSchema")
            .map(|schema| {
                if !schema.is_object()
                    || schema.get("type").and_then(Value::as_str) != Some("object")
                {
                    return Err(error(
                        "mcp_schema",
                        "MCP output schema must be an object schema",
                    ));
                }
                jsonschema::validator_for(schema)
                    .map(Arc::new)
                    .map_err(|_| error("mcp_schema", "invalid MCP output schema"))
            })
            .transpose()?;
        let definition = ToolDefinition::new(name, description, schema.clone())
            .map_err(|_| error("mcp_schema", "invalid MCP tool definition"))?;
        Ok(Self {
            client,
            remote_name: remote.into(),
            definition,
            input_validator,
            output_validator,
        })
    }

    #[must_use]
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }

    fn decode(&self, result: &Value) -> ToolResult<ToolResultOutput> {
        let failed = match result.get("isError") {
            None => false,
            Some(Value::Bool(b)) => *b,
            _ => return Err(error("mcp_result", "invalid MCP isError flag")),
        };
        let content = result
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| error("mcp_result", "missing MCP result content"))?;
        let mut texts = Vec::new();
        for block in content {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                return Err(error(
                    "mcp_content",
                    "MCP v1 adapter supports text/structured results only; unsupported content was not silently discarded",
                ));
            }
            texts.push(
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| error("mcp_result", "invalid MCP text result"))?
                    .to_owned(),
            );
        }
        let structured = result.get("structuredContent");
        if let Some(value) = structured {
            if !value.is_object() {
                return Err(error(
                    "mcp_result",
                    "MCP structured content must be an object",
                ));
            }
        }
        if !failed {
            if let Some(validator) = &self.output_validator {
                let value = structured.ok_or_else(|| {
                    error(
                        "mcp_result",
                        "MCP structured output is required by output schema",
                    )
                })?;
                if !validator.is_valid(value) {
                    return Err(error("mcp_result", "MCP structured output violates schema"));
                }
            }
        }
        let text = if let Some(value) = structured {
            json!({"content":texts,"structuredContent":value}).to_string()
        } else {
            texts.join("\n")
        };
        if failed {
            return Err(error("mcp_tool_error", &text));
        }
        Ok(text.into())
    }
}

impl Tool for McpTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    fn execute(&self, input: Value, _context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        Box::pin(async move {
            if !self.input_validator.is_valid(&input) {
                return Err(error("mcp_input", "MCP arguments do not match tool schema"));
            }
            let result = self
                .client
                .request(
                    "tools/call",
                    json!({"name":self.remote_name,"arguments":input}),
                )
                .await?;
            self.decode(&result).map_err(|error| {
                if error.code.as_deref() == Some("mcp_tool_error") { error }
                else { agentscope::ToolError::in_doubt(format!(
                    "MCP returned unusable tool output ({}); execution may have succeeded, reconcile before retrying",
                    error.code.as_deref().unwrap_or("mcp_result")
                )) }
            })
        })
    }
}
