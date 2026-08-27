//! Object-safe asynchronous tool interface.

use std::{fmt, future::Future, pin::Pin};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    message::{Metadata, ToolCallBlock, ToolResultBlock, ToolResultOutput},
    model::ToolDefinition,
};

/// Result returned by a tool operation.
pub type ToolResult<T> = Result<T, ToolError>;

/// A boxed asynchronous tool operation.
pub type ToolFuture<'a, T> = Pin<Box<dyn Future<Output = ToolResult<T>> + Send + 'a>>;

/// Application data made available to one tool invocation.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolContext {
    /// Session, tracing, authorization, or other caller-provided metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

impl ToolContext {
    /// Creates an empty invocation context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the caller-provided metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// A structured tool definition or execution failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolError {
    /// Human-readable failure description.
    pub message: String,
    /// Stable application or tool-specific error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Whether invoking the tool again may succeed.
    #[serde(default)]
    pub retryable: bool,
}

impl ToolError {
    /// Creates a non-retryable tool error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
            retryable: false,
        }
    }

    /// Attaches a stable error code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Marks whether retrying the invocation may succeed.
    #[must_use]
    pub const fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.code {
            Some(code) => write!(formatter, "tool error {code}: {}", self.message),
            None => write!(formatter, "tool error: {}", self.message),
        }
    }
}

impl std::error::Error for ToolError {}

/// An object-safe asynchronous tool.
///
/// Implementations provide [`Self::execute`]. The default [`Self::invoke`]
/// adapter validates the model-produced call, executes the tool, and preserves
/// its identity in a successful [`ToolResultBlock`].
pub trait Tool: Send + Sync {
    /// Returns the model-visible, validated tool definition.
    fn definition(&self) -> &ToolDefinition;

    /// Executes parsed JSON input.
    fn execute(&self, input: Value, context: ToolContext) -> ToolFuture<'_, ToolResultOutput>;

    /// Validates and executes a model-produced tool call.
    fn invoke<'a>(
        &'a self,
        call: &ToolCallBlock,
        context: ToolContext,
    ) -> ToolFuture<'a, ToolResultBlock> {
        let expected = &self.definition().name;
        if call.name() != expected {
            let error = ToolError::new(format!(
                "tool call requested `{}`, but this tool is `{expected}`",
                call.name()
            ))
            .with_code("tool_name_mismatch");
            return Box::pin(async move { Err(error) });
        }
        let input = match call.parsed_input() {
            Ok(input) => input,
            Err(error) => {
                let error = ToolError::new(error.to_string()).with_code("invalid_tool_input");
                return Box::pin(async move { Err(error) });
            }
        };
        let call_id = call.id().to_owned();
        let tool_name = call.name().to_owned();
        Box::pin(async move {
            let output = self.execute(input, context).await?;
            ToolResultBlock::success(call_id, tool_name, output)
                .map_err(|error| ToolError::new(error.to_string()).with_code("invalid_tool_result"))
        })
    }
}
