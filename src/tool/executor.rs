//! Batch tool-call execution with stable result ordering.

use std::{fmt, sync::Arc};

use futures_util::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::message::{Metadata, ToolCallBlock, ToolResultBlock, ToolResultState};

use super::{ToolContext, ToolError, ToolFuture, ToolRegistry, ToolResult};

/// The scheduling strategy used for a batch of tool calls.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionMode {
    /// Wait for each call to finish before starting the next call.
    #[default]
    Sequential,
    /// Start all calls together while preserving their input order in results.
    Concurrent,
}

/// Executes model-produced tool calls through a shared [`ToolRegistry`].
///
/// Ordinary dispatch failures are represented as terminal error
/// [`ToolResultBlock`] values. This lets a batch retain one result per input
/// call even when some calls fail.
#[derive(Clone)]
pub struct ToolExecutor {
    registry: Arc<ToolRegistry>,
    mode: ToolExecutionMode,
}

impl ToolExecutor {
    /// Creates a sequential executor owning the supplied registry.
    #[must_use]
    pub fn new(registry: ToolRegistry) -> Self {
        Self::from_shared(Arc::new(registry))
    }

    /// Creates a sequential executor backed by a shared registry.
    #[must_use]
    pub const fn from_shared(registry: Arc<ToolRegistry>) -> Self {
        Self {
            registry,
            mode: ToolExecutionMode::Sequential,
        }
    }

    /// Selects the scheduling strategy for batches.
    #[must_use]
    pub const fn with_mode(mut self, mode: ToolExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns the selected scheduling strategy.
    #[must_use]
    pub const fn mode(&self) -> ToolExecutionMode {
        self.mode
    }

    /// Returns the executor's shared tool registry.
    #[must_use]
    pub const fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }

    /// Executes one call and converts dispatch failures into error results.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] only if a result cannot be constructed from the
    /// already-validated call identity.
    #[must_use]
    pub fn execute_one<'a>(
        &'a self,
        call: &'a ToolCallBlock,
        context: ToolContext,
    ) -> ToolFuture<'a, ToolResultBlock> {
        Box::pin(async move {
            match self.registry.invoke(call, context).await {
                Ok(result) => Ok(result),
                Err(error) => error_result(call, error),
            }
        })
    }

    /// Executes a batch and returns exactly one result per call in input order.
    ///
    /// Sequential mode is the default because tools may have side effects.
    /// Concurrent mode starts every invocation together, but still returns
    /// results in the same order as `calls`.
    ///
    /// # Errors
    ///
    /// Returns [`ToolError`] only if a result cannot be constructed from an
    /// already-validated call identity. Tool lookup, validation, and execution
    /// failures are returned as error result blocks instead.
    #[must_use]
    pub fn execute_all<'a>(
        &'a self,
        calls: &'a [ToolCallBlock],
        context: ToolContext,
    ) -> ToolFuture<'a, Vec<ToolResultBlock>> {
        Box::pin(async move {
            match self.mode {
                ToolExecutionMode::Sequential => {
                    let mut results = Vec::with_capacity(calls.len());
                    for call in calls {
                        results.push(self.execute_one(call, context.clone()).await?);
                    }
                    Ok(results)
                }
                ToolExecutionMode::Concurrent => join_all(
                    calls
                        .iter()
                        .map(|call| self.execute_one(call, context.clone())),
                )
                .await
                .into_iter()
                .collect(),
            }
        })
    }
}

impl fmt::Debug for ToolExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolExecutor")
            .field("registry", &self.registry)
            .field("mode", &self.mode)
            .finish()
    }
}

fn error_result(call: &ToolCallBlock, error: ToolError) -> ToolResult<ToolResultBlock> {
    let mut metadata = Metadata::new();
    metadata.insert(
        "error".to_owned(),
        json!({
            "message": &error.message,
            "code": &error.code,
            "retryable": error.retryable,
        }),
    );
    ToolResultBlock::finished(
        call.id(),
        call.name(),
        error.message,
        ToolResultState::Error,
    )
    .map(|result| result.with_metadata(metadata))
    .map_err(|result_error| {
        ToolError::new(result_error.to_string()).with_code("invalid_tool_result")
    })
}
