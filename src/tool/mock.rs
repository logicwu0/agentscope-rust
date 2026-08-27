//! Deterministic mock tool for tests and examples.

use std::sync::{Mutex, MutexGuard};
use std::{collections::VecDeque, fmt};

use serde_json::Value;

use crate::{message::ToolResultOutput, model::ToolDefinition};

use super::{Tool, ToolContext, ToolError, ToolFuture, ToolResult};

/// One invocation recorded by [`MockTool`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolInvocation {
    /// Parsed JSON arguments received by the tool.
    pub input: Value,
    /// Caller-provided invocation context.
    pub context: ToolContext,
}

/// A deterministic tool backed by a queue of scripted outputs.
pub struct MockTool {
    definition: ToolDefinition,
    outputs: Mutex<VecDeque<ToolResult<ToolResultOutput>>>,
    invocations: Mutex<Vec<ToolInvocation>>,
}

impl MockTool {
    /// Creates an empty mock from a validated tool definition.
    #[must_use]
    pub fn new(definition: ToolDefinition) -> Self {
        Self {
            definition,
            outputs: Mutex::new(VecDeque::new()),
            invocations: Mutex::new(Vec::new()),
        }
    }

    /// Adds a successful output to the script.
    #[must_use]
    pub fn with_output(self, output: impl Into<ToolResultOutput>) -> Self {
        self.push_output(Ok(output.into()));
        self
    }

    /// Adds an execution failure to the script.
    #[must_use]
    pub fn with_error(self, error: ToolError) -> Self {
        self.push_output(Err(error));
        self
    }

    /// Queues one invocation result.
    pub fn push_output(&self, output: ToolResult<ToolResultOutput>) {
        lock(&self.outputs).push_back(output);
    }

    /// Returns all invocations in execution order.
    #[must_use]
    pub fn recorded_invocations(&self) -> Vec<ToolInvocation> {
        lock(&self.invocations).clone()
    }
}

impl fmt::Debug for MockTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MockTool")
            .field("definition", &self.definition)
            .field("remaining_outputs", &lock(&self.outputs).len())
            .field("invocation_count", &lock(&self.invocations).len())
            .finish()
    }
}

impl Tool for MockTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn execute(&self, input: Value, context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        lock(&self.invocations).push(ToolInvocation { input, context });
        let output = lock(&self.outputs).pop_front().unwrap_or_else(|| {
            Err(ToolError::new("mock has no scripted tool output remaining")
                .with_code("mock_exhausted"))
        });
        Box::pin(async move { output })
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
