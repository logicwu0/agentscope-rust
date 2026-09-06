//! Asynchronous tool interfaces and deterministic test doubles.

mod core;
mod executor;
mod idempotent;
mod mock;
mod registry;

pub use crate::model::ToolDefinition;
pub use core::{TOOL_IDEMPOTENCY_KEY, Tool, ToolContext, ToolError, ToolFuture, ToolResult};
pub use executor::{ToolExecutionMode, ToolExecutor};
pub use idempotent::IdempotentTool;
pub use mock::{MockTool, ToolInvocation};
pub use registry::ToolRegistry;

#[cfg(test)]
mod tests;
