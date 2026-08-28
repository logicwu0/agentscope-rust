//! Asynchronous tool interfaces and deterministic test doubles.

mod core;
mod executor;
mod mock;
mod registry;

pub use crate::model::ToolDefinition;
pub use core::{Tool, ToolContext, ToolError, ToolFuture, ToolResult};
pub use executor::{ToolExecutionMode, ToolExecutor};
pub use mock::{MockTool, ToolInvocation};
pub use registry::ToolRegistry;

#[cfg(test)]
mod tests;
