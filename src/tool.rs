//! Asynchronous tool interfaces and deterministic test doubles.

mod core;
mod mock;

pub use crate::model::ToolDefinition;
pub use core::{Tool, ToolContext, ToolError, ToolFuture, ToolResult};
pub use mock::{MockTool, ToolInvocation};

#[cfg(test)]
mod tests;
