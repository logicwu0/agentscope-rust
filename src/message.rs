//! Message types exchanged between agents, models, tools, and users.

mod content;
mod data;
mod msg;
mod role;
mod structured_output;
mod thinking;
mod tool_call;
mod tool_result;
mod usage;

use chrono::Local;
use serde_json::{Map, Value};
use uuid::Uuid;

pub use content::{ContentBlock, TextBlock};
pub use data::{Base64Source, DataBlock, DataBlockError, DataSource, UrlSource};
pub use msg::Msg;
pub use role::Role;
pub use structured_output::{StructuredOutputBlock, StructuredOutputError, StructuredOutputState};
pub use thinking::{ThinkingBlock, ThinkingBlockError};
pub use tool_call::{
    PermissionBehavior, PermissionRule, ToolCallBlock, ToolCallError, ToolCallState,
};
pub use tool_result::{
    ToolResultBlock, ToolResultContent, ToolResultError, ToolResultOutput, ToolResultState,
};
pub use usage::Usage;

/// Arbitrary JSON metadata attached to a message.
pub type Metadata = Map<String, Value>;

pub(crate) fn generate_id() -> String {
    Uuid::new_v4().simple().to_string()
}

pub(crate) fn generate_timestamp() -> String {
    Local::now()
        .naive_local()
        .format("%Y-%m-%dT%H:%M:%S%.6f")
        .to_string()
}

#[cfg(test)]
mod tests;
