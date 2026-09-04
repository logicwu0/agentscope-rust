//! `AgentScope` for Rust.

#![forbid(unsafe_code)]

pub mod agent;
pub mod memory;
pub mod message;
pub mod model;
pub mod tool;

pub use agent::{
    AGENT_STATE_VERSION, Agent, AgentError, AgentEvent, AgentEventStream, AgentFuture, AgentHook,
    AgentHookError, AgentHookEvent, AgentHookFuture, AgentHookResult, AgentInterruptHandle,
    AgentResult, AgentState, InMemoryStateStore, PendingToolCalls, ReActAgent, StateKey,
    StateRecord, StateStore, StateStoreError, StateStoreFuture, StateStoreResult, ToolConfirmation,
    ToolConfirmationDecision,
};
#[cfg(feature = "sqlite")]
pub use memory::SQLiteMemory;
pub use memory::{InMemoryMemory, Memory, MemoryError, MemoryFuture, MemoryResult};

pub use model::{
    ChatEvent, ChatEventStream, ChatModel, ChatRequest, ChatRequestError, ChatResponse,
    ChatResponseAccumulator, ChatStreamError, FinishReason, GenerateOptions, MockChatModel,
    ModelCapabilities, ModelCapability, ModelError, ModelFuture, ModelResult, OpenAIChatModel,
    OpenAIChatModelBuilder, OpenAIConfigError, RetryPolicy, ToolDefinition,
};

pub use message::{
    Base64Source, ContentBlock, DataBlock, DataBlockError, DataSource, Metadata, Msg,
    PermissionBehavior, PermissionRule, Role, StructuredOutputBlock, StructuredOutputError,
    StructuredOutputState, TextBlock, ThinkingBlock, ThinkingBlockError, ToolCallBlock,
    ToolCallError, ToolCallState, ToolResultBlock, ToolResultContent, ToolResultError,
    ToolResultOutput, ToolResultState, UrlSource, Usage,
};

pub use tool::{
    MockTool, Tool, ToolContext, ToolError, ToolExecutionMode, ToolExecutor, ToolFuture,
    ToolInvocation, ToolRegistry, ToolResult,
};

/// The current crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_package_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
