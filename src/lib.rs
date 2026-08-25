//! `AgentScope` for Rust.

#![forbid(unsafe_code)]

pub mod message;
pub mod model;

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
