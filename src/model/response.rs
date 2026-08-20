//! Provider-neutral chat model responses.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::message::{
    ContentBlock, Metadata, Msg, Role, ToolCallBlock, Usage, generate_id, generate_timestamp,
};

/// Why a chat model stopped generating output.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model completed the response normally.
    Completed,
    /// The model reached an output or context length limit.
    Length,
    /// The model stopped to request one or more tool calls.
    ToolCalls,
    /// The provider's content filter stopped generation.
    ContentFilter,
    /// The caller interrupted generation.
    Interrupted,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Completed => "completed",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Interrupted => "interrupted",
        })
    }
}

/// A complete or partial provider-neutral chat model response.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ChatResponse {
    #[serde(rename = "type", default)]
    response_type: ChatResponseType,
    /// Content accumulated in this response.
    pub content: Vec<ContentBlock>,
    /// Whether this response contains the complete model output.
    pub is_last: bool,
    /// The unique response identifier.
    #[serde(default = "generate_id")]
    pub id: String,
    /// The local creation time in ISO 8601 format.
    #[serde(default = "generate_timestamp")]
    pub created_at: String,
    /// Token usage, when reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Why generation stopped, available on a final response.
    #[serde(
        default,
        rename = "finished_reason",
        skip_serializing_if = "Option::is_none"
    )]
    pub finish_reason: Option<FinishReason>,
    /// Provider-specific or application metadata.
    #[serde(default)]
    pub metadata: Metadata,
}

impl ChatResponse {
    /// Creates a partial response without a finish reason.
    #[must_use]
    pub fn partial(content: impl IntoIterator<Item = ContentBlock>) -> Self {
        Self::new(content, false, None)
    }

    /// Creates a final response with the given finish reason.
    #[must_use]
    pub fn finished(content: impl IntoIterator<Item = ContentBlock>, reason: FinishReason) -> Self {
        Self::new(content, true, Some(reason))
    }

    /// Creates a normally completed final response.
    #[must_use]
    pub fn completed(content: impl IntoIterator<Item = ContentBlock>) -> Self {
        Self::finished(content, FinishReason::Completed)
    }

    /// Attaches provider-reported token usage.
    #[must_use]
    pub const fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Replaces provider-specific or application metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Concatenates all plain-text blocks using `separator`.
    #[must_use]
    pub fn text_content(&self, separator: &str) -> Option<String> {
        let text = self
            .content
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect::<Vec<_>>();

        (!text.is_empty()).then(|| text.join(separator))
    }

    /// Iterates over tool calls in their original content order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCallBlock> {
        self.content.iter().filter_map(|block| match block {
            ContentBlock::ToolCall(tool_call) => Some(tool_call),
            ContentBlock::Text(_)
            | ContentBlock::Thinking(_)
            | ContentBlock::ToolResult(_)
            | ContentBlock::StructuredOutput(_)
            | ContentBlock::Data(_) => None,
        })
    }

    /// Converts this response into an assistant message.
    #[must_use]
    pub fn into_assistant_msg(self, name: impl Into<String>) -> Msg {
        let mut message = Msg::new(name, Role::Assistant, self.content);
        message.metadata = self.metadata;
        message.usage = self.usage;
        message
    }

    fn new(
        content: impl IntoIterator<Item = ContentBlock>,
        is_last: bool,
        finish_reason: Option<FinishReason>,
    ) -> Self {
        Self {
            response_type: ChatResponseType::ChatResponse,
            content: content.into_iter().collect(),
            is_last,
            id: generate_id(),
            created_at: generate_timestamp(),
            usage: None,
            finish_reason,
            metadata: Metadata::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ChatResponseType {
    #[default]
    ChatResponse,
}
