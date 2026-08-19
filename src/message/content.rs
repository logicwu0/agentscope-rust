//! Text and top-level content block types.

use serde::{Deserialize, Serialize};

use super::{
    DataBlock, ThinkingBlock, ToolCallBlock, ToolResultBlock, generate_id, generate_timestamp,
};

/// A plain-text message content block.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TextBlock {
    /// The text contained in this block.
    pub text: String,
    /// The unique block identifier.
    pub id: String,
    /// The local creation time in ISO 8601 format.
    pub created_at: String,
    /// The completion time for streamed content, when available.
    pub finished_at: Option<String>,
}

impl TextBlock {
    /// Creates a text block.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        let created_at = generate_timestamp();
        Self {
            text: text.into(),
            id: generate_id(),
            finished_at: None,
            created_at,
        }
    }
}

/// A typed block within a message.
///
/// Additional variants will be introduced as multimodal and tool support is
/// implemented.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain-text content.
    Text(TextBlock),
    /// Model reasoning content.
    Thinking(ThinkingBlock),
    /// A request to invoke a tool.
    ToolCall(ToolCallBlock),
    /// Output produced by a tool invocation.
    ToolResult(ToolResultBlock),
    /// Binary or multimodal data.
    Data(DataBlock),
}

impl ContentBlock {
    /// Returns the text when this is a text block.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(block) => Some(block.text.as_str()),
            Self::Thinking(_) | Self::ToolCall(_) | Self::ToolResult(_) | Self::Data(_) => None,
        }
    }
}

impl From<TextBlock> for ContentBlock {
    fn from(block: TextBlock) -> Self {
        Self::Text(block)
    }
}

impl From<ThinkingBlock> for ContentBlock {
    fn from(block: ThinkingBlock) -> Self {
        Self::Thinking(block)
    }
}

impl From<ToolCallBlock> for ContentBlock {
    fn from(block: ToolCallBlock) -> Self {
        Self::ToolCall(block)
    }
}

impl From<ToolResultBlock> for ContentBlock {
    fn from(block: ToolResultBlock) -> Self {
        Self::ToolResult(block)
    }
}

impl From<DataBlock> for ContentBlock {
    fn from(block: DataBlock) -> Self {
        Self::Data(block)
    }
}

impl From<String> for ContentBlock {
    fn from(text: String) -> Self {
        Self::Text(TextBlock::new(text))
    }
}

impl From<&str> for ContentBlock {
    fn from(text: &str) -> Self {
        Self::Text(TextBlock::new(text))
    }
}
