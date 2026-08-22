//! Streaming chat response accumulation.

use std::fmt;

use serde_json::Value;

use crate::message::{
    ContentBlock, StructuredOutputBlock, StructuredOutputError, TextBlock, ThinkingBlock,
    ToolCallBlock, ToolCallError, ToolCallState, Usage, generate_timestamp,
};

use super::{ChatEvent, ChatResponse, FinishReason, ModelError};

/// An error returned while accumulating a chat event stream.
#[derive(Debug)]
pub enum ChatStreamError {
    /// An event arrived after a terminal event.
    AlreadyFinished,
    /// The response was requested before a terminal event arrived.
    NotFinished,
    /// A content delta contained a blank block identifier.
    EmptyBlockId,
    /// A block identifier was reused for a different content type.
    BlockTypeMismatch(String),
    /// Deltas for one tool call disagreed about its name.
    ToolNameMismatch {
        /// The tool call identifier.
        tool_call_id: String,
        /// The name supplied by the first delta.
        expected: String,
        /// The conflicting name.
        actual: String,
    },
    /// Deltas for one structured-output block used different schemas.
    SchemaMismatch(String),
    /// A tool call could not be constructed or completed.
    ToolCall(ToolCallError),
    /// Structured output could not be constructed or completed.
    StructuredOutput(StructuredOutputError),
    /// The model stream ended with a provider failure.
    Model(ModelError),
}

impl fmt::Display for ChatStreamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyFinished => formatter.write_str("chat stream is already finished"),
            Self::NotFinished => formatter.write_str("chat stream has not finished"),
            Self::EmptyBlockId => formatter.write_str("content block id cannot be empty"),
            Self::BlockTypeMismatch(block_id) => {
                write!(formatter, "content block `{block_id}` changed type")
            }
            Self::ToolNameMismatch {
                tool_call_id,
                expected,
                actual,
            } => write!(
                formatter,
                "tool call `{tool_call_id}` changed name from `{expected}` to `{actual}`"
            ),
            Self::SchemaMismatch(block_id) => {
                write!(
                    formatter,
                    "structured-output block `{block_id}` changed schema"
                )
            }
            Self::ToolCall(error) => write!(formatter, "invalid streamed tool call: {error}"),
            Self::StructuredOutput(error) => {
                write!(formatter, "invalid streamed structured output: {error}")
            }
            Self::Model(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ChatStreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ToolCall(error) => Some(error),
            Self::StructuredOutput(error) => Some(error),
            Self::Model(error) => Some(error),
            Self::AlreadyFinished
            | Self::NotFinished
            | Self::EmptyBlockId
            | Self::BlockTypeMismatch(_)
            | Self::ToolNameMismatch { .. }
            | Self::SchemaMismatch(_) => None,
        }
    }
}

impl From<ToolCallError> for ChatStreamError {
    fn from(error: ToolCallError) -> Self {
        Self::ToolCall(error)
    }
}

impl From<StructuredOutputError> for ChatStreamError {
    fn from(error: StructuredOutputError) -> Self {
        Self::StructuredOutput(error)
    }
}

/// Accumulates incremental chat events into one final [`ChatResponse`].
#[derive(Debug)]
pub struct ChatResponseAccumulator {
    response: ChatResponse,
    pending_usage: Option<Usage>,
    terminal_error: Option<ModelError>,
    finished: bool,
}

impl ChatResponseAccumulator {
    /// Creates an empty response accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            response: ChatResponse::partial([]),
            pending_usage: None,
            terminal_error: None,
            finished: false,
        }
    }

    /// Applies one streaming event.
    ///
    /// # Errors
    ///
    /// Returns [`ChatStreamError`] when an event violates stream ordering,
    /// changes a block's identity, or completes malformed JSON.
    pub fn apply(&mut self, event: ChatEvent) -> Result<(), ChatStreamError> {
        if self.finished || self.terminal_error.is_some() {
            return Err(ChatStreamError::AlreadyFinished);
        }

        match event {
            ChatEvent::TextDelta { block_id, delta } => self.apply_text_delta(block_id, &delta),
            ChatEvent::ThinkingDelta { block_id, delta } => {
                self.apply_thinking_delta(block_id, &delta)
            }
            ChatEvent::ToolCallDelta {
                tool_call_id,
                tool_name,
                delta,
            } => self.apply_tool_call_delta(tool_call_id, tool_name, &delta),
            ChatEvent::StructuredOutputDelta {
                block_id,
                schema,
                delta,
            } => self.apply_structured_output_delta(block_id, schema, &delta),
            ChatEvent::Usage { usage } => {
                self.pending_usage = Some(self.pending_usage.unwrap_or_default() + usage);
                Ok(())
            }
            ChatEvent::Finished { reason } => self.finish(reason),
            ChatEvent::Error { error } => {
                self.terminal_error = Some(error);
                Ok(())
            }
        }
    }

    /// Returns the response accumulated so far.
    ///
    /// Pending usage is intentionally hidden until a successful finish event.
    #[must_use]
    pub const fn response(&self) -> &ChatResponse {
        &self.response
    }

    /// Returns whether a successful or error terminal event has arrived.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finished || self.terminal_error.is_some()
    }

    /// Consumes the accumulator and returns the final response.
    ///
    /// # Errors
    ///
    /// Returns [`ChatStreamError::NotFinished`] before a terminal event or
    /// [`ChatStreamError::Model`] after a model error event.
    pub fn into_response(self) -> Result<ChatResponse, ChatStreamError> {
        if let Some(error) = self.terminal_error {
            Err(ChatStreamError::Model(error))
        } else if self.finished {
            Ok(self.response)
        } else {
            Err(ChatStreamError::NotFinished)
        }
    }

    fn apply_text_delta(&mut self, block_id: String, delta: &str) -> Result<(), ChatStreamError> {
        validate_block_id(&block_id)?;
        if let Some(index) = find_block(&self.response.content, &block_id) {
            match &mut self.response.content[index] {
                ContentBlock::Text(block) => {
                    block.text.push_str(delta);
                    Ok(())
                }
                _ => Err(ChatStreamError::BlockTypeMismatch(block_id)),
            }
        } else {
            let mut block = TextBlock::new(delta);
            block.id = block_id;
            self.response.content.push(ContentBlock::Text(block));
            Ok(())
        }
    }

    fn apply_thinking_delta(
        &mut self,
        block_id: String,
        delta: &str,
    ) -> Result<(), ChatStreamError> {
        validate_block_id(&block_id)?;
        if let Some(index) = find_block(&self.response.content, &block_id) {
            match &mut self.response.content[index] {
                ContentBlock::Thinking(block) => {
                    block.thinking.push_str(delta);
                    Ok(())
                }
                _ => Err(ChatStreamError::BlockTypeMismatch(block_id)),
            }
        } else {
            let mut block = ThinkingBlock::new(delta);
            block.id = block_id;
            self.response.content.push(ContentBlock::Thinking(block));
            Ok(())
        }
    }

    fn apply_tool_call_delta(
        &mut self,
        tool_call_id: String,
        tool_name: String,
        delta: &str,
    ) -> Result<(), ChatStreamError> {
        if let Some(index) = find_block(&self.response.content, &tool_call_id) {
            match &mut self.response.content[index] {
                ContentBlock::ToolCall(block) if block.name() == tool_name => {
                    block.append_input(delta);
                    Ok(())
                }
                ContentBlock::ToolCall(block) => Err(ChatStreamError::ToolNameMismatch {
                    tool_call_id,
                    expected: block.name().to_owned(),
                    actual: tool_name,
                }),
                _ => Err(ChatStreamError::BlockTypeMismatch(tool_call_id)),
            }
        } else {
            let mut block = ToolCallBlock::streaming(&tool_call_id, tool_name)?;
            block.append_input(delta);
            self.response.content.push(ContentBlock::ToolCall(block));
            Ok(())
        }
    }

    fn apply_structured_output_delta(
        &mut self,
        block_id: String,
        schema: Value,
        delta: &str,
    ) -> Result<(), ChatStreamError> {
        validate_block_id(&block_id)?;
        if let Some(index) = find_block(&self.response.content, &block_id) {
            match &mut self.response.content[index] {
                ContentBlock::StructuredOutput(block) if block.schema() == &schema => {
                    block.append_output_delta(delta)?;
                    Ok(())
                }
                ContentBlock::StructuredOutput(_) => Err(ChatStreamError::SchemaMismatch(block_id)),
                _ => Err(ChatStreamError::BlockTypeMismatch(block_id)),
            }
        } else {
            let mut block = StructuredOutputBlock::streaming(schema)?;
            block.id = block_id;
            block.append_output_delta(delta)?;
            self.response
                .content
                .push(ContentBlock::StructuredOutput(block));
            Ok(())
        }
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), ChatStreamError> {
        let mut completed_content = self.response.content.clone();
        finish_blocks(&mut completed_content)?;

        self.response.content = completed_content;
        self.response.is_last = true;
        self.response.finish_reason = Some(reason);
        self.response.usage = self.pending_usage;
        self.finished = true;
        Ok(())
    }
}

impl Default for ChatResponseAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_block_id(block_id: &str) -> Result<(), ChatStreamError> {
    if block_id.trim().is_empty() {
        Err(ChatStreamError::EmptyBlockId)
    } else {
        Ok(())
    }
}

fn find_block(content: &[ContentBlock], block_id: &str) -> Option<usize> {
    content
        .iter()
        .position(|block| content_block_id(block) == block_id)
}

fn content_block_id(block: &ContentBlock) -> &str {
    match block {
        ContentBlock::Text(block) => &block.id,
        ContentBlock::Thinking(block) => &block.id,
        ContentBlock::ToolCall(block) => block.id(),
        ContentBlock::ToolResult(block) => block.id(),
        ContentBlock::StructuredOutput(block) => &block.id,
        ContentBlock::Data(block) => &block.id,
    }
}

fn finish_blocks(content: &mut [ContentBlock]) -> Result<(), ChatStreamError> {
    let finished_at = generate_timestamp();
    for block in content {
        match block {
            ContentBlock::Text(block) => block.finished_at = Some(finished_at.clone()),
            ContentBlock::Thinking(block) => block.finished_at = Some(finished_at.clone()),
            ContentBlock::ToolCall(block) => {
                block.parsed_input()?;
                block.state = ToolCallState::Finished;
                block.finished_at = Some(finished_at.clone());
            }
            ContentBlock::StructuredOutput(block) => block.finish()?,
            ContentBlock::Data(block) => block.finished_at = Some(finished_at.clone()),
            ContentBlock::ToolResult(_) => {}
        }
    }
    Ok(())
}
