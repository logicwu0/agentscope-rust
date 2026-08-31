//! Provider-neutral streaming agent lifecycle events.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Msg, ToolCallBlock, ToolResultBlock, Usage,
    model::{ChatEvent, FinishReason},
};

use super::AgentError;

/// An incremental event emitted while an agent produces a reply.
///
/// `step` values are one-based model-call numbers within a single reply.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Incremental plain-text model content.
    TextDelta {
        /// The one-based model-call number.
        step: usize,
        /// The content block being updated.
        block_id: String,
        /// The text fragment to append.
        delta: String,
    },
    /// Incremental model reasoning content.
    ThinkingDelta {
        /// The one-based model-call number.
        step: usize,
        /// The content block being updated.
        block_id: String,
        /// The reasoning fragment to append.
        delta: String,
    },
    /// Incremental JSON arguments for a tool call.
    ToolCallDelta {
        /// The one-based model-call number.
        step: usize,
        /// The provider-assigned tool call identifier.
        tool_call_id: String,
        /// The tool being called.
        tool_name: String,
        /// The raw JSON fragment to append.
        delta: String,
    },
    /// Incremental JSON for a structured response.
    StructuredOutputDelta {
        /// The one-based model-call number.
        step: usize,
        /// The content block being updated.
        block_id: String,
        /// The JSON Schema requested from the model.
        schema: Value,
        /// The raw JSON fragment to append.
        delta: String,
    },
    /// Provider-reported token usage for one model call.
    Usage {
        /// The one-based model-call number.
        step: usize,
        /// Usage reported by the model provider.
        usage: Usage,
    },
    /// One model call completed and its output was accumulated.
    StepFinished {
        /// The one-based model-call number.
        step: usize,
        /// Why the model stopped generating.
        reason: FinishReason,
    },
    /// The agent is about to execute a complete tool call.
    ToolStarted {
        /// The model-call number that requested the tool.
        step: usize,
        /// The complete tool call being executed.
        call: ToolCallBlock,
    },
    /// A tool call produced a terminal result.
    ToolFinished {
        /// The model-call number that requested the tool.
        step: usize,
        /// The successful or failed tool result.
        result: ToolResultBlock,
    },
    /// The agent completed its reply.
    Finished {
        /// The number of model calls used by this reply.
        steps: usize,
        /// The final assistant message.
        message: Msg,
    },
    /// The agent terminated with a structured error.
    Error {
        /// The active model-call number, when execution had started.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<usize>,
        /// The terminal failure.
        error: AgentError,
    },
}

impl AgentEvent {
    /// Lifts a model-layer streaming event into one agent step.
    #[must_use]
    pub fn from_chat_event(step: usize, event: ChatEvent) -> Self {
        match event {
            ChatEvent::TextDelta { block_id, delta } => Self::TextDelta {
                step,
                block_id,
                delta,
            },
            ChatEvent::ThinkingDelta { block_id, delta } => Self::ThinkingDelta {
                step,
                block_id,
                delta,
            },
            ChatEvent::ToolCallDelta {
                tool_call_id,
                tool_name,
                delta,
            } => Self::ToolCallDelta {
                step,
                tool_call_id,
                tool_name,
                delta,
            },
            ChatEvent::StructuredOutputDelta {
                block_id,
                schema,
                delta,
            } => Self::StructuredOutputDelta {
                step,
                block_id,
                schema,
                delta,
            },
            ChatEvent::Usage { usage } => Self::Usage { step, usage },
            ChatEvent::Finished { reason } => Self::StepFinished { step, reason },
            ChatEvent::Error { error } => Self::Error {
                step: Some(step),
                error: AgentError::Model(error),
            },
        }
    }
}

#[cfg(test)]
mod tests;
