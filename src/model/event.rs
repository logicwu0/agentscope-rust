//! Streaming chat events and response accumulation.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::Usage;

use super::FinishReason;

/// An incremental event emitted by a chat model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    /// Incremental plain-text content.
    TextDelta {
        /// The content block being updated.
        block_id: String,
        /// The text fragment to append.
        delta: String,
    },
    /// Incremental model reasoning content.
    ThinkingDelta {
        /// The content block being updated.
        block_id: String,
        /// The reasoning fragment to append.
        delta: String,
    },
    /// Incremental JSON arguments for a tool call.
    ToolCallDelta {
        /// The provider-assigned tool call identifier.
        tool_call_id: String,
        /// The tool being called.
        tool_name: String,
        /// The raw JSON fragment to append.
        delta: String,
    },
    /// Incremental JSON for a structured response.
    StructuredOutputDelta {
        /// The content block being updated.
        block_id: String,
        /// The JSON Schema requested from the model.
        schema: Value,
        /// The raw JSON fragment to append.
        delta: String,
    },
    /// Provider-reported token usage.
    Usage {
        /// Usage to accumulate for the final response.
        usage: Usage,
    },
    /// Successful end of the model stream.
    Finished {
        /// Why the provider stopped generating output.
        reason: FinishReason,
    },
    /// Terminal model or provider failure.
    Error {
        /// Structured information about the failure.
        error: ModelError,
    },
}

/// A structured model or provider failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelError {
    /// A human-readable error description.
    pub message: String,
    /// An optional provider or application error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Whether retrying the call may succeed.
    #[serde(default)]
    pub retryable: bool,
}

impl ModelError {
    /// Creates a non-retryable model error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
            retryable: false,
        }
    }

    /// Attaches a provider or application error code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Marks whether retrying the call may succeed.
    #[must_use]
    pub const fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.code {
            Some(code) => write!(formatter, "model error {code}: {}", self.message),
            None => write!(formatter, "model error: {}", self.message),
        }
    }
}

impl std::error::Error for ModelError {}
