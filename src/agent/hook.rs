//! Read-only asynchronous agent lifecycle hooks.

use std::{fmt, future::Future, pin::Pin};

use serde::{Deserialize, Serialize};

use crate::{ChatRequest, ChatResponse, Msg, ToolCallBlock, ToolResultBlock};

/// Result returned by an agent hook.
pub type AgentHookResult<T> = Result<T, AgentHookError>;

/// A boxed asynchronous hook operation.
pub type AgentHookFuture<'a> = Pin<Box<dyn Future<Output = AgentHookResult<()>> + Send + 'a>>;

/// A read-only lifecycle notification emitted by an agent.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentHookEvent {
    /// The agent accepted a new input message.
    BeforeReply {
        /// The input that will start the reply.
        message: Msg,
    },
    /// The agent is about to call its model.
    BeforeModelCall {
        /// The one-based model-call number.
        step: usize,
        /// The complete provider-neutral model request.
        request: ChatRequest,
    },
    /// The model returned a complete response.
    AfterModelCall {
        /// The one-based model-call number.
        step: usize,
        /// The complete provider-neutral model response.
        response: ChatResponse,
    },
    /// The agent is about to execute one tool call.
    BeforeToolCall {
        /// The model-call number that requested the tool.
        step: usize,
        /// The complete tool call.
        call: ToolCallBlock,
    },
    /// One tool call produced a terminal result.
    AfterToolCall {
        /// The model-call number that requested the tool.
        step: usize,
        /// The successful or failed tool result.
        result: ToolResultBlock,
    },
    /// The agent produced its final reply and updated configured memory.
    AfterReply {
        /// The number of model calls used by the reply.
        steps: usize,
        /// The complete assistant message.
        message: Msg,
    },
}

/// A structured failure returned by an [`AgentHook`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AgentHookError {
    /// Human-readable failure description.
    pub message: String,
    /// Stable application-specific error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl AgentHookError {
    /// Creates a hook error without a stable code.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
        }
    }

    /// Attaches a stable application-specific error code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
}

impl fmt::Display for AgentHookError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.code {
            Some(code) => write!(formatter, "agent hook error {code}: {}", self.message),
            None => write!(formatter, "agent hook error: {}", self.message),
        }
    }
}

impl std::error::Error for AgentHookError {}

/// An object-safe, read-only asynchronous agent lifecycle hook.
///
/// Hooks execute in registration order. Returning an error stops the current
/// agent operation; hooks cannot mutate lifecycle data.
pub trait AgentHook: Send + Sync {
    /// Observes one lifecycle event.
    fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a>;
}

#[cfg(test)]
mod tests;
