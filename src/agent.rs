//! Agent interfaces and built-in implementations.

mod event;
mod react;

use std::{fmt, future::Future, pin::Pin};

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::{Msg, memory::MemoryError, model::ModelError, tool::ToolError};

pub use event::AgentEvent;
pub use react::ReActAgent;

/// Result returned by an agent operation.
pub type AgentResult<T> = Result<T, AgentError>;

/// A boxed asynchronous agent operation.
pub type AgentFuture<'a, T> = Pin<Box<dyn Future<Output = AgentResult<T>> + Send + 'a>>;

/// A boxed asynchronous stream of agent lifecycle events.
pub type AgentEventStream<'a> = Pin<Box<dyn Stream<Item = AgentResult<AgentEvent>> + Send + 'a>>;

/// An object-safe asynchronous agent.
pub trait Agent: Send + Sync {
    /// Returns the agent's display name.
    fn name(&self) -> &str;

    /// Produces one reply to an input message.
    fn reply(&self, message: Msg) -> AgentFuture<'_, Msg>;
}

/// A configuration or runtime agent failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum AgentError {
    /// The configured agent name was empty.
    EmptyName,
    /// The configured model/tool iteration limit was zero.
    ZeroMaxSteps,
    /// The model failed.
    Model(ModelError),
    /// Tool execution infrastructure failed.
    Tool(ToolError),
    /// Conversation memory failed.
    Memory(MemoryError),
    /// The model produced a response that cannot drive the agent loop.
    InvalidModelResponse(String),
    /// The model continued requesting tools after the configured limit.
    MaxStepsExceeded {
        /// The configured maximum number of model calls.
        max_steps: usize,
    },
}

impl fmt::Display for AgentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => formatter.write_str("agent name cannot be empty"),
            Self::ZeroMaxSteps => formatter.write_str("agent max_steps must be greater than zero"),
            Self::Model(error) => write!(formatter, "agent model failed: {error}"),
            Self::Tool(error) => write!(formatter, "agent tool execution failed: {error}"),
            Self::Memory(error) => write!(formatter, "agent memory failed: {error}"),
            Self::InvalidModelResponse(message) => {
                write!(formatter, "invalid model response: {message}")
            }
            Self::MaxStepsExceeded { max_steps } => {
                write!(
                    formatter,
                    "agent exceeded its limit of {max_steps} model calls"
                )
            }
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Tool(error) => Some(error),
            Self::Memory(error) => Some(error),
            Self::EmptyName
            | Self::ZeroMaxSteps
            | Self::InvalidModelResponse(_)
            | Self::MaxStepsExceeded { .. } => None,
        }
    }
}

impl From<ModelError> for AgentError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<ToolError> for AgentError {
    fn from(error: ToolError) -> Self {
        Self::Tool(error)
    }
}

impl From<MemoryError> for AgentError {
    fn from(error: MemoryError) -> Self {
        Self::Memory(error)
    }
}

#[cfg(test)]
mod tests;
