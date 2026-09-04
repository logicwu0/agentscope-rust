//! Agent interfaces and built-in implementations.

mod confirmation;
mod event;
mod hook;
mod interrupt;
mod react;
mod state;
mod store;

use std::{fmt, future::Future, pin::Pin};

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::{
    Msg,
    memory::MemoryError,
    model::{ChatStreamError, ModelError},
    tool::ToolError,
};

pub use confirmation::{PendingToolCalls, ToolConfirmation, ToolConfirmationDecision};
pub use event::AgentEvent;
pub use hook::{AgentHook, AgentHookError, AgentHookEvent, AgentHookFuture, AgentHookResult};
pub use interrupt::AgentInterruptHandle;
pub use react::ReActAgent;
pub use state::{AGENT_STATE_VERSION, AgentState};
pub use store::{
    InMemoryStateStore, StateKey, StateRecord, StateStore, StateStoreError, StateStoreFuture,
    StateStoreResult,
};

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

    /// Streams one reply as model, tool, and lifecycle events.
    fn stream(&self, message: Msg) -> AgentFuture<'_, AgentEventStream<'_>>;

    /// Stores an external message in conversation memory without replying.
    fn observe(&self, message: Msg) -> AgentFuture<'_, ()>;

    /// Captures a versioned snapshot of configured conversation state.
    fn snapshot(&self) -> AgentFuture<'_, AgentState>;

    /// Atomically restores a previously captured conversation state.
    fn restore(&self, state: AgentState) -> AgentFuture<'_, ()>;

    /// Resumes a reply paused for explicit tool-call confirmation.
    fn resume_tool_calls(
        &self,
        reply_id: String,
        confirmations: Vec<ToolConfirmation>,
    ) -> AgentFuture<'_, Msg>;

    /// Returns a handle that can interrupt in-flight replies on this agent.
    fn interrupt_handle(&self) -> AgentInterruptHandle;
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
    /// An operation that requires conversation memory had none configured.
    MemoryNotConfigured,
    /// Execution paused until explicit decisions are supplied for tool calls.
    ToolConfirmationRequired {
        /// The persisted checkpoint needed to resume safely.
        checkpoint: PendingToolCalls,
    },
    /// No tool-confirmation checkpoint exists for this session.
    NoPendingToolConfirmation,
    /// Confirmation input did not match the persisted checkpoint.
    InvalidToolConfirmation(String),
    /// Per-session agent state could not be loaded or saved.
    StateStore(StateStoreError),
    /// The state snapshot uses a format this crate cannot restore.
    UnsupportedStateVersion {
        /// The version found in the snapshot.
        found: u32,
        /// The only version supported by this crate.
        supported: u32,
    },
    /// The state snapshot belongs to a differently named agent.
    StateAgentMismatch {
        /// The target agent name.
        expected: String,
        /// The agent name stored in the snapshot.
        found: String,
    },
    /// A lifecycle hook failed.
    Hook(AgentHookError),
    /// The caller interrupted the in-flight agent operation.
    Interrupted,
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
            Self::MemoryNotConfigured => {
                formatter.write_str("agent operation requires configured conversation memory")
            }
            Self::ToolConfirmationRequired { checkpoint } => write!(
                formatter,
                "agent is waiting for confirmation of reply {}",
                checkpoint.reply_id()
            ),
            Self::NoPendingToolConfirmation => {
                formatter.write_str("agent has no pending tool confirmation")
            }
            Self::InvalidToolConfirmation(message) => {
                write!(formatter, "invalid tool confirmation: {message}")
            }
            Self::StateStore(error) => write!(formatter, "agent state persistence failed: {error}"),
            Self::UnsupportedStateVersion { found, supported } => write!(
                formatter,
                "unsupported agent state version {found}; expected {supported}"
            ),
            Self::StateAgentMismatch { expected, found } => write!(
                formatter,
                "agent state belongs to {found:?}, but target agent is {expected:?}"
            ),
            Self::Hook(error) => write!(formatter, "agent lifecycle hook failed: {error}"),
            Self::Interrupted => formatter.write_str("agent operation was interrupted"),
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
            Self::StateStore(error) => Some(error),
            Self::Hook(error) => Some(error),
            Self::EmptyName
            | Self::ZeroMaxSteps
            | Self::MemoryNotConfigured
            | Self::ToolConfirmationRequired { .. }
            | Self::NoPendingToolConfirmation
            | Self::InvalidToolConfirmation(_)
            | Self::UnsupportedStateVersion { .. }
            | Self::StateAgentMismatch { .. }
            | Self::Interrupted
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

impl From<StateStoreError> for AgentError {
    fn from(error: StateStoreError) -> Self {
        Self::StateStore(error)
    }
}

impl From<AgentHookError> for AgentError {
    fn from(error: AgentHookError) -> Self {
        Self::Hook(error)
    }
}

impl From<ChatStreamError> for AgentError {
    fn from(error: ChatStreamError) -> Self {
        match error {
            ChatStreamError::Model(error) => Self::Model(error),
            error => Self::InvalidModelResponse(error.to_string()),
        }
    }
}

#[cfg(test)]
mod tests;
