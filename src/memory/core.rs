//! Object-safe asynchronous conversation memory interface.

use std::{fmt, future::Future, pin::Pin};

use serde::{Deserialize, Serialize};

use crate::Msg;

/// Result returned by a memory operation.
pub type MemoryResult<T> = Result<T, MemoryError>;

/// A boxed asynchronous memory operation.
pub type MemoryFuture<'a, T> = Pin<Box<dyn Future<Output = MemoryResult<T>> + Send + 'a>>;

/// Conversation history storage used by agents.
pub trait Memory: Send + Sync {
    /// Returns a snapshot of all stored messages in conversation order.
    fn messages(&self) -> MemoryFuture<'_, Vec<Msg>>;

    /// Appends messages in the supplied order.
    fn append(&self, messages: Vec<Msg>) -> MemoryFuture<'_, ()>;

    /// Atomically replaces all stored messages with the supplied history.
    ///
    /// On failure, the previously stored history must remain unchanged.
    fn replace(&self, messages: Vec<Msg>) -> MemoryFuture<'_, ()>;

    /// Removes all stored messages.
    fn clear(&self) -> MemoryFuture<'_, ()>;
}

/// A structured memory backend failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MemoryError {
    /// Human-readable failure description.
    pub message: String,
    /// Stable backend or application error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Whether retrying the operation may succeed.
    #[serde(default)]
    pub retryable: bool,
}

impl MemoryError {
    /// Creates a non-retryable memory error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
            retryable: false,
        }
    }

    /// Attaches a stable error code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Marks whether retrying the operation may succeed.
    #[must_use]
    pub const fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }
}

impl fmt::Display for MemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.code {
            Some(code) => write!(formatter, "memory error {code}: {}", self.message),
            None => write!(formatter, "memory error: {}", self.message),
        }
    }
}

impl std::error::Error for MemoryError {}
