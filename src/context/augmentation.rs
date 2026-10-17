//! Request-local context retrieval without authority over tools or history.

use crate::Msg;
use serde::{Deserialize, Serialize};
use std::{fmt, future::Future, pin::Pin};

/// Read-only input to an augmentation algorithm, before token-budget trimming.
pub struct ContextInput<'a> {
    /// Agent display name (not a tenant or authorization identifier).
    pub agent_name: &'a str,
    /// One-based model step, including steps after recovery.
    pub step: usize,
    /// Selected, summarized and offloaded model-visible messages.
    pub messages: &'a [Msg],
}

/// Failure retrieving supplementary context.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ContextAugmentationError {
    /// Human-readable failure description; do not include credentials.
    pub message: String,
}

impl fmt::Display for ContextAugmentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ContextAugmentationError {}

/// Asynchronous retrieval result. Empty output means no additional context.
pub type ContextAugmentationFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<String>, ContextAugmentationError>> + Send + 'a>>;

/// Retrieves supplementary text for each actual model step.
///
/// Providers run in registration order and see the same unaugmented input.
/// Results are inserted as labeled user-role data after the system/summary
/// prefix, pinned for token accounting, and never appended to durable history.
/// They are visible to model-call hooks. Labels are not a prompt-injection defense.
///
/// Providers are runtime configuration, not serialized state. Reattach on restore.
/// Retrieval runs again after recovery; implement read-only, cancellation-safe
/// operations and bind tenant/session authorization in the provider itself.
/// Summary generation and compaction budget probes do not invoke providers.
pub trait ContextAugmenter: Send + Sync {
    /// Retrieve context without mutating agent state, permissions or tools.
    fn augment<'a>(&'a self, input: ContextInput<'a>) -> ContextAugmentationFuture<'a>;
}

#[cfg(test)]
mod tests;
