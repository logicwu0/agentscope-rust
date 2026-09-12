//! Model-input history selection, independent of durable conversation storage.

use std::{collections::BTreeMap, fmt, num::NonZeroUsize};

use crate::{ContentBlock, Msg, Role};

mod budget;
pub use budget::{
    HeuristicTokenCounter, TokenBudget, TokenBudgetError, TokenCount, TokenCountAccuracy,
    TokenCounter,
};

/// Selects history for each model call without modifying stored messages.
///
/// The configured agent system prompt is prepended separately. Implementations
/// must preserve message order, the active turn, and tool-call/result pairs.
/// Policies are synchronous, should be inexpensive, and must not perform I/O.
/// This interface selects history; it does not perform LLM-based summarization.
pub trait ContextPolicy: Send + Sync {
    /// Returns the model-visible history, leaving the input unchanged.
    fn select_messages(&self, history: &[Msg]) -> Vec<Msg>;
}

/// Includes the full history (the default agent behavior).
#[derive(Clone, Copy, Debug, Default)]
pub struct FullContext;

impl ContextPolicy for FullContext {
    fn select_messages(&self, history: &[Msg]) -> Vec<Msg> {
        history.to_vec()
    }
}

/// Keeps the most recent user turns, including the current turn.
///
/// A turn starts at a user message and includes all following messages up to
/// the next user message. System messages are always retained. If there are no
/// more than the requested number of user messages, all history is retained.
/// A cut crossing a tool-call/result pair is moved backwards to retain the
/// complete exchange. Consequently this is not a strict message/token limit.
/// Existing malformed history is not repaired or validated by this policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecentTurns {
    max_turns: NonZeroUsize,
}

impl RecentTurns {
    /// Creates a policy retaining at least one user turn.
    ///
    /// # Errors
    /// Returns [`ZeroContextTurns`] if `max_turns` is zero.
    pub fn new(max_turns: usize) -> Result<Self, ZeroContextTurns> {
        Ok(Self {
            max_turns: NonZeroUsize::new(max_turns).ok_or(ZeroContextTurns)?,
        })
    }

    /// Returns the requested number of recent user turns.
    #[must_use]
    pub const fn max_turns(&self) -> usize {
        self.max_turns.get()
    }
}

impl ContextPolicy for RecentTurns {
    fn select_messages(&self, history: &[Msg]) -> Vec<Msg> {
        let mut turns = history
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, message)| message.role == Role::User);
        let Some((mut start, _)) = turns.nth(self.max_turns.get() - 1) else {
            return history.to_vec();
        };
        if turns.next().is_none() {
            return history.to_vec();
        }

        // Record backwards dependencies, then expand the suffix in one reverse
        // pass. This also handles results observed after a new user message.
        let mut calls = BTreeMap::new();
        let mut dependencies = Vec::with_capacity(history.len());
        let mut turn_start = 0;
        for (index, message) in history.iter().enumerate() {
            if message.role == Role::User {
                turn_start = index;
            }
            let mut dependency = index;
            for block in &message.content {
                match block {
                    ContentBlock::ToolCall(call) => {
                        calls.insert((call.id(), call.name()), turn_start);
                    }
                    ContentBlock::ToolResult(result) => {
                        if let Some(&call_index) = calls.get(&(result.id(), result.name())) {
                            dependency = dependency.min(call_index);
                        }
                    }
                    _ => {}
                }
            }
            dependencies.push(dependency);
        }
        for (index, &dependency) in dependencies.iter().enumerate().rev() {
            if index >= start || (history[index].role == Role::System && dependency < index) {
                start = start.min(dependency);
            }
        }
        history
            .iter()
            .enumerate()
            .filter(|(index, message)| *index >= start || message.role == Role::System)
            .map(|(_, message)| message.clone())
            .collect()
    }
}

/// A recent-turn policy cannot discard the current user turn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZeroContextTurns;

impl fmt::Display for ZeroContextTurns {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("context must retain at least one user turn")
    }
}

impl std::error::Error for ZeroContextTurns {}

#[cfg(test)]
mod tests;
