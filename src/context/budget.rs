//! Per-request context-window budgeting (not cumulative billing limits).

use std::{fmt, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{ChatRequest, ContentBlock, Role, ToolResultContent, ToolResultOutput};

use super::{ContextPolicy, RecentTurns};

/// Whether a counter estimates or exactly tokenizes its supported model input.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenCountAccuracy {
    /// An approximation, not a guarantee about provider token usage.
    Estimated,
    /// Exact for the model/encoding supported by the counter implementation.
    Exact,
}

/// Input token count and its declared accuracy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenCount {
    /// Number of input tokens; excludes reserved output tokens.
    pub tokens: u64,
    /// Whether the counter used a heuristic or model-specific tokenizer.
    pub accuracy: TokenCountAccuracy,
}

/// Counts the entire request input, including system messages and tool schemas.
///
/// Implementations must be synchronous, deterministic, and perform no I/O.
/// An exact counter must account for the target model's actual wire formatting,
/// templates, schemas, and supported modalities, not just tokenize text blocks.
/// Return an error for inputs that cannot be counted reliably enough to use.
pub trait TokenCounter: Send + Sync {
    /// Counts input tokens, excluding generation output.
    ///
    /// # Errors
    /// Returns an error when the request cannot be counted.
    fn count(&self, request: &ChatRequest) -> Result<TokenCount, TokenBudgetError>;
}

/// An offline heuristic: ceil(serialized input UTF-8 bytes / 3) + 16.
///
/// Includes roles, names, content, tools, structured-output schema and extra
/// provider options. Ignores message IDs/timestamps/usage and generation controls.
/// This is neither a tokenizer nor a guaranteed upper bound; leave headroom or
/// install a model-specific counter. Binary/image/audio/video content is rejected
/// instead of estimating its cost from a URL or base64 length.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeuristicTokenCounter;

impl TokenCounter for HeuristicTokenCounter {
    fn count(&self, request: &ChatRequest) -> Result<TokenCount, TokenBudgetError> {
        for message in &request.messages {
            for block in &message.content {
                let unsupported = match block {
                    ContentBlock::Data(_) => true,
                    ContentBlock::ToolResult(result) => match result.output() {
                        ToolResultOutput::Blocks(blocks) => blocks
                            .iter()
                            .any(|block| matches!(block, ToolResultContent::Data(_))),
                        ToolResultOutput::Text(_) => false,
                    },
                    _ => false,
                };
                if unsupported {
                    return Err(TokenBudgetError::UnsupportedContent);
                }
            }
        }
        let messages = request
            .messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role, "name": message.name, "content": message.content,
                })
            })
            .collect::<Vec<_>>();
        let input = json!({
            "messages": messages, "tools": request.tools,
            "structured_output_schema": request.structured_output_schema,
            "extra": request.options.extra,
        });
        let bytes = u64::try_from(input.to_string().len())
            .map_err(|_| TokenBudgetError::Counter("input size overflow".into()))?;
        Ok(TokenCount {
            tokens: bytes.div_ceil(3).saturating_add(16),
            accuracy: TokenCountAccuracy::Estimated,
        })
    }
}

/// Limits each model request to a context window with reserved output space.
///
/// Applied after the agent's context policy. Complete old user turns are removed
/// as needed; system messages and the current turn are never discarded. Tool
/// dependencies may force additional turns to remain. Stored history is untouched.
/// This configuration and the counter are not part of persisted agent state.
#[derive(Clone)]
pub struct TokenBudget {
    context_window: u64,
    reserved_output: u32,
    counter: Arc<dyn TokenCounter>,
}

impl TokenBudget {
    /// Creates a budget using the approximate offline counter.
    ///
    /// # Errors
    /// Rejects zero output reservation or a window no larger than the reservation.
    pub fn new(context_window: u64, reserved_output: u32) -> Result<Self, TokenBudgetError> {
        if reserved_output == 0 || context_window <= u64::from(reserved_output) {
            return Err(TokenBudgetError::InvalidConfiguration);
        }
        Ok(Self {
            context_window,
            reserved_output,
            counter: Arc::new(HeuristicTokenCounter),
        })
    }

    /// Replaces the input counter with an application/model-specific implementation.
    #[must_use]
    pub fn with_counter<C: TokenCounter + 'static>(mut self, counter: C) -> Self {
        self.counter = Arc::new(counter);
        self
    }

    /// Attaches a shared counter.
    #[must_use]
    pub fn with_shared_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.counter = counter;
        self
    }

    /// Returns the configured total context window.
    #[must_use]
    pub const fn context_window(&self) -> u64 {
        self.context_window
    }

    /// Returns reserved output tokens (also the default `max_tokens`).
    #[must_use]
    pub const fn reserved_output(&self) -> u32 {
        self.reserved_output
    }

    /// Returns the input allowance, excluding reserved output space.
    #[must_use]
    pub fn input_limit(&self) -> u64 {
        self.context_window - u64::from(self.reserved_output)
    }

    /// Selects a fitting request without changing the source conversation.
    ///
    /// Sets `max_tokens` to the reservation if absent; preserves smaller explicit
    /// values, and rejects zero/larger values. Provider-specific output controls
    /// must not override this cap. Input is counted again after each whole-turn
    /// cut. On failure no model request should be sent.
    ///
    /// # Errors
    /// Returns a counter error, invalid output limit, or [`TokenBudgetError::Exceeded`]
    /// if the protected content alone is too large. Estimated counts do not
    /// guarantee that the provider will accept the resulting context length.
    pub fn apply(&self, mut request: ChatRequest) -> Result<ChatRequest, TokenBudgetError> {
        let output = request.options.max_tokens.unwrap_or(self.reserved_output);
        if output == 0 || output > self.reserved_output {
            return Err(TokenBudgetError::InvalidOutputLimit {
                requested: output,
                reserved: self.reserved_output,
            });
        }
        request.options.max_tokens = Some(output);
        let mut count = self.counter.count(&request)?;
        if count.tokens <= self.input_limit() {
            return Ok(request);
        }
        let history = request.messages.clone();
        let turns = history
            .iter()
            .filter(|message| message.role == Role::User)
            .count();
        for keep in (1..turns).rev() {
            let selected = RecentTurns::new(keep)
                .map_err(|_| TokenBudgetError::InvalidConfiguration)?
                .select_messages(&history);
            if selected == request.messages {
                continue;
            }
            request.messages = selected;
            count = self.counter.count(&request)?;
            if count.tokens <= self.input_limit() {
                return Ok(request);
            }
        }
        Err(TokenBudgetError::Exceeded {
            input: count,
            input_limit: self.input_limit(),
            reserved_output: self.reserved_output,
        })
    }
}

impl fmt::Debug for TokenBudget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TokenBudget")
            .field("context_window", &self.context_window)
            .field("reserved_output", &self.reserved_output)
            .finish_non_exhaustive()
    }
}

/// Configuration, counting, or context-window overflow failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum TokenBudgetError {
    /// Reservation must be positive and smaller than the context window.
    InvalidConfiguration,
    /// Explicit output limit must be positive and no larger than the reservation.
    InvalidOutputLimit {
        /// Configured generation limit.
        requested: u32,
        /// Available output reservation.
        reserved: u32,
    },
    /// The default counter cannot price multimodal input.
    UnsupportedContent,
    /// An application counter failed; do not include secrets or message content.
    Counter(String),
    /// Even the protected history exceeds the input allowance.
    Exceeded {
        /// Count of the smallest retained request and its accuracy.
        input: TokenCount,
        /// Maximum allowed input tokens.
        input_limit: u64,
        /// Output space reserved within the window.
        reserved_output: u32,
    },
}

impl fmt::Display for TokenBudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration => {
                formatter.write_str("context window must exceed a positive output reservation")
            }
            Self::InvalidOutputLimit {
                requested,
                reserved,
            } => write!(
                formatter,
                "output limit {requested} must be positive and no larger than reserved output {reserved}"
            ),
            Self::UnsupportedContent => {
                formatter.write_str("multimodal input requires a model-specific token counter")
            }
            Self::Counter(message) => write!(formatter, "token counter failed: {message}"),
            Self::Exceeded {
                input,
                input_limit,
                reserved_output,
            } => write!(
                formatter,
                "protected context needs {} input tokens ({:?}), exceeding {input_limit}; {reserved_output} output tokens reserved",
                input.tokens, input.accuracy
            ),
        }
    }
}

impl std::error::Error for TokenBudgetError {}

#[cfg(test)]
mod tests;
