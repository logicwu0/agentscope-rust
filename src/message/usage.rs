//! Token usage reported by model calls.

use std::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};

/// Token usage associated with a message or model call.
///
/// Reasoning and cached input tokens are optional provider details and are
/// normally already included in `output_tokens` and `input_tokens`,
/// respectively. Consequently, [`Self::total_tokens`] only adds the two core
/// counters and does not double-count the details.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Usage {
    /// The number of input tokens consumed.
    pub input_tokens: u64,
    /// The number of output tokens generated.
    pub output_tokens: u64,
    /// Reasoning tokens included in `output_tokens`, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// Cached tokens included in `input_tokens`, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
}

impl Usage {
    /// Creates usage containing the provider-neutral core counters.
    #[must_use]
    pub const fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            reasoning_tokens: None,
            cached_input_tokens: None,
        }
    }

    /// Adds a provider-reported reasoning-token detail.
    #[must_use]
    pub const fn with_reasoning_tokens(mut self, reasoning_tokens: u64) -> Self {
        self.reasoning_tokens = Some(reasoning_tokens);
        self
    }

    /// Adds a provider-reported cached-input-token detail.
    #[must_use]
    pub const fn with_cached_input_tokens(mut self, cached_input_tokens: u64) -> Self {
        self.cached_input_tokens = Some(cached_input_tokens);
        self
    }

    /// Returns the sum of input and output tokens.
    #[must_use]
    pub const fn total_tokens(self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.reasoning_tokens = add_optional(self.reasoning_tokens, other.reasoning_tokens);
        self.cached_input_tokens =
            add_optional(self.cached_input_tokens, other.cached_input_tokens);
    }
}

impl Add for Usage {
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

fn add_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (None, None) => None,
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
    }
}
