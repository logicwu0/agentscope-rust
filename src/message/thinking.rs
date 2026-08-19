//! Model reasoning content blocks.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{Metadata, generate_id, generate_timestamp};

/// An error returned when adding provider-specific thinking fields.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThinkingBlockError {
    /// An extension attempted to replace a standard thinking block field.
    ReservedField(String),
}

impl fmt::Display for ThinkingBlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReservedField(field) => {
                write!(formatter, "provider field `{field}` is reserved")
            }
        }
    }
}

impl std::error::Error for ThinkingBlockError {}

/// A model's reasoning content.
///
/// Provider-specific fields such as Anthropic's `signature` and
/// `redacted_thinking_data` are preserved in the flattened JSON object.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ThinkingBlock {
    /// The reasoning text. This can be empty for redacted thinking blocks.
    pub thinking: String,
    /// The unique block identifier.
    pub id: String,
    /// The local creation time in ISO 8601 format.
    pub created_at: String,
    /// The completion time for streamed content, when available.
    pub finished_at: Option<String>,
    /// Additional provider-specific JSON fields.
    #[serde(default, flatten)]
    extra: Metadata,
}

impl ThinkingBlock {
    /// Creates a thinking block.
    #[must_use]
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
            id: generate_id(),
            created_at: generate_timestamp(),
            finished_at: None,
            extra: Metadata::new(),
        }
    }

    /// Adds one provider-specific JSON field.
    ///
    /// # Errors
    ///
    /// Returns [`ThinkingBlockError::ReservedField`] if `key` is one of the
    /// standard serialized fields.
    pub fn with_extra_field(
        mut self,
        key: impl Into<String>,
        value: impl Into<Value>,
    ) -> Result<Self, ThinkingBlockError> {
        let key = key.into();
        validate_thinking_extra_key(&key)?;
        self.extra.insert(key, value.into());
        Ok(self)
    }

    /// Replaces the provider-specific JSON fields.
    ///
    /// # Errors
    ///
    /// Returns [`ThinkingBlockError::ReservedField`] if any key is one of the
    /// standard serialized fields.
    pub fn with_extra_fields(mut self, extra: Metadata) -> Result<Self, ThinkingBlockError> {
        for key in extra.keys() {
            validate_thinking_extra_key(key)?;
        }
        self.extra = extra;
        Ok(self)
    }

    /// Returns the provider-specific JSON fields.
    #[must_use]
    pub const fn extra_fields(&self) -> &Metadata {
        &self.extra
    }
}

fn validate_thinking_extra_key(key: &str) -> Result<(), ThinkingBlockError> {
    const RESERVED_FIELDS: [&str; 5] = ["type", "thinking", "id", "created_at", "finished_at"];
    if RESERVED_FIELDS.contains(&key) {
        Err(ThinkingBlockError::ReservedField(key.to_owned()))
    } else {
        Ok(())
    }
}
