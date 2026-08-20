//! Messages exchanged between agents, models, tools, and users.

use serde::{Deserialize, Serialize};

use super::{ContentBlock, Metadata, Role, Usage, generate_id, generate_timestamp};

/// A message exchanged within an `AgentScope` application.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Msg {
    /// The sender's display name or identity.
    pub name: String,
    /// The message content.
    pub content: Vec<ContentBlock>,
    /// The sender's role.
    pub role: Role,
    /// The unique message identifier.
    pub id: String,
    /// Arbitrary application metadata.
    #[serde(default)]
    pub metadata: Metadata,
    /// Token usage associated with this message, when reported by a model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// The local creation time in ISO 8601 format.
    pub created_at: String,
}

impl Msg {
    /// Creates a message from typed content blocks.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        role: Role,
        content: impl IntoIterator<Item = ContentBlock>,
    ) -> Self {
        Self {
            name: name.into(),
            content: content.into_iter().collect(),
            role,
            id: generate_id(),
            metadata: Metadata::new(),
            usage: None,
            created_at: generate_timestamp(),
        }
    }

    /// Creates a user message with the default sender name `user`.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self::text("user", Role::User, text)
    }

    /// Creates an assistant message.
    #[must_use]
    pub fn assistant(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self::text(name, Role::Assistant, text)
    }

    /// Creates a system message with the default sender name `system`.
    #[must_use]
    pub fn system(text: impl Into<String>) -> Self {
        Self::text("system", Role::System, text)
    }

    /// Replaces this message's metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Attaches model token usage to this message.
    #[must_use]
    pub const fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Concatenates all text blocks using `separator`.
    #[must_use]
    pub fn text_content(&self, separator: &str) -> Option<String> {
        let text = self
            .content
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect::<Vec<_>>();

        (!text.is_empty()).then(|| text.join(separator))
    }

    fn text(name: impl Into<String>, role: Role, text: impl Into<String>) -> Self {
        Self::new(name, role, [ContentBlock::from(text.into())])
    }
}
