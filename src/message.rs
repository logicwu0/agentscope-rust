//! Message types exchanged between agents, models, tools, and users.

use std::fmt;

use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

/// Arbitrary JSON metadata attached to a message.
pub type Metadata = Map<String, Value>;

/// The role of a message sender.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions supplied by the application.
    System,
    /// Input supplied by an end user.
    User,
    /// Output produced by an agent or model.
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        })
    }
}

/// A plain-text message content block.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TextBlock {
    /// The text contained in this block.
    pub text: String,
    /// The unique block identifier.
    pub id: String,
    /// The local creation time in ISO 8601 format.
    pub created_at: String,
    /// The completion time for streamed content, when available.
    pub finished_at: Option<String>,
}

impl TextBlock {
    /// Creates a completed text block.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        let created_at = generate_timestamp();
        Self {
            text: text.into(),
            id: generate_id(),
            finished_at: None,
            created_at,
        }
    }
}

/// A typed block within a message.
///
/// Additional variants will be introduced as multimodal and tool support is
/// implemented.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain-text content.
    Text(TextBlock),
}

impl ContentBlock {
    /// Returns the text when this is a text block.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(block) => Some(block.text.as_str()),
        }
    }
}

impl From<TextBlock> for ContentBlock {
    fn from(block: TextBlock) -> Self {
        Self::Text(block)
    }
}

impl From<String> for ContentBlock {
    fn from(text: String) -> Self {
        Self::Text(TextBlock::new(text))
    }
}

impl From<&str> for ContentBlock {
    fn from(text: &str) -> Self {
        Self::Text(TextBlock::new(text))
    }
}

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

fn generate_id() -> String {
    Uuid::new_v4().simple().to_string()
}

fn generate_timestamp() -> String {
    Local::now()
        .naive_local()
        .format("%Y-%m-%dT%H:%M:%S%.6f")
        .to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ContentBlock, Metadata, Msg, Role};

    #[test]
    fn roles_use_agentscope_wire_values() {
        assert_eq!(serde_json::to_value(Role::System).unwrap(), "system");
        assert_eq!(serde_json::to_value(Role::User).unwrap(), "user");
        assert_eq!(serde_json::to_value(Role::Assistant).unwrap(), "assistant");
    }

    #[test]
    fn user_message_has_generated_identity_and_text_block() {
        let message = Msg::user("hello");

        assert_eq!(message.name, "user");
        assert_eq!(message.role, Role::User);
        assert_eq!(message.id.len(), 32);
        assert!(!message.created_at.is_empty());
        assert_eq!(message.text_content("\n").as_deref(), Some("hello"));

        let value = serde_json::to_value(message).unwrap();
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][0]["text"], "hello");
    }

    #[test]
    fn json_round_trip_preserves_message_and_metadata() {
        let mut metadata = Metadata::new();
        metadata.insert("request_id".into(), json!("req-42"));
        metadata.insert("attempt".into(), json!(2));
        let original = Msg::assistant("Friday", "Done").with_metadata(metadata);

        let json = serde_json::to_string(&original).unwrap();
        let restored: Msg = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, original);
        assert_eq!(restored.metadata["request_id"], "req-42");
        assert_eq!(restored.metadata["attempt"], 2);
    }

    #[test]
    fn deserialization_accepts_missing_metadata() {
        let message: Msg = serde_json::from_value(json!({
            "name": "system",
            "content": [{
                "type": "text",
                "text": "Follow the policy.",
                "id": "block-1",
                "created_at": "2026-08-12T12:00:00.000000",
                "finished_at": "2026-08-12T12:00:00.000000"
            }],
            "role": "system",
            "id": "message-1",
            "created_at": "2026-08-12T12:00:00.000000"
        }))
        .unwrap();

        assert!(message.metadata.is_empty());
        assert!(matches!(message.content[0], ContentBlock::Text(_)));
    }

    #[test]
    fn empty_message_has_no_text_content() {
        let message = Msg::new("Friday", Role::Assistant, []);

        assert_eq!(message.text_content("\n"), None);
        assert_eq!(
            serde_json::to_value(message).unwrap()["content"],
            Value::Array(vec![])
        );
    }
}
