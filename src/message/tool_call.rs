//! Streaming-aware tool call content blocks.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;

use super::generate_timestamp;

/// The behavior applied when a permission rule matches a tool call.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionBehavior {
    /// Permit the tool call without prompting.
    Allow,
    /// Reject the tool call.
    Deny,
    /// Ask the user before executing the tool call.
    Ask,
}

/// A permission suggestion associated with a tool call.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PermissionRule {
    /// The tool this rule applies to.
    pub tool_name: String,
    /// An optional tool-specific match expression.
    pub rule_content: Option<String>,
    /// The behavior to apply when the rule matches.
    pub behavior: PermissionBehavior,
    /// The origin of this rule.
    pub source: String,
}

/// The lifecycle state of a tool call.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallState {
    /// The call has not yet passed through the permission system.
    #[default]
    Pending,
    /// The call is waiting for user confirmation.
    Asking,
    /// The call is permitted and waiting for execution.
    Allowed,
    /// The call was submitted for external execution.
    Submitted,
    /// The call lifecycle has ended.
    Finished,
}

impl fmt::Display for ToolCallState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Pending => "pending",
            Self::Asking => "asking",
            Self::Allowed => "allowed",
            Self::Submitted => "submitted",
            Self::Finished => "finished",
        })
    }
}

/// An error returned when constructing or validating a tool call.
#[derive(Debug)]
pub enum ToolCallError {
    /// The provider supplied an empty tool call identifier.
    EmptyId,
    /// The tool name was empty or contained only whitespace.
    EmptyName,
    /// A completed tool call contained malformed JSON input.
    InvalidInput(serde_json::Error),
}

impl fmt::Display for ToolCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => formatter.write_str("tool call id cannot be empty"),
            Self::EmptyName => formatter.write_str("tool name cannot be empty"),
            Self::InvalidInput(error) => write!(formatter, "invalid tool input JSON: {error}"),
        }
    }
}

impl std::error::Error for ToolCallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInput(error) => Some(error),
            Self::EmptyId | Self::EmptyName => None,
        }
    }
}

/// A model request to invoke a tool.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ToolCallBlock {
    /// The provider-assigned tool call identifier.
    id: String,
    /// The tool to invoke.
    name: String,
    /// Raw JSON input, which may be incomplete while streaming.
    input: String,
    /// The tool call's current lifecycle state.
    pub state: ToolCallState,
    /// Permission rules suggested while requesting confirmation.
    pub suggested_rules: Vec<PermissionRule>,
    /// The local creation time in ISO 8601 format.
    pub created_at: String,
    /// The completion time, when available.
    pub finished_at: Option<String>,
}

impl ToolCallBlock {
    /// Creates a tool call whose input will arrive incrementally.
    ///
    /// # Errors
    ///
    /// Returns [`ToolCallError::EmptyId`] or [`ToolCallError::EmptyName`] when
    /// the corresponding value is blank.
    pub fn streaming(
        id: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Self, ToolCallError> {
        let id = id.into();
        let name = name.into();
        validate_tool_call_identity(&id, &name)?;

        Ok(Self {
            id,
            name,
            input: String::new(),
            state: ToolCallState::Pending,
            suggested_rules: Vec::new(),
            created_at: generate_timestamp(),
            finished_at: None,
        })
    }

    /// Creates a tool call with complete, validated JSON input.
    ///
    /// # Errors
    ///
    /// Returns [`ToolCallError::EmptyId`] or [`ToolCallError::EmptyName`] when
    /// the corresponding value is blank, or [`ToolCallError::InvalidInput`]
    /// when `input` is not valid JSON.
    pub fn complete(
        id: impl Into<String>,
        name: impl Into<String>,
        input: impl Into<String>,
    ) -> Result<Self, ToolCallError> {
        let mut block = Self::streaming(id, name)?;
        block.input = input.into();
        block.validate_input()?;
        Ok(block)
    }

    /// Returns the tool name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the provider-assigned tool call identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the raw JSON input accumulated so far.
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Appends a raw input fragment received from a streaming model response.
    pub fn append_input(&mut self, delta: &str) {
        self.input.push_str(delta);
    }

    /// Parses the current input as JSON.
    ///
    /// # Errors
    ///
    /// Returns [`ToolCallError::InvalidInput`] while the input is incomplete
    /// or malformed.
    pub fn parsed_input(&self) -> Result<Value, ToolCallError> {
        serde_json::from_str(&self.input).map_err(ToolCallError::InvalidInput)
    }

    /// Replaces the suggested permission rules.
    #[must_use]
    pub fn with_suggested_rules(mut self, rules: Vec<PermissionRule>) -> Self {
        self.suggested_rules = rules;
        self
    }

    fn validate_input(&self) -> Result<(), ToolCallError> {
        self.parsed_input().map(|_| ())
    }
}

impl<'de> Deserialize<'de> for ToolCallBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireBlock {
            id: String,
            name: String,
            input: String,
            #[serde(default)]
            state: ToolCallState,
            #[serde(default)]
            suggested_rules: Vec<PermissionRule>,
            #[serde(default = "generate_timestamp")]
            created_at: String,
            #[serde(default)]
            finished_at: Option<String>,
        }

        let block = WireBlock::deserialize(deserializer)?;
        validate_tool_call_identity(&block.id, &block.name).map_err(de::Error::custom)?;
        Ok(Self {
            id: block.id,
            name: block.name,
            input: block.input,
            state: block.state,
            suggested_rules: block.suggested_rules,
            created_at: block.created_at,
            finished_at: block.finished_at,
        })
    }
}

fn validate_tool_call_identity(id: &str, name: &str) -> Result<(), ToolCallError> {
    if id.trim().is_empty() {
        Err(ToolCallError::EmptyId)
    } else if name.trim().is_empty() {
        Err(ToolCallError::EmptyName)
    } else {
        Ok(())
    }
}
