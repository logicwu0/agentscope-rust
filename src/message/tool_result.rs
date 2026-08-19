//! Streaming and multimodal tool result content blocks.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};

use super::{DataBlock, Metadata, TextBlock, generate_timestamp};

/// The execution state of a tool result.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolResultState {
    /// The tool is still executing or streaming output.
    #[default]
    Running,
    /// The tool completed successfully.
    Success,
    /// The tool failed.
    Error,
    /// The tool was interrupted before completion.
    Interrupted,
    /// Execution was denied by the permission system or user.
    Denied,
}

impl ToolResultState {
    /// Returns whether this state ends the tool execution lifecycle.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }
}

impl fmt::Display for ToolResultState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Running => "running",
            Self::Success => "success",
            Self::Error => "error",
            Self::Interrupted => "interrupted",
            Self::Denied => "denied",
        })
    }
}

/// A content block allowed inside structured tool output.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContent {
    /// Plain-text tool output.
    Text(TextBlock),
    /// Binary or multimodal tool output.
    Data(DataBlock),
}

impl From<TextBlock> for ToolResultContent {
    fn from(block: TextBlock) -> Self {
        Self::Text(block)
    }
}

impl From<DataBlock> for ToolResultContent {
    fn from(block: DataBlock) -> Self {
        Self::Data(block)
    }
}

/// The output carried by a tool result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolResultOutput {
    /// A raw text result.
    Text(String),
    /// Structured text and multimodal result blocks.
    Blocks(Vec<ToolResultContent>),
}

impl From<String> for ToolResultOutput {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for ToolResultOutput {
    fn from(text: &str) -> Self {
        Self::Text(text.to_owned())
    }
}

impl From<Vec<ToolResultContent>> for ToolResultOutput {
    fn from(blocks: Vec<ToolResultContent>) -> Self {
        Self::Blocks(blocks)
    }
}

/// An error returned when constructing or finishing a tool result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolResultError {
    /// The tool call identifier was empty.
    EmptyId,
    /// The tool name was empty or contained only whitespace.
    EmptyName,
    /// `running` was supplied where a terminal state was required.
    NonTerminalState,
}

impl fmt::Display for ToolResultError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => formatter.write_str("tool result id cannot be empty"),
            Self::EmptyName => formatter.write_str("tool name cannot be empty"),
            Self::NonTerminalState => {
                formatter.write_str("running is not a terminal tool result state")
            }
        }
    }
}

impl std::error::Error for ToolResultError {}

/// The output produced by a tool invocation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ToolResultBlock {
    /// The identifier shared with the corresponding tool call.
    id: String,
    /// The invoked tool's name.
    name: String,
    /// The raw or structured tool output.
    output: ToolResultOutput,
    /// The current execution state.
    state: ToolResultState,
    /// Arbitrary result metadata.
    #[serde(default)]
    pub metadata: Metadata,
    /// The local creation time in ISO 8601 format.
    pub created_at: String,
    /// The completion time, when available.
    pub finished_at: Option<String>,
}

impl ToolResultBlock {
    /// Creates an empty result for a running tool call.
    ///
    /// # Errors
    ///
    /// Returns [`ToolResultError::EmptyId`] or [`ToolResultError::EmptyName`]
    /// when the corresponding value is blank.
    pub fn running(
        id: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Self, ToolResultError> {
        let id = id.into();
        let name = name.into();
        validate_tool_result_identity(&id, &name)?;
        Ok(Self {
            id,
            name,
            output: ToolResultOutput::Blocks(Vec::new()),
            state: ToolResultState::Running,
            metadata: Metadata::new(),
            created_at: generate_timestamp(),
            finished_at: None,
        })
    }

    /// Creates a finished result with a terminal state.
    ///
    /// # Errors
    ///
    /// Returns an identity error when `id` or `name` is blank, or
    /// [`ToolResultError::NonTerminalState`] when `state` is `running`.
    pub fn finished(
        id: impl Into<String>,
        name: impl Into<String>,
        output: impl Into<ToolResultOutput>,
        state: ToolResultState,
    ) -> Result<Self, ToolResultError> {
        let mut block = Self::running(id, name)?;
        block.output = output.into();
        block.finish(state)?;
        Ok(block)
    }

    /// Creates a successfully finished result.
    ///
    /// # Errors
    ///
    /// Returns [`ToolResultError::EmptyId`] or [`ToolResultError::EmptyName`]
    /// when the corresponding value is blank.
    pub fn success(
        id: impl Into<String>,
        name: impl Into<String>,
        output: impl Into<ToolResultOutput>,
    ) -> Result<Self, ToolResultError> {
        Self::finished(id, name, output, ToolResultState::Success)
    }

    /// Returns the corresponding tool call identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the tool name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the current output.
    #[must_use]
    pub const fn output(&self) -> &ToolResultOutput {
        &self.output
    }

    /// Returns the current execution state.
    #[must_use]
    pub const fn state(&self) -> ToolResultState {
        self.state
    }

    /// Appends a streaming text fragment, merging adjacent text blocks.
    pub fn append_text_delta(&mut self, delta: &str) {
        let blocks = self.output_blocks_mut();
        match blocks.last_mut() {
            Some(ToolResultContent::Text(block)) => block.text.push_str(delta),
            Some(ToolResultContent::Data(_)) | None => {
                blocks.push(ToolResultContent::Text(TextBlock::new(delta)));
            }
        }
    }

    /// Appends multimodal data to the structured output.
    pub fn append_data(&mut self, block: DataBlock) {
        self.output_blocks_mut()
            .push(ToolResultContent::Data(block));
    }

    /// Replaces the result metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Marks this result as finished with a terminal state.
    ///
    /// # Errors
    ///
    /// Returns [`ToolResultError::NonTerminalState`] when `state` is
    /// `running`.
    pub fn finish(&mut self, state: ToolResultState) -> Result<(), ToolResultError> {
        if !state.is_terminal() {
            return Err(ToolResultError::NonTerminalState);
        }
        self.state = state;
        self.finished_at = Some(generate_timestamp());
        Ok(())
    }

    fn output_blocks_mut(&mut self) -> &mut Vec<ToolResultContent> {
        if let ToolResultOutput::Text(_) = &self.output {
            let ToolResultOutput::Text(text) =
                std::mem::replace(&mut self.output, ToolResultOutput::Blocks(Vec::new()))
            else {
                unreachable!("tool output was checked as text")
            };
            if !text.is_empty() {
                self.output =
                    ToolResultOutput::Blocks(vec![ToolResultContent::Text(TextBlock::new(text))]);
            }
        }
        let ToolResultOutput::Blocks(blocks) = &mut self.output else {
            unreachable!("raw tool output was converted to blocks")
        };
        blocks
    }
}

impl<'de> Deserialize<'de> for ToolResultBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireBlock {
            id: String,
            name: String,
            output: ToolResultOutput,
            #[serde(default)]
            state: ToolResultState,
            #[serde(default)]
            metadata: Metadata,
            #[serde(default = "generate_timestamp")]
            created_at: String,
            #[serde(default)]
            finished_at: Option<String>,
        }

        let block = WireBlock::deserialize(deserializer)?;
        validate_tool_result_identity(&block.id, &block.name).map_err(de::Error::custom)?;
        Ok(Self {
            id: block.id,
            name: block.name,
            output: block.output,
            state: block.state,
            metadata: block.metadata,
            created_at: block.created_at,
            finished_at: block.finished_at,
        })
    }
}

fn validate_tool_result_identity(id: &str, name: &str) -> Result<(), ToolResultError> {
    if id.trim().is_empty() {
        Err(ToolResultError::EmptyId)
    } else if name.trim().is_empty() {
        Err(ToolResultError::EmptyName)
    } else {
        Ok(())
    }
}
