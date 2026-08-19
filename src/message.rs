//! Message types exchanged between agents, models, tools, and users.

use std::fmt;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use chrono::Local;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Map, Value};
use url::Url;
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
    /// Creates a text block.
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

/// An error returned when constructing or decoding a data block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DataBlockError {
    /// The media type was empty or contained only whitespace.
    EmptyMediaType,
    /// The provided Base64 data was malformed.
    InvalidBase64(base64::DecodeError),
    /// The provided URL was malformed or relative.
    InvalidUrl(url::ParseError),
}

impl fmt::Display for DataBlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMediaType => formatter.write_str("media type cannot be empty"),
            Self::InvalidBase64(error) => write!(formatter, "invalid Base64 data: {error}"),
            Self::InvalidUrl(error) => write!(formatter, "invalid URL: {error}"),
        }
    }
}

impl std::error::Error for DataBlockError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EmptyMediaType => None,
            Self::InvalidBase64(error) => Some(error),
            Self::InvalidUrl(error) => Some(error),
        }
    }
}

/// Inline Base64-encoded binary data.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Base64Source {
    /// The Base64-encoded payload.
    data: String,
    /// The payload's media type, such as `image/png`.
    media_type: String,
}

impl Base64Source {
    /// Creates a validated Base64 source.
    ///
    /// # Errors
    ///
    /// Returns [`DataBlockError::InvalidBase64`] when `data` is malformed,
    /// or [`DataBlockError::EmptyMediaType`] when `media_type` is blank.
    pub fn new(
        data: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Result<Self, DataBlockError> {
        let data = data.into();
        STANDARD
            .decode(&data)
            .map_err(DataBlockError::InvalidBase64)?;

        Ok(Self {
            data,
            media_type: validate_media_type(media_type.into())?,
        })
    }

    /// Returns the Base64-encoded payload.
    #[must_use]
    pub fn data(&self) -> &str {
        &self.data
    }

    /// Returns the payload's media type.
    #[must_use]
    pub fn media_type(&self) -> &str {
        &self.media_type
    }
}

impl<'de> Deserialize<'de> for Base64Source {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireSource {
            data: String,
            media_type: String,
        }

        let source = WireSource::deserialize(deserializer)?;
        Self::new(source.data, source.media_type).map_err(de::Error::custom)
    }
}

/// Binary data addressed by an absolute URL.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct UrlSource {
    /// The absolute data URL.
    url: Url,
    /// The payload's media type, such as `audio/mpeg`.
    media_type: String,
}

impl UrlSource {
    /// Creates a validated URL source.
    ///
    /// # Errors
    ///
    /// Returns [`DataBlockError::InvalidUrl`] when `url` is not an absolute
    /// URL, or [`DataBlockError::EmptyMediaType`] when `media_type` is blank.
    pub fn new(
        url: impl AsRef<str>,
        media_type: impl Into<String>,
    ) -> Result<Self, DataBlockError> {
        Ok(Self {
            url: Url::parse(url.as_ref()).map_err(DataBlockError::InvalidUrl)?,
            media_type: validate_media_type(media_type.into())?,
        })
    }

    /// Returns the absolute data URL.
    #[must_use]
    pub const fn url(&self) -> &Url {
        &self.url
    }

    /// Returns the payload's media type.
    #[must_use]
    pub fn media_type(&self) -> &str {
        &self.media_type
    }
}

impl<'de> Deserialize<'de> for UrlSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireSource {
            url: String,
            media_type: String,
        }

        let source = WireSource::deserialize(deserializer)?;
        Self::new(source.url, source.media_type).map_err(de::Error::custom)
    }
}

/// The source of a binary data block.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DataSource {
    /// Inline Base64-encoded data.
    Base64(Base64Source),
    /// Data available at an absolute URL.
    Url(UrlSource),
}

/// A binary or multimodal message content block.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DataBlock {
    /// The unique block identifier.
    pub id: String,
    /// The source of the block's binary data.
    pub source: DataSource,
    /// An optional file or display name.
    pub name: Option<String>,
    /// The local creation time in ISO 8601 format.
    pub created_at: String,
    /// The completion time for streamed content, when available.
    pub finished_at: Option<String>,
}

impl DataBlock {
    /// Creates a block containing inline Base64-encoded data.
    ///
    /// # Errors
    ///
    /// Returns [`DataBlockError::InvalidBase64`] when `data` is malformed,
    /// or [`DataBlockError::EmptyMediaType`] when `media_type` is blank.
    pub fn base64(
        data: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Result<Self, DataBlockError> {
        Ok(Self::new(DataSource::Base64(Base64Source::new(
            data, media_type,
        )?)))
    }

    /// Creates a block referring to data at an absolute URL.
    ///
    /// # Errors
    ///
    /// Returns [`DataBlockError::InvalidUrl`] when `url` is not an absolute
    /// URL, or [`DataBlockError::EmptyMediaType`] when `media_type` is blank.
    pub fn url(
        url: impl AsRef<str>,
        media_type: impl Into<String>,
    ) -> Result<Self, DataBlockError> {
        Ok(Self::new(DataSource::Url(UrlSource::new(url, media_type)?)))
    }

    /// Assigns a file or display name to this block.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    fn new(source: DataSource) -> Self {
        Self {
            id: generate_id(),
            source,
            name: None,
            created_at: generate_timestamp(),
            finished_at: None,
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
    /// Model reasoning content.
    Thinking(ThinkingBlock),
    /// A request to invoke a tool.
    ToolCall(ToolCallBlock),
    /// Output produced by a tool invocation.
    ToolResult(ToolResultBlock),
    /// Binary or multimodal data.
    Data(DataBlock),
}

impl ContentBlock {
    /// Returns the text when this is a text block.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(block) => Some(block.text.as_str()),
            Self::Thinking(_) | Self::ToolCall(_) | Self::ToolResult(_) | Self::Data(_) => None,
        }
    }
}

impl From<TextBlock> for ContentBlock {
    fn from(block: TextBlock) -> Self {
        Self::Text(block)
    }
}

impl From<ThinkingBlock> for ContentBlock {
    fn from(block: ThinkingBlock) -> Self {
        Self::Thinking(block)
    }
}

impl From<ToolCallBlock> for ContentBlock {
    fn from(block: ToolCallBlock) -> Self {
        Self::ToolCall(block)
    }
}

impl From<ToolResultBlock> for ContentBlock {
    fn from(block: ToolResultBlock) -> Self {
        Self::ToolResult(block)
    }
}

impl From<DataBlock> for ContentBlock {
    fn from(block: DataBlock) -> Self {
        Self::Data(block)
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

fn validate_media_type(media_type: String) -> Result<String, DataBlockError> {
    if media_type.trim().is_empty() {
        Err(DataBlockError::EmptyMediaType)
    } else {
        Ok(media_type)
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

fn validate_tool_call_identity(id: &str, name: &str) -> Result<(), ToolCallError> {
    if id.trim().is_empty() {
        Err(ToolCallError::EmptyId)
    } else if name.trim().is_empty() {
        Err(ToolCallError::EmptyName)
    } else {
        Ok(())
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

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{
        ContentBlock, DataBlock, DataBlockError, DataSource, Metadata, Msg, PermissionBehavior,
        PermissionRule, Role, ThinkingBlock, ThinkingBlockError, ToolCallBlock, ToolCallError,
        ToolCallState, ToolResultBlock, ToolResultContent, ToolResultError, ToolResultOutput,
        ToolResultState,
    };

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

    #[test]
    fn url_data_block_uses_agentscope_wire_format() {
        let block = DataBlock::url("https://example.com/image.png", "image/png")
            .unwrap()
            .with_name("diagram.png");
        let DataSource::Url(source) = &block.source else {
            panic!("expected URL source");
        };
        assert_eq!(source.url().as_str(), "https://example.com/image.png");
        assert_eq!(source.media_type(), "image/png");
        let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

        assert_eq!(value["type"], "data");
        assert_eq!(value["source"]["type"], "url");
        assert_eq!(value["source"]["url"], "https://example.com/image.png");
        assert_eq!(value["source"]["media_type"], "image/png");
        assert_eq!(value["name"], "diagram.png");
    }

    #[test]
    fn base64_data_block_uses_agentscope_wire_format() {
        let block = DataBlock::base64("aGVsbG8=", "text/plain").unwrap();
        let DataSource::Base64(source) = &block.source else {
            panic!("expected Base64 source");
        };
        assert_eq!(source.data(), "aGVsbG8=");
        assert_eq!(source.media_type(), "text/plain");
        let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

        assert_eq!(value["type"], "data");
        assert_eq!(value["source"]["type"], "base64");
        assert_eq!(value["source"]["data"], "aGVsbG8=");
        assert_eq!(value["source"]["media_type"], "text/plain");
    }

    #[test]
    fn mixed_message_round_trip_preserves_text_and_data() {
        let original = Msg::new(
            "user",
            Role::User,
            [
                ContentBlock::from("Describe this image"),
                ContentBlock::from(
                    DataBlock::url("https://example.com/cat.jpg", "image/jpeg").unwrap(),
                ),
            ],
        );

        let json = serde_json::to_string(&original).unwrap();
        let restored: Msg = serde_json::from_str(&json).unwrap();

        assert_eq!(restored, original);
        assert_eq!(
            restored.text_content("\n").as_deref(),
            Some("Describe this image")
        );
        assert!(matches!(restored.content[1], ContentBlock::Data(_)));
    }

    #[test]
    fn constructors_reject_invalid_data_sources() {
        assert!(matches!(
            DataBlock::url("relative/image.png", "image/png"),
            Err(DataBlockError::InvalidUrl(_))
        ));
        assert!(matches!(
            DataBlock::base64("not base64!", "image/png"),
            Err(DataBlockError::InvalidBase64(_))
        ));
        assert_eq!(
            DataBlock::base64("", "  ").unwrap_err(),
            DataBlockError::EmptyMediaType
        );
    }

    #[test]
    fn deserialization_rejects_invalid_data_sources() {
        let invalid_url = json!({
            "type": "data",
            "id": "block-1",
            "source": {
                "type": "url",
                "url": "relative/image.png",
                "media_type": "image/png"
            },
            "name": null,
            "created_at": "2026-08-12T12:00:00.000000",
            "finished_at": null
        });
        let invalid_base64 = json!({
            "type": "data",
            "id": "block-2",
            "source": {
                "type": "base64",
                "data": "not base64!",
                "media_type": "image/png"
            },
            "name": null,
            "created_at": "2026-08-12T12:00:00.000000",
            "finished_at": null
        });

        assert!(serde_json::from_value::<ContentBlock>(invalid_url).is_err());
        assert!(serde_json::from_value::<ContentBlock>(invalid_base64).is_err());
    }

    #[test]
    fn thinking_block_uses_agentscope_wire_format() {
        let block = ThinkingBlock::new("I should inspect the request first.");
        let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

        assert_eq!(value["type"], "thinking");
        assert_eq!(value["thinking"], "I should inspect the request first.");
        assert_eq!(value["id"].as_str().unwrap().len(), 32);
        assert!(value["created_at"].is_string());
        assert!(value["finished_at"].is_null());
    }

    #[test]
    fn thinking_provider_fields_survive_json_round_trip() {
        let block = ThinkingBlock::new("")
            .with_extra_field("signature", "sig-123")
            .unwrap()
            .with_extra_field("redacted_thinking_data", "encrypted-payload")
            .unwrap();
        let original = Msg::new("Friday", Role::Assistant, [ContentBlock::from(block)]);

        let json = serde_json::to_string(&original).unwrap();
        let restored: Msg = serde_json::from_str(&json).unwrap();
        let ContentBlock::Thinking(restored_block) = &restored.content[0] else {
            panic!("expected thinking block");
        };

        assert_eq!(restored, original);
        assert_eq!(restored_block.extra_fields()["signature"], "sig-123");
        assert_eq!(
            restored_block.extra_fields()["redacted_thinking_data"],
            "encrypted-payload"
        );
    }

    #[test]
    fn text_content_excludes_thinking_blocks() {
        let message = Msg::new(
            "Friday",
            Role::Assistant,
            [
                ContentBlock::from(ThinkingBlock::new("Internal reasoning")),
                ContentBlock::from("Final answer"),
            ],
        );

        assert_eq!(message.text_content("\n").as_deref(), Some("Final answer"));
    }

    #[test]
    fn thinking_extensions_cannot_replace_standard_fields() {
        for field in ["type", "thinking", "id", "created_at", "finished_at"] {
            assert_eq!(
                ThinkingBlock::new("reasoning")
                    .with_extra_field(field, "replacement")
                    .unwrap_err(),
                ThinkingBlockError::ReservedField(field.to_owned())
            );
        }
    }

    #[test]
    fn tool_call_states_use_agentscope_wire_values() {
        for (state, expected) in [
            (ToolCallState::Pending, "pending"),
            (ToolCallState::Asking, "asking"),
            (ToolCallState::Allowed, "allowed"),
            (ToolCallState::Submitted, "submitted"),
            (ToolCallState::Finished, "finished"),
        ] {
            assert_eq!(serde_json::to_value(state).unwrap(), expected);
            assert_eq!(state.to_string(), expected);
        }
    }

    #[test]
    fn complete_tool_call_uses_agentscope_wire_format() {
        let rule = PermissionRule {
            tool_name: "get_weather".into(),
            rule_content: Some("Hangzhou".into()),
            behavior: PermissionBehavior::Ask,
            source: "model".into(),
        };
        let block = ToolCallBlock::complete("call-123", "get_weather", r#"{"city":"Hangzhou"}"#)
            .unwrap()
            .with_suggested_rules(vec![rule]);
        assert_eq!(block.id(), "call-123");
        assert_eq!(block.name(), "get_weather");
        let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

        assert_eq!(value["type"], "tool_call");
        assert_eq!(value["id"], "call-123");
        assert_eq!(value["name"], "get_weather");
        assert_eq!(value["input"], r#"{"city":"Hangzhou"}"#);
        assert_eq!(value["state"], "pending");
        assert_eq!(value["suggested_rules"][0]["behavior"], "ask");
    }

    #[test]
    fn streaming_tool_call_accepts_partial_input() {
        let mut block = ToolCallBlock::streaming("call-1", "get_weather").unwrap();
        block.append_input("{\"city\":\"");
        assert!(matches!(
            block.parsed_input(),
            Err(ToolCallError::InvalidInput(_))
        ));

        block.append_input("Hangzhou\"}");
        assert_eq!(block.parsed_input().unwrap(), json!({"city": "Hangzhou"}));
    }

    #[test]
    fn completed_tool_calls_require_valid_identity_and_json() {
        assert!(matches!(
            ToolCallBlock::complete("", "get_weather", "{}"),
            Err(ToolCallError::EmptyId)
        ));
        assert!(matches!(
            ToolCallBlock::complete("call-1", "  ", "{}"),
            Err(ToolCallError::EmptyName)
        ));
        assert!(matches!(
            ToolCallBlock::complete("call-1", "get_weather", "{"),
            Err(ToolCallError::InvalidInput(_))
        ));
    }

    #[test]
    fn tool_call_json_round_trip_preserves_streaming_input_and_rules() {
        let json = json!({
            "type": "tool_call",
            "id": "call-7",
            "name": "search",
            "input": "{\"query\":",
            "state": "asking",
            "suggested_rules": [{
                "tool_name": "search",
                "rule_content": null,
                "behavior": "allow",
                "source": "userSettings"
            }],
            "created_at": "2026-08-19T12:00:00.000000",
            "finished_at": null
        });

        let block: ContentBlock = serde_json::from_value(json.clone()).unwrap();
        let ContentBlock::ToolCall(tool_call) = &block else {
            panic!("expected tool call block");
        };
        assert_eq!(tool_call.name(), "search");
        assert_eq!(tool_call.input(), "{\"query\":");
        assert_eq!(tool_call.state, ToolCallState::Asking);
        assert_eq!(serde_json::to_value(block).unwrap(), json);
    }

    #[test]
    fn deserialization_rejects_blank_tool_identity() {
        for (id, name) in [("", "search"), ("call-1", " ")] {
            let value = json!({
                "type": "tool_call",
                "id": id,
                "name": name,
                "input": "",
                "state": "pending",
                "suggested_rules": [],
                "created_at": "2026-08-19T12:00:00.000000",
                "finished_at": null
            });
            assert!(serde_json::from_value::<ContentBlock>(value).is_err());
        }
    }

    #[test]
    fn text_content_excludes_tool_calls() {
        let message = Msg::new(
            "Friday",
            Role::Assistant,
            [
                ContentBlock::from(ToolCallBlock::complete("call-1", "search", "{}").unwrap()),
                ContentBlock::from("I will search for that."),
            ],
        );

        assert_eq!(
            message.text_content("\n").as_deref(),
            Some("I will search for that.")
        );
    }

    #[test]
    fn tool_result_states_use_agentscope_wire_values() {
        for (state, expected) in [
            (ToolResultState::Running, "running"),
            (ToolResultState::Success, "success"),
            (ToolResultState::Error, "error"),
            (ToolResultState::Interrupted, "interrupted"),
            (ToolResultState::Denied, "denied"),
        ] {
            assert_eq!(serde_json::to_value(state).unwrap(), expected);
            assert_eq!(state.to_string(), expected);
        }
    }

    #[test]
    fn running_tool_result_uses_agentscope_wire_format() {
        let block = ToolResultBlock::running("call-1", "get_weather").unwrap();
        assert_eq!(block.id(), "call-1");
        assert_eq!(block.name(), "get_weather");
        assert_eq!(block.state(), ToolResultState::Running);
        let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

        assert_eq!(value["type"], "tool_result");
        assert_eq!(value["output"], json!([]));
        assert_eq!(value["state"], "running");
        assert_eq!(value["metadata"], json!({}));
        assert!(value["finished_at"].is_null());
    }

    #[test]
    fn successful_raw_tool_result_round_trips() {
        let original = ToolResultBlock::success("call-2", "get_weather", "Sunny").unwrap();
        assert_eq!(original.state(), ToolResultState::Success);
        assert!(original.finished_at.is_some());
        assert_eq!(original.output(), &ToolResultOutput::Text("Sunny".into()));

        let json = serde_json::to_string(&ContentBlock::from(original.clone())).unwrap();
        let restored: ContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, ContentBlock::from(original));
    }

    #[test]
    fn streaming_tool_result_merges_text_and_preserves_multimodal_output() {
        let mut block = ToolResultBlock::running("call-3", "inspect_image").unwrap();
        block.append_text_delta("Found ");
        block.append_text_delta("a cat.");
        block.append_data(DataBlock::url("https://example.com/cat.jpg", "image/jpeg").unwrap());
        block.append_text_delta(" High confidence.");
        block.finish(ToolResultState::Success).unwrap();

        let value = serde_json::to_value(ContentBlock::from(block.clone())).unwrap();
        assert_eq!(value["output"][0]["type"], "text");
        assert_eq!(value["output"][0]["text"], "Found a cat.");
        assert_eq!(value["output"][1]["type"], "data");
        assert_eq!(value["output"][2]["text"], " High confidence.");

        let restored: ContentBlock = serde_json::from_value(value).unwrap();
        assert_eq!(restored, ContentBlock::from(block));
    }

    #[test]
    fn appending_to_raw_output_converts_it_to_structured_blocks() {
        let json = json!({
            "type": "tool_result",
            "id": "call-4",
            "name": "search",
            "output": "First",
            "state": "running",
            "metadata": {},
            "created_at": "2026-08-19T12:00:00.000000",
            "finished_at": null
        });
        let ContentBlock::ToolResult(mut block) =
            serde_json::from_value::<ContentBlock>(json).unwrap()
        else {
            panic!("expected tool result block");
        };

        block.append_text_delta(" second");
        let ToolResultOutput::Blocks(blocks) = block.output() else {
            panic!("expected structured output");
        };
        let ToolResultContent::Text(text) = &blocks[0] else {
            panic!("expected text output");
        };
        assert_eq!(text.text, "First second");
    }

    #[test]
    fn tool_result_finish_requires_terminal_state() {
        let mut block = ToolResultBlock::running("call-5", "search").unwrap();
        assert_eq!(
            block.finish(ToolResultState::Running).unwrap_err(),
            ToolResultError::NonTerminalState
        );
        assert_eq!(block.state(), ToolResultState::Running);
        assert!(block.finished_at.is_none());
    }

    #[test]
    fn tool_result_metadata_and_terminal_state_round_trip() {
        let mut metadata = Metadata::new();
        metadata.insert("status_code".into(), json!(403));
        let original = ToolResultBlock::finished(
            "call-6",
            "write_file",
            "Permission denied",
            ToolResultState::Denied,
        )
        .unwrap()
        .with_metadata(metadata);

        let value = serde_json::to_value(ContentBlock::from(original.clone())).unwrap();
        assert_eq!(value["state"], "denied");
        assert_eq!(value["metadata"]["status_code"], 403);
        assert_eq!(
            serde_json::from_value::<ContentBlock>(value).unwrap(),
            ContentBlock::from(original)
        );
    }

    #[test]
    fn tool_result_rejects_blank_identity() {
        assert_eq!(
            ToolResultBlock::running("", "search").unwrap_err(),
            ToolResultError::EmptyId
        );
        assert_eq!(
            ToolResultBlock::running("call-1", "  ").unwrap_err(),
            ToolResultError::EmptyName
        );

        let invalid = json!({
            "type": "tool_result",
            "id": "call-1",
            "name": "",
            "output": "",
            "state": "running",
            "metadata": {},
            "created_at": "2026-08-19T12:00:00.000000",
            "finished_at": null
        });
        assert!(serde_json::from_value::<ContentBlock>(invalid).is_err());
    }

    #[test]
    fn text_content_excludes_tool_results() {
        let message = Msg::new(
            "Friday",
            Role::Assistant,
            [
                ContentBlock::from(
                    ToolResultBlock::success("call-7", "search", "internal result").unwrap(),
                ),
                ContentBlock::from("Here is the answer."),
            ],
        );

        assert_eq!(
            message.text_content("\n").as_deref(),
            Some("Here is the answer.")
        );
    }
}
