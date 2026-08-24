//! Provider-neutral chat requests and generation options.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;

use crate::message::{Metadata, Msg};

/// Provider-neutral generation parameters.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct GenerateOptions {
    /// Sampling temperature, interpreted by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum number of output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling probability, interpreted by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Optional deterministic sampling seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Sequences that stop generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Additional provider-specific JSON options.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub extra: Metadata,
}

impl GenerateOptions {
    /// Creates empty generation options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the sampling temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum output token count.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets nucleus sampling probability.
    #[must_use]
    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets a deterministic sampling seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Replaces the stop sequences.
    #[must_use]
    pub fn with_stop(mut self, stop: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.stop = stop.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces provider-specific options.
    #[must_use]
    pub fn with_extra(mut self, extra: Metadata) -> Self {
        self.extra = extra;
        self
    }
}

/// A tool schema made available to a chat model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ToolDefinition {
    /// The model-visible tool name.
    pub name: String,
    /// A model-visible explanation of what the tool does.
    pub description: String,
    /// JSON Schema describing the tool arguments.
    pub input_schema: Value,
}

impl<'de> Deserialize<'de> for ToolDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireToolDefinition {
            name: String,
            description: String,
            input_schema: Value,
        }

        let tool = WireToolDefinition::deserialize(deserializer)?;
        Self::new(tool.name, tool.description, tool.input_schema).map_err(de::Error::custom)
    }
}

impl ToolDefinition {
    /// Creates a tool definition.
    ///
    /// # Errors
    ///
    /// Returns [`ChatRequestError`] when the name is blank or the schema root
    /// is neither an object nor a boolean.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Result<Self, ChatRequestError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ChatRequestError::EmptyToolName);
        }
        validate_schema(&input_schema)?;
        Ok(Self {
            name,
            description: description.into(),
            input_schema,
        })
    }
}

/// Input passed to a [`super::ChatModel`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatRequest {
    /// Conversation messages supplied to the model.
    pub messages: Vec<Msg>,
    /// Provider-neutral generation parameters.
    #[serde(default)]
    pub options: GenerateOptions,
    /// Tools available to the model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// Requested structured-output JSON Schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_output_schema: Option<Value>,
}

impl<'de> Deserialize<'de> for ChatRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireChatRequest {
            messages: Vec<Msg>,
            #[serde(default)]
            options: GenerateOptions,
            #[serde(default)]
            tools: Vec<ToolDefinition>,
            #[serde(default)]
            structured_output_schema: Option<Value>,
        }

        let request = WireChatRequest::deserialize(deserializer)?;
        if let Some(schema) = &request.structured_output_schema {
            validate_schema(schema).map_err(de::Error::custom)?;
        }
        Ok(Self {
            messages: request.messages,
            options: request.options,
            tools: request.tools,
            structured_output_schema: request.structured_output_schema,
        })
    }
}

impl ChatRequest {
    /// Creates a request from conversation messages.
    #[must_use]
    pub fn new(messages: impl IntoIterator<Item = Msg>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            options: GenerateOptions::default(),
            tools: Vec::new(),
            structured_output_schema: None,
        }
    }

    /// Replaces the generation options.
    #[must_use]
    pub fn with_options(mut self, options: GenerateOptions) -> Self {
        self.options = options;
        self
    }

    /// Replaces the available tools.
    #[must_use]
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }

    /// Requests output conforming to a JSON Schema.
    ///
    /// # Errors
    ///
    /// Returns [`ChatRequestError::InvalidSchema`] when the schema root is
    /// neither an object nor a boolean.
    pub fn with_structured_output_schema(
        mut self,
        schema: Value,
    ) -> Result<Self, ChatRequestError> {
        validate_schema(&schema)?;
        self.structured_output_schema = Some(schema);
        Ok(self)
    }
}

/// An error returned while constructing a chat request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRequestError {
    /// A tool definition used a blank name.
    EmptyToolName,
    /// A JSON Schema root was neither an object nor a boolean.
    InvalidSchema,
}

impl fmt::Display for ChatRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyToolName => formatter.write_str("tool name cannot be empty"),
            Self::InvalidSchema => {
                formatter.write_str("JSON Schema root must be an object or boolean")
            }
        }
    }
}

impl std::error::Error for ChatRequestError {}

fn validate_schema(schema: &Value) -> Result<(), ChatRequestError> {
    if schema.is_object() || schema.is_boolean() {
        Ok(())
    } else {
        Err(ChatRequestError::InvalidSchema)
    }
}
