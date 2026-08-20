//! Streaming structured-output content blocks.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;

use super::{generate_id, generate_timestamp};

/// The lifecycle state of structured model output.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum StructuredOutputState {
    /// JSON text is still arriving from the model.
    #[default]
    Streaming,
    /// The accumulated output is complete and valid JSON.
    Complete,
}

impl fmt::Display for StructuredOutputState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Streaming => "streaming",
            Self::Complete => "complete",
        })
    }
}

/// An error returned when constructing or updating structured output.
#[derive(Debug)]
pub enum StructuredOutputError {
    /// A JSON Schema root was neither an object nor a boolean.
    InvalidSchema,
    /// Completed output was not valid JSON.
    InvalidOutput(serde_json::Error),
    /// A streaming update was attempted after completion.
    AlreadyComplete,
}

impl fmt::Display for StructuredOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema => {
                formatter.write_str("JSON Schema root must be an object or boolean")
            }
            Self::InvalidOutput(error) => {
                write!(formatter, "invalid structured output JSON: {error}")
            }
            Self::AlreadyComplete => formatter.write_str("structured output is already complete"),
        }
    }
}

impl std::error::Error for StructuredOutputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidOutput(error) => Some(error),
            Self::InvalidSchema | Self::AlreadyComplete => None,
        }
    }
}

/// JSON output produced according to a requested JSON Schema.
///
/// The output is stored as raw text so that incomplete JSON fragments can be
/// preserved while a model response is streaming. Calling [`Self::finish`]
/// validates the accumulated text as JSON, but schema-instance validation is
/// intentionally left to the model layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StructuredOutputBlock {
    /// The JSON Schema requested from the model.
    schema: Value,
    /// Raw JSON text, which may be incomplete while streaming.
    output: String,
    /// The block's current lifecycle state.
    state: StructuredOutputState,
    /// The unique block identifier.
    pub id: String,
    /// The local creation time in ISO 8601 format.
    pub created_at: String,
    /// The completion time, when available.
    pub finished_at: Option<String>,
}

impl StructuredOutputBlock {
    /// Creates an empty block whose JSON output will arrive incrementally.
    ///
    /// # Errors
    ///
    /// Returns [`StructuredOutputError::InvalidSchema`] when `schema` is not a
    /// valid JSON Schema root shape.
    pub fn streaming(schema: Value) -> Result<Self, StructuredOutputError> {
        validate_schema(&schema)?;
        Ok(Self {
            schema,
            output: String::new(),
            state: StructuredOutputState::Streaming,
            id: generate_id(),
            created_at: generate_timestamp(),
            finished_at: None,
        })
    }

    /// Creates a completed block from a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`StructuredOutputError::InvalidSchema`] when `schema` is not a
    /// valid JSON Schema root shape.
    pub fn complete(
        schema: Value,
        output: impl Into<Value>,
    ) -> Result<Self, StructuredOutputError> {
        let mut block = Self::streaming(schema)?;
        block.output =
            serde_json::to_string(&output.into()).map_err(StructuredOutputError::InvalidOutput)?;
        block.state = StructuredOutputState::Complete;
        block.finished_at = Some(generate_timestamp());
        Ok(block)
    }

    /// Returns the requested JSON Schema.
    #[must_use]
    pub const fn schema(&self) -> &Value {
        &self.schema
    }

    /// Returns the raw JSON accumulated so far.
    #[must_use]
    pub fn raw_output(&self) -> &str {
        &self.output
    }

    /// Returns the current lifecycle state.
    #[must_use]
    pub const fn state(&self) -> StructuredOutputState {
        self.state
    }

    /// Parses the current output as JSON.
    ///
    /// # Errors
    ///
    /// Returns [`StructuredOutputError::InvalidOutput`] while the output is
    /// incomplete or malformed.
    pub fn parsed_output(&self) -> Result<Value, StructuredOutputError> {
        serde_json::from_str(&self.output).map_err(StructuredOutputError::InvalidOutput)
    }

    /// Appends a raw JSON fragment received from a streaming model response.
    ///
    /// # Errors
    ///
    /// Returns [`StructuredOutputError::AlreadyComplete`] after the block has
    /// been completed.
    pub fn append_output_delta(&mut self, delta: &str) -> Result<(), StructuredOutputError> {
        if self.state == StructuredOutputState::Complete {
            return Err(StructuredOutputError::AlreadyComplete);
        }
        self.output.push_str(delta);
        Ok(())
    }

    /// Validates the accumulated JSON and marks the block complete.
    ///
    /// # Errors
    ///
    /// Returns [`StructuredOutputError::InvalidOutput`] when the accumulated
    /// output is not valid JSON, or [`StructuredOutputError::AlreadyComplete`]
    /// when the block has already been completed.
    pub fn finish(&mut self) -> Result<(), StructuredOutputError> {
        if self.state == StructuredOutputState::Complete {
            return Err(StructuredOutputError::AlreadyComplete);
        }
        self.parsed_output()?;
        self.state = StructuredOutputState::Complete;
        self.finished_at = Some(generate_timestamp());
        Ok(())
    }
}

impl<'de> Deserialize<'de> for StructuredOutputBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireBlock {
            schema: Value,
            output: String,
            #[serde(default)]
            state: StructuredOutputState,
            #[serde(default = "generate_id")]
            id: String,
            #[serde(default = "generate_timestamp")]
            created_at: String,
            #[serde(default)]
            finished_at: Option<String>,
        }

        let block = WireBlock::deserialize(deserializer)?;
        validate_schema(&block.schema).map_err(de::Error::custom)?;
        if block.state == StructuredOutputState::Complete {
            serde_json::from_str::<Value>(&block.output).map_err(de::Error::custom)?;
        }
        Ok(Self {
            schema: block.schema,
            output: block.output,
            state: block.state,
            id: block.id,
            created_at: block.created_at,
            finished_at: block.finished_at,
        })
    }
}

fn validate_schema(schema: &Value) -> Result<(), StructuredOutputError> {
    if schema.is_object() || schema.is_boolean() {
        Ok(())
    } else {
        Err(StructuredOutputError::InvalidSchema)
    }
}
