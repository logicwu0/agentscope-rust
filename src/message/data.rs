//! Validated multimodal data content blocks.

use std::fmt;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Deserializer, Serialize, de};
use url::Url;

use super::{generate_id, generate_timestamp};

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

fn validate_media_type(media_type: String) -> Result<String, DataBlockError> {
    if media_type.trim().is_empty() {
        Err(DataBlockError::EmptyMediaType)
    } else {
        Ok(media_type)
    }
}
