//! OpenAI-compatible chat-completions client.

mod retry;
mod sse;
mod wire;

use std::{env, fmt, time::Duration};

use reqwest::{Client, Response, StatusCode, Url};
use serde_json::Value;

use super::{
    ChatEventStream, ChatModel, ChatRequest, ChatResponse, ModelCapabilities, ModelCapability,
    ModelError, ModelFuture,
};

pub use retry::RetryPolicy;

/// A streaming-capable client for OpenAI-compatible chat-completions APIs.
#[derive(Clone)]
pub struct OpenAIChatModel {
    model: String,
    api_key: SecretString,
    endpoint: Url,
    client: Client,
    retry_policy: RetryPolicy,
}

impl OpenAIChatModel {
    /// Starts building an OpenAI-compatible model.
    #[must_use]
    pub fn builder() -> OpenAIChatModelBuilder {
        OpenAIChatModelBuilder::default()
    }

    /// Creates a model using the default `OpenAI` base URL.
    ///
    /// # Errors
    ///
    /// Returns [`OpenAIConfigError`] if the model name or API key is empty, or
    /// if the HTTP client cannot be constructed.
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self, OpenAIConfigError> {
        Self::builder().model(model).api_key(api_key).build()
    }

    /// Loads the API key from `OPENAI_API_KEY`.
    ///
    /// # Errors
    ///
    /// Returns [`OpenAIConfigError`] if the environment variable is absent or
    /// invalid, or if the remaining configuration is invalid.
    pub fn from_env(model: impl Into<String>) -> Result<Self, OpenAIConfigError> {
        Self::builder()
            .model(model)
            .api_key_from_env("OPENAI_API_KEY")?
            .build()
    }
}

impl fmt::Debug for OpenAIChatModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAIChatModel")
            .field("model", &self.model)
            .field("api_key", &self.api_key)
            .field("endpoint", &self.endpoint)
            .field("retry_policy", &self.retry_policy)
            .finish_non_exhaustive()
    }
}

impl ChatModel for OpenAIChatModel {
    fn name(&self) -> &str {
        &self.model
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::new()
            .with(ModelCapability::Streaming, true)
            .with(ModelCapability::ToolCalls, true)
            .with(ModelCapability::StructuredOutput, true)
    }

    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async move {
            let body = wire::encode_request(&self.model, &request, false)?;
            let response = self.send_with_retries(&body).await?;

            let value = response.json::<Value>().await.map_err(|error| {
                ModelError::new(format!("provider returned invalid JSON: {error}"))
                    .with_code("invalid_response")
            })?;
            wire::decode_response(&value, request.structured_output_schema.as_ref())
        })
    }

    fn stream(&self, request: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async move {
            let body = wire::encode_request(&self.model, &request, true)?;
            let response = self.send_with_retries(&body).await?;

            Ok(sse::decode_stream(
                response,
                request.structured_output_schema,
            ))
        })
    }
}

impl OpenAIChatModel {
    async fn send_with_retries(&self, body: &Value) -> Result<Response, ModelError> {
        let mut retries = 0;
        loop {
            let result = self
                .client
                .post(self.endpoint.clone())
                .bearer_auth(self.api_key.expose())
                .json(body)
                .send()
                .await;
            match result {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let status = response.status();
                    let retry_after = retry::retry_after(response.headers());
                    let response_body = response.json::<Value>().await.ok();
                    let error = provider_error(status, response_body.as_ref());
                    if !self.retry_policy.can_retry(retries, &error) {
                        return Err(error);
                    }
                    self.retry_policy.wait(retries, retry_after).await;
                }
                Err(error) => {
                    let error = transport_error(&error);
                    if !self.retry_policy.can_retry(retries, &error) {
                        return Err(error);
                    }
                    self.retry_policy.wait(retries, None).await;
                }
            }
            retries += 1;
        }
    }
}

/// Builder for [`OpenAIChatModel`].
#[derive(Clone, Debug)]
pub struct OpenAIChatModelBuilder {
    model: Option<String>,
    api_key: Option<SecretString>,
    base_url: String,
    timeout: Duration,
    retry_policy: RetryPolicy,
}

impl Default for OpenAIChatModelBuilder {
    fn default() -> Self {
        Self {
            model: None,
            api_key: None,
            base_url: "https://api.openai.com/v1".to_owned(),
            timeout: Duration::from_secs(60),
            retry_policy: RetryPolicy::default(),
        }
    }
}

impl OpenAIChatModelBuilder {
    /// Sets the provider model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the API key in memory.
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString(api_key.into()));
        self
    }

    /// Reads the API key from an environment variable.
    ///
    /// # Errors
    ///
    /// Returns [`OpenAIConfigError::MissingEnvironmentVariable`] when the
    /// variable is absent or is not valid Unicode.
    pub fn api_key_from_env(mut self, variable: &str) -> Result<Self, OpenAIConfigError> {
        let key = env::var(variable)
            .map_err(|_| OpenAIConfigError::MissingEnvironmentVariable(variable.to_owned()))?;
        self.api_key = Some(SecretString(key));
        Ok(self)
    }

    /// Sets the API base URL, for example `https://api.deepseek.com`.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the complete request timeout.
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the retry and rate-limit policy.
    #[must_use]
    pub const fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Constructs the configured model.
    ///
    /// # Errors
    ///
    /// Returns [`OpenAIConfigError`] when a required field or URL is invalid,
    /// or when the HTTP client cannot be constructed.
    pub fn build(self) -> Result<OpenAIChatModel, OpenAIConfigError> {
        let model = require_nonempty(self.model, OpenAIConfigError::MissingModel)?;
        let api_key = self.api_key.ok_or(OpenAIConfigError::MissingApiKey)?;
        if api_key.expose().trim().is_empty() {
            return Err(OpenAIConfigError::MissingApiKey);
        }
        let endpoint = Url::parse(&format!(
            "{}/chat/completions",
            self.base_url.trim_end_matches('/')
        ))
        .map_err(OpenAIConfigError::InvalidBaseUrl)?;
        if !matches!(endpoint.scheme(), "http" | "https") {
            return Err(OpenAIConfigError::UnsupportedUrlScheme);
        }
        let client = Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(OpenAIConfigError::HttpClient)?;
        Ok(OpenAIChatModel {
            model,
            api_key,
            endpoint,
            client,
            retry_policy: self.retry_policy,
        })
    }
}

/// Invalid OpenAI-compatible model configuration.
#[derive(Debug)]
pub enum OpenAIConfigError {
    /// No model name was configured.
    MissingModel,
    /// No non-empty API key was configured.
    MissingApiKey,
    /// An environment variable containing the key was unavailable.
    MissingEnvironmentVariable(String),
    /// The base URL was malformed.
    InvalidBaseUrl(url::ParseError),
    /// The base URL did not use HTTP or HTTPS.
    UnsupportedUrlScheme,
    /// The underlying HTTP client could not be created.
    HttpClient(reqwest::Error),
}

impl fmt::Display for OpenAIConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingModel => formatter.write_str("model name cannot be empty"),
            Self::MissingApiKey => formatter.write_str("API key cannot be empty"),
            Self::MissingEnvironmentVariable(variable) => {
                write!(
                    formatter,
                    "environment variable `{variable}` is not available"
                )
            }
            Self::InvalidBaseUrl(error) => write!(formatter, "invalid API base URL: {error}"),
            Self::UnsupportedUrlScheme => {
                formatter.write_str("API base URL must use HTTP or HTTPS")
            }
            Self::HttpClient(error) => write!(formatter, "failed to create HTTP client: {error}"),
        }
    }
}

impl std::error::Error for OpenAIConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBaseUrl(error) => Some(error),
            Self::HttpClient(error) => Some(error),
            Self::MissingModel
            | Self::MissingApiKey
            | Self::MissingEnvironmentVariable(_)
            | Self::UnsupportedUrlScheme => None,
        }
    }
}

#[derive(Clone)]
struct SecretString(String);

impl SecretString {
    fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[REDACTED]")
    }
}

fn require_nonempty(
    value: Option<String>,
    error: OpenAIConfigError,
) -> Result<String, OpenAIConfigError> {
    match value {
        Some(value) if !value.trim().is_empty() => Ok(value),
        Some(_) | None => Err(error),
    }
}

fn transport_error(error: &reqwest::Error) -> ModelError {
    ModelError::new(format!("OpenAI-compatible request failed: {error}"))
        .with_code("transport_error")
        .with_retryable(error.is_timeout() || error.is_connect())
}

fn provider_error(status: StatusCode, body: Option<&Value>) -> ModelError {
    let provider = body.and_then(|value| value.get("error"));
    let message = provider
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map_or_else(
            || format!("provider returned HTTP {status}"),
            ToOwned::to_owned,
        );
    let code = provider
        .and_then(|error| error.get("code").or_else(|| error.get("type")))
        .and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            Value::Number(value) => Some(value.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| format!("http_{}", status.as_u16()));
    ModelError::new(message).with_code(code).with_retryable(
        status == StatusCode::REQUEST_TIMEOUT
            || status == StatusCode::CONFLICT
            || status == StatusCode::TOO_MANY_REQUESTS
            || status.is_server_error(),
    )
}

#[cfg(test)]
mod tests;
