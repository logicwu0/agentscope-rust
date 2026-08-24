//! Object-safe asynchronous chat model interface.

use std::{future::Future, pin::Pin};

use futures_core::Stream;

use super::{ChatEvent, ChatRequest, ChatResponse, ModelCapabilities, ModelError};

/// Result returned by provider-neutral model operations.
pub type ModelResult<T> = Result<T, ModelError>;

/// A boxed asynchronous model operation.
pub type ModelFuture<'a, T> = Pin<Box<dyn Future<Output = ModelResult<T>> + Send + 'a>>;

/// A boxed asynchronous stream of chat events.
pub type ChatEventStream<'a> = Pin<Box<dyn Stream<Item = ModelResult<ChatEvent>> + Send + 'a>>;

/// Provider-neutral asynchronous chat model.
///
/// Boxed futures keep this trait object-safe, allowing models to be stored as
/// `Arc<dyn ChatModel>` while still supporting asynchronous implementations.
pub trait ChatModel: Send + Sync {
    /// Returns the configured model name.
    fn name(&self) -> &str;

    /// Returns capabilities supported by this model implementation.
    fn capabilities(&self) -> ModelCapabilities;

    /// Generates one complete response.
    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse>;

    /// Starts an incremental response stream.
    fn stream(&self, request: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>>;
}
