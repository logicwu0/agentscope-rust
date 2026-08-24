//! Deterministic mock chat model for tests and examples.

use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{Mutex, MutexGuard},
    task::{Context, Poll},
};

use futures_core::Stream;

use super::{
    ChatEvent, ChatEventStream, ChatModel, ChatRequest, ChatResponse, ModelCapabilities,
    ModelError, ModelFuture, ModelResult,
};

/// A deterministic model backed by queues of scripted outputs.
#[derive(Debug)]
pub struct MockChatModel {
    name: String,
    capabilities: ModelCapabilities,
    responses: Mutex<VecDeque<ModelResult<ChatResponse>>>,
    streams: Mutex<VecDeque<ModelResult<Vec<ModelResult<ChatEvent>>>>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl MockChatModel {
    /// Creates a mock with all current capabilities enabled.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            capabilities: ModelCapabilities::all(),
            responses: Mutex::new(VecDeque::new()),
            streams: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Replaces the capabilities advertised by this mock.
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: ModelCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Adds a successful complete response to the script.
    #[must_use]
    pub fn with_response(self, response: ChatResponse) -> Self {
        self.push_response(Ok(response));
        self
    }

    /// Adds a complete-call failure to the script.
    #[must_use]
    pub fn with_error(self, error: ModelError) -> Self {
        self.push_response(Err(error));
        self
    }

    /// Adds a successful event stream to the script.
    #[must_use]
    pub fn with_stream(self, events: impl IntoIterator<Item = ModelResult<ChatEvent>>) -> Self {
        self.push_stream(Ok(events.into_iter().collect()));
        self
    }

    /// Adds a failure that occurs while starting a stream.
    #[must_use]
    pub fn with_stream_error(self, error: ModelError) -> Self {
        self.push_stream(Err(error));
        self
    }

    /// Queues a complete-call result.
    pub fn push_response(&self, response: ModelResult<ChatResponse>) {
        lock(&self.responses).push_back(response);
    }

    /// Queues a stream-start result.
    pub fn push_stream(&self, stream: ModelResult<Vec<ModelResult<ChatEvent>>>) {
        lock(&self.streams).push_back(stream);
    }

    /// Returns all requests in invocation order.
    #[must_use]
    pub fn recorded_requests(&self) -> Vec<ChatRequest> {
        lock(&self.requests).clone()
    }

    fn record(&self, request: ChatRequest) {
        lock(&self.requests).push(request);
    }
}

impl ChatModel for MockChatModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.capabilities.clone()
    }

    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        self.record(request);
        let response = lock(&self.responses).pop_front().unwrap_or_else(|| {
            Err(ModelError::new(
                "mock has no scripted complete response remaining",
            ))
        });
        Box::pin(async move { response })
    }

    fn stream(&self, request: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        self.record(request);
        let stream = lock(&self.streams).pop_front().unwrap_or_else(|| {
            Err(ModelError::new(
                "mock has no scripted event stream remaining",
            ))
        });
        Box::pin(async move {
            stream.map(|events| {
                Box::pin(MockEventStream {
                    events: events.into_iter(),
                }) as ChatEventStream<'_>
            })
        })
    }
}

struct MockEventStream {
    events: std::vec::IntoIter<ModelResult<ChatEvent>>,
}

impl Stream for MockEventStream {
    type Item = ModelResult<ChatEvent>;

    fn poll_next(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.events.next())
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
