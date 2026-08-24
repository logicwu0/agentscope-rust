//! Provider-neutral model interfaces and response types.

mod accumulator;
mod capabilities;
mod chat_model;
mod event;
mod mock;
mod request;
mod response;

pub use accumulator::{ChatResponseAccumulator, ChatStreamError};
pub use capabilities::{ModelCapabilities, ModelCapability};
pub use chat_model::{ChatEventStream, ChatModel, ModelFuture, ModelResult};
pub use event::{ChatEvent, ModelError};
pub use mock::MockChatModel;
pub use request::{ChatRequest, ChatRequestError, GenerateOptions, ToolDefinition};
pub use response::{ChatResponse, FinishReason};

#[cfg(test)]
mod tests;
