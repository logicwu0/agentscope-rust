//! Provider-neutral model interfaces and response types.

mod accumulator;
mod event;
mod response;

pub use accumulator::{ChatResponseAccumulator, ChatStreamError};
pub use event::{ChatEvent, ModelError};
pub use response::{ChatResponse, FinishReason};

#[cfg(test)]
mod tests;
