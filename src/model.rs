//! Provider-neutral model interfaces and response types.

mod response;

pub use response::{ChatResponse, FinishReason};

#[cfg(test)]
mod tests;
