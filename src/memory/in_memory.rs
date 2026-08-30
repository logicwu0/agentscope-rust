//! Thread-safe process-local conversation memory.

use std::{fmt, sync::Mutex};

use crate::Msg;

use super::{Memory, MemoryFuture};

/// An unbounded, process-local conversation history.
#[derive(Default)]
pub struct InMemoryMemory {
    messages: Mutex<Vec<Msg>>,
}

impl InMemoryMemory {
    /// Creates an empty memory.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
        }
    }

    /// Creates memory initialized with existing conversation history.
    #[must_use]
    pub fn from_messages(messages: impl IntoIterator<Item = Msg>) -> Self {
        Self {
            messages: Mutex::new(messages.into_iter().collect()),
        }
    }
}

impl Memory for InMemoryMemory {
    fn messages(&self) -> MemoryFuture<'_, Vec<Msg>> {
        Box::pin(async move { Ok(lock(&self.messages).clone()) })
    }

    fn append(&self, messages: Vec<Msg>) -> MemoryFuture<'_, ()> {
        Box::pin(async move {
            lock(&self.messages).extend(messages);
            Ok(())
        })
    }

    fn clear(&self) -> MemoryFuture<'_, ()> {
        Box::pin(async move {
            lock(&self.messages).clear();
            Ok(())
        })
    }
}

impl fmt::Debug for InMemoryMemory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InMemoryMemory")
            .field("message_count", &lock(&self.messages).len())
            .finish()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
