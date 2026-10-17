//! Cooperative interruption primitives for in-flight agent operations.

use tokio::sync::watch;

/// A cloneable handle for interrupting in-flight replies on one agent.
///
/// Each operation captures the current signal generation when it starts. An
/// interrupt wakes all operations that were already active, while operations
/// started afterward are unaffected. Dropping model or tool futures cannot
/// undo external side effects they already performed, so tools remain
/// responsible for their own cancellation safety.
#[derive(Clone, Debug)]
pub struct AgentInterruptHandle {
    generation: watch::Sender<u64>,
}

impl AgentInterruptHandle {
    /// Creates an interrupt handle with no pending signal.
    #[must_use]
    pub fn new() -> Self {
        let (generation, _receiver) = watch::channel(0);
        Self { generation }
    }

    /// Requests cooperative interruption of all currently active replies.
    pub fn interrupt(&self) {
        self.generation
            .send_modify(|generation| *generation = generation.wrapping_add(1));
    }

    /// Captures an operation-scoped cancellation token. Capture before starting work;
    /// later operations are not cancelled by earlier interrupt requests.
    #[must_use]
    pub fn token(&self) -> AgentInterruptToken {
        let receiver = self.generation.subscribe();
        let baseline = *receiver.borrow();
        AgentInterruptToken { receiver, baseline }
    }
}

impl Default for AgentInterruptHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation-scoped cancellation signal for custom agent implementations.
/// Keep a handle alive while waiting; dropping a future does not undo tool effects.
#[derive(Clone, Debug)]
pub struct AgentInterruptToken {
    receiver: watch::Receiver<u64>,
    baseline: u64,
}

impl AgentInterruptToken {
    /// Reports whether an interrupt occurred since this token was captured.
    #[must_use]
    pub fn is_interrupted(&self) -> bool {
        *self.receiver.borrow() != self.baseline
    }

    /// Waits until an interrupt occurs.
    ///
    /// # Panics
    /// Panics if all interrupt handles are dropped before cancellation.
    pub async fn cancelled(&mut self) {
        while !self.is_interrupted() {
            self.receiver
                .changed()
                .await
                .expect("the interrupt handle remains owned by the agent");
        }
    }
}

#[cfg(test)]
mod tests {
    use futures_executor::block_on;

    use super::AgentInterruptHandle;

    #[test]
    fn interrupt_wakes_active_tokens_without_affecting_future_operations() {
        let handle = AgentInterruptHandle::new();
        let cloned = handle.clone();
        let mut active = handle.token();

        cloned.interrupt();

        assert!(active.is_interrupted());
        block_on(active.cancelled());
        assert!(!handle.token().is_interrupted());
    }
}
