//! Independently implementable, object-safe agent capabilities.

use super::{AgentEventStream, AgentFuture, AgentInterruptHandle, AgentState, ToolConfirmation};
use crate::Msg;

/// Minimal asynchronous execution contract used by pipelines.
/// Custom algorithms need not implement persistence, compaction or tool recovery.
/// Implementations must honor their interruption handle during in-flight work.
///
/// Optional capabilities are deliberately unavailable on the minimal trait:
/// ```compile_fail
/// use agentscope::Agent;
/// fn snapshot(agent: &dyn Agent) {
///     let _ = agent.snapshot(); // require StatefulAgent instead
/// }
/// ```
pub trait Agent: Send + Sync {
    /// Returns the agent's display name.
    fn name(&self) -> &str;

    /// Produces one reply to an input message.
    fn reply(&self, message: Msg) -> AgentFuture<'_, Msg>;

    /// Streams one reply as model, tool, and lifecycle events.
    fn stream(&self, message: Msg) -> AgentFuture<'_, AgentEventStream<'_>>;

    /// Returns a handle that can interrupt in-flight replies on this agent.
    fn interrupt_handle(&self) -> AgentInterruptHandle;
}

/// Optional conversation observation and snapshot/restore capability.
/// The current snapshot format is [`AgentState`]; richer algorithm state may need a future schema.
pub trait StatefulAgent: Agent {
    /// Stores an external message in conversation memory without replying.
    fn observe(&self, message: Msg) -> AgentFuture<'_, ()>;

    /// Captures a versioned snapshot of configured conversation state.
    fn snapshot(&self) -> AgentFuture<'_, AgentState>;

    /// Atomically restores a previously captured conversation state.
    fn restore(&self, state: AgentState) -> AgentFuture<'_, ()>;
}

/// Optional explicit context-compaction capability.
pub trait ContextManagedAgent: Agent {
    /// Explicitly summarizes old completed turns, retaining at least one recent turn.
    /// Returns `None` if there is no newly eligible prefix.
    fn compact_context(
        &self,
        keep_recent_turns: usize,
    ) -> AgentFuture<'_, Option<crate::ContextSummary>>;

    /// Removes only the summary, making original history available again.
    fn clear_context_summary(&self) -> AgentFuture<'_, ()>;
}

/// Optional tool confirmation and uncertain-execution recovery capability.
/// Implementations must preserve explicit approval and idempotency boundaries.
pub trait RecoverableAgent: Agent {
    /// Resumes a reply paused for explicit tool-call confirmation.
    fn resume_tool_calls(
        &self,
        reply_id: String,
        confirmations: Vec<ToolConfirmation>,
    ) -> AgentFuture<'_, Msg>;

    /// Resolves an execution with externally reconciled terminal tool results.
    fn resolve_tool_execution(
        &self,
        reply_id: String,
        results: Vec<crate::ToolResultBlock>,
    ) -> AgentFuture<'_, Msg>;

    /// Explicitly retries all approved calls of an uncertain execution with
    /// their original idempotency keys. Tools must honor those keys to deduplicate effects.
    fn retry_tool_execution(&self, reply_id: String) -> AgentFuture<'_, Msg>;

    /// Approves/denies pending calls and streams tools and the continuing reply.
    fn stream_resume_tool_calls(
        &self,
        reply_id: String,
        confirmations: Vec<ToolConfirmation>,
    ) -> AgentFuture<'_, AgentEventStream<'_>>;

    /// Explicitly retries uncertain calls and streams the continuing reply.
    fn stream_retry_tool_execution(
        &self,
        reply_id: String,
    ) -> AgentFuture<'_, AgentEventStream<'_>>;

    /// Supplies verified tool results and streams the continuing reply.
    fn stream_resolve_tool_execution(
        &self,
        reply_id: String,
        results: Vec<crate::ToolResultBlock>,
    ) -> AgentFuture<'_, AgentEventStream<'_>>;
}
