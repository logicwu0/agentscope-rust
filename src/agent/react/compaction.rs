//! Explicit summary generation with commit-after-validation semantics.

use super::{ReActAgent, StateOperation, ensure_not_interrupted, lock};
use crate::{
    AgentError, AgentFuture, AgentResult, ContextPolicy, ContextSummarizer, ContextSummary, Msg,
    RecentTurns, Role, SummaryError,
};
use std::{collections::BTreeSet, sync::Arc};

impl ReActAgent {
    /// Configures a summarizer. Ordinary replies invoke it only when automatic
    /// compaction is explicitly enabled and its budget check triggers.
    #[must_use]
    pub fn with_summarizer<S: ContextSummarizer + 'static>(mut self, summarizer: S) -> Self {
        self.summarizer = Some(Arc::new(summarizer));
        self
    }

    /// Shares an explicit summarizer, which is not persisted with agent state.
    #[must_use]
    pub fn with_shared_summarizer(mut self, summarizer: Arc<dyn ContextSummarizer>) -> Self {
        self.summarizer = Some(summarizer);
        self
    }

    /// Summarizes a completed old prefix, retaining at least `keep_recent_turns`
    /// user turns verbatim. Never modifies raw memory. A repeated request with
    /// no newly eligible prefix returns `None` without calling the summarizer.
    ///
    /// Only call on an idle agent. Active operations/clones return `Busy`;
    /// pending tool checkpoints are rejected. Shared memory must not be modified
    /// externally during this operation. Independent processes require store CAS.
    /// All generation/validation failures leave the prior summary unchanged.
    /// Store writes are committed before installing the new runtime summary.
    /// An uncertain store acknowledgement must be resolved by reloading its state.
    #[must_use]
    pub fn compact_context(
        &self,
        keep_recent_turns: usize,
    ) -> AgentFuture<'_, Option<ContextSummary>> {
        Box::pin(async move {
            RecentTurns::new(keep_recent_turns)
                .map_err(|_| AgentError::Summary(SummaryError::ZeroRecentTurns))?;
            let _guard = self
                .context_operations
                .clone()
                .try_write_owned()
                .map_err(|_| AgentError::Summary(SummaryError::Busy))?;
            let mut operation = self.begin_store_operation().await?;
            let original = self.snapshot_memory().await?;
            let interrupt = self.interrupt.token();
            let Some(summary) = self
                .summary_candidate(&original, keep_recent_turns, interrupt)
                .await?
            else {
                return Ok(None);
            };
            let system = self.system_prompt.as_ref().map(Msg::system);
            self.chat_request_with_summary(original.messages(), system.as_ref(), Some(&summary))
                .await?;
            if self.snapshot_memory().await? != original {
                return Err(AgentError::Summary(SummaryError::StaleSource));
            }
            self.commit_summary(
                &mut operation,
                original.with_context_summary(Some(summary.clone())),
            )
            .await?;
            Ok(Some(summary))
        })
    }

    pub(super) async fn summary_candidate(
        &self,
        original: &crate::AgentState,
        keep_recent_turns: usize,
        mut interrupt: crate::agent::interrupt::AgentInterruptToken,
    ) -> AgentResult<Option<ContextSummary>> {
        let policy = RecentTurns::new(keep_recent_turns)
            .map_err(|_| AgentError::Summary(SummaryError::ZeroRecentTurns))?;
        if original.pending_tool_calls().is_some() || original.pending_tool_execution().is_some() {
            return Err(AgentError::Summary(SummaryError::PendingTools));
        }
        if let Some(summary) = original.context_summary() {
            summary
                .validate(original.messages())
                .map_err(AgentError::Summary)?;
        }
        let selected = policy.select_messages(original.messages());
        let ids = selected
            .iter()
            .map(|m| m.id.as_str())
            .collect::<BTreeSet<_>>();
        let end = original
            .messages()
            .iter()
            .position(|m| m.role != Role::System && ids.contains(m.id.as_str()))
            .unwrap_or(0);
        if !original.messages()[..end]
            .iter()
            .any(|m| m.role != Role::System)
            || original
                .context_summary()
                .is_some_and(|s| s.covered_messages() >= end)
        {
            return Ok(None);
        }
        // Validate boundaries before spending tokens.
        ContextSummary::new("validation".into(), &original.messages()[..end])
            .map_err(AgentError::Summary)?;
        let summarizer = self
            .summarizer
            .as_ref()
            .ok_or(AgentError::Summary(SummaryError::NotConfigured))?;
        ensure_not_interrupted(&interrupt)?;
        let text = tokio::select! {
            () = interrupt.cancelled() => return Err(AgentError::Interrupted),
            result = summarizer.summarize(&original.messages()[..end]) => result.map_err(AgentError::Summary)?,
        };
        ensure_not_interrupted(&interrupt)?;
        let summary =
            ContextSummary::new(text, &original.messages()[..end]).map_err(AgentError::Summary)?;
        let old = Self::summary_input(original.messages(), original.context_summary())?;
        let new = Self::summary_input(original.messages(), Some(&summary))?;
        if serde_json::to_vec(&new)
            .map_err(|_| AgentError::Summary(SummaryError::InvalidResponse))?
            .len()
            >= serde_json::to_vec(&old)
                .map_err(|_| AgentError::Summary(SummaryError::InvalidResponse))?
                .len()
        {
            return Err(AgentError::Summary(SummaryError::NotSmaller));
        }
        Ok(Some(summary))
    }

    /// Removes only the summary. Raw history remains available, subject to the
    /// configured context policy and token budget. Does not call any model.
    #[must_use]
    pub fn clear_context_summary(&self) -> AgentFuture<'_, ()> {
        Box::pin(async move {
            let _guard = self
                .context_operations
                .clone()
                .try_write_owned()
                .map_err(|_| AgentError::Summary(SummaryError::Busy))?;
            let mut operation = self.begin_store_operation().await?;
            let state = self.snapshot_memory().await?.with_context_summary(None);
            self.commit_summary(&mut operation, state).await
        })
    }

    pub(super) async fn commit_summary(
        &self,
        operation: &mut StateOperation,
        state: crate::AgentState,
    ) -> AgentResult<()> {
        let summary = state.context_summary().cloned();
        if let Some(binding) = &operation.binding {
            let record = binding
                .store
                .save(binding.key.clone(), operation.expected_revision, state)
                .await?;
            operation.expected_revision = Some(record.revision());
        }
        // No await between a successful durable write and installing the summary.
        *lock(&self.context_summary) = summary;
        Ok(())
    }

    pub(super) fn summary_projection(
        history: &[Msg],
        summary: Option<&ContextSummary>,
    ) -> AgentResult<Vec<Msg>> {
        let Some(summary) = summary else {
            return Ok(history.to_vec());
        };
        summary.validate(history).map_err(AgentError::Summary)?;
        Ok(history[..summary.covered_messages()]
            .iter()
            .filter(|m| m.role == Role::System)
            .chain(history[summary.covered_messages()..].iter())
            .cloned()
            .collect())
    }

    fn summary_input(history: &[Msg], summary: Option<&ContextSummary>) -> AgentResult<Vec<Msg>> {
        Ok(summary
            .map(ContextSummary::message)
            .into_iter()
            .chain(Self::summary_projection(history, summary)?)
            .collect())
    }
}
