//! Opt-in, once-per-new-reply compaction; never entered by recovery or tool steps.

use super::{ReActAgent, StateOperation, ensure_not_interrupted, lock};
use crate::agent::interrupt::AgentInterruptToken;
use crate::{
    AgentError, AgentEvent, AgentEventStream, AgentFuture, AgentHookEvent, AgentResult, Msg,
    SummaryError, TokenBudgetError,
};
use async_stream::stream;
use futures_util::StreamExt;

impl ReActAgent {
    /// Enables one automatic compaction attempt before each new reply if the
    /// selected context plus incoming message exceeds the configured token budget.
    /// Retains this many existing recent turns, plus the incoming message. Requires
    /// memory, an explicit summarizer and token budget. Disabled by default.
    ///
    /// This opt-in permits extra summary-model calls/cost. It is runtime config,
    /// not persisted. Recovery paths and intermediate tool steps do not trigger it.
    ///
    /// # Errors
    /// Returns a summary configuration error for zero retained turns.
    pub fn with_auto_compaction(mut self, keep_recent_turns: usize) -> AgentResult<Self> {
        if keep_recent_turns == 0 {
            return Err(AgentError::Summary(SummaryError::ZeroRecentTurns));
        }
        self.auto_compaction = Some(keep_recent_turns);
        Ok(self)
    }

    /// Disables automatic calls without removing summaries or the summarizer.
    #[must_use]
    pub fn without_auto_compaction(mut self) -> Self {
        self.auto_compaction = None;
        self
    }

    async fn begin_auto_reply(
        &self,
        message: &Msg,
    ) -> AgentResult<(StateOperation, AgentInterruptToken)> {
        // Hold exclusive access from budget check through the reply. This avoids
        // read-to-write lock upgrades and racing another clone's compaction.
        let guard = self.context_operations.clone().write_owned().await;
        let mut operation = self.begin_store_operation().await?;
        operation.auto_guard = Some(guard);
        let interrupt = self.interrupt.token();
        if let Some(checkpoint) = lock(&self.pending_tool_execution).clone() {
            return Err(AgentError::ToolExecutionInDoubt { checkpoint });
        }
        if let Some(checkpoint) = lock(&self.pending_tool_calls).clone() {
            return Err(AgentError::ToolConfirmationRequired { checkpoint });
        }
        self.notify_hooks(&AgentHookEvent::BeforeReply {
            message: message.clone(),
        })
        .await?;
        ensure_not_interrupted(&interrupt)?;
        Ok((operation, interrupt))
    }

    pub(super) fn auto_reply(&self, message: Msg) -> AgentFuture<'_, Msg> {
        Box::pin(async move {
            let (mut operation, interrupt) = self.begin_auto_reply(&message).await?;
            {
                let mut events =
                    self.auto_compaction_events(&message, &mut operation, interrupt.clone());
                while let Some(event) = events.next().await {
                    event?;
                }
            }
            let result = async {
                let history = self.append_auto_input(message, &interrupt).await?;
                self.continue_reply(history, interrupt, 0).await
            }
            .await;
            self.finish_state_operation(Some(operation)).await?;
            result
        })
    }

    pub(super) fn auto_stream(&self, message: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
        Box::pin(async move {
            let (mut operation, interrupt) = self.begin_auto_reply(&message).await?;
            Ok(Box::pin(stream! {
                let mut failure = None;
                {
                    let mut events = self.auto_compaction_events(&message, &mut operation, interrupt.clone());
                    while let Some(event) = events.next().await {
                        match event {
                            Ok(event) => yield Ok(event),
                            Err(error) => { failure = Some(error); break; }
                        }
                    }
                }
                if let Some(error) = failure {
                    yield Ok(AgentEvent::Error { step: None, error });
                    return;
                }
                let history = match self.append_auto_input(message, &interrupt).await {
                    Ok(history) => history,
                    Err(error) => { yield Ok(AgentEvent::Error { step: None, error }); return; }
                };
                let events = self.agent_event_stream(history, self.system_prompt.as_ref().map(Msg::system), interrupt, 0);
                let mut events = self.finish_reply_stream(events, Some(operation));
                while let Some(event) = events.next().await { yield event; }
            }) as AgentEventStream<'_>)
        })
    }

    async fn append_auto_input(
        &self,
        message: Msg,
        interrupt: &AgentInterruptToken,
    ) -> AgentResult<Vec<Msg>> {
        ensure_not_interrupted(interrupt)?;
        let memory = self
            .memory
            .as_ref()
            .ok_or(AgentError::MemoryNotConfigured)?;
        let mut history = memory.messages().await?;
        memory.append(vec![message.clone()]).await?;
        history.push(message);
        Ok(history)
    }

    fn auto_compaction_events<'a>(
        &'a self,
        message: &'a Msg,
        operation: &'a mut StateOperation,
        interrupt: AgentInterruptToken,
    ) -> AgentEventStream<'a> {
        Box::pin(stream! {
            let result = self.auto_check(message).await;
            let (original, overflow) = match result {
                Ok(Some(plan)) => plan,
                Ok(None) => return,
                Err(error) => { yield Err(error); return; }
            };
            let keep = self.auto_compaction.unwrap_or(1);
            yield Ok(AgentEvent::ContextCompactionStarted { keep_recent_turns: keep });
            let result = async {
                let summary = self.summary_candidate(&original, keep, interrupt.clone()).await?
                    .ok_or(AgentError::TokenBudget(overflow))?;
                let mut history = original.messages().to_vec();
                history.push(message.clone());
                self.check_auto_budget(&history, Some(&summary)).await?;
                ensure_not_interrupted(&interrupt)?;
                if self.snapshot_memory().await? != original { return Err(AgentError::Summary(SummaryError::StaleSource)); }
                self.commit_summary(operation, original.with_context_summary(Some(summary.clone()))).await?;
                Ok(summary.covered_messages())
            }.await;
            match result {
                Ok(covered_messages) => yield Ok(AgentEvent::ContextCompactionCompleted { covered_messages }),
                Err(error) => {
                    yield Ok(AgentEvent::ContextCompactionFailed { error: error.clone() });
                    yield Err(error);
                }
            }
        })
    }

    async fn auto_check(
        &self,
        message: &Msg,
    ) -> AgentResult<Option<(crate::AgentState, TokenBudgetError)>> {
        if self.summarizer.is_none() {
            return Err(AgentError::Summary(SummaryError::NotConfigured));
        }
        let original = self.snapshot_memory().await?;
        let mut history = original.messages().to_vec();
        history.push(message.clone());
        match self
            .check_auto_budget(&history, original.context_summary())
            .await
        {
            Ok(()) => Ok(None),
            Err(AgentError::TokenBudget(error @ TokenBudgetError::Exceeded { .. })) => {
                Ok(Some((original, error)))
            }
            Err(error) => Err(error),
        }
    }

    async fn check_auto_budget(
        &self,
        history: &[Msg],
        summary: Option<&crate::ContextSummary>,
    ) -> AgentResult<()> {
        let budget = self
            .token_budget
            .as_ref()
            .ok_or(AgentError::Summary(SummaryError::AutoRequiresBudget))?;
        let system = self.system_prompt.as_ref().map(Msg::system);
        let (request, _) = self
            .unbudgeted_request(history, system.as_ref(), summary)
            .await?;
        budget
            .check_without_trimming(request)
            .map_err(AgentError::TokenBudget)
    }
}
