//! Streaming confirmation, retry, and externally reconciled continuation.

use super::{
    ReActAgent, StateOperation, confirmation_map, ensure_not_interrupted, finish_checkpoint_calls,
    lock, reconciled_result_map, validate_checkpoint_message,
};
use crate::agent::{
    AgentError, AgentFuture, AgentResult, PendingToolCalls, PendingToolExecution, ToolConfirmation,
    ToolConfirmationDecision, interrupt::AgentInterruptToken,
};
use crate::{
    AgentEvent, AgentEventStream, AgentHookEvent, ContentBlock, Msg, Role, ToolCallBlock,
    ToolResultBlock,
};
use async_stream::stream;
use futures_util::StreamExt;
use std::collections::BTreeMap;

enum Recovery {
    Confirm(Vec<ToolConfirmation>),
    Retry,
    Resolve(Vec<ToolResultBlock>),
}

struct Plan {
    checkpoint: PendingToolCalls,
    execution: Option<PendingToolExecution>,
    decisions: BTreeMap<String, ToolConfirmationDecision>,
    approved: Vec<ToolCallBlock>,
    resolved: Option<Vec<ToolResultBlock>>,
}

impl ReActAgent {
    /// Streams tool execution and the reply after explicit confirmation.
    ///
    /// Validation errors are returned before a stream is created. Poll the
    /// stream to execute tools and through its terminal event to save final
    /// state. Dropping during tools retains the durable execution checkpoint;
    /// completed tool observations are saved before subsequent model deltas.
    #[must_use]
    pub fn stream_resume_tool_calls(
        &self,
        reply_id: impl Into<String>,
        confirmations: Vec<ToolConfirmation>,
    ) -> AgentFuture<'_, AgentEventStream<'_>> {
        self.start_recovery_stream(reply_id.into(), Recovery::Confirm(confirmations))
    }

    /// Explicitly retries approved calls with the original idempotency keys and
    /// streams their events and the continuing reply. Tools must deduplicate
    /// external effects. Poll through the terminal event for the final save.
    #[must_use]
    pub fn stream_retry_tool_execution(
        &self,
        reply_id: impl Into<String>,
    ) -> AgentFuture<'_, AgentEventStream<'_>> {
        self.start_recovery_stream(reply_id.into(), Recovery::Retry)
    }

    /// Supplies verified terminal results without executing tools, then streams
    /// the remaining reply. This updates agent state, not the idempotency store.
    /// Poll through the terminal event for the final save.
    #[must_use]
    pub fn stream_resolve_tool_execution(
        &self,
        reply_id: impl Into<String>,
        results: Vec<ToolResultBlock>,
    ) -> AgentFuture<'_, AgentEventStream<'_>> {
        self.start_recovery_stream(reply_id.into(), Recovery::Resolve(results))
    }

    fn start_recovery_stream(
        &self,
        reply_id: String,
        recovery: Recovery,
    ) -> AgentFuture<'_, AgentEventStream<'_>> {
        Box::pin(async move {
            let mut operation = self.begin_state_operation().await?;
            let plan = self.recovery_plan(&reply_id, recovery).await?;
            let interrupt = self.interrupt.token();
            Ok(Box::pin(stream! {
                let mut events = self.recovery_events(plan, &mut operation, interrupt);
                while let Some(event) = events.next().await {
                    let terminal = event.as_ref().map_or(true, |event| matches!(event,
                        AgentEvent::Finished { .. } | AgentEvent::ToolConfirmationRequired { .. } | AgentEvent::Error { .. }));
                    if terminal {
                        drop(events);
                        match self.finish_state_operation(operation).await {
                            Ok(()) => yield event,
                            Err(error) => yield Ok(AgentEvent::Error { step: None, error }),
                        }
                        return;
                    }
                    yield event;
                }
                drop(events);
                if let Err(error) = self.finish_state_operation(operation).await {
                    yield Ok(AgentEvent::Error { step: None, error });
                }
            }) as AgentEventStream<'_>)
        })
    }

    async fn recovery_plan(&self, reply_id: &str, recovery: Recovery) -> AgentResult<Plan> {
        let (checkpoint, execution, confirmations, supplied) = match recovery {
            Recovery::Confirm(confirmations) => {
                if let Some(checkpoint) = lock(&self.pending_tool_execution).clone() {
                    return Err(AgentError::ToolExecutionInDoubt { checkpoint });
                }
                let checkpoint = lock(&self.pending_tool_calls)
                    .clone()
                    .ok_or(AgentError::NoPendingToolConfirmation)?;
                if checkpoint.reply_id() != reply_id {
                    return Err(AgentError::InvalidToolConfirmation(
                        "reply id does not match pending confirmation".into(),
                    ));
                }
                confirmation_map(&checkpoint, confirmations.clone())?;
                let execution = confirmations
                    .iter()
                    .any(|decision| {
                        matches!(decision.decision(), ToolConfirmationDecision::Approve)
                    })
                    .then(|| {
                        PendingToolExecution::new(
                            uuid::Uuid::new_v4().simple().to_string(),
                            checkpoint.clone(),
                            confirmations.clone(),
                        )
                    });
                (checkpoint, execution, confirmations, None)
            }
            Recovery::Retry | Recovery::Resolve(_) => {
                let execution = lock(&self.pending_tool_execution)
                    .clone()
                    .ok_or(AgentError::NoPendingToolExecution)?;
                if execution.confirmation().reply_id() != reply_id {
                    return Err(AgentError::InvalidToolExecutionResolution(
                        "reply id does not match uncertain execution".into(),
                    ));
                }
                let supplied = if let Recovery::Resolve(results) = recovery {
                    Some(results)
                } else {
                    None
                };
                (
                    execution.confirmation().clone(),
                    Some(execution.clone()),
                    execution.decisions().to_vec(),
                    supplied,
                )
            }
        };
        let memory = self
            .memory
            .as_ref()
            .ok_or(AgentError::MemoryNotConfigured)?;
        validate_checkpoint_message(&memory.messages().await?, &checkpoint)?;
        if checkpoint.step() >= self.max_steps {
            return Err(AgentError::MaxStepsExceeded {
                max_steps: self.max_steps,
            });
        }
        let decisions = confirmation_map(&checkpoint, confirmations)?;
        let resolved = supplied
            .map(|results| reconciled_result_map(&checkpoint, &decisions, results))
            .transpose()?;
        let approved = checkpoint
            .calls()
            .iter()
            .filter(|call| {
                matches!(
                    decisions.get(call.id()),
                    Some(ToolConfirmationDecision::Approve)
                )
            })
            .cloned()
            .collect();
        Ok(Plan {
            checkpoint,
            execution,
            decisions,
            approved,
            resolved,
        })
    }

    fn recovery_events<'a>(
        &'a self,
        plan: Plan,
        operation: &'a mut Option<StateOperation>,
        mut interrupt: AgentInterruptToken,
    ) -> AgentEventStream<'a> {
        Box::pin(stream! {
            let step = plan.checkpoint.step();
            let setup = async {
                ensure_not_interrupted(&interrupt)?;
                if plan.resolved.is_none() {
                    self.notify_before_tool_calls(step, &plan.approved).await?;
                    ensure_not_interrupted(&interrupt)?;
                    if let Some(execution) = &plan.execution {
                        *lock(&self.pending_tool_calls) = None;
                        *lock(&self.pending_tool_execution) = Some(execution.clone());
                        self.persist_state_operation(operation).await?;
                    }
                }
                Ok::<_, AgentError>(())
            }.await;
            if let Err(error) = setup { yield Ok(failure(step, error)); return; }
            let results = if let Some(results) = plan.resolved {
                results
            } else {
                for call in &plan.approved {
                    yield Ok(AgentEvent::ToolStarted { step, call: call.clone() });
                }
                let execution = tokio::select! {
                    biased;
                    () = interrupt.cancelled(), if !plan.approved.is_empty() => {
                        Err(AgentError::ToolExecutionInDoubt { checkpoint: plan.execution.clone().expect("approved calls have a checkpoint") })
                    }
                    results = self.execute_confirmed_tools(&plan.approved, plan.execution.as_ref()) => results,
                };
                match execution.and_then(|results| reconciled_result_map(&plan.checkpoint, &plan.decisions, results)) {
                    Ok(results) => results,
                    Err(error) => { yield Ok(failure(step, error)); return; }
                }
            };
            let history = match self.commit_recovery_results(&plan.checkpoint, &results, operation).await {
                Ok(history) => history,
                Err(error) => { yield Ok(failure(step, error)); return; }
            };
            for result in results {
                if let Err(error) = self.notify_hooks(&AgentHookEvent::AfterToolCall { step, result: result.clone() }).await {
                    yield Ok(failure(step, error)); return;
                }
                yield Ok(AgentEvent::ToolFinished { step, result });
            }
            let mut continuation = self.agent_event_stream(history, self.system_prompt.as_ref().map(Msg::system), interrupt, step);
            while let Some(event) = continuation.next().await { yield event; }
        })
    }

    async fn commit_recovery_results(
        &self,
        checkpoint: &PendingToolCalls,
        results: &[ToolResultBlock],
        operation: &mut Option<StateOperation>,
    ) -> AgentResult<Vec<Msg>> {
        let memory = self
            .memory
            .as_ref()
            .ok_or(AgentError::MemoryNotConfigured)?;
        let mut history = memory.messages().await?;
        validate_checkpoint_message(&history, checkpoint)?;
        finish_checkpoint_calls(&mut history, checkpoint);
        history.push(Msg::new(
            "tool",
            Role::Assistant,
            results.iter().cloned().map(ContentBlock::from),
        ));
        // One atomic replacement keeps checkpoints and observations consistent
        // even when a later hook fails.
        memory.replace(history.clone()).await?;
        *lock(&self.pending_tool_calls) = None;
        *lock(&self.pending_tool_execution) = None;
        self.persist_state_operation(operation).await?;
        Ok(history)
    }
}

fn failure(step: usize, error: AgentError) -> AgentEvent {
    AgentEvent::Error {
        step: Some(step),
        error,
    }
}
