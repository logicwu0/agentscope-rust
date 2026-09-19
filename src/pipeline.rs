//! Minimal sequential agent orchestration. No shared memory or automatic replay.

mod checkpoint;
mod error;
mod event;
pub use checkpoint::{
    InMemoryPipelineStore, PIPELINE_CHECKPOINT_VERSION, PipelineCheckpoint,
    PipelineCheckpointStatus, PipelineRecord, PipelineStore, PipelineStoreError,
    PipelineStoreFuture,
};
pub use error::{PipelineConfigError, PipelineError, PipelineFailure};
pub use event::PipelineEvent;

use crate::{
    Agent, AgentError, AgentEvent, AgentInterruptHandle, ContentBlock, Msg, Role, StateKey,
};
use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, future::Future, pin::Pin, sync::Arc};
use tokio::sync::Mutex;

/// One completed stage. Indices are one-based pipeline stages, not `ReAct` steps.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PipelineStep {
    /// One-based position in the fixed stage list.
    pub step: usize,
    /// Configured stage agent name, captured when constructing the pipeline.
    pub agent_name: String,
    /// Original reply; may include content that is not passed to the next agent.
    pub message: Msg,
}

/// Completed run with the last reply and all observed stage results.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PipelineOutput {
    /// Last agent's original reply, not a synthesized message.
    pub message: Msg,
    /// Original completed outputs; may contain private metadata/thinking.
    pub steps: Vec<PipelineStep>,
}

/// A lazy, cancellable sequential run.
pub type PipelineFuture<'a> =
    Pin<Box<dyn Future<Output = Result<PipelineOutput, PipelineError>> + Send + 'a>>;

/// Pipeline event stream. Runtime failures are terminal [`PipelineEvent::Error`]
/// values so already emitted stage events remain available to the caller.
pub type PipelineEventStream<'a> = Pin<Box<dyn Stream<Item = PipelineEvent> + Send + 'a>>;

/// Lazy preparation of a pipeline stream. Only an overlapping-run `Busy` error
/// is returned before a stream exists; agents start when the stream is polled.
pub type PipelineStreamFuture<'a> =
    Pin<Box<dyn Future<Output = Result<PipelineEventStream<'a>, PipelineError>> + Send + 'a>>;

struct Stage {
    name: String,
    agent: Arc<dyn Agent>,
}

/// Runs each agent once in order, forwarding only visible text as new user input.
///
/// This is an orchestrator, not an `Agent`: no combined history, state store or
/// recovery API is implied. Supply independent agents/memories/session keys.
/// Deliberately shared backing memory cannot be detected through `dyn Agent`.
/// Do not run the same agents independently while this pipeline uses them.
/// Clones share a run lock and interrupt handle; overlapping runs fail as Busy.
#[derive(Clone)]
pub struct SequentialPipeline {
    stages: Arc<Vec<Stage>>,
    operation: Arc<Mutex<()>>,
    interrupt: AgentInterruptHandle,
}

impl SequentialPipeline {
    /// Creates a fixed-order pipeline with nonempty, unique agent names.
    /// # Errors
    /// Rejects an empty list or blank/duplicate names. Does not invoke agents.
    pub fn new(agents: Vec<Arc<dyn Agent>>) -> Result<Self, PipelineConfigError> {
        if agents.is_empty() {
            return Err(PipelineConfigError::Empty);
        }
        let mut names = BTreeSet::new();
        let mut stages = Vec::with_capacity(agents.len());
        for (index, agent) in agents.into_iter().enumerate() {
            let name = agent.name().to_owned();
            if name.trim().is_empty() {
                return Err(PipelineConfigError::EmptyName { step: index + 1 });
            }
            if !names.insert(name.clone()) {
                return Err(PipelineConfigError::DuplicateName(name));
            }
            stages.push(Stage { name, agent });
        }
        Ok(Self {
            stages: Arc::new(stages),
            operation: Arc::new(Mutex::new(())),
            interrupt: AgentInterruptHandle::new(),
        })
    }

    /// Interrupts active pipeline runs only. Future runs use a new signal baseline.
    /// It drops the active reply future, not unrelated uses of an agent's handle.
    /// Already committed state/tool effects are not undone.
    #[must_use]
    pub fn interrupt_handle(&self) -> AgentInterruptHandle {
        self.interrupt.clone()
    }

    /// Runs from stage one; every invocation is a NEW run, never a resume.
    ///
    /// The first agent receives the supplied message unchanged. Later agents get
    /// a fresh User message named after the preceding stage, containing its text
    /// blocks joined by newline. Thinking, metadata, usage and history are not
    /// forwarded. Mixed non-text outputs are rejected rather than silently lost.
    ///
    /// No agent is called until this future is polled. Dropping it stops dispatch
    /// and releases the lock, but cannot undo completed or in-flight side effects.
    /// Agent-level persistence/cancellation guarantees still apply; there is no
    /// pipeline transaction or automatic checkpoint/save/rollback on cancellation.
    /// # Errors
    /// Busy, interruption, agent failure (including confirmation/in-doubt), or an
    /// unusable intermediate reply. Completed replies accompany every failure.
    #[must_use]
    pub fn run(&self, input: Msg) -> PipelineFuture<'_> {
        Box::pin(async move {
            let _guard = self.operation.try_lock().map_err(|_| PipelineError {
                step: None,
                agent_name: None,
                completed: Vec::new(),
                cause: PipelineFailure::Busy,
            })?;
            let mut interrupt = self.interrupt.token();
            let mut completed = Vec::with_capacity(self.stages.len());
            let mut incoming = input;
            for (index, stage) in self.stages.iter().enumerate() {
                // `biased` ensures an observed cancellation wins over a ready
                // reply and prevents dispatch of subsequent stages.
                let result = tokio::select! {
                    biased;
                    () = interrupt.cancelled() => Err(PipelineFailure::Interrupted),
                    result = async { stage.agent.reply(incoming).await } => result.map_err(|e| PipelineFailure::Agent(Box::new(e))),
                };
                let message = result.map_err(|cause| PipelineError {
                    step: Some(index + 1),
                    agent_name: Some(stage.name.clone()),
                    completed: std::mem::take(&mut completed),
                    cause,
                })?;
                completed.push(PipelineStep {
                    step: index + 1,
                    agent_name: stage.name.clone(),
                    message: message.clone(),
                });
                if index + 1 == self.stages.len() {
                    return Ok(PipelineOutput {
                        message,
                        steps: completed,
                    });
                }
                incoming = handoff(&stage.name, &message).map_err(|reason| PipelineError {
                    // This stage completed but its output cannot be handed off.
                    step: Some(index + 1),
                    agent_name: Some(stage.name.clone()),
                    completed: std::mem::take(&mut completed),
                    cause: PipelineFailure::InvalidHandoff(reason),
                })?;
            }
            unreachable!("constructor rejects empty pipelines")
        })
    }

    /// Streams every stage through the same fixed-order and handoff rules as
    /// [`Self::run`]. The stream owns the pipeline run lock until it terminates or
    /// is dropped. Awaiting this method reserves the run but invokes no agent;
    /// the first [`PipelineEvent::StageStarted`] and the agent start are lazy.
    ///
    /// Wrapped [`AgentEvent`] values preserve their agent-local `ReAct` `step`.
    /// `pipeline_step` is separately one-based. An agent `Finished` event is
    /// followed by `StageCompleted`; after the last stage, `Finished` contains
    /// the complete output. Confirmation and agent `Error` events are forwarded,
    /// then followed by one terminal pipeline `Error`. No later stage starts.
    ///
    /// Consumers must poll through a terminal pipeline event for agent state-store
    /// finalization. Dropping the stream stops dispatch and releases the run lock,
    /// but cannot roll back memory, durable state, or external tool effects.
    /// # Errors
    /// Returns `Busy` only when another run/stream owns this pipeline or a clone.
    #[must_use]
    pub fn stream(&self, input: Msg) -> PipelineStreamFuture<'_> {
        Box::pin(async move {
            let guard = self.operation.try_lock().map_err(|_| busy_error())?;
            let mut interrupt = self.interrupt.token();
            Ok(Box::pin(stream! {
                let _guard = guard;
                let mut completed = Vec::with_capacity(self.stages.len());
                let mut incoming = input;
                for (index, stage) in self.stages.iter().enumerate() {
                    let pipeline_step = index + 1;
                    let agent_name = stage.name.clone();
                    if interrupt.is_interrupted() {
                        yield pipeline_error_event(pipeline_step, &agent_name, &mut completed, PipelineFailure::Interrupted);
                        return;
                    }
                    yield PipelineEvent::StageStarted { pipeline_step, agent_name: agent_name.clone() };
                    let started = tokio::select! {
                        biased;
                        () = interrupt.cancelled() => Err(PipelineFailure::Interrupted),
                        result = stage.agent.stream(incoming) => result.map_err(|e| PipelineFailure::Agent(Box::new(e))),
                    };
                    let mut events = match started {
                        Ok(events) => events,
                        Err(cause) => {
                            yield pipeline_error_event(pipeline_step, &agent_name, &mut completed, cause);
                            return;
                        }
                    };
                    let message = loop {
                        let item = tokio::select! {
                            biased;
                            () = interrupt.cancelled() => {
                                yield pipeline_error_event(pipeline_step, &agent_name, &mut completed, PipelineFailure::Interrupted);
                                return;
                            }
                            item = events.next() => item,
                        };
                        let Some(item) = item else {
                            yield pipeline_error_event(
                                pipeline_step, &agent_name, &mut completed,
                                PipelineFailure::Agent(Box::new(AgentError::InvalidModelResponse(
                                    "agent stream ended without a terminal event".into(),
                                ))),
                            );
                            return;
                        };
                        match item {
                            Err(error) => {
                                yield pipeline_error_event(
                                    pipeline_step, &agent_name, &mut completed,
                                    PipelineFailure::Agent(Box::new(error)),
                                );
                                return;
                            }
                            Ok(event) => {
                                let terminal = agent_terminal(&event);
                                yield PipelineEvent::Agent {
                                    pipeline_step,
                                    agent_name: agent_name.clone(),
                                    event,
                                };
                                match terminal {
                                    Some(Ok(message)) => break message,
                                    Some(Err(cause)) => {
                                        yield pipeline_error_event(
                                            pipeline_step, &agent_name, &mut completed, cause,
                                        );
                                        return;
                                    }
                                    None => {}
                                }
                            }
                        }
                    };
                    drop(events);
                    let stage_result = PipelineStep {
                        step: pipeline_step,
                        agent_name: agent_name.clone(),
                        message: message.clone(),
                    };
                    completed.push(stage_result.clone());
                    yield PipelineEvent::StageCompleted { stage: stage_result };
                    if pipeline_step == self.stages.len() {
                        yield PipelineEvent::Finished {
                            output: PipelineOutput { message, steps: completed },
                        };
                        return;
                    }
                    incoming = match handoff(&agent_name, &message) {
                        Ok(message) => message,
                        Err(reason) => {
                            yield pipeline_error_event(
                                pipeline_step, &agent_name, &mut completed,
                                PipelineFailure::InvalidHandoff(reason),
                            );
                            return;
                        }
                    };
                }
            }) as PipelineEventStream<'_>)
        })
    }

    /// Starts a NEW revisioned run. Stages are marked in flight before dispatch
    /// and recorded complete before the next stage starts. A store conflict,
    /// crash, cancellation, or agent failure never authorizes automatic replay.
    /// Each agent must also have its own durable state binding if required.
    /// # Errors
    /// Busy, store errors, agent failures, or invalid handoffs.
    #[must_use]
    pub fn run_checkpointed<'a>(
        &'a self,
        store: &'a dyn PipelineStore,
        key: StateKey,
        input: Msg,
    ) -> PipelineFuture<'a> {
        Box::pin(async move {
            let _guard = self.operation.try_lock().map_err(|_| busy_error())?;
            let interrupt = self.interrupt.token();
            let names = self.stages.iter().map(|stage| stage.name.clone()).collect();
            let checkpoint = PipelineCheckpoint {
                version: PIPELINE_CHECKPOINT_VERSION,
                agent_names: names,
                completed: Vec::new(),
                next_input: input,
                status: PipelineCheckpointStatus::Ready,
            };
            let record = store
                .save(key.clone(), None, checkpoint)
                .await
                .map_err(|error| {
                    checkpoint_error(
                        None,
                        None,
                        Vec::new(),
                        PipelineFailure::Store(error.to_string()),
                    )
                })?;
            self.execute_checkpointed(store, key, record, interrupt)
                .await
        })
    }

    /// Resumes only from a committed `Ready` boundary. An `InFlight` checkpoint
    /// is deliberately rejected because that stage may have had side effects.
    /// This never resumes an agent's own pending confirmation or execution.
    /// # Errors
    /// Busy, missing/corrupt/incompatible/unsafe checkpoint, store or agent error.
    #[must_use]
    pub fn resume_checkpointed<'a>(
        &'a self,
        store: &'a dyn PipelineStore,
        key: StateKey,
    ) -> PipelineFuture<'a> {
        Box::pin(async move {
            let _guard = self.operation.try_lock().map_err(|_| busy_error())?;
            let interrupt = self.interrupt.token();
            let record = store
                .load(&key)
                .await
                .map_err(|error| {
                    checkpoint_error(
                        None,
                        None,
                        Vec::new(),
                        PipelineFailure::Store(error.to_string()),
                    )
                })?
                .ok_or_else(|| {
                    checkpoint_error(
                        None,
                        None,
                        Vec::new(),
                        PipelineFailure::UnsafeResume("checkpoint does not exist".into()),
                    )
                })?;
            if record.revision == 0 {
                return Err(checkpoint_error(
                    None,
                    None,
                    record.checkpoint.completed.clone(),
                    PipelineFailure::UnsafeResume("checkpoint revision is zero".into()),
                ));
            }
            self.validate_checkpoint(&record.checkpoint)?;
            self.execute_checkpointed(store, key, record, interrupt)
                .await
        })
    }

    fn validate_checkpoint(&self, checkpoint: &PipelineCheckpoint) -> Result<(), PipelineError> {
        let names: Vec<_> = self.stages.iter().map(|stage| stage.name.clone()).collect();
        let valid = checkpoint.version == PIPELINE_CHECKPOINT_VERSION
            && checkpoint.agent_names == names
            && checkpoint.status == PipelineCheckpointStatus::Ready
            && checkpoint.completed.len() < self.stages.len()
            && checkpoint
                .completed
                .iter()
                .enumerate()
                .all(|(index, step)| step.step == index + 1 && step.agent_name == names[index]);
        if !valid {
            return Err(checkpoint_error(None, None, checkpoint.completed.clone(),
                PipelineFailure::UnsafeResume("checkpoint is in flight, finished, corrupt, or configured for another pipeline".into())));
        }
        Ok(())
    }

    async fn execute_checkpointed(
        &self,
        store: &dyn PipelineStore,
        key: StateKey,
        mut record: PipelineRecord,
        mut interrupt: crate::agent::AgentInterruptToken,
    ) -> Result<PipelineOutput, PipelineError> {
        let start = record.checkpoint.completed.len();
        for index in start..self.stages.len() {
            let stage = &self.stages[index];
            let completed = record.checkpoint.completed.clone();
            let failure = |cause| {
                checkpoint_error(
                    Some(index + 1),
                    Some(stage.name.clone()),
                    completed.clone(),
                    cause,
                )
            };
            if interrupt.is_interrupted() {
                return Err(failure(PipelineFailure::Interrupted));
            }
            record.checkpoint.status = PipelineCheckpointStatus::InFlight;
            record = store
                .save(key.clone(), Some(record.revision), record.checkpoint)
                .await
                .map_err(|error| failure(PipelineFailure::Store(error.to_string())))?;
            let input = record.checkpoint.next_input.clone();
            let message = tokio::select! {
                biased;
                () = interrupt.cancelled() => Err(PipelineFailure::Interrupted),
                result = stage.agent.reply(input) => result.map_err(|error| PipelineFailure::Agent(Box::new(error))),
            }.map_err(failure)?;
            let next_input = if index + 1 == self.stages.len() {
                record.checkpoint.next_input.clone()
            } else {
                handoff(&stage.name, &message)
                    .map_err(|reason| failure(PipelineFailure::InvalidHandoff(reason)))?
            };
            record.checkpoint.completed.push(PipelineStep {
                step: index + 1,
                agent_name: stage.name.clone(),
                message: message.clone(),
            });
            record.checkpoint.next_input = next_input;
            record.checkpoint.status = if index + 1 == self.stages.len() {
                PipelineCheckpointStatus::Finished
            } else {
                PipelineCheckpointStatus::Ready
            };
            record = store
                .save(key.clone(), Some(record.revision), record.checkpoint)
                .await
                .map_err(|error| {
                    checkpoint_error(
                        Some(index + 1),
                        Some(stage.name.clone()),
                        completed,
                        PipelineFailure::Store(error.to_string()),
                    )
                })?;
            if index + 1 == self.stages.len() {
                return Ok(PipelineOutput {
                    message,
                    steps: record.checkpoint.completed,
                });
            }
        }
        unreachable!("validated checkpoint has a remaining stage")
    }
}

fn checkpoint_error(
    step: Option<usize>,
    agent_name: Option<String>,
    completed: Vec<PipelineStep>,
    cause: PipelineFailure,
) -> PipelineError {
    PipelineError {
        step,
        agent_name,
        completed,
        cause,
    }
}

fn busy_error() -> PipelineError {
    PipelineError {
        step: None,
        agent_name: None,
        completed: Vec::new(),
        cause: PipelineFailure::Busy,
    }
}

fn agent_terminal(event: &AgentEvent) -> Option<Result<Msg, PipelineFailure>> {
    match event {
        AgentEvent::Finished { message, .. } => Some(Ok(message.clone())),
        AgentEvent::ToolConfirmationRequired { checkpoint } => Some(Err(PipelineFailure::Agent(
            Box::new(AgentError::ToolConfirmationRequired {
                checkpoint: checkpoint.clone(),
            }),
        ))),
        AgentEvent::Error { error, .. } => {
            Some(Err(PipelineFailure::Agent(Box::new(error.clone()))))
        }
        _ => None,
    }
}

fn pipeline_error_event(
    step: usize,
    name: &str,
    completed: &mut Vec<PipelineStep>,
    cause: PipelineFailure,
) -> PipelineEvent {
    PipelineEvent::Error {
        error: pipeline_error(step, name, completed, cause),
    }
}

fn pipeline_error(
    step: usize,
    name: &str,
    completed: &mut Vec<PipelineStep>,
    cause: PipelineFailure,
) -> PipelineError {
    PipelineError {
        step: Some(step),
        agent_name: Some(name.to_owned()),
        completed: std::mem::take(completed),
        cause,
    }
}

fn handoff(source: &str, message: &Msg) -> Result<Msg, String> {
    let mut texts = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text(text) => texts.push(text.text.as_str()),
            ContentBlock::Thinking(_) => {}
            ContentBlock::ToolCall(_)
            | ContentBlock::ToolResult(_)
            | ContentBlock::Data(_)
            | ContentBlock::StructuredOutput(_) => {
                return Err("v1 pipeline handoff supports visible text only".into());
            }
        }
    }
    let text = texts.join("\n");
    if text.trim().is_empty() {
        return Err("agent reply has no visible text to hand off".into());
    }
    Ok(Msg::new(source, Role::User, [ContentBlock::from(text)]))
}

#[cfg(test)]
mod tests;
