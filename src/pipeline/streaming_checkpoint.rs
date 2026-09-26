//! Streaming execution backed by durable stage-boundary checkpoints.

use super::{
    PIPELINE_CHECKPOINT_VERSION, PipelineCheckpoint, PipelineCheckpointStatus, PipelineEvent,
    PipelineEventStream, PipelineFailure, PipelineOutput, PipelineRecord, PipelineStep,
    PipelineStore, PipelineStreamFuture, SequentialPipeline, Stage, agent_terminal, busy_error,
    checkpoint_error, handoff,
};
use crate::{AgentError, AgentEvent, Msg, StateKey, agent::AgentInterruptToken};
use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use std::pin::Pin;
use tokio::sync::MutexGuard;

enum StageItem {
    Event(AgentEvent),
    Finished(Msg),
    Failed(PipelineFailure),
}

type StageItemStream<'a> = Pin<Box<dyn Stream<Item = StageItem> + Send + 'a>>;

impl SequentialPipeline {
    /// Starts a new checkpointed streaming run. The initial `Ready` checkpoint
    /// is created while awaiting this method, but no agent is invoked until the
    /// returned stream is polled. Before each `StageStarted`, `InFlight` is
    /// committed; `StageCompleted` is emitted only after the completed boundary
    /// is durably committed. Dropping an active stage leaves `InFlight`.
    /// # Errors
    /// Busy or failure to create the initial checkpoint.
    #[must_use]
    pub fn stream_checkpointed<'a>(
        &'a self,
        store: &'a dyn PipelineStore,
        key: StateKey,
        input: Msg,
    ) -> PipelineStreamFuture<'a> {
        Box::pin(async move {
            let guard = self.operation.try_lock().map_err(|_| busy_error())?;
            let interrupt = self.interrupt.token();
            let checkpoint = PipelineCheckpoint {
                version: PIPELINE_CHECKPOINT_VERSION,
                agent_names: self.stages.iter().map(|stage| stage.name.clone()).collect(),
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
            Ok(checkpoint_event_stream(
                self, store, key, record, guard, interrupt,
            ))
        })
    }

    /// Streams from a committed `Ready` boundary without replaying completed
    /// stages. Awaiting reserves the run and validates the checkpoint; agents
    /// remain lazy until the returned stream is polled.
    /// # Errors
    /// Busy, missing/corrupt/incompatible/unsafe checkpoint, or store error.
    #[must_use]
    pub fn resume_checkpointed_stream<'a>(
        &'a self,
        store: &'a dyn PipelineStore,
        key: StateKey,
    ) -> PipelineStreamFuture<'a> {
        Box::pin(async move {
            let guard = self.operation.try_lock().map_err(|_| busy_error())?;
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
            self.validate_checkpoint(&record.checkpoint, PipelineCheckpointStatus::Ready)?;
            Ok(checkpoint_event_stream(
                self, store, key, record, guard, interrupt,
            ))
        })
    }
}

fn checkpoint_event_stream<'a>(
    pipeline: &'a SequentialPipeline,
    store: &'a dyn PipelineStore,
    key: StateKey,
    mut record: PipelineRecord,
    guard: MutexGuard<'a, ()>,
    interrupt: AgentInterruptToken,
) -> PipelineEventStream<'a> {
    Box::pin(stream! {
        let _guard = guard;
        let start = record.checkpoint.completed.len();
        for index in start..pipeline.stages.len() {
            let stage = &pipeline.stages[index];
            let pipeline_step = index + 1;
            let agent_name = stage.name.clone();
            let completed_before = record.checkpoint.completed.clone();
            if interrupt.is_interrupted() {
                yield failure_event(&record, pipeline_step, &agent_name, PipelineFailure::Interrupted);
                return;
            }
            record.checkpoint.status = PipelineCheckpointStatus::InFlight;
            record = match store.save(key.clone(), Some(record.revision), record.checkpoint).await {
                Ok(record) => record,
                Err(error) => {
                    yield PipelineEvent::Error { error: checkpoint_error(
                        Some(pipeline_step), Some(agent_name), completed_before,
                        PipelineFailure::Store(error.to_string()),
                    ) };
                    return;
                }
            };
            yield PipelineEvent::StageStarted { pipeline_step, agent_name: agent_name.clone() };
            let input = record.checkpoint.next_input.clone();
            let mut stage_events = agent_stage_stream(stage, input, interrupt.clone());
            let message = loop {
                match stage_events.next().await {
                    Some(StageItem::Event(event)) => yield PipelineEvent::Agent {
                        pipeline_step,
                        agent_name: agent_name.clone(),
                        event,
                    },
                    Some(StageItem::Finished(message)) => break message,
                    Some(StageItem::Failed(cause)) => {
                        yield failure_event(&record, pipeline_step, &agent_name, cause);
                        return;
                    }
                    None => unreachable!("stage stream always emits a terminal item"),
                }
            };
            drop(stage_events);
            let next_input = if pipeline_step == pipeline.stages.len() {
                record.checkpoint.next_input.clone()
            } else {
                match handoff(&agent_name, &message) {
                    Ok(input) => input,
                    Err(reason) => {
                        yield failure_event(&record, pipeline_step, &agent_name, PipelineFailure::InvalidHandoff(reason));
                        return;
                    }
                }
            };
            let stage_result = PipelineStep {
                step: pipeline_step,
                agent_name: agent_name.clone(),
                message: message.clone(),
            };
            record.checkpoint.completed.push(stage_result.clone());
            record.checkpoint.next_input = next_input;
            record.checkpoint.status = if pipeline_step == pipeline.stages.len() {
                PipelineCheckpointStatus::Finished
            } else {
                PipelineCheckpointStatus::Ready
            };
            record = match store.save(key.clone(), Some(record.revision), record.checkpoint).await {
                Ok(record) => record,
                Err(error) => {
                    yield PipelineEvent::Error { error: checkpoint_error(
                        Some(pipeline_step), Some(agent_name),
                        completed_before,
                        PipelineFailure::Store(error.to_string()),
                    ) };
                    return;
                }
            };
            yield PipelineEvent::StageCompleted { stage: stage_result };
            if pipeline_step == pipeline.stages.len() {
                yield PipelineEvent::Finished { output: PipelineOutput {
                    message,
                    steps: record.checkpoint.completed,
                } };
                return;
            }
        }
    })
}

fn agent_stage_stream(
    stage: &Stage,
    input: Msg,
    mut interrupt: AgentInterruptToken,
) -> StageItemStream<'_> {
    Box::pin(stream! {
        let started = tokio::select! {
            biased;
            () = interrupt.cancelled() => Err(PipelineFailure::Interrupted),
            result = stage.agent.stream(input) => result.map_err(|error| PipelineFailure::Agent(Box::new(error))),
        };
        let mut events = match started {
            Ok(events) => events,
            Err(cause) => { yield StageItem::Failed(cause); return; }
        };
        loop {
            let item = tokio::select! {
                biased;
                () = interrupt.cancelled() => { yield StageItem::Failed(PipelineFailure::Interrupted); return; }
                item = events.next() => item,
            };
            let Some(item) = item else {
                yield StageItem::Failed(PipelineFailure::Agent(Box::new(
                    AgentError::InvalidModelResponse("agent stream ended without a terminal event".into()),
                )));
                return;
            };
            match item {
                Err(error) => { yield StageItem::Failed(PipelineFailure::Agent(Box::new(error))); return; }
                Ok(event) => {
                    let terminal = agent_terminal(&event);
                    yield StageItem::Event(event);
                    match terminal {
                        Some(Ok(message)) => { yield StageItem::Finished(message); return; }
                        Some(Err(cause)) => { yield StageItem::Failed(cause); return; }
                        None => {}
                    }
                }
            }
        }
    })
}

fn failure_event(
    record: &PipelineRecord,
    step: usize,
    name: &str,
    cause: PipelineFailure,
) -> PipelineEvent {
    PipelineEvent::Error {
        error: checkpoint_error(
            Some(step),
            Some(name.to_owned()),
            record.checkpoint.completed.clone(),
            cause,
        ),
    }
}
