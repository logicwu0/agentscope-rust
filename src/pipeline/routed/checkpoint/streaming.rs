//! Selected-agent streaming with durable dispatch and terminal-result boundaries.

use super::super::streaming::terminal_outcome;
use super::{
    ROUTED_CHECKPOINT_VERSION, RoutedCheckpoint, RoutedCheckpointStatus, RoutedRecord, RoutedStore,
    Stage, checkpoint_error, load_record, save_record,
};
use crate::{
    AgentError, AgentEvent, Msg, RoutedError, RoutedEvent, RoutedEventStream, RoutedFailure,
    RoutedPipeline, RoutedStreamFuture, StateKey, agent::AgentInterruptToken,
};
use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use std::pin::Pin;
use tokio::sync::MutexGuard;

enum SelectedItem {
    Event(AgentEvent),
    Finished(Result<Msg, RoutedFailure>),
}

type SelectedStream<'a> = Pin<Box<dyn Stream<Item = SelectedItem> + Send + 'a>>;

impl RoutedPipeline {
    /// Starts NEW checkpointed streaming at an unused key. Awaiting validates
    /// the exact route, reserves the shared run/stream lock and creates `Ready`,
    /// without invoking an agent. Unknown routes are rejected before any store
    /// access, even while busy. Polling commits `InFlight` before `RouteStarted`;
    /// only a later poll invokes the selected agent's stream.
    ///
    /// Original agent events are forwarded unchanged. An agent terminal event
    /// alone is NOT a durable acknowledgement: continue polling to commit its
    /// `Completed`/`Failed` outcome and receive the routed `Finished`/`Error`.
    /// Once observed, the agent's terminal outcome survives a later interrupt.
    /// Startup errors, error items and EOF without an agent terminal also commit
    /// `Failed`. Pipeline interruption leaves undispatched `Ready` or fenced
    /// `InFlight` progress instead; neither interruption nor drop authorizes retry.
    ///
    /// Store writes are awaited, not raced against interruption. A failed or
    /// dropped write may already have committed: inspect the actual record.
    /// Agent state and route checkpoints are separate transactions. No tasks
    /// are spawned; consumer polling drives execution. The shared lock is
    /// released before the routed terminal event, or when the stream is dropped.
    /// # Errors
    /// Unknown route, Busy or initial storage failure. Runtime failures are events.
    #[must_use]
    pub fn stream_checkpointed<'a>(
        &'a self,
        store: &'a dyn RoutedStore,
        key: StateKey,
        route: impl Into<String>,
        input: Msg,
    ) -> RoutedStreamFuture<'a> {
        let route = route.into();
        Box::pin(async move {
            let Some(stage) = self.routes.get(&route) else {
                return Err(RoutedError {
                    route,
                    agent_name: None,
                    cause: RoutedFailure::UnknownRoute,
                });
            };
            let checkpoint = RoutedCheckpoint {
                version: ROUTED_CHECKPOINT_VERSION,
                route,
                agent_name: stage.name.clone(),
                input,
                status: RoutedCheckpointStatus::Ready,
            };
            let guard = self
                .operation
                .try_lock()
                .map_err(|_| checkpoint_error(Some(&checkpoint), RoutedFailure::Busy))?;
            let interrupt = self.interrupt.token();
            let record = save_record(store, key.clone(), None, checkpoint).await?;
            Ok(checkpoint_event_stream(
                stage, store, key, record, guard, interrupt,
            ))
        })
    }

    /// Streams ONLY a validated, committed `Ready` selection using its stored
    /// route and original input. Awaiting loads and validates progress while
    /// reserving the shared lock; the selected agent remains lazy until polling.
    /// Ready checkpoints created by non-streaming runs are compatible as well.
    ///
    /// `InFlight` requires external reconciliation; terminal progress is read
    /// with [`RoutedCheckpoint::finished_result`] rather than replayed. This
    /// does not restore agent state, approve tools or retry failed execution.
    /// # Errors
    /// Busy, storage failure, or missing/malformed/incompatible/unsafe progress.
    #[must_use]
    pub fn resume_checkpointed_stream<'a>(
        &'a self,
        store: &'a dyn RoutedStore,
        key: StateKey,
    ) -> RoutedStreamFuture<'a> {
        Box::pin(async move {
            let guard = self
                .operation
                .try_lock()
                .map_err(|_| checkpoint_error(None, RoutedFailure::Busy))?;
            let interrupt = self.interrupt.token();
            let record = load_record(store, &key).await?;
            let stage = self.validate_ready_checkpoint(&record)?;
            Ok(checkpoint_event_stream(
                stage, store, key, record, guard, interrupt,
            ))
        })
    }
}

fn checkpoint_event_stream<'a>(
    stage: &'a Stage,
    store: &'a dyn RoutedStore,
    key: StateKey,
    mut record: RoutedRecord,
    guard: MutexGuard<'a, ()>,
    interrupt: AgentInterruptToken,
) -> RoutedEventStream<'a> {
    Box::pin(stream! {
        let result = 'run: {
            if interrupt.is_interrupted() {
                break 'run Err(checkpoint_error(Some(&record.checkpoint), RoutedFailure::Interrupted));
            }
            if let Err(error) = commit_status(store, &key, &mut record, RoutedCheckpointStatus::InFlight).await {
                break 'run Err(error);
            }
            if interrupt.is_interrupted() {
                break 'run Err(checkpoint_error(Some(&record.checkpoint), RoutedFailure::Interrupted));
            }
            yield RoutedEvent::RouteStarted {
                route: record.checkpoint.route.clone(), agent_name: stage.name.clone(),
            };
            let mut selected = selected_stream(stage, record.checkpoint.input.clone(), interrupt);
            let outcome = loop {
                match selected.next().await {
                    Some(SelectedItem::Event(event)) => yield RoutedEvent::Agent {
                        route: record.checkpoint.route.clone(), agent_name: stage.name.clone(), event,
                    },
                    Some(SelectedItem::Finished(outcome)) => break outcome,
                    None => unreachable!("selected stream always yields a terminal outcome"),
                }
            };
            drop(selected);
            let status = match outcome {
                Ok(message) => RoutedCheckpointStatus::Completed(message),
                Err(RoutedFailure::Agent(error)) => RoutedCheckpointStatus::Failed(error),
                Err(cause) => break 'run Err(checkpoint_error(Some(&record.checkpoint), cause)),
            };
            // An observed terminal result is retained across the preceding yield
            // and this store await, even if the interrupt generation has changed.
            if let Err(error) = commit_status(store, &key, &mut record, status).await {
                break 'run Err(error);
            }
            record.checkpoint.finished_result().unwrap_or_else(|| {
                Err(checkpoint_error(Some(&record.checkpoint), RoutedFailure::UnsafeResume(
                    "checkpoint is not structurally complete".into(),
                )))
            })
        };
        drop(guard);
        yield match result {
            Ok(output) => RoutedEvent::Finished { output },
            Err(error) => RoutedEvent::Error { error },
        };
    })
}

fn selected_stream(
    stage: &Stage,
    input: Msg,
    mut interrupt: AgentInterruptToken,
) -> SelectedStream<'_> {
    Box::pin(stream! {
        let opened = tokio::select! {
            biased;
            () = interrupt.cancelled() => Err(RoutedFailure::Interrupted),
            // Avoid even synchronous stream() side effects when cancellation wins.
            opened = async { stage.agent.stream(input).await } => opened
                .map_err(|error| RoutedFailure::Agent(Box::new(error))),
        };
        let mut events = match opened {
            Ok(events) => events,
            Err(cause) => {
                yield SelectedItem::Finished(Err(cause));
                return;
            }
        };
        let outcome = loop {
            let item = tokio::select! {
                biased;
                () = interrupt.cancelled() => break Err(RoutedFailure::Interrupted),
                item = events.next() => item,
            };
            match item {
                None => break Err(RoutedFailure::Agent(Box::new(AgentError::InvalidModelResponse(
                    "agent stream ended without a terminal event".into(),
                )))),
                Some(Err(error)) => break Err(RoutedFailure::Agent(Box::new(error))),
                Some(Ok(event)) => {
                    if let Some(outcome) = terminal_outcome(&event) {
                        drop(events);
                        yield SelectedItem::Event(event);
                        // Do not consult cancellation or poll the child again:
                        // its observed terminal outcome must reach the commit.
                        yield SelectedItem::Finished(outcome);
                        return;
                    }
                    yield SelectedItem::Event(event);
                }
            }
        };
        drop(events);
        yield SelectedItem::Finished(outcome);
    })
}

async fn commit_status(
    store: &dyn RoutedStore,
    key: &StateKey,
    record: &mut RoutedRecord,
    status: RoutedCheckpointStatus,
) -> Result<(), RoutedError> {
    let mut checkpoint = record.checkpoint.clone();
    checkpoint.status = status;
    *record = save_record(store, key.clone(), Some(record.revision), checkpoint).await?;
    Ok(())
}

#[cfg(test)]
mod tests;
