//! Lazy multiplexing with serialized durable branch-boundary commits.

use super::super::{
    ParallelBranchOutcome, ParallelBranchResult, ParallelError, ParallelEvent, ParallelEventStream,
    ParallelFailure, ParallelPipeline, ParallelStreamFuture,
    streaming::{BranchItem, branch_stream, failed_stream},
};
use super::{
    ParallelBranchCheckpoint, ParallelRecord, ParallelStore, checkpoint_error, load_record,
};
use crate::{Msg, StateKey, agent::AgentInterruptToken};
use async_stream::stream;
use futures_util::{StreamExt, stream::FuturesUnordered};
use std::sync::atomic::AtomicBool;
use tokio::sync::MutexGuard;

impl ParallelPipeline {
    /// Starts NEW checkpointed parallel streaming at an unused key. Awaiting
    /// creates all `Ready` branches and reserves the shared lock, without calling
    /// agents. Polling commits `InFlight` before each `BranchStarted`/dispatch.
    /// Original agent events interleave; `BranchFinished` is emitted only after
    /// its `Completed`/`Failed` state is committed. Agent-local terminal events
    /// alone are not durable acknowledgements. Failures do not cancel siblings.
    ///
    /// Dropping, interrupting or a store failure cancels active streams and stops
    /// queued dispatch. Uncommitted branches remain `InFlight`, never replayed
    /// automatically. Error diagnostics use the last acknowledged record. Store
    /// writes are awaited, not raced against interruption; after an ambiguous
    /// write error, reload and inspect the actual record before reconciliation.
    /// Pipeline and child-agent state stores are separate transactions.
    /// No background tasks or event queues are created; consumer polling drives
    /// execution. Poll through the pipeline terminal event for finalization.
    /// # Errors
    /// Busy or failure to create the initial checkpoint; runtime errors are events.
    #[must_use]
    pub fn stream_checkpointed<'a>(
        &'a self,
        store: &'a dyn ParallelStore,
        key: StateKey,
        input: Msg,
    ) -> ParallelStreamFuture<'a> {
        Box::pin(async move {
            let guard = self
                .operation
                .try_lock()
                .map_err(|_| checkpoint_error(None, ParallelFailure::Busy))?;
            let interrupt = self.interrupt.token();
            let record = store
                .save(key.clone(), None, self.new_checkpoint(input))
                .await
                .map_err(|error| {
                    checkpoint_error(None, ParallelFailure::Store(error.to_string()))
                })?;
            Ok(checkpoint_event_stream(
                self, store, key, record, guard, interrupt,
            ))
        })
    }

    /// Streams only committed `Ready` branches with the original input; completed
    /// and failed branches are skipped. Awaiting validates and reserves the run;
    /// no agent is called until polling. Any `InFlight` branch blocks all resume
    /// until externally verified and explicitly reconciled. Terminal progress
    /// is read with [`super::ParallelCheckpoint::finished_result`], not replayed.
    /// This does not restore child state, approve tools, or retry failed work.
    /// # Errors
    /// Busy, store error, or missing/incompatible/unsafe progress.
    #[must_use]
    pub fn resume_checkpointed_stream<'a>(
        &'a self,
        store: &'a dyn ParallelStore,
        key: StateKey,
    ) -> ParallelStreamFuture<'a> {
        Box::pin(async move {
            let guard = self
                .operation
                .try_lock()
                .map_err(|_| checkpoint_error(None, ParallelFailure::Busy))?;
            let interrupt = self.interrupt.token();
            let record = load_record(store, &key).await?;
            self.validate_resume(&record)?;
            Ok(checkpoint_event_stream(
                self, store, key, record, guard, interrupt,
            ))
        })
    }
}

fn checkpoint_event_stream<'a>(
    pipeline: &'a ParallelPipeline,
    store: &'a dyn ParallelStore,
    key: StateKey,
    mut record: ParallelRecord,
    guard: MutexGuard<'a, ()>,
    mut interrupt: AgentInterruptToken,
) -> ParallelEventStream<'a> {
    Box::pin(stream! {
        let _guard = guard;
        let ready: Vec<_> = record.checkpoint.branches.iter().enumerate()
            .filter_map(|(index, branch)| matches!(branch, ParallelBranchCheckpoint::Ready).then_some(index))
            .collect();
        let started: Vec<_> = pipeline.branches.iter().map(|_| AtomicBool::new(false)).collect();
        let mut active = FuturesUnordered::new();
        let mut next = 0;
        let failure = 'run: loop {
            if interrupt.is_interrupted() {
                break Some(checkpoint_error(Some(&record.checkpoint), ParallelFailure::Interrupted));
            }
            while next < ready.len() && active.len() < pipeline.max_concurrency {
                if interrupt.is_interrupted() {
                    break 'run Some(checkpoint_error(Some(&record.checkpoint), ParallelFailure::Interrupted));
                }
                let index = ready[next];
                if let Err(error) = commit_branch(store, &key, &mut record, index, ParallelBranchCheckpoint::InFlight).await {
                    break 'run Some(error);
                }
                if interrupt.is_interrupted() {
                    break 'run Some(checkpoint_error(Some(&record.checkpoint), ParallelFailure::Interrupted));
                }
                active.push(branch_stream(index, &pipeline.branches[index], record.checkpoint.input.clone(),
                    &started[index], interrupt.clone()).into_future());
                next += 1;
            }
            if active.is_empty() { break None; }
            let selected = tokio::select! {
                biased;
                () = interrupt.cancelled() => break Some(checkpoint_error(Some(&record.checkpoint), ParallelFailure::Interrupted)),
                selected = active.next() => selected,
            };
            let Some((item, events)) = selected else { continue; };
            let index = events.index;
            let branch = index + 1;
            let agent_name = pipeline.branches[index].name.clone();
            let outcome = match item {
                Some(BranchItem::Started) => {
                    active.push(events.into_future());
                    yield ParallelEvent::BranchStarted { branch, agent_name };
                    continue;
                }
                Some(BranchItem::Agent(event)) => {
                    active.push(events.into_future());
                    yield ParallelEvent::Agent { branch, agent_name, event };
                    continue;
                }
                Some(BranchItem::Finished(outcome)) => outcome,
                None => {
                    if interrupt.is_interrupted() {
                        break Some(checkpoint_error(Some(&record.checkpoint), ParallelFailure::Interrupted));
                    }
                    failed_stream()
                }
            };
            drop(events);
            let state = match &outcome {
                ParallelBranchOutcome::Completed(message) => ParallelBranchCheckpoint::Completed(message.clone()),
                ParallelBranchOutcome::Failed(error) => ParallelBranchCheckpoint::Failed(error.clone()),
                _ => unreachable!("branch stream only returns terminal outcomes"),
            };
            if let Err(error) = commit_branch(store, &key, &mut record, index, state).await {
                break Some(error);
            }
            yield ParallelEvent::BranchFinished { result: ParallelBranchResult { branch, agent_name, outcome } };
        };
        // Cancel siblings BEFORE yielding a terminal event, even if the consumer
        // keeps the exhausted stream alive without polling it again.
        drop(active);
        let result = failure.map_or_else(|| record.checkpoint.finished_result().unwrap_or_else(|| {
            Err(checkpoint_error(Some(&record.checkpoint), ParallelFailure::UnsafeResume(
                "checkpoint is not structurally complete".into(),
            )))
        }), Err);
        match result {
            Ok(output) => yield ParallelEvent::Finished { output },
            Err(error) => yield ParallelEvent::Error { error },
        }
    })
}

async fn commit_branch(
    store: &dyn ParallelStore,
    key: &StateKey,
    record: &mut ParallelRecord,
    index: usize,
    state: ParallelBranchCheckpoint,
) -> Result<(), ParallelError> {
    let mut checkpoint = record.checkpoint.clone();
    checkpoint.branches[index] = state;
    *record = store
        .save(key.clone(), Some(record.revision), checkpoint)
        .await
        .map_err(|error| {
            checkpoint_error(
                Some(&record.checkpoint),
                ParallelFailure::Store(error.to_string()),
            )
        })?;
    Ok(())
}

#[cfg(test)]
mod tests;
