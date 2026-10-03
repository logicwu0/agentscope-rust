//! Single-writer checkpoint fences around bounded concurrent replies.

use super::super::{
    ParallelError, ParallelFailure, ParallelFuture, ParallelOutput, ParallelPipeline, branch_reply,
};
use super::{
    ParallelBranchCheckpoint, ParallelRecord, ParallelStore, checkpoint_error, load_record,
};
use crate::{Msg, StateKey, agent::AgentInterruptToken};
use futures_util::{StreamExt, stream::FuturesUnordered};
use std::sync::atomic::AtomicBool;

impl ParallelPipeline {
    /// Starts NEW non-streaming progress at an unused key. Each branch is marked
    /// `InFlight` before dispatch and its reply/error committed before reporting.
    /// Pipeline and agent stores are not one transaction; bind each agent to its
    /// own durable session if its conversation state must also survive restart.
    /// # Errors
    /// Busy, store errors, interruption or collected agent failures.
    #[must_use]
    pub fn run_checkpointed<'a>(
        &'a self,
        store: &'a dyn ParallelStore,
        key: StateKey,
        input: Msg,
    ) -> ParallelFuture<'a> {
        Box::pin(async move {
            let _guard = self
                .operation
                .try_lock()
                .map_err(|_| checkpoint_error(None, ParallelFailure::Busy))?;
            let interrupt = self.interrupt.token();
            let checkpoint = self.new_checkpoint(input);
            let record = store
                .save(key.clone(), None, checkpoint)
                .await
                .map_err(|error| {
                    checkpoint_error(None, ParallelFailure::Store(error.to_string()))
                })?;
            self.execute_checkpointed(store, key, record, interrupt)
                .await
        })
    }

    /// Resumes only undispatched `Ready` branches, with the original input.
    /// Completed and failed branches are skipped. Any `InFlight` branch blocks
    /// the entire resume until explicitly reconciled; terminal records are read
    /// using [`super::ParallelCheckpoint::finished_result`] instead of replayed.
    /// This does not approve tools, retry failures or restore agent state itself.
    /// # Errors
    /// Busy, storage failure, missing/incompatible/unsafe progress, interruption,
    /// or collected agent errors, including failures retained from earlier runs.
    #[must_use]
    pub fn resume_checkpointed<'a>(
        &'a self,
        store: &'a dyn ParallelStore,
        key: StateKey,
    ) -> ParallelFuture<'a> {
        Box::pin(async move {
            let _guard = self
                .operation
                .try_lock()
                .map_err(|_| checkpoint_error(None, ParallelFailure::Busy))?;
            let interrupt = self.interrupt.token();
            let record = load_record(store, &key).await?;
            self.validate_resume(&record)?;
            self.execute_checkpointed(store, key, record, interrupt)
                .await
        })
    }

    async fn execute_checkpointed(
        &self,
        store: &dyn ParallelStore,
        key: StateKey,
        mut record: ParallelRecord,
        mut interrupt: AgentInterruptToken,
    ) -> Result<ParallelOutput, ParallelError> {
        let ready: Vec<_> = record
            .checkpoint
            .branches
            .iter()
            .enumerate()
            .filter_map(|(index, branch)| {
                matches!(branch, ParallelBranchCheckpoint::Ready).then_some(index)
            })
            .collect();
        let started: Vec<_> = self
            .branches
            .iter()
            .map(|_| AtomicBool::new(false))
            .collect();
        let mut active = FuturesUnordered::new();
        let mut next = 0;
        loop {
            if interrupt.is_interrupted() {
                return Err(checkpoint_error(
                    Some(&record.checkpoint),
                    ParallelFailure::Interrupted,
                ));
            }
            while next < ready.len() && active.len() < self.max_concurrency {
                if interrupt.is_interrupted() {
                    return Err(checkpoint_error(
                        Some(&record.checkpoint),
                        ParallelFailure::Interrupted,
                    ));
                }
                let index = ready[next];
                let mut checkpoint = record.checkpoint.clone();
                checkpoint.branches[index] = ParallelBranchCheckpoint::InFlight;
                record = store
                    .save(key.clone(), Some(record.revision), checkpoint)
                    .await
                    .map_err(|error| {
                        checkpoint_error(
                            Some(&record.checkpoint),
                            ParallelFailure::Store(error.to_string()),
                        )
                    })?;
                // Keep the committed marker even when cancellation wins before
                // invocation. It is conservative evidence, never replay authority.
                if interrupt.is_interrupted() {
                    return Err(checkpoint_error(
                        Some(&record.checkpoint),
                        ParallelFailure::Interrupted,
                    ));
                }
                active.push(branch_reply(
                    index,
                    &self.branches[index],
                    record.checkpoint.input.clone(),
                    &started[index],
                    interrupt.clone(),
                ));
                next += 1;
            }
            if active.is_empty() {
                return record.checkpoint.finished_result().unwrap_or_else(|| {
                    Err(checkpoint_error(
                        Some(&record.checkpoint),
                        ParallelFailure::UnsafeResume(
                            "checkpoint is not structurally complete".into(),
                        ),
                    ))
                });
            }
            let item = tokio::select! {
                biased;
                () = interrupt.cancelled() => return Err(checkpoint_error(Some(&record.checkpoint), ParallelFailure::Interrupted)),
                item = active.next() => item,
            };
            let Some((index, result)) = item else {
                continue;
            };
            let Some(result) = result else {
                return Err(checkpoint_error(
                    Some(&record.checkpoint),
                    ParallelFailure::Interrupted,
                ));
            };
            let mut checkpoint = record.checkpoint.clone();
            checkpoint.branches[index] = match result {
                Ok(message) => ParallelBranchCheckpoint::Completed(message),
                Err(error) => ParallelBranchCheckpoint::Failed(Box::new(error)),
            };
            record = store
                .save(key.clone(), Some(record.revision), checkpoint)
                .await
                .map_err(|error| {
                    checkpoint_error(
                        Some(&record.checkpoint),
                        ParallelFailure::Store(error.to_string()),
                    )
                })?;
        }
    }
}
