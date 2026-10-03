//! Revisioned progress for independent parallel branches.

mod execution;
mod reconciliation;
mod streaming;

use super::{
    ParallelBranchOutcome, ParallelBranchResult, ParallelError, ParallelFailure, ParallelOutput,
    ParallelPipeline,
};
use crate::{AgentError, Msg, PipelineStoreError, PipelineStoreFuture, StateKey};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Mutex,
};

/// Version of the independent parallel checkpoint format.
pub const PARALLEL_CHECKPOINT_VERSION: u32 = 1;

/// Durable state of one branch. Dispatch requires committing `InFlight` first.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum ParallelBranchCheckpoint {
    /// Never authorized for dispatch; safe to execute on explicit resume.
    Ready,
    /// May have had effects, even if its reply was not observed. Never replayed.
    InFlight,
    /// Original reply was observed and durably committed.
    Completed(Msg),
    /// Original error was durably committed. Never retried by pipeline resume.
    Failed(Box<AgentError>),
}

/// Durable progress, independent of each agent's own memory and state store.
///
/// Original input/replies/errors may contain private data. Agent names and their
/// order must match when resuming; the concurrency limit may be changed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ParallelCheckpoint {
    pub version: u32,
    pub agent_names: Vec<String>,
    /// Unchanged input supplied to every newly dispatched branch.
    pub input: Msg,
    /// One state per configured agent, in the same order as `agent_names`.
    pub branches: Vec<ParallelBranchCheckpoint>,
}

impl ParallelCheckpoint {
    /// Reads a fully committed successful or failed result without dispatching.
    /// Returns `None` for invalid metadata or any `Ready`/`InFlight` branch.
    /// Original completed messages are not reinterpreted or rewritten.
    #[must_use]
    pub fn finished_result(&self) -> Option<Result<ParallelOutput, ParallelError>> {
        if !self.valid_structure()
            || self.branches.iter().any(|branch| {
                matches!(
                    branch,
                    ParallelBranchCheckpoint::Ready | ParallelBranchCheckpoint::InFlight,
                )
            })
        {
            return None;
        }
        let branches = self.branch_results();
        Some(
            if self
                .branches
                .iter()
                .any(|branch| matches!(branch, ParallelBranchCheckpoint::Failed(_)))
            {
                Err(ParallelError {
                    cause: ParallelFailure::AgentFailures,
                    branches,
                })
            } else {
                Ok(ParallelOutput { branches })
            },
        )
    }

    fn valid_structure(&self) -> bool {
        let names: BTreeSet<_> = self.agent_names.iter().collect();
        self.version == PARALLEL_CHECKPOINT_VERSION
            && !self.agent_names.is_empty()
            && self.branches.len() == self.agent_names.len()
            && names.len() == self.agent_names.len()
            && self.agent_names.iter().all(|name| !name.trim().is_empty())
    }

    fn branch_results(&self) -> Vec<ParallelBranchResult> {
        self.agent_names
            .iter()
            .zip(&self.branches)
            .enumerate()
            .map(|(index, (name, state))| ParallelBranchResult {
                branch: index + 1,
                agent_name: name.clone(),
                outcome: match state {
                    ParallelBranchCheckpoint::Ready => ParallelBranchOutcome::NotStarted,
                    ParallelBranchCheckpoint::InFlight => ParallelBranchOutcome::Interrupted,
                    ParallelBranchCheckpoint::Completed(message) => {
                        ParallelBranchOutcome::Completed(message.clone())
                    }
                    ParallelBranchCheckpoint::Failed(error) => {
                        ParallelBranchOutcome::Failed(error.clone())
                    }
                },
            })
            .collect()
    }
}

/// One optimistic-concurrency revision of parallel progress.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ParallelRecord {
    pub revision: u64,
    pub checkpoint: ParallelCheckpoint,
}

/// Parallel storage, separate from sequential pipeline and agent state stores.
pub trait ParallelStore: Send + Sync {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<ParallelRecord>>;

    /// Compare-and-swap progress. `None` creates only if absent. A successful save
    /// returns the supplied checkpoint with its incremented positive revision.
    /// Storage does not validate workflow metadata; pipeline APIs do that.
    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        checkpoint: ParallelCheckpoint,
    ) -> PipelineStoreFuture<'_, ParallelRecord>;
}

/// Process-local parallel checkpoint store; not durable across process restarts.
#[derive(Default)]
pub struct InMemoryParallelStore {
    records: Mutex<BTreeMap<StateKey, ParallelRecord>>,
}

impl InMemoryParallelStore {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Mutex::new(BTreeMap::new()),
        }
    }
}

impl ParallelStore for InMemoryParallelStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<ParallelRecord>> {
        Box::pin(async move {
            Ok(self
                .records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(key)
                .cloned())
        })
    }

    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        checkpoint: ParallelCheckpoint,
    ) -> PipelineStoreFuture<'_, ParallelRecord> {
        Box::pin(async move {
            let mut records = self
                .records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let actual = records.get(&key).map(|record| record.revision);
            if actual != expected_revision {
                return Err(PipelineStoreError::new(format!(
                    "parallel revision conflict: expected {expected_revision:?}, actual {actual:?}",
                )));
            }
            let revision = actual
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| PipelineStoreError::new("parallel revision overflow"))?;
            let record = ParallelRecord {
                revision,
                checkpoint,
            };
            records.insert(key, record.clone());
            Ok(record)
        })
    }
}

impl ParallelPipeline {
    fn new_checkpoint(&self, input: Msg) -> ParallelCheckpoint {
        ParallelCheckpoint {
            version: PARALLEL_CHECKPOINT_VERSION,
            agent_names: self
                .branches
                .iter()
                .map(|stage| stage.name.clone())
                .collect(),
            input,
            branches: self
                .branches
                .iter()
                .map(|_| ParallelBranchCheckpoint::Ready)
                .collect(),
        }
    }

    fn validate_resume(&self, record: &ParallelRecord) -> Result<(), ParallelError> {
        self.validate_checkpoint(record)?;
        if record
            .checkpoint
            .branches
            .iter()
            .any(|branch| matches!(branch, ParallelBranchCheckpoint::InFlight))
            || !record
                .checkpoint
                .branches
                .iter()
                .any(|branch| matches!(branch, ParallelBranchCheckpoint::Ready))
        {
            return Err(checkpoint_error(Some(&record.checkpoint), ParallelFailure::UnsafeResume(
                "in-flight branches require reconciliation; terminal progress cannot be replayed".into(),
            )));
        }
        Ok(())
    }

    fn validate_checkpoint(&self, record: &ParallelRecord) -> Result<(), ParallelError> {
        if record.revision == 0
            || !record.checkpoint.valid_structure()
            || !record
                .checkpoint
                .agent_names
                .iter()
                .zip(self.branches.iter())
                .all(|(name, stage)| name == &stage.name)
            || record.checkpoint.agent_names.len() != self.branches.len()
        {
            return Err(checkpoint_error(
                Some(&record.checkpoint),
                ParallelFailure::UnsafeResume(
                    "checkpoint revision, version, structure, or pipeline configuration is invalid"
                        .into(),
                ),
            ));
        }
        Ok(())
    }
}

async fn load_record(
    store: &dyn ParallelStore,
    key: &StateKey,
) -> Result<ParallelRecord, ParallelError> {
    store
        .load(key)
        .await
        .map_err(|error| checkpoint_error(None, ParallelFailure::Store(error.to_string())))?
        .ok_or_else(|| {
            checkpoint_error(
                None,
                ParallelFailure::UnsafeResume("checkpoint does not exist".into()),
            )
        })
}

fn checkpoint_error(
    checkpoint: Option<&ParallelCheckpoint>,
    cause: ParallelFailure,
) -> ParallelError {
    ParallelError {
        cause,
        branches: checkpoint.map_or_else(Vec::new, ParallelCheckpoint::branch_results),
    }
}

#[cfg(test)]
mod tests;
