//! Revisioned dispatch fences for a single, explicitly selected agent.

mod execution;
mod reconciliation;
mod streaming;

use super::{RoutedError, RoutedFailure, RoutedOutput, RoutedPipeline, Stage};
use crate::{AgentError, Msg, PipelineStoreError, PipelineStoreFuture, StateKey};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Mutex};

/// Version of persisted explicit-route checkpoints.
pub const ROUTED_CHECKPOINT_VERSION: u32 = 1;

/// Durable dispatch state. Only `Ready` can be explicitly resumed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum RoutedCheckpointStatus {
    /// Not authorized for dispatch yet; explicit resume can start this input.
    Ready,
    /// Dispatch may have had effects. Never automatically replayed.
    InFlight,
    /// Original reply was observed and committed, or externally reconciled.
    Completed(Msg),
    /// Original agent error was observed and committed. Never retried on resume.
    Failed(Box<AgentError>),
}

/// Selected route and unchanged input, independent of the agent's own snapshot.
///
/// Names are symbolic compatibility checks, not proof of agent identity, model,
/// credentials, tool policy or state binding. Keep those bindings compatible
/// across restart. Unrelated routes may change without changing this selection.
/// Original input, replies and errors can contain private data; protect storage
/// like the selected agent's own state. This is not a tool replay authorization.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RoutedCheckpoint {
    pub version: u32,
    pub route: String,
    pub agent_name: String,
    pub input: Msg,
    pub status: RoutedCheckpointStatus,
}

impl RoutedCheckpoint {
    /// Reads a committed result without dispatching or filtering its payload.
    /// Returns `None` for invalid metadata or `Ready`/`InFlight` progress.
    #[must_use]
    pub fn finished_result(&self) -> Option<Result<RoutedOutput, RoutedError>> {
        if !self.valid_structure() {
            return None;
        }
        match &self.status {
            RoutedCheckpointStatus::Completed(message) => Some(Ok(RoutedOutput {
                route: self.route.clone(),
                agent_name: self.agent_name.clone(),
                message: message.clone(),
            })),
            RoutedCheckpointStatus::Failed(error) => Some(Err(checkpoint_error(
                Some(self),
                RoutedFailure::Agent(error.clone()),
            ))),
            RoutedCheckpointStatus::Ready | RoutedCheckpointStatus::InFlight => None,
        }
    }

    fn valid_structure(&self) -> bool {
        self.version == ROUTED_CHECKPOINT_VERSION
            && !self.route.trim().is_empty()
            && !self.agent_name.trim().is_empty()
    }
}

/// An optimistic-concurrency revision of one selected-route checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RoutedRecord {
    pub revision: u64,
    pub checkpoint: RoutedCheckpoint,
}

/// Route checkpoint storage, separate from agent, sequential and parallel state.
pub trait RoutedStore: Send + Sync {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<RoutedRecord>>;

    /// Compare-and-swap progress; `None` creates only if absent. Successful saves
    /// return the supplied checkpoint unchanged at the incremented positive
    /// revision. Stores validate storage, not workflow metadata or external truth.
    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        checkpoint: RoutedCheckpoint,
    ) -> PipelineStoreFuture<'_, RoutedRecord>;
}

/// Process-local route checkpoints; not durable across process restarts.
#[derive(Default)]
pub struct InMemoryRoutedStore {
    records: Mutex<BTreeMap<StateKey, RoutedRecord>>,
}

impl InMemoryRoutedStore {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Mutex::new(BTreeMap::new()),
        }
    }
}

impl RoutedStore for InMemoryRoutedStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<RoutedRecord>> {
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
        checkpoint: RoutedCheckpoint,
    ) -> PipelineStoreFuture<'_, RoutedRecord> {
        Box::pin(async move {
            let mut records = self
                .records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let actual = records.get(&key).map(|record| record.revision);
            if actual != expected_revision {
                return Err(PipelineStoreError::new(format!(
                    "routed revision conflict: expected {expected_revision:?}, actual {actual:?}",
                )));
            }
            let revision = actual
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| PipelineStoreError::new("routed revision overflow"))?;
            let record = RoutedRecord {
                revision,
                checkpoint,
            };
            records.insert(key, record.clone());
            Ok(record)
        })
    }
}

impl RoutedPipeline {
    fn validate_ready_checkpoint<'a>(
        &'a self,
        record: &RoutedRecord,
    ) -> Result<&'a Stage, RoutedError> {
        let stage = self.validate_checkpoint(record)?;
        if record.checkpoint.status != RoutedCheckpointStatus::Ready {
            return Err(checkpoint_error(
                Some(&record.checkpoint),
                RoutedFailure::UnsafeResume(
                    "only ready progress can resume; in-flight or terminal work cannot be replayed"
                        .into(),
                ),
            ));
        }
        Ok(stage)
    }

    fn validate_checkpoint<'a>(&'a self, record: &RoutedRecord) -> Result<&'a Stage, RoutedError> {
        let checkpoint = &record.checkpoint;
        let failure = || {
            checkpoint_error(
                Some(checkpoint),
                RoutedFailure::UnsafeResume(
                    "checkpoint revision, version, selection or route binding is invalid".into(),
                ),
            )
        };
        if record.revision == 0 || !checkpoint.valid_structure() {
            return Err(failure());
        }
        let Some(stage) = self.routes.get(&checkpoint.route) else {
            return Err(failure());
        };
        if stage.name != checkpoint.agent_name {
            return Err(failure());
        }
        Ok(stage)
    }
}

fn checkpoint_error(checkpoint: Option<&RoutedCheckpoint>, cause: RoutedFailure) -> RoutedError {
    RoutedError {
        route: checkpoint.map_or_else(String::new, |saved| saved.route.clone()),
        agent_name: checkpoint.map(|saved| saved.agent_name.clone()),
        cause,
    }
}

async fn load_record(store: &dyn RoutedStore, key: &StateKey) -> Result<RoutedRecord, RoutedError> {
    store
        .load(key)
        .await
        .map_err(|error| checkpoint_error(None, RoutedFailure::Store(error.to_string())))?
        .ok_or_else(|| {
            checkpoint_error(
                None,
                RoutedFailure::UnsafeResume("checkpoint does not exist".into()),
            )
        })
}

async fn save_record(
    store: &dyn RoutedStore,
    key: StateKey,
    expected_revision: Option<u64>,
    checkpoint: RoutedCheckpoint,
) -> Result<RoutedRecord, RoutedError> {
    let saved = store
        .save(key, expected_revision, checkpoint.clone())
        .await
        .map_err(|error| {
            checkpoint_error(Some(&checkpoint), RoutedFailure::Store(error.to_string()))
        })?;
    if expected_revision.unwrap_or(0).checked_add(1) != Some(saved.revision)
        || saved.checkpoint != checkpoint
    {
        return Err(checkpoint_error(
            Some(&checkpoint),
            RoutedFailure::Store(
                "store returned an incompatible checkpoint acknowledgement".into(),
            ),
        ));
    }
    Ok(saved)
}

#[cfg(test)]
mod tests;
