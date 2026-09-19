//! Explicit, revisioned stage-boundary checkpoints.

use super::PipelineStep;
use crate::{Msg, StateKey};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt, future::Future, pin::Pin, sync::Mutex};

/// Version of the persisted pipeline checkpoint format.
pub const PIPELINE_CHECKPOINT_VERSION: u32 = 1;

/// Whether it is safe to automatically dispatch the next stage.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineCheckpointStatus {
    /// No stage is running; resume may dispatch the next stage.
    Ready,
    /// A stage may have performed effects; manual reconciliation is required.
    InFlight,
    /// Every stage has completed and its result was committed.
    Finished,
}

/// Durable progress for one fixed sequential pipeline.
///
/// This contains original stage replies, possibly including private metadata.
/// Store it with the same access control as agent state. It is not an agent
/// snapshot: each agent still needs its own durable state binding.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PipelineCheckpoint {
    pub version: u32,
    pub agent_names: Vec<String>,
    pub completed: Vec<PipelineStep>,
    pub next_input: Msg,
    pub status: PipelineCheckpointStatus,
}

/// One optimistic-concurrency revision of a pipeline checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PipelineRecord {
    pub revision: u64,
    pub checkpoint: PipelineCheckpoint,
}

/// Pipeline checkpoint storage failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PipelineStoreError {
    pub message: String,
}

impl PipelineStoreError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for PipelineStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for PipelineStoreError {}

/// A boxed, asynchronous checkpoint-store operation.
pub type PipelineStoreFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<T, PipelineStoreError>> + Send + 'a>>;

/// Revisioned storage for pipeline progress, independent of agent state storage.
pub trait PipelineStore: Send + Sync {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<PipelineRecord>>;

    /// Compare-and-swap a checkpoint. `None` creates only if absent.
    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        checkpoint: PipelineCheckpoint,
    ) -> PipelineStoreFuture<'_, PipelineRecord>;
}

/// Process-local store for tests and non-durable workflows.
#[derive(Default)]
pub struct InMemoryPipelineStore {
    records: Mutex<BTreeMap<StateKey, PipelineRecord>>,
}

impl InMemoryPipelineStore {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Mutex::new(BTreeMap::new()),
        }
    }
}

impl PipelineStore for InMemoryPipelineStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<PipelineRecord>> {
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
        checkpoint: PipelineCheckpoint,
    ) -> PipelineStoreFuture<'_, PipelineRecord> {
        Box::pin(async move {
            let mut records = self
                .records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let actual = records.get(&key).map(|record| record.revision);
            if actual != expected_revision {
                return Err(PipelineStoreError::new(format!(
                    "pipeline revision conflict: expected {expected_revision:?}, actual {actual:?}"
                )));
            }
            let revision = actual
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| PipelineStoreError::new("pipeline revision overflow"))?;
            let record = PipelineRecord {
                revision,
                checkpoint,
            };
            records.insert(key, record.clone());
            Ok(record)
        })
    }
}
