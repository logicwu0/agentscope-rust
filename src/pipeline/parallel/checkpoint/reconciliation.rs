//! Explicit, revision-checked submission of externally verified branch replies.

use super::super::{ParallelError, ParallelFailure, ParallelPipeline};
use super::{
    ParallelBranchCheckpoint, ParallelRecord, ParallelStore, checkpoint_error, load_record,
};
use crate::{Msg, Role, StateKey};

impl ParallelPipeline {
    /// Commits an externally verified reply for ONE `InFlight` or `Failed`
    /// branch without calling any agent or dispatching queued work. Resolve
    /// external tool effects and the agent's own state before using this API.
    /// The caller supplies the revision actually inspected, preventing stale or
    /// duplicate writes. After reconciling every in-flight branch, explicitly
    /// resume remaining `Ready` work or read the committed terminal result.
    ///
    /// `branch` is one-based. Evidence must be an Assistant message named for
    /// that branch. This checks structure and identity, not the external outcome.
    /// `Ready` and `Completed` states cannot be overwritten through this method.
    /// # Errors
    /// Busy, load/save failure, stale/malformed progress or invalid evidence.
    pub async fn reconcile_checkpointed(
        &self,
        store: &dyn ParallelStore,
        key: StateKey,
        expected_revision: u64,
        branch: usize,
        message: Msg,
    ) -> Result<ParallelRecord, ParallelError> {
        let _guard = self
            .operation
            .try_lock()
            .map_err(|_| checkpoint_error(None, ParallelFailure::Busy))?;
        let record = load_record(store, &key).await?;
        self.validate_checkpoint(&record)?;
        let failure = |reason: &str| {
            checkpoint_error(
                Some(&record.checkpoint),
                ParallelFailure::UnsafeResume(reason.into()),
            )
        };
        if record.revision != expected_revision {
            return Err(failure("checkpoint revision changed; inspect it again"));
        }
        let Some(index) = branch
            .checked_sub(1)
            .filter(|index| *index < self.branches.len())
        else {
            return Err(failure(
                "branch number is outside the configured agent list",
            ));
        };
        if !matches!(
            record.checkpoint.branches[index],
            ParallelBranchCheckpoint::InFlight | ParallelBranchCheckpoint::Failed(_)
        ) {
            return Err(failure(
                "only an in-flight or failed branch can be reconciled",
            ));
        }
        if message.role != Role::Assistant || message.name != self.branches[index].name {
            return Err(failure(
                "verified reply must be an Assistant message from the selected agent",
            ));
        }
        let mut checkpoint = record.checkpoint.clone();
        checkpoint.branches[index] = ParallelBranchCheckpoint::Completed(message);
        store
            .save(key, Some(expected_revision), checkpoint)
            .await
            .map_err(|error| {
                checkpoint_error(
                    Some(&record.checkpoint),
                    ParallelFailure::Store(error.to_string()),
                )
            })
    }
}
