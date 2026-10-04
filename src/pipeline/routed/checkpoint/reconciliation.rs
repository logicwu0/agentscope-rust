//! Explicit submission of an externally verified selected-agent reply.

use super::{
    RoutedCheckpointStatus, RoutedRecord, RoutedStore, checkpoint_error, load_record, save_record,
};
use crate::{Msg, Role, RoutedError, RoutedFailure, RoutedPipeline, StateKey};

impl RoutedPipeline {
    /// Commits an externally verified reply for `InFlight` or `Failed` progress,
    /// without calling an agent, approving tools or dispatching any new work.
    /// Stop or externally fence the original worker, then reconcile external
    /// effects and the selected agent's own state first.
    /// The revision actually inspected prevents stale or duplicate decisions.
    ///
    /// Evidence must be an Assistant message named for the captured selected
    /// agent. This checks structure and symbolic identity, not external truth.
    /// `Ready` and `Completed` progress cannot be overwritten. The entire reply
    /// is retained unchanged; completed progress is read, never auto-replayed.
    /// # Errors
    /// Busy, load/save failure, stale/malformed/incompatible progress or invalid evidence.
    pub async fn reconcile_checkpointed(
        &self,
        store: &dyn RoutedStore,
        key: StateKey,
        expected_revision: u64,
        message: Msg,
    ) -> Result<RoutedRecord, RoutedError> {
        let _guard = self
            .operation
            .try_lock()
            .map_err(|_| checkpoint_error(None, RoutedFailure::Busy))?;
        let record = load_record(store, &key).await?;
        self.validate_checkpoint(&record)?;
        let failure = |reason: &str| {
            checkpoint_error(
                Some(&record.checkpoint),
                RoutedFailure::UnsafeResume(reason.into()),
            )
        };
        if record.revision != expected_revision {
            return Err(failure("checkpoint revision changed; inspect it again"));
        }
        if !matches!(
            record.checkpoint.status,
            RoutedCheckpointStatus::InFlight | RoutedCheckpointStatus::Failed(_)
        ) {
            return Err(failure(
                "only in-flight or failed progress can be reconciled",
            ));
        }
        if message.role != Role::Assistant || message.name != record.checkpoint.agent_name {
            return Err(failure(
                "verified reply must be an Assistant message from the selected agent",
            ));
        }
        let mut checkpoint = record.checkpoint.clone();
        checkpoint.status = RoutedCheckpointStatus::Completed(message);
        save_record(store, key, Some(expected_revision), checkpoint).await
    }
}
