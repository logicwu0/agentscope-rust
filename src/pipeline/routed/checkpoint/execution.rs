//! Non-streaming dispatch guarded by committed route checkpoints.

use super::{
    RoutedCheckpoint, RoutedCheckpointStatus, RoutedRecord, RoutedStore, checkpoint_error,
    load_record, save_record,
};
use crate::{
    Msg, ROUTED_CHECKPOINT_VERSION, RoutedError, RoutedFailure, RoutedFuture, RoutedOutput,
    RoutedPipeline, StateKey, agent::AgentInterruptToken,
};

impl RoutedPipeline {
    /// Starts NEW non-streaming progress at an unused key. Route lookup occurs
    /// before the shared run/stream lock and storage, so an unknown route never
    /// invokes storage or an agent, even while another route is busy.
    ///
    /// Creates `Ready`, then commits `InFlight` BEFORE invoking the selected
    /// agent. An observed reply/error is committed before reporting it. Storage
    /// failures, cancellation or dropping the future never authorize replay.
    /// Checkpoint storage and the agent's own durable state are not one atomic
    /// transaction; bind each independent agent to its own durable session.
    /// # Errors
    /// Unknown route, Busy, storage failure, interruption or original agent error.
    #[must_use]
    pub fn run_checkpointed<'a>(
        &'a self,
        store: &'a dyn RoutedStore,
        key: StateKey,
        route: impl Into<String>,
        input: Msg,
    ) -> RoutedFuture<'a> {
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
            let _guard = self
                .operation
                .try_lock()
                .map_err(|_| checkpoint_error(Some(&checkpoint), RoutedFailure::Busy))?;
            let interrupt = self.interrupt.token();
            let record = save_record(store, key.clone(), None, checkpoint).await?;
            self.execute_checkpointed(store, key, record, interrupt)
                .await
        })
    }

    /// Resumes ONLY a committed, compatible `Ready` selection, with its saved
    /// route and unchanged input. No new route can be substituted here.
    /// `InFlight` must be externally reconciled; `Completed`/`Failed` results are
    /// read via [`RoutedCheckpoint::finished_result`] instead of replayed.
    ///
    /// This does not restore child state, approve tools or retry agent failures.
    /// Only the selected route/name binding is checked; unrelated routes can
    /// change. Names alone cannot prove model, credentials or policy identity.
    /// # Errors
    /// Busy, load/save failure, missing/malformed/incompatible/unsafe progress,
    /// interruption or the original selected agent error.
    #[must_use]
    pub fn resume_checkpointed<'a>(
        &'a self,
        store: &'a dyn RoutedStore,
        key: StateKey,
    ) -> RoutedFuture<'a> {
        Box::pin(async move {
            let _guard = self
                .operation
                .try_lock()
                .map_err(|_| checkpoint_error(None, RoutedFailure::Busy))?;
            let interrupt = self.interrupt.token();
            let record = load_record(store, &key).await?;
            self.execute_checkpointed(store, key, record, interrupt)
                .await
        })
    }

    async fn execute_checkpointed(
        &self,
        store: &dyn RoutedStore,
        key: StateKey,
        mut record: RoutedRecord,
        mut interrupt: AgentInterruptToken,
    ) -> Result<RoutedOutput, RoutedError> {
        let stage = self.validate_checkpoint(&record)?;
        if record.checkpoint.status != RoutedCheckpointStatus::Ready {
            return Err(checkpoint_error(
                Some(&record.checkpoint),
                RoutedFailure::UnsafeResume(
                    "only ready progress can resume; in-flight or terminal work cannot be replayed"
                        .into(),
                ),
            ));
        }
        if interrupt.is_interrupted() {
            return Err(checkpoint_error(
                Some(&record.checkpoint),
                RoutedFailure::Interrupted,
            ));
        }
        let mut checkpoint = record.checkpoint.clone();
        checkpoint.status = RoutedCheckpointStatus::InFlight;
        record = save_record(store, key.clone(), Some(record.revision), checkpoint).await?;
        // Keep the fence even if cancellation wins before invocation: a saved
        // in-flight marker is conservative evidence, never replay authority.
        let result = tokio::select! {
            biased;
            () = interrupt.cancelled() => return Err(checkpoint_error(
                Some(&record.checkpoint), RoutedFailure::Interrupted,
            )),
            result = async { stage.agent.reply(record.checkpoint.input.clone()).await } => result,
        };
        let mut checkpoint = record.checkpoint.clone();
        checkpoint.status = match result {
            Ok(message) => RoutedCheckpointStatus::Completed(message),
            Err(error) => RoutedCheckpointStatus::Failed(Box::new(error)),
        };
        // Once observed, persist the terminal outcome even if interruption
        // arrives while this save is pending. Never report an uncommitted reply.
        record = save_record(store, key, Some(record.revision), checkpoint).await?;
        record.checkpoint.finished_result().ok_or_else(|| {
            checkpoint_error(
                Some(&record.checkpoint),
                RoutedFailure::UnsafeResume("checkpoint is not structurally complete".into()),
            )
        })?
    }
}
