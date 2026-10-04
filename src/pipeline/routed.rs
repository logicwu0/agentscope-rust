//! Exact, caller-selected routing to one agent without fallback or broadcast.

mod checkpoint;
mod error;
mod event;
mod model_router;
mod streaming;

#[cfg(test)]
mod streaming_tests_support;

pub use checkpoint::{
    InMemoryRoutedStore, ROUTED_CHECKPOINT_VERSION, RoutedCheckpoint, RoutedCheckpointStatus,
    RoutedRecord, RoutedStore,
};
pub use error::{RoutedConfigError, RoutedError, RoutedFailure};
pub use event::RoutedEvent;
pub use model_router::{
    ModelRouter, ModelRouterConfigError, RouteSelection, RouteSelectionError, RouteSelectionFuture,
};
pub use streaming::{RoutedEventStream, RoutedStreamFuture};

use super::Stage;
use crate::{Agent, AgentInterruptHandle, Msg};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, future::Future, pin::Pin, sync::Arc};
use tokio::sync::Mutex;

/// The selected route and its original agent reply, without a synthesized handoff.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RoutedOutput {
    /// Exact caller-selected route key, distinct from the agent's name.
    pub route: String,
    /// Agent name captured when the pipeline was constructed.
    pub agent_name: String,
    /// Original reply, including any private thinking, metadata or data blocks.
    pub message: Msg,
}

/// A lazy run; no agent is invoked until this future is polled.
pub type RoutedFuture<'a> =
    Pin<Box<dyn Future<Output = Result<RoutedOutput, RoutedError>> + Send + 'a>>;

/// Routes one NEW reply to the explicitly selected agent.
///
/// This is an orchestrator, not an `Agent`. [`ModelRouter`] adds opt-in model
/// selection over an explicit allowlist. Routes
/// are immutable, exact keys with no case folding, whitespace normalization,
/// fallback or broadcast. Different keys can intentionally alias the same
/// agent; agent names need not be unique because route keys identify selections.
///
/// Runs, streams and clones share one lock across ALL routes and one interrupt
/// handle. Each target retains its own memory and durable state: no history is
/// combined and no child state is restored by this pipeline. Avoid independently running the
/// same target agents while using this orchestrator. Shared backing memories
/// cannot be detected through `dyn Agent`.
///
/// Ordinary runs and streams have no route checkpoint. Checkpointed
/// runs and streams fence dispatch with revisioned storage and resume only
/// undispatched work. No operation automatically retries, restores child state
/// or rolls back effects. Pending tool confirmation and uncertain execution
/// remain original agent errors for the caller to resolve using the selected
/// agent's own APIs; resolving them does not resume an orchestration.
#[derive(Clone)]
pub struct RoutedPipeline {
    routes: Arc<BTreeMap<String, Stage>>,
    operation: Arc<Mutex<()>>,
    interrupt: AgentInterruptHandle,
}

impl RoutedPipeline {
    /// Registers fixed, nonblank, unique route keys and captures each agent name.
    /// Keys are separate from names; aliases and repeated agent names are valid.
    /// Configuration only reads names and does not run any agent operation.
    /// # Errors
    /// Rejects empty configuration, blank keys, duplicate exact keys, or blank
    /// agent names. It never silently replaces a duplicate route.
    pub fn new(routes: Vec<(String, Arc<dyn Agent>)>) -> Result<Self, RoutedConfigError> {
        if routes.is_empty() {
            return Err(RoutedConfigError::Empty);
        }
        let mut registered = BTreeMap::new();
        for (index, (route, agent)) in routes.into_iter().enumerate() {
            if route.trim().is_empty() {
                return Err(RoutedConfigError::EmptyRoute { entry: index + 1 });
            }
            if registered.contains_key(&route) {
                return Err(RoutedConfigError::DuplicateRoute(route));
            }
            let name = agent.name().to_owned();
            if name.trim().is_empty() {
                return Err(RoutedConfigError::EmptyAgentName { route });
            }
            registered.insert(route, Stage { name, agent });
        }
        Ok(Self {
            routes: Arc::new(registered),
            operation: Arc::new(Mutex::new(())),
            interrupt: AgentInterruptHandle::new(),
        })
    }

    /// Interrupts the active run or stream by dropping its operation, without
    /// calling the selected agent's own interrupt handle or affecting unrelated
    /// users of that handle. Later runs/streams capture a fresh signal baseline.
    /// Observed terminal outcomes and external tool effects are not undone.
    #[must_use]
    pub fn interrupt_handle(&self) -> AgentInterruptHandle {
        self.interrupt.clone()
    }

    /// Invokes exactly one registered target with `input` unchanged. Original
    /// message role, content, identity and metadata are neither filtered nor
    /// rewritten, and the returned reply is retained unchanged as well.
    ///
    /// Every invocation is a NEW run. No target is called until polling. Route
    /// lookup precedes lock acquisition: an unknown key always returns
    /// `UnknownRoute`, even during another run, without dispatching anything.
    /// Known routes compete for one shared run lock, not a per-route lock.
    ///
    /// An already-observed interrupt takes priority over polling a ready reply,
    /// including on resumed polls. Dropping the run stops its active operation
    /// and releases the lock without spawning tasks, saving a pipeline state,
    /// authorizing a retry or undoing side effects. The selected agent's own
    /// persistence and cancellation guarantees continue to apply.
    /// # Errors
    /// Unknown route, Busy, interruption, or the original selected agent error,
    /// including tool confirmation and uncertain execution checkpoints.
    #[must_use]
    pub fn run(&self, route: impl Into<String>, input: Msg) -> RoutedFuture<'_> {
        let route = route.into();
        Box::pin(async move {
            let Some(stage) = self.routes.get(&route) else {
                return Err(RoutedError {
                    route,
                    agent_name: None,
                    cause: RoutedFailure::UnknownRoute,
                });
            };
            let error = |cause| RoutedError {
                route: route.clone(),
                agent_name: Some(stage.name.clone()),
                cause,
            };
            let _guard = self
                .operation
                .try_lock()
                .map_err(|_| error(RoutedFailure::Busy))?;
            let mut interrupt = self.interrupt.token();
            let message = tokio::select! {
                biased;
                () = interrupt.cancelled() => Err(error(RoutedFailure::Interrupted)),
                // Delay even constructing reply() until this branch is polled;
                // a trait implementation may have synchronous start effects.
                reply = async { stage.agent.reply(input).await } => reply
                    .map_err(|cause| error(RoutedFailure::Agent(Box::new(cause)))),
            }?;
            Ok(RoutedOutput {
                route,
                agent_name: stage.name.clone(),
                message,
            })
        })
    }
}

#[cfg(test)]
mod tests;
