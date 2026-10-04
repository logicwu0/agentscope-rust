//! One model selection followed by at most one explicitly allowlisted target.

mod error;
mod selection;

pub use error::{ModelRouterConfigError, RouteSelectionError};

use crate::{
    AgentInterruptHandle, ChatModel, Msg, RoutedError, RoutedFailure, RoutedFuture, RoutedOutput,
    RoutedPipeline, Usage, agent::AgentInterruptToken,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, future::Future, pin::Pin, sync::Arc};

/// A validated selection, not a checkpoint or authorization to execute tools.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteSelection {
    /// Exact allowlisted route key, without normalization.
    pub route: String,
    /// Target name captured by the underlying pipeline.
    pub agent_name: String,
    /// Usage of the selection call only, when reported by the model.
    pub usage: Option<Usage>,
}

/// Lazy model-only selection. No target agent is invoked.
pub type RouteSelectionFuture<'a> =
    Pin<Box<dyn Future<Output = Result<RouteSelection, RoutedError>> + Send + 'a>>;

/// Non-streaming model selection over an explicit subset of registered routes.
///
/// The catalog is immutable, caller-supplied policy: keys must already exist in
/// the supplied pipeline and descriptions must be nonblank. Only visible text
/// from the input is sent to the selector, as user data; no message role, name,
/// metadata, thinking, media, tools or agent history is sent. All visible text
/// is sent, so choose an appropriate model and data-sharing policy.
///
/// Selection is not permission checking. A model may choose the wrong allowed
/// route, including due to prompt injection. Expose only appropriate targets;
/// their tool confirmation and authorization policies still apply. There is no
/// fallback, retry loop, delegation, shared agent memory or automatic checkpoint.
/// The configured model may perform its own transport retries.
///
/// Clones and the original pipeline share one operation lock and interrupt
/// handle. No background tasks are spawned. Dropping cancels local polling,
/// not remote model work or effects already performed by a target.
#[derive(Clone)]
pub struct ModelRouter {
    pipeline: RoutedPipeline,
    model: Arc<dyn ChatModel>,
    routes: Arc<BTreeMap<String, String>>,
}

impl ModelRouter {
    /// Creates a router with an owned model and an explicit route/description catalog.
    /// # Errors
    /// Rejects empty catalogs, unknown or duplicate keys, and blank descriptions.
    pub fn new<M: ChatModel + 'static>(
        pipeline: RoutedPipeline,
        model: M,
        routes: Vec<(String, String)>,
    ) -> Result<Self, ModelRouterConfigError> {
        Self::from_shared(pipeline, Arc::new(model), routes)
    }

    /// Creates a router sharing a model. Configuration does not invoke the model
    /// or agents. The catalog may intentionally exclude registered targets.
    /// # Errors
    /// Rejects empty catalogs, unknown or duplicate keys, and blank descriptions.
    pub fn from_shared(
        pipeline: RoutedPipeline,
        model: Arc<dyn ChatModel>,
        routes: Vec<(String, String)>,
    ) -> Result<Self, ModelRouterConfigError> {
        if routes.is_empty() {
            return Err(ModelRouterConfigError::Empty);
        }
        let mut catalog = BTreeMap::new();
        for (route, description) in routes {
            if !pipeline.routes.contains_key(&route) {
                return Err(ModelRouterConfigError::UnknownRoute(route));
            }
            if catalog.contains_key(&route) {
                return Err(ModelRouterConfigError::DuplicateRoute(route));
            }
            if description.trim().is_empty() {
                return Err(ModelRouterConfigError::EmptyDescription { route });
            }
            catalog.insert(route, description);
        }
        Ok(Self {
            pipeline,
            model,
            routes: Arc::new(catalog),
        })
    }

    /// Returns the handle shared with the original pipeline and all clones.
    #[must_use]
    pub fn interrupt_handle(&self) -> AgentInterruptHandle {
        self.pipeline.interrupt_handle()
    }

    /// Chooses a route without invoking any target. Holds the shared pipeline
    /// lock during selection and releases it before returning. A later explicit
    /// run/stream/checkpoint call is a separate operation, not an atomic handoff.
    /// Save the selected route through existing checkpoint APIs when needed;
    /// resuming that checkpoint must not call the selector again.
    /// # Errors
    /// Busy, interruption, unsupported model, empty visible text, no match,
    /// malformed/disallowed selection, or the original model error.
    #[must_use]
    pub fn select(&self, input: Msg) -> RouteSelectionFuture<'_> {
        Box::pin(async move {
            let _guard = self
                .pipeline
                .operation
                .try_lock()
                .map_err(|_| unselected_error(RoutedFailure::Busy))?;
            let mut interrupt = self.pipeline.interrupt.token();
            self.select_locked(&input, &mut interrupt).await
        })
    }

    /// Selects once, then invokes exactly one target with the complete original
    /// message unchanged. Selection and execution hold the SAME pipeline lock
    /// and interrupt baseline. The child is not called if selection fails or an
    /// interrupt is observed before dispatch. The original reply/error is retained.
    ///
    /// This is a NEW non-checkpointed run, not a resume. The returned output's
    /// usage belongs to the target; selector usage is not added to its message.
    /// Use [`Self::select`] when separate selection accounting is needed.
    /// # Errors
    /// Selection failures, Busy, interruption, or the original target error.
    #[must_use]
    pub fn run(&self, input: Msg) -> RoutedFuture<'_> {
        Box::pin(async move {
            let _guard = self
                .pipeline
                .operation
                .try_lock()
                .map_err(|_| unselected_error(RoutedFailure::Busy))?;
            let mut interrupt = self.pipeline.interrupt.token();
            let selection = self.select_locked(&input, &mut interrupt).await?;
            let stage = &self.pipeline.routes[&selection.route];
            let attributed = |cause| RoutedError {
                route: selection.route.clone(),
                agent_name: Some(selection.agent_name.clone()),
                cause,
            };
            let message = tokio::select! {
                biased;
                () = interrupt.cancelled() => Err(attributed(RoutedFailure::Interrupted)),
                reply = async { stage.agent.reply(input).await } => reply
                    .map_err(|error| attributed(RoutedFailure::Agent(Box::new(error)))),
            }?;
            Ok(RoutedOutput {
                route: selection.route,
                agent_name: selection.agent_name,
                message,
            })
        })
    }

    async fn select_locked(
        &self,
        input: &Msg,
        interrupt: &mut AgentInterruptToken,
    ) -> Result<RouteSelection, RoutedError> {
        let request = self.request(input).map_err(selection_error)?;
        let response = tokio::select! {
            biased;
            () = interrupt.cancelled() => return Err(unselected_error(RoutedFailure::Interrupted)),
            // Do not construct generate() until this branch is actually polled.
            response = async { self.model.generate(request).await } => response,
        };
        // A model may trigger cancellation in the same poll that returns output.
        if interrupt.is_interrupted() {
            return Err(unselected_error(RoutedFailure::Interrupted));
        }
        let response =
            response.map_err(|error| selection_error(RouteSelectionError::Model(error)))?;
        self.decode_selection(&response).map_err(selection_error)
    }
}

fn unselected_error(cause: RoutedFailure) -> RoutedError {
    RoutedError {
        route: String::new(),
        agent_name: None,
        cause,
    }
}

fn selection_error(error: RouteSelectionError) -> RoutedError {
    unselected_error(RoutedFailure::Selection(error))
}

#[cfg(test)]
mod tests;
