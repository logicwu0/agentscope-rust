//! Lazy selected-agent streams, driven entirely by consumer polling.

use super::{RoutedError, RoutedEvent, RoutedFailure, RoutedOutput, RoutedPipeline};
use crate::{AgentError, AgentEvent, Msg, agent::AgentInterruptToken, pipeline::Stage};
use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use std::{future::Future, pin::Pin};
use tokio::sync::MutexGuard;

/// Selected-route events. Runtime failures become terminal event values.
pub type RoutedEventStream<'a> = Pin<Box<dyn Stream<Item = RoutedEvent> + Send + 'a>>;

/// Lazy route validation and shared-lock reservation; no agent runs here.
/// Checkpointed methods also create or validate stored progress during preparation.
/// Model-router streams defer classification until their first event poll.
pub type RoutedStreamFuture<'a> =
    Pin<Box<dyn Future<Output = Result<RoutedEventStream<'a>, RoutedError>> + Send + 'a>>;

impl RoutedPipeline {
    /// Streams a NEW reply from exactly one explicitly selected agent.
    ///
    /// Awaiting validates the route before reserving the shared run/stream lock,
    /// without invoking any agent. Unless already interrupted, the first stream
    /// poll emits `RouteStarted`; only a later poll invokes the selected agent's
    /// `stream` with unchanged input. A start event announces selection, not
    /// dispatch. Unknown routes never fall back or emit a start event.
    ///
    /// Original agent events retain their local step and payload. `Finished`,
    /// `Error` and `ToolConfirmationRequired` are forwarded before one terminal
    /// routed event. Startup failures, error items and EOF without an agent
    /// terminal event produce a routed `Error`. Nothing after an agent terminal
    /// event is polled. Once observed, that terminal outcome is retained even
    /// if interrupted between yielding it and the routed terminal event.
    ///
    /// Already-observed interruption takes priority over opening or polling the
    /// agent, including on resumed polls. No tasks are spawned and the child's
    /// interrupt handle is not invoked. The shared lock is released BEFORE the
    /// routed terminal event is yielded, or when the stream is dropped.
    /// Poll through the routed terminal event for the agent's state finalization.
    /// Dropping emits no terminal event, saves no route checkpoint and does not
    /// undo effects, approve confirmations or authorize a retry.
    /// # Errors
    /// Unknown route (even while busy), or Busy across any run/stream or clone.
    #[must_use]
    pub fn stream(&self, route: impl Into<String>, input: Msg) -> RoutedStreamFuture<'_> {
        let route = route.into();
        Box::pin(async move {
            let Some(stage) = self.routes.get(&route) else {
                return Err(RoutedError {
                    route,
                    agent_name: None,
                    cause: RoutedFailure::UnknownRoute,
                });
            };
            let guard = self
                .operation
                .try_lock()
                .map_err(|_| attributed_error(&route, &stage.name, RoutedFailure::Busy))?;
            let interrupt = self.interrupt.token();
            Ok(selected_event_stream(stage, route, input, guard, interrupt))
        })
    }
}

/// Keeps explicit and model-selected target streams on the same execution path.
pub(super) fn selected_event_stream<'a>(
    stage: &'a Stage,
    route: String,
    input: Msg,
    guard: MutexGuard<'a, ()>,
    mut interrupt: AgentInterruptToken,
) -> RoutedEventStream<'a> {
    Box::pin(stream! {
        let outcome = if interrupt.is_interrupted() {
            Err(RoutedFailure::Interrupted)
        } else {
            yield RoutedEvent::RouteStarted {
                route: route.clone(), agent_name: stage.name.clone(),
            };
            let opened = tokio::select! {
                biased;
                () = interrupt.cancelled() => Err(RoutedFailure::Interrupted),
                // Delay even synchronous stream() side effects until polled.
                opened = async { stage.agent.stream(input).await } => opened
                    .map_err(|error| RoutedFailure::Agent(Box::new(error))),
            };
            match opened {
                Err(cause) => Err(cause),
                Ok(mut events) => loop {
                    let item = tokio::select! {
                        biased;
                        () = interrupt.cancelled() => break Err(RoutedFailure::Interrupted),
                        item = events.next() => item,
                    };
                    match item {
                        None => break Err(RoutedFailure::Agent(Box::new(
                            AgentError::InvalidModelResponse(
                                "agent stream ended without a terminal event".into(),
                            ),
                        ))),
                        Some(Err(error)) => break Err(RoutedFailure::Agent(Box::new(error))),
                        Some(Ok(event)) => {
                            if let Some(outcome) = terminal_outcome(&event) {
                                // Keep an observed terminal across the yield gap;
                                // never poll the child's tail or overwrite it.
                                drop(events);
                                yield RoutedEvent::Agent {
                                    route: route.clone(), agent_name: stage.name.clone(), event,
                                };
                                break outcome;
                            }
                            yield RoutedEvent::Agent {
                                route: route.clone(), agent_name: stage.name.clone(), event,
                            };
                        }
                    }
                },
            }
        };
        // Retaining a terminal stream must not block later explicit work.
        drop(guard);
        yield match outcome {
            Ok(message) => RoutedEvent::Finished { output: RoutedOutput {
                route, agent_name: stage.name.clone(), message,
            } },
            Err(cause) => RoutedEvent::Error {
                error: attributed_error(&route, &stage.name, cause),
            },
        };
    })
}

fn attributed_error(route: &str, name: &str, cause: RoutedFailure) -> RoutedError {
    RoutedError {
        route: route.to_owned(),
        agent_name: Some(name.to_owned()),
        cause,
    }
}

pub(super) fn terminal_outcome(event: &AgentEvent) -> Option<Result<Msg, RoutedFailure>> {
    match event {
        AgentEvent::Finished { message, .. } => Some(Ok(message.clone())),
        AgentEvent::Error { error, .. } => Some(Err(RoutedFailure::Agent(Box::new(error.clone())))),
        AgentEvent::ToolConfirmationRequired { checkpoint } => Some(Err(RoutedFailure::Agent(
            Box::new(AgentError::ToolConfirmationRequired {
                checkpoint: checkpoint.clone(),
            }),
        ))),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
