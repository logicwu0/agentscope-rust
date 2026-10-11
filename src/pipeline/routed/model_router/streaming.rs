//! Lazy classification and shared selected-target streaming.

use super::{ModelRouter, unselected_error};
use crate::{Msg, RoutedEvent, RoutedFailure, RoutedStreamFuture};
use async_stream::stream;
use futures_util::StreamExt;

impl ModelRouter {
    /// Streams a new model-selected target reply. Awaiting only reserves the
    /// shared pipeline lock; no selector or agent operation is invoked yet.
    /// The first event poll performs one non-streaming classification with the
    /// same validation and text projection as [`Self::select`]. On success it
    /// emits `RouteStarted`; a later poll starts the selected agent's stream
    /// with the complete original input. No selector deltas or usage are added
    /// to target events or messages.
    ///
    /// Selection failures emit one unattributed routed `Error`, without a start
    /// event or target call. After selection, original agent events, terminal
    /// outcomes, interrupt priority and stream finalization follow the explicit
    /// routed stream contract. The shared lock and interrupt baseline cover
    /// selection through execution, across clones and the original pipeline.
    /// The lock is released before the routed terminal event or on drop.
    ///
    /// Dropping stops local polling and emits no terminal event. This method
    /// creates no checkpoint, background task, fallback or automatic retry;
    /// observed terminal outcomes and external effects are not undone.
    /// # Errors
    /// Busy during preparation. Selection and execution failures are events.
    #[must_use]
    pub fn stream(&self, input: Msg) -> RoutedStreamFuture<'_> {
        Box::pin(async move {
            let guard = self
                .pipeline
                .operation
                .try_lock()
                .map_err(|_| unselected_error(RoutedFailure::Busy))?;
            let mut interrupt = self.pipeline.interrupt.token();
            Ok(Box::pin(stream! {
                let selection = match self.select_locked(&input, &mut interrupt).await {
                    Ok(selection) => selection,
                    Err(error) => {
                        drop(guard);
                        yield RoutedEvent::Error { error };
                        return;
                    }
                };
                let stage = &self.pipeline.routes[&selection.route];
                let mut events = super::super::streaming::selected_event_stream(
                    stage, selection.route, input, guard, interrupt,
                );
                while let Some(event) = events.next().await {
                    yield event;
                }
            }) as crate::RoutedEventStream<'_>)
        })
    }
}

#[cfg(test)]
mod tests;
