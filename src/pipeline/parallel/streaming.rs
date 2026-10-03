//! Lazy branch streams multiplexed without background tasks or event queues.

use super::{
    ParallelBranchOutcome, ParallelBranchResult, ParallelError, ParallelEvent, ParallelFailure,
    ParallelOutput, ParallelPipeline,
};
use crate::{AgentError, AgentEvent, Msg, agent::AgentInterruptToken, pipeline::Stage};
use async_stream::stream;
use futures_core::Stream;
use futures_util::{StreamExt, stream::FuturesUnordered};
use std::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicBool, Ordering},
    task::{Context, Poll},
};

/// Interleaved parallel events. Runtime failures are terminal event values.
pub type ParallelEventStream<'a> = Pin<Box<dyn Stream<Item = ParallelEvent> + Send + 'a>>;

/// Lazy stream preparation; only a conflicting active run returns an error here.
pub type ParallelStreamFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ParallelEventStream<'a>, ParallelError>> + Send + 'a>>;

enum BranchItem {
    Started,
    Agent(AgentEvent),
    Finished(ParallelBranchOutcome),
}

struct BranchStream<'a> {
    index: usize,
    events: Pin<Box<dyn Stream<Item = BranchItem> + Send + 'a>>,
}

impl Stream for BranchStream<'_> {
    type Item = BranchItem;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.events.as_mut().poll_next(cx)
    }
}

impl ParallelPipeline {
    /// Streams a NEW bounded fan-out with the same input and failure policy as
    /// [`Self::run`]. Awaiting reserves the shared run lock without invoking any
    /// agent. Branch selection and agent operations start only when polled.
    ///
    /// Each branch emits `BranchStarted`, its original `AgentEvent` values, then
    /// `BranchFinished` with `Completed` or `Failed`. Events from different
    /// branches may interleave; their final outcomes stay in configured order.
    /// Ordinary failures, including tool confirmation, do not cancel siblings.
    /// The fully consumed stream ends with one `Finished` or aggregate `Error`.
    ///
    /// Interruption drops active operations and stops queued dispatch, preserving
    /// terminal replies/errors already observed. `BranchStarted` selects a slot
    /// before invoking the agent, so interruption immediately afterward may
    /// still report `NotStarted`. Dropping the stream emits no terminal event.
    /// Effects are not rolled back, and there is no automatic retry or checkpoint.
    /// Poll through the pipeline terminal event for agent state finalization.
    /// No agent tasks are spawned; consumer polling drives the entire run.
    /// # Errors
    /// Returns `Busy` when this pipeline or a clone already owns a run/stream.
    #[must_use]
    pub fn stream(&self, input: Msg) -> ParallelStreamFuture<'_> {
        Box::pin(async move {
            let guard = self.operation.try_lock().map_err(|_| ParallelError {
                cause: ParallelFailure::Busy,
                branches: Vec::new(),
            })?;
            let interrupt = self.interrupt.token();
            Ok(Box::pin(stream! {
                let _guard = guard;
                let started: Vec<_> = self.branches.iter().map(|_| AtomicBool::new(false)).collect();
                let mut branches: Vec<_> = self.branches.iter().enumerate().map(|(index, stage)| {
                    ParallelBranchResult {
                        branch: index + 1,
                        agent_name: stage.name.clone(),
                        outcome: ParallelBranchOutcome::NotStarted,
                    }
                }).collect();
                let mut interrupt = interrupt;
                let mut active = FuturesUnordered::new();
                let mut next = 0;
                let interrupted = loop {
                    if interrupt.is_interrupted() { break true; }
                    while next < self.branches.len() && active.len() < self.max_concurrency {
                        active.push(branch_stream(
                            next, &self.branches[next], input.clone(), &started[next], interrupt.clone(),
                        ).into_future());
                        next += 1;
                    }
                    if active.is_empty() { break false; }
                    let selected = tokio::select! {
                        biased;
                        () = interrupt.cancelled() => break true,
                        selected = active.next() => selected,
                    };
                    let Some((item, events)) = selected else { continue; };
                    let index = events.index;
                    let branch = branches[index].branch;
                    let agent_name = branches[index].agent_name.clone();
                    match item {
                        Some(BranchItem::Started) => {
                            active.push(events.into_future());
                            yield ParallelEvent::BranchStarted { branch, agent_name };
                        }
                        Some(BranchItem::Agent(event)) => {
                            // Record terminal observations BEFORE yielding, so
                            // an interrupt before BranchFinished retains them.
                            if let Some(outcome) = terminal_outcome(&event) {
                                branches[index].outcome = outcome;
                            }
                            active.push(events.into_future());
                            yield ParallelEvent::Agent { branch, agent_name, event };
                        }
                        Some(BranchItem::Finished(outcome)) => {
                            branches[index].outcome = outcome;
                            drop(events);
                            yield ParallelEvent::BranchFinished { result: branches[index].clone() };
                        }
                        None => {
                            drop(events);
                            if interrupt.is_interrupted() { break true; }
                            branches[index].outcome = failed_stream();
                            yield ParallelEvent::BranchFinished { result: branches[index].clone() };
                        }
                    }
                };
                drop(active);
                if interrupted {
                    for (branch, started) in branches.iter_mut().zip(&started) {
                        if matches!(branch.outcome, ParallelBranchOutcome::NotStarted)
                            && started.load(Ordering::Relaxed)
                        {
                            branch.outcome = ParallelBranchOutcome::Interrupted;
                        }
                    }
                    yield ParallelEvent::Error { error: ParallelError {
                        cause: ParallelFailure::Interrupted, branches,
                    } };
                } else if branches.iter().any(|branch| matches!(branch.outcome, ParallelBranchOutcome::Failed(_))) {
                    yield ParallelEvent::Error { error: ParallelError {
                        cause: ParallelFailure::AgentFailures, branches,
                    } };
                } else {
                    yield ParallelEvent::Finished { output: ParallelOutput { branches } };
                }
            }) as ParallelEventStream<'_>)
        })
    }
}

fn branch_stream<'a>(
    index: usize,
    stage: &'a Stage,
    input: Msg,
    started: &'a AtomicBool,
    mut interrupt: AgentInterruptToken,
) -> BranchStream<'a> {
    BranchStream {
        index,
        events: Box::pin(stream! {
            if interrupt.is_interrupted() { return; }
            yield BranchItem::Started;
            if interrupt.is_interrupted() { return; }
            started.store(true, Ordering::Relaxed);
            let opened = tokio::select! {
                biased;
                () = interrupt.cancelled() => return,
                result = stage.agent.stream(input) => result,
            };
            let mut events = match opened {
                Ok(events) => events,
                Err(error) => {
                    yield BranchItem::Finished(ParallelBranchOutcome::Failed(Box::new(error)));
                    return;
                }
            };
            loop {
                // Siblings can trigger cancellation during one multiplexing
                // poll. Do not poll another agent afterward in that same poll.
                if interrupt.is_interrupted() { return; }
                let item = tokio::select! {
                    biased;
                    () = interrupt.cancelled() => return,
                    item = events.next() => item,
                };
                match item {
                    None => {
                        drop(events);
                        yield BranchItem::Finished(failed_stream());
                        return;
                    }
                    Some(Err(error)) => {
                        drop(events);
                        yield BranchItem::Finished(ParallelBranchOutcome::Failed(Box::new(error)));
                        return;
                    }
                    Some(Ok(event)) => {
                        let terminal = terminal_outcome(&event);
                        yield BranchItem::Agent(event);
                        if let Some(outcome) = terminal {
                            drop(events);
                            yield BranchItem::Finished(outcome);
                            return;
                        }
                    }
                }
            }
        }),
    }
}

fn terminal_outcome(event: &AgentEvent) -> Option<ParallelBranchOutcome> {
    match event {
        AgentEvent::Finished { message, .. } => {
            Some(ParallelBranchOutcome::Completed(message.clone()))
        }
        AgentEvent::Error { error, .. } => {
            Some(ParallelBranchOutcome::Failed(Box::new(error.clone())))
        }
        AgentEvent::ToolConfirmationRequired { checkpoint } => Some(ParallelBranchOutcome::Failed(
            Box::new(AgentError::ToolConfirmationRequired {
                checkpoint: checkpoint.clone(),
            }),
        )),
        _ => None,
    }
}

fn failed_stream() -> ParallelBranchOutcome {
    ParallelBranchOutcome::Failed(Box::new(AgentError::InvalidModelResponse(
        "agent stream ended without a terminal event".into(),
    )))
}

#[cfg(test)]
mod tests;
