//! Bounded fan-out with ordered branch outcomes and cooperative interruption.

use super::{PipelineConfigError, Stage, stages};
use crate::{
    Agent, AgentError, AgentInterruptHandle, AgentResult, Msg, agent::AgentInterruptToken,
};
use futures_util::{StreamExt, stream::FuturesUnordered};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::Mutex;

/// The observed outcome of one configured branch.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum ParallelBranchOutcome {
    /// Original reply, including any private metadata or thinking.
    Completed(Msg),
    /// Agent failure, including pending confirmation or uncertain tool execution.
    Failed(Box<AgentError>),
    /// Reply was invoked but no result was observed before pipeline interruption.
    /// External effects and agent state may still require reconciliation.
    Interrupted,
    /// This branch's reply was never invoked.
    NotStarted,
}

/// One branch's original result, identified independently of completion order.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ParallelBranchResult {
    /// One-based position in the configured agent list, not a `ReAct` step.
    pub branch: usize,
    /// Agent name captured when the pipeline was constructed.
    pub agent_name: String,
    pub outcome: ParallelBranchOutcome,
}

/// All branch replies, in configured order. There is no synthesized final reply.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ParallelOutput {
    /// A successful run contains only `Completed` outcomes.
    pub branches: Vec<ParallelBranchResult>,
}

/// Why the fan-out did not complete successfully.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelFailure {
    /// Another run on this pipeline or a clone holds the run lock.
    Busy,
    /// The pipeline handle interrupted dispatch and active replies.
    Interrupted,
    /// All branches were attempted; at least one agent returned an error.
    AgentFailures,
}

/// Failed run with all observed outcomes in configured order.
///
/// This is diagnostic data, not a checkpoint or authorization to replay effects.
/// Agent failures retain confirmation and uncertain-execution details for the
/// caller to resolve through the individual agent APIs.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ParallelError {
    pub cause: ParallelFailure,
    /// One outcome per branch except for `Busy`, which dispatches nothing and
    /// returns an empty list. Interrupted runs also distinguish unstarted work.
    pub branches: Vec<ParallelBranchResult>,
}

impl fmt::Display for ParallelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cause {
            ParallelFailure::Busy => f.write_str("parallel pipeline already has an active run"),
            ParallelFailure::Interrupted => {
                f.write_str("parallel pipeline interrupted; prior effects are not rolled back")
            }
            ParallelFailure::AgentFailures => {
                let failed = self
                    .branches
                    .iter()
                    .filter(|branch| matches!(branch.outcome, ParallelBranchOutcome::Failed(_)))
                    .count();
                write!(
                    f,
                    "parallel pipeline completed with {failed} agent failure(s)"
                )
            }
        }
    }
}
impl std::error::Error for ParallelError {}

/// A lazy run; no agent is invoked until this future is polled.
pub type ParallelFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ParallelOutput, ParallelError>> + Send + 'a>>;

type BranchFuture<'a> =
    Pin<Box<dyn Future<Output = (usize, Option<AgentResult<Msg>>)> + Send + 'a>>;

/// Sends the same input to independent agents with a fixed concurrency limit.
///
/// Ordinary agent failures do not cancel siblings or skip queued branches. All
/// observed outcomes are returned in configured order. The caller can explicitly
/// feed selected visible replies to a separate summarizing agent afterward.
/// Supply independent memories and durable session keys; shared backing state
/// cannot be detected through `dyn Agent`. Avoid independent concurrent use of
/// these same agents. Clones share the run lock and interrupt handle.
///
/// This orchestrator has no streaming, combined memory, checkpoint or resume API.
/// Every run is new work. Dropping its future stops dispatch and drops active
/// reply futures without spawning background tasks or undoing external effects.
#[derive(Clone)]
pub struct ParallelPipeline {
    branches: Arc<Vec<Stage>>,
    max_concurrency: usize,
    operation: Arc<Mutex<()>>,
    interrupt: AgentInterruptHandle,
}

impl ParallelPipeline {
    /// Creates a fan-out with nonempty, unique agent names and a positive limit.
    /// Limits larger than the agent count are allowed.
    /// # Errors
    /// Rejects zero concurrency, an empty list, or blank/duplicate agent names.
    pub fn new(
        agents: Vec<Arc<dyn Agent>>,
        max_concurrency: usize,
    ) -> Result<Self, PipelineConfigError> {
        if max_concurrency == 0 {
            return Err(PipelineConfigError::ZeroConcurrency);
        }
        Ok(Self {
            branches: Arc::new(stages(agents)?),
            max_concurrency,
            operation: Arc::new(Mutex::new(())),
            interrupt: AgentInterruptHandle::new(),
        })
    }

    /// Interrupts the active run; later runs capture a fresh signal baseline.
    /// Drops active reply futures without calling individual agent handles.
    #[must_use]
    pub fn interrupt_handle(&self) -> AgentInterruptHandle {
        self.interrupt.clone()
    }

    /// Invokes every branch once with an unchanged clone of `input`.
    ///
    /// At most `max_concurrency` reply futures are active. Failure of one agent
    /// (even pending approval) does not affect siblings. Pipeline interruption
    /// preserves results already observed, drops active work, and leaves queued
    /// branches unstarted. Completed or in-flight effects are never rolled back.
    /// Dropping this future does not return partial results or save a checkpoint.
    /// # Errors
    /// Busy, interruption, or one or more agent failures with ordered outcomes.
    #[must_use]
    pub fn run(&self, input: Msg) -> ParallelFuture<'_> {
        Box::pin(async move {
            let _guard = self.operation.try_lock().map_err(|_| ParallelError {
                cause: ParallelFailure::Busy,
                branches: Vec::new(),
            })?;
            let mut interrupt = self.interrupt.token();
            // Mark invocation inside each lazy future, rather than when queued,
            // so interruption can distinguish dispatched work from untouched work.
            let started: Vec<_> = self
                .branches
                .iter()
                .map(|_| AtomicBool::new(false))
                .collect();
            let mut branches: Vec<_> = self
                .branches
                .iter()
                .enumerate()
                .map(|(index, stage)| ParallelBranchResult {
                    branch: index + 1,
                    agent_name: stage.name.clone(),
                    outcome: ParallelBranchOutcome::NotStarted,
                })
                .collect();
            let mut active = FuturesUnordered::new();
            let mut next = 0;
            let cause = loop {
                if interrupt.is_interrupted() {
                    break Some(ParallelFailure::Interrupted);
                }
                while next < self.branches.len() && active.len() < self.max_concurrency {
                    active.push(branch_reply(
                        next,
                        &self.branches[next],
                        input.clone(),
                        &started[next],
                        interrupt.clone(),
                    ));
                    next += 1;
                }
                if active.is_empty() {
                    break branches
                        .iter()
                        .any(|branch| matches!(branch.outcome, ParallelBranchOutcome::Failed(_)))
                        .then_some(ParallelFailure::AgentFailures);
                }
                let item = tokio::select! {
                    biased;
                    () = interrupt.cancelled() => break Some(ParallelFailure::Interrupted),
                    result = active.next() => result,
                };
                let Some((index, result)) = item else {
                    continue;
                };
                let Some(result) = result else {
                    break Some(ParallelFailure::Interrupted);
                };
                branches[index].outcome = match result {
                    Ok(message) => ParallelBranchOutcome::Completed(message),
                    Err(error) => ParallelBranchOutcome::Failed(Box::new(error)),
                };
            };
            drop(active);
            if cause == Some(ParallelFailure::Interrupted) {
                for (branch, started) in branches.iter_mut().zip(&started) {
                    if matches!(branch.outcome, ParallelBranchOutcome::NotStarted)
                        && started.load(Ordering::Relaxed)
                    {
                        branch.outcome = ParallelBranchOutcome::Interrupted;
                    }
                }
            }
            match cause {
                Some(cause) => Err(ParallelError { cause, branches }),
                None => Ok(ParallelOutput { branches }),
            }
        })
    }
}

fn branch_reply<'a>(
    index: usize,
    stage: &'a Stage,
    input: Msg,
    started: &'a AtomicBool,
    interrupt: AgentInterruptToken,
) -> BranchFuture<'a> {
    Box::pin(async move {
        // Another branch can synchronously trigger an interrupt while the
        // FuturesUnordered poll is in progress. Do not invoke a sibling afterward.
        if interrupt.is_interrupted() {
            return (index, None);
        }
        started.store(true, Ordering::Relaxed);
        (index, Some(stage.agent.reply(input).await))
    })
}

#[cfg(test)]
mod tests;
