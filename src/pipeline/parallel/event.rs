//! Branch-aware events for one parallel pipeline run.

use super::{ParallelBranchResult, ParallelError, ParallelOutput};
use crate::AgentEvent;
use serde::{Deserialize, Serialize};

/// Serializable, interleaved lifecycle of a bounded parallel run.
///
/// Events preserve each branch's order, but there is no fixed order across
/// branches. Fully consuming the stream yields exactly one terminal `Finished`
/// or `Error`. Dropping the stream emits no synthetic terminal event.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParallelEvent {
    /// A concurrency slot was assigned, before invoking the agent stream.
    /// A plain stream can still leave the branch `NotStarted` on interruption.
    /// Checkpointed streams commit `InFlight` before emitting this event.
    BranchStarted {
        /// One-based configured branch, distinct from an agent's `ReAct` step.
        branch: usize,
        /// Agent name captured when constructing the pipeline.
        agent_name: String,
    },
    /// Original event from one branch, without rewriting its agent-local step.
    Agent {
        branch: usize,
        agent_name: String,
        /// May contain private reasoning or metadata; consumers select what to show.
        event: AgentEvent,
    },
    /// One branch produced a completed reply or failed; siblings still continue.
    /// Interrupted and unstarted branches are reported only in terminal `Error`.
    /// In checkpointed streams this acknowledges a durable terminal commit;
    /// an earlier agent-local terminal event alone does not.
    BranchFinished {
        /// Original branch outcome, including any structured agent error.
        result: ParallelBranchResult,
    },
    /// All configured agents completed successfully.
    Finished {
        /// Original replies in configured branch order.
        output: ParallelOutput,
    },
    /// Aggregate agent failures, interruption, or checkpoint storage failure.
    Error {
        /// Outcomes in configured order. Checkpointed streams report only the
        /// last acknowledged record, not uncommitted agent observations.
        error: ParallelError,
    },
}
