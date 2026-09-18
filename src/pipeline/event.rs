use serde::{Deserialize, Serialize};

use super::{PipelineError, PipelineOutput, PipelineStep};
use crate::AgentEvent;

/// Serializable lifecycle of one sequential pipeline run.
///
/// Exactly one terminal [`Self::Finished`] or [`Self::Error`] is emitted when the
/// stream is fully consumed. Dropping the stream emits no synthetic terminal event.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PipelineEvent {
    /// A stage is selected, immediately before starting its agent stream.
    StageStarted {
        /// One-based pipeline stage, distinct from an agent's `ReAct` step.
        pipeline_step: usize,
        /// Agent name captured when constructing the pipeline.
        agent_name: String,
    },
    /// An unmodified event emitted by the active agent.
    Agent {
        /// One-based pipeline stage.
        pipeline_step: usize,
        /// Agent name captured when constructing the pipeline.
        agent_name: String,
        /// Original event, including its agent-local model step where applicable.
        event: AgentEvent,
    },
    /// The agent emitted `Finished` and its output was recorded for handoff.
    StageCompleted {
        /// Complete original stage output; may contain private metadata/thinking.
        stage: PipelineStep,
    },
    /// Every configured stage completed.
    Finished {
        /// Final reply and all original stage outputs.
        output: PipelineOutput,
    },
    /// Terminal failure; no subsequent stage is started automatically.
    Error {
        /// Cause and all stages completed before termination.
        error: PipelineError,
    },
}
