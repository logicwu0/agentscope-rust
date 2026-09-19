use super::PipelineStep;
use crate::AgentError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Invalid fixed-stage pipeline configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelineConfigError {
    Empty,
    EmptyName { step: usize },
    DuplicateName(String),
}

impl fmt::Display for PipelineConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("pipeline requires at least one agent"),
            Self::EmptyName { step } => write!(f, "pipeline stage {step} has a blank agent name"),
            Self::DuplicateName(name) => write!(f, "pipeline agent name {name:?} is duplicated"),
        }
    }
}
impl std::error::Error for PipelineConfigError {}

/// Why a run stopped. No variant authorizes retrying completed work.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum PipelineFailure {
    /// Another run on this pipeline or a clone is active; nothing dispatched.
    Busy,
    /// The pipeline handle interrupted the run; active effects may be uncertain.
    Interrupted,
    /// An agent failed or requires explicit approval/reconciliation.
    Agent(Box<AgentError>),
    /// A completed intermediate output cannot be safely mapped to text input.
    InvalidHandoff(String),
    /// Checkpoint storage failed; the last active stage may have had effects.
    Store(String),
    /// Stored progress is incompatible, in flight, or already finished.
    UnsafeResume(String),
}

/// Run failure with an ordered record of already-observed completed replies.
/// The failing agent may have additional state or external effects; inspect it
/// separately. This value is diagnostic data, not a resumable checkpoint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PipelineError {
    /// One-based active stage (or source of invalid handoff); None for Busy.
    pub step: Option<usize>,
    pub agent_name: Option<String>,
    pub completed: Vec<PipelineStep>,
    pub cause: PipelineFailure,
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(step) = self.step {
            write!(
                f,
                "pipeline stage {step} ({}) stopped: ",
                self.agent_name.as_deref().unwrap_or("unknown")
            )?;
        }
        match &self.cause {
            PipelineFailure::Busy => f.write_str("pipeline already has an active run"),
            PipelineFailure::Interrupted => {
                f.write_str("pipeline interrupted; prior effects are not rolled back")
            }
            PipelineFailure::Agent(error) => write!(f, "{error}"),
            PipelineFailure::InvalidHandoff(reason) => write!(f, "invalid handoff: {reason}"),
            PipelineFailure::Store(reason) => write!(f, "checkpoint store: {reason}"),
            PipelineFailure::UnsafeResume(reason) => write!(f, "unsafe pipeline resume: {reason}"),
        }
    }
}
impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            PipelineFailure::Agent(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}
