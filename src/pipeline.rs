//! Minimal sequential agent orchestration. No shared memory or automatic replay.

mod error;
pub use error::{PipelineConfigError, PipelineError, PipelineFailure};

use crate::{Agent, AgentInterruptHandle, ContentBlock, Msg, Role};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, future::Future, pin::Pin, sync::Arc};
use tokio::sync::Mutex;

/// One completed stage. Indices are one-based pipeline stages, not `ReAct` steps.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PipelineStep {
    /// One-based position in the fixed stage list.
    pub step: usize,
    /// Configured stage agent name, captured when constructing the pipeline.
    pub agent_name: String,
    /// Original reply; may include content that is not passed to the next agent.
    pub message: Msg,
}

/// Completed run with the last reply and all observed stage results.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PipelineOutput {
    /// Last agent's original reply, not a synthesized message.
    pub message: Msg,
    /// Original completed outputs; may contain private metadata/thinking.
    pub steps: Vec<PipelineStep>,
}

/// A lazy, cancellable sequential run.
pub type PipelineFuture<'a> =
    Pin<Box<dyn Future<Output = Result<PipelineOutput, PipelineError>> + Send + 'a>>;

struct Stage {
    name: String,
    agent: Arc<dyn Agent>,
}

/// Runs each agent once in order, forwarding only visible text as new user input.
///
/// This is an orchestrator, not an `Agent`: no combined history, state store or
/// recovery API is implied. Supply independent agents/memories/session keys.
/// Deliberately shared backing memory cannot be detected through `dyn Agent`.
/// Do not run the same agents independently while this pipeline uses them.
/// Clones share a run lock and interrupt handle; overlapping runs fail as Busy.
#[derive(Clone)]
pub struct SequentialPipeline {
    stages: Arc<Vec<Stage>>,
    operation: Arc<Mutex<()>>,
    interrupt: AgentInterruptHandle,
}

impl SequentialPipeline {
    /// Creates a fixed-order pipeline with nonempty, unique agent names.
    /// # Errors
    /// Rejects an empty list or blank/duplicate names. Does not invoke agents.
    pub fn new(agents: Vec<Arc<dyn Agent>>) -> Result<Self, PipelineConfigError> {
        if agents.is_empty() {
            return Err(PipelineConfigError::Empty);
        }
        let mut names = BTreeSet::new();
        let mut stages = Vec::with_capacity(agents.len());
        for (index, agent) in agents.into_iter().enumerate() {
            let name = agent.name().to_owned();
            if name.trim().is_empty() {
                return Err(PipelineConfigError::EmptyName { step: index + 1 });
            }
            if !names.insert(name.clone()) {
                return Err(PipelineConfigError::DuplicateName(name));
            }
            stages.push(Stage { name, agent });
        }
        Ok(Self {
            stages: Arc::new(stages),
            operation: Arc::new(Mutex::new(())),
            interrupt: AgentInterruptHandle::new(),
        })
    }

    /// Interrupts active pipeline runs only. Future runs use a new signal baseline.
    /// It drops the active reply future, not unrelated uses of an agent's handle.
    /// Already committed state/tool effects are not undone.
    #[must_use]
    pub fn interrupt_handle(&self) -> AgentInterruptHandle {
        self.interrupt.clone()
    }

    /// Runs from stage one; every invocation is a NEW run, never a resume.
    ///
    /// The first agent receives the supplied message unchanged. Later agents get
    /// a fresh User message named after the preceding stage, containing its text
    /// blocks joined by newline. Thinking, metadata, usage and history are not
    /// forwarded. Mixed non-text outputs are rejected rather than silently lost.
    ///
    /// No agent is called until this future is polled. Dropping it stops dispatch
    /// and releases the lock, but cannot undo completed or in-flight side effects.
    /// Agent-level persistence/cancellation guarantees still apply; there is no
    /// pipeline transaction or automatic checkpoint/save/rollback on cancellation.
    /// # Errors
    /// Busy, interruption, agent failure (including confirmation/in-doubt), or an
    /// unusable intermediate reply. Completed replies accompany every failure.
    #[must_use]
    pub fn run(&self, input: Msg) -> PipelineFuture<'_> {
        Box::pin(async move {
            let _guard = self.operation.try_lock().map_err(|_| PipelineError {
                step: None,
                agent_name: None,
                completed: Vec::new(),
                cause: PipelineFailure::Busy,
            })?;
            let mut interrupt = self.interrupt.token();
            let mut completed = Vec::with_capacity(self.stages.len());
            let mut incoming = input;
            for (index, stage) in self.stages.iter().enumerate() {
                // `biased` ensures an observed cancellation wins over a ready
                // reply and prevents dispatch of subsequent stages.
                let result = tokio::select! {
                    biased;
                    () = interrupt.cancelled() => Err(PipelineFailure::Interrupted),
                    result = async { stage.agent.reply(incoming).await } => result.map_err(|e| PipelineFailure::Agent(Box::new(e))),
                };
                let message = result.map_err(|cause| PipelineError {
                    step: Some(index + 1),
                    agent_name: Some(stage.name.clone()),
                    completed: std::mem::take(&mut completed),
                    cause,
                })?;
                completed.push(PipelineStep {
                    step: index + 1,
                    agent_name: stage.name.clone(),
                    message: message.clone(),
                });
                if index + 1 == self.stages.len() {
                    return Ok(PipelineOutput {
                        message,
                        steps: completed,
                    });
                }
                incoming = handoff(&stage.name, &message).map_err(|reason| PipelineError {
                    // This stage completed but its output cannot be handed off.
                    step: Some(index + 1),
                    agent_name: Some(stage.name.clone()),
                    completed: std::mem::take(&mut completed),
                    cause: PipelineFailure::InvalidHandoff(reason),
                })?;
            }
            unreachable!("constructor rejects empty pipelines")
        })
    }
}

fn handoff(source: &str, message: &Msg) -> Result<Msg, String> {
    let mut texts = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text(text) => texts.push(text.text.as_str()),
            ContentBlock::Thinking(_) => {}
            ContentBlock::ToolCall(_)
            | ContentBlock::ToolResult(_)
            | ContentBlock::Data(_)
            | ContentBlock::StructuredOutput(_) => {
                return Err("v1 pipeline handoff supports visible text only".into());
            }
        }
    }
    let text = texts.join("\n");
    if text.trim().is_empty() {
        return Err("agent reply has no visible text to hand off".into());
    }
    Ok(Msg::new(source, Role::User, [ContentBlock::from(text)]))
}

#[cfg(test)]
mod tests;
