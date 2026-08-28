//! Minimal non-streaming `ReAct` agent loop.

use std::{fmt, sync::Arc};

use crate::{
    ContentBlock, GenerateOptions, Msg, Role, ToolCallBlock,
    model::{ChatModel, ChatRequest, FinishReason},
    tool::{ToolContext, ToolExecutor},
};

use super::{Agent, AgentError, AgentFuture, AgentResult};

const DEFAULT_MAX_STEPS: usize = 8;

/// A minimal reason-act-observe agent.
///
/// Each [`Self::reply`] call owns an isolated, temporary conversation. The
/// model is called until it returns no tool calls or reaches `max_steps`.
#[derive(Clone)]
pub struct ReActAgent {
    name: String,
    model: Arc<dyn ChatModel>,
    tools: ToolExecutor,
    max_steps: usize,
    system_prompt: Option<String>,
    options: GenerateOptions,
}

impl ReActAgent {
    /// Creates an agent from an owned model and tool executor.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::EmptyName`] when `name` is blank.
    pub fn new<M>(name: impl Into<String>, model: M, tools: ToolExecutor) -> AgentResult<Self>
    where
        M: ChatModel + 'static,
    {
        Self::from_shared(name, Arc::new(model), tools)
    }

    /// Creates an agent from a shared model trait object and tool executor.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::EmptyName`] when `name` is blank.
    pub fn from_shared(
        name: impl Into<String>,
        model: Arc<dyn ChatModel>,
        tools: ToolExecutor,
    ) -> AgentResult<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(AgentError::EmptyName);
        }
        Ok(Self {
            name,
            model,
            tools,
            max_steps: DEFAULT_MAX_STEPS,
            system_prompt: None,
            options: GenerateOptions::new(),
        })
    }

    /// Sets the maximum number of model calls in one reply.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::ZeroMaxSteps`] when `max_steps` is zero.
    pub fn with_max_steps(mut self, max_steps: usize) -> AgentResult<Self> {
        if max_steps == 0 {
            return Err(AgentError::ZeroMaxSteps);
        }
        self.max_steps = max_steps;
        Ok(self)
    }

    /// Sets instructions prepended to every isolated conversation.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets generation options used for every model call.
    #[must_use]
    pub fn with_options(mut self, options: GenerateOptions) -> Self {
        self.options = options;
        self
    }

    /// Returns the model/tool iteration limit.
    #[must_use]
    pub const fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Returns the configured tool executor.
    #[must_use]
    pub const fn tools(&self) -> &ToolExecutor {
        &self.tools
    }

    /// Produces one reply using an isolated `ReAct` conversation.
    #[must_use]
    pub fn reply(&self, message: Msg) -> AgentFuture<'_, Msg> {
        Box::pin(async move {
            let mut history = Vec::new();
            if let Some(prompt) = &self.system_prompt {
                history.push(Msg::system(prompt));
            }
            history.push(message);

            for step in 0..self.max_steps {
                let request = ChatRequest::new(history.clone())
                    .with_options(self.options.clone())
                    .with_tools(self.tools.registry().definitions());
                let response = self.model.generate(request).await?;
                if !response.is_last {
                    return Err(AgentError::InvalidModelResponse(
                        "complete generation returned a partial response".to_owned(),
                    ));
                }
                let finish_reason = response.finish_reason;
                let calls = response
                    .tool_calls()
                    .cloned()
                    .collect::<Vec<ToolCallBlock>>();
                let assistant_message = response.into_assistant_msg(&self.name);

                if calls.is_empty() {
                    if finish_reason == Some(FinishReason::ToolCalls) {
                        return Err(AgentError::InvalidModelResponse(
                            "finish reason requested tools, but no tool calls were present"
                                .to_owned(),
                        ));
                    }
                    return Ok(assistant_message);
                }

                history.push(assistant_message);
                if step + 1 == self.max_steps {
                    return Err(AgentError::MaxStepsExceeded {
                        max_steps: self.max_steps,
                    });
                }
                let results = self.tools.execute_all(&calls, ToolContext::new()).await?;
                history.push(Msg::new(
                    "tool",
                    Role::Assistant,
                    results.into_iter().map(ContentBlock::from),
                ));
            }

            unreachable!("positive max_steps and loop exits cover all responses")
        })
    }
}

impl Agent for ReActAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn reply(&self, message: Msg) -> AgentFuture<'_, Msg> {
        Self::reply(self, message)
    }
}

impl fmt::Debug for ReActAgent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReActAgent")
            .field("name", &self.name)
            .field("model", &self.model.name())
            .field("tools", &self.tools)
            .field("max_steps", &self.max_steps)
            .field("system_prompt", &self.system_prompt)
            .field("options", &self.options)
            .finish()
    }
}
