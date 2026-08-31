//! Minimal non-streaming `ReAct` agent loop.

mod streaming;

use std::{fmt, sync::Arc};

use crate::{
    AgentEventStream, ContentBlock, GenerateOptions, Msg, Role, ToolCallBlock,
    memory::Memory,
    model::{ChatModel, ChatRequest, FinishReason},
    tool::{ToolContext, ToolExecutor},
};

use super::{Agent, AgentError, AgentFuture, AgentResult};

const DEFAULT_MAX_STEPS: usize = 8;

/// A minimal reason-act-observe agent.
///
/// The model is called until it returns no tool calls or reaches `max_steps`.
/// Conversations are isolated unless a [`Memory`] implementation is attached.
#[derive(Clone)]
pub struct ReActAgent {
    name: String,
    model: Arc<dyn ChatModel>,
    tools: ToolExecutor,
    max_steps: usize,
    system_prompt: Option<String>,
    options: GenerateOptions,
    memory: Option<Arc<dyn Memory>>,
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
            memory: None,
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

    /// Sets instructions prepended to every model request without persisting them.
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

    /// Attaches owned conversation memory shared across replies.
    #[must_use]
    pub fn with_memory<M>(mut self, memory: M) -> Self
    where
        M: Memory + 'static,
    {
        self.memory = Some(Arc::new(memory));
        self
    }

    /// Attaches shared conversation memory across replies.
    #[must_use]
    pub fn with_shared_memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Returns the attached conversation memory, when configured.
    #[must_use]
    pub const fn memory(&self) -> Option<&Arc<dyn Memory>> {
        self.memory.as_ref()
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

    /// Produces one reply using a `ReAct` conversation.
    #[must_use]
    pub fn reply(&self, message: Msg) -> AgentFuture<'_, Msg> {
        Box::pin(async move {
            let mut history = match &self.memory {
                Some(memory) => memory.messages().await?,
                None => Vec::new(),
            };
            if let Some(memory) = &self.memory {
                memory.append(vec![message.clone()]).await?;
            }
            history.push(message);

            let system_prompt = self.system_prompt.as_ref().map(Msg::system);

            for step in 0..self.max_steps {
                let request_messages = system_prompt.iter().cloned().chain(history.iter().cloned());
                let request = ChatRequest::new(request_messages)
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
                    if let Some(memory) = &self.memory {
                        memory.append(vec![assistant_message.clone()]).await?;
                    }
                    return Ok(assistant_message);
                }

                if let Some(memory) = &self.memory {
                    memory.append(vec![assistant_message.clone()]).await?;
                }
                history.push(assistant_message);
                if step + 1 == self.max_steps {
                    return Err(AgentError::MaxStepsExceeded {
                        max_steps: self.max_steps,
                    });
                }
                let results = self.tools.execute_all(&calls, ToolContext::new()).await?;
                let observation = Msg::new(
                    "tool",
                    Role::Assistant,
                    results.into_iter().map(ContentBlock::from),
                );
                if let Some(memory) = &self.memory {
                    memory.append(vec![observation.clone()]).await?;
                }
                history.push(observation);
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

    fn stream(&self, message: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
        Self::stream(self, message)
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
            .field("has_memory", &self.memory.is_some())
            .finish()
    }
}
