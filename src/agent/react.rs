//! Minimal non-streaming `ReAct` agent loop.

mod streaming;

use std::{fmt, sync::Arc};

use crate::{
    AgentEventStream, AgentHook, AgentHookEvent, ContentBlock, GenerateOptions, Msg, Role,
    ToolCallBlock,
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
    hooks: Vec<Arc<dyn AgentHook>>,
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
            hooks: Vec::new(),
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

    /// Adds an owned read-only lifecycle hook.
    #[must_use]
    pub fn with_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// Adds a shared read-only lifecycle hook.
    #[must_use]
    pub fn with_shared_hook(mut self, hook: Arc<dyn AgentHook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Returns lifecycle hooks in execution order.
    #[must_use]
    pub fn hooks(&self) -> &[Arc<dyn AgentHook>] {
        &self.hooks
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

    /// Stores an external message in configured memory without calling the model.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::MemoryNotConfigured`] when no memory is attached,
    /// or propagates hook and memory failures.
    #[must_use]
    pub fn observe(&self, message: Msg) -> AgentFuture<'_, ()> {
        Box::pin(async move {
            let memory = self
                .memory
                .as_ref()
                .ok_or(AgentError::MemoryNotConfigured)?;
            self.notify_hooks(&AgentHookEvent::BeforeObserve {
                message: message.clone(),
            })
            .await?;
            memory.append(vec![message.clone()]).await?;
            self.notify_hooks(&AgentHookEvent::AfterObserve { message })
                .await?;
            Ok(())
        })
    }

    /// Produces one reply using a `ReAct` conversation.
    #[must_use]
    pub fn reply(&self, message: Msg) -> AgentFuture<'_, Msg> {
        Box::pin(async move {
            self.notify_hooks(&AgentHookEvent::BeforeReply {
                message: message.clone(),
            })
            .await?;
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
                self.notify_hooks(&AgentHookEvent::BeforeModelCall {
                    step: step + 1,
                    request: request.clone(),
                })
                .await?;
                let response = self.model.generate(request).await?;
                if !response.is_last {
                    return Err(AgentError::InvalidModelResponse(
                        "complete generation returned a partial response".to_owned(),
                    ));
                }
                self.notify_hooks(&AgentHookEvent::AfterModelCall {
                    step: step + 1,
                    response: response.clone(),
                })
                .await?;
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
                    self.notify_hooks(&AgentHookEvent::AfterReply {
                        steps: step + 1,
                        message: assistant_message.clone(),
                    })
                    .await?;
                    return Ok(assistant_message);
                }

                if step + 1 == self.max_steps {
                    if let Some(memory) = &self.memory {
                        memory.append(vec![assistant_message]).await?;
                    }
                    return Err(AgentError::MaxStepsExceeded {
                        max_steps: self.max_steps,
                    });
                }
                for call in &calls {
                    self.notify_hooks(&AgentHookEvent::BeforeToolCall {
                        step: step + 1,
                        call: call.clone(),
                    })
                    .await?;
                }
                if let Some(memory) = &self.memory {
                    memory.append(vec![assistant_message.clone()]).await?;
                }
                history.push(assistant_message);
                let results = self.tools.execute_all(&calls, ToolContext::new()).await?;
                let observation = Msg::new(
                    "tool",
                    Role::Assistant,
                    results.iter().cloned().map(ContentBlock::from),
                );
                if let Some(memory) = &self.memory {
                    memory.append(vec![observation.clone()]).await?;
                }
                for result in &results {
                    self.notify_hooks(&AgentHookEvent::AfterToolCall {
                        step: step + 1,
                        result: result.clone(),
                    })
                    .await?;
                }
                history.push(observation);
            }

            unreachable!("positive max_steps and loop exits cover all responses")
        })
    }

    pub(super) fn notify_hooks<'a>(&'a self, event: &'a AgentHookEvent) -> AgentFuture<'a, ()> {
        Box::pin(async move {
            for hook in &self.hooks {
                hook.on_event(event).await?;
            }
            Ok(())
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

    fn observe(&self, message: Msg) -> AgentFuture<'_, ()> {
        Self::observe(self, message)
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
            .field("hooks", &self.hooks.len())
            .finish()
    }
}
