//! Minimal non-streaming `ReAct` agent loop.

mod streaming;

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::{Arc, Mutex},
};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::{
    AgentEventStream, AgentHook, AgentHookEvent, ContentBlock, GenerateOptions, Msg, Role,
    ToolCallBlock, ToolCallState, ToolResultBlock, ToolResultState,
    memory::Memory,
    model::{ChatModel, ChatRequest, FinishReason},
    tool::{ToolContext, ToolExecutor},
};

use super::{
    AGENT_STATE_VERSION, Agent, AgentError, AgentFuture, AgentInterruptHandle, AgentResult,
    AgentState, PendingToolCalls, StateKey, StateStore, ToolConfirmation, ToolConfirmationDecision,
};

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
    interrupt: AgentInterruptHandle,
    state_binding: Option<Arc<StateBinding>>,
    confirmation_tools: BTreeSet<String>,
    pending_tool_calls: Arc<Mutex<Option<PendingToolCalls>>>,
}

struct StateBinding {
    key: StateKey,
    store: Arc<dyn StateStore>,
    operation: Arc<AsyncMutex<()>>,
}

struct StateOperation {
    binding: Arc<StateBinding>,
    expected_revision: Option<u64>,
    _guard: OwnedMutexGuard<()>,
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
            interrupt: AgentInterruptHandle::new(),
            state_binding: None,
            confirmation_tools: BTreeSet::new(),
            pending_tool_calls: Arc::new(Mutex::new(None)),
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

    /// Binds this agent to automatic state loading and saving for one session.
    ///
    /// Conversation memory must also be configured. Clones of this agent share
    /// an operation lock; independent agent instances rely on store revisions
    /// to detect concurrent updates.
    #[must_use]
    pub fn with_state_store<S>(mut self, key: StateKey, store: S) -> Self
    where
        S: StateStore + 'static,
    {
        self.state_binding = Some(Arc::new(StateBinding::new(key, Arc::new(store))));
        self
    }

    /// Binds this agent to a shared state store for one session.
    #[must_use]
    pub fn with_shared_state_store(mut self, key: StateKey, store: Arc<dyn StateStore>) -> Self {
        self.state_binding = Some(Arc::new(StateBinding::new(key, store)));
        self
    }

    /// Returns the bound state key, when automatic persistence is configured.
    #[must_use]
    pub fn state_key(&self) -> Option<&StateKey> {
        self.state_binding.as_ref().map(|binding| &binding.key)
    }

    /// Requires explicit confirmation before a named tool can execute.
    #[must_use]
    pub fn with_tool_confirmation_required(mut self, tool_name: impl Into<String>) -> Self {
        self.confirmation_tools.insert(tool_name.into());
        self
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

    /// Returns a handle for interrupting replies already running on this agent.
    #[must_use]
    pub fn interrupt_handle(&self) -> AgentInterruptHandle {
        self.interrupt.clone()
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
            let operation = self.begin_state_operation().await?;
            let result = self.observe_without_state_store(message).await;
            self.finish_state_operation(operation).await?;
            result
        })
    }

    /// Captures the complete configured conversation history.
    #[must_use]
    pub fn snapshot(&self) -> AgentFuture<'_, AgentState> {
        Box::pin(async move {
            let operation = self.begin_state_operation().await?;
            let state = self.snapshot_memory().await;
            drop(operation);
            state
        })
    }

    /// Validates and atomically restores a conversation snapshot.
    #[must_use]
    pub fn restore(&self, state: AgentState) -> AgentFuture<'_, ()> {
        Box::pin(async move {
            let operation = self.begin_state_operation().await?;
            let result = self.restore_memory(state).await;
            if result.is_ok() {
                self.finish_state_operation(operation).await?;
            }
            result
        })
    }

    /// Produces one reply using a `ReAct` conversation.
    #[must_use]
    pub fn reply(&self, message: Msg) -> AgentFuture<'_, Msg> {
        Box::pin(async move {
            let operation = self.begin_state_operation().await?;
            let result = self.reply_without_state_store(message).await;
            self.finish_state_operation(operation).await?;
            result
        })
    }

    /// Resolves every tool call in a persisted confirmation checkpoint and
    /// continues the original reply.
    #[must_use]
    pub fn resume_tool_calls(
        &self,
        reply_id: impl Into<String>,
        confirmations: Vec<ToolConfirmation>,
    ) -> AgentFuture<'_, Msg> {
        let reply_id = reply_id.into();
        Box::pin(async move {
            let operation = self.begin_state_operation().await?;
            let result = self
                .resume_tool_calls_without_state_store(&reply_id, confirmations)
                .await;
            self.finish_state_operation(operation).await?;
            result
        })
    }

    fn reply_without_state_store(&self, message: Msg) -> AgentFuture<'_, Msg> {
        Box::pin(async move {
            let interrupt = self.interrupt.token();
            if let Some(pending) = lock(&self.pending_tool_calls).clone() {
                return Err(AgentError::ToolConfirmationRequired {
                    checkpoint: pending,
                });
            }
            self.notify_hooks(&AgentHookEvent::BeforeReply {
                message: message.clone(),
            })
            .await?;
            ensure_not_interrupted(&interrupt)?;
            let mut history = match &self.memory {
                Some(memory) => memory.messages().await?,
                None => Vec::new(),
            };
            if let Some(memory) = &self.memory {
                memory.append(vec![message.clone()]).await?;
            }
            history.push(message);

            self.continue_reply(history, interrupt, 0).await
        })
    }

    fn continue_reply(
        &self,
        mut history: Vec<Msg>,
        mut interrupt: super::interrupt::AgentInterruptToken,
        start_step: usize,
    ) -> AgentFuture<'_, Msg> {
        Box::pin(async move {
            let system_prompt = self.system_prompt.as_ref().map(Msg::system);
            for step in start_step..self.max_steps {
                ensure_not_interrupted(&interrupt)?;
                let request_messages = system_prompt.iter().cloned().chain(history.iter().cloned());
                let request = ChatRequest::new(request_messages)
                    .with_options(self.options.clone())
                    .with_tools(self.tools.registry().definitions());
                self.notify_hooks(&AgentHookEvent::BeforeModelCall {
                    step: step + 1,
                    request: request.clone(),
                })
                .await?;
                ensure_not_interrupted(&interrupt)?;
                let response = self.generate_response(request, &mut interrupt).await?;
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
                ensure_not_interrupted(&interrupt)?;
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
                    ensure_not_interrupted(&interrupt)?;
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
                if self.requires_confirmation(&calls) {
                    let checkpoint = self
                        .pause_for_confirmation(step + 1, assistant_message)
                        .await?;
                    return Err(AgentError::ToolConfirmationRequired { checkpoint });
                }
                self.notify_before_tool_calls(step + 1, &calls).await?;
                ensure_not_interrupted(&interrupt)?;
                if let Some(memory) = &self.memory {
                    memory.append(vec![assistant_message.clone()]).await?;
                }
                history.push(assistant_message);
                let results = tokio::select! {
                    biased;
                    () = interrupt.cancelled() => {
                        let results = interrupted_tool_results(&calls)?;
                        self.record_tool_results(step + 1, &results).await?;
                        return Err(AgentError::Interrupted);
                    }
                    results = self.tools.execute_all(&calls, ToolContext::new()) => results?,
                };
                let observation = self.record_tool_results(step + 1, &results).await?;
                history.push(observation);
            }

            unreachable!("positive max_steps and loop exits cover all responses")
        })
    }

    async fn resume_tool_calls_without_state_store(
        &self,
        reply_id: &str,
        confirmations: Vec<ToolConfirmation>,
    ) -> AgentResult<Msg> {
        let checkpoint = lock(&self.pending_tool_calls)
            .clone()
            .ok_or(AgentError::NoPendingToolConfirmation)?;
        if checkpoint.reply_id() != reply_id {
            return Err(AgentError::InvalidToolConfirmation(format!(
                "reply id `{reply_id}` does not match pending reply `{}`",
                checkpoint.reply_id()
            )));
        }
        let decisions = confirmation_map(&checkpoint, confirmations)?;
        let memory = self
            .memory
            .as_ref()
            .ok_or(AgentError::MemoryNotConfigured)?;
        let mut history = memory.messages().await?;
        validate_checkpoint_message(&history, &checkpoint)?;

        let approved = checkpoint
            .calls()
            .iter()
            .filter(|call| {
                matches!(
                    decisions.get(call.id()),
                    Some(ToolConfirmationDecision::Approve)
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        self.notify_before_tool_calls(checkpoint.step(), &approved)
            .await?;
        let mut interrupt = self.interrupt.token();
        ensure_not_interrupted(&interrupt)?;
        let (approved_results, interrupted) = tokio::select! {
            biased;
            () = interrupt.cancelled(), if !approved.is_empty() => {
                (interrupted_tool_results(&approved)?, true)
            }
            results = self.tools.execute_all(&approved, ToolContext::new()) => {
                (results?, false)
            }
        };
        let approved_results = approved_results
            .into_iter()
            .map(|result| (result.id().to_owned(), result))
            .collect::<BTreeMap<_, _>>();
        let mut results = Vec::with_capacity(checkpoint.calls().len());
        for call in checkpoint.calls() {
            match decisions.get(call.id()) {
                Some(ToolConfirmationDecision::Approve) => {
                    results.push(approved_results.get(call.id()).cloned().ok_or_else(|| {
                        AgentError::InvalidModelResponse(format!(
                            "approved tool `{}` produced no result",
                            call.id()
                        ))
                    })?);
                }
                Some(ToolConfirmationDecision::Deny { reason }) => {
                    results.push(
                        ToolResultBlock::finished(
                            call.id(),
                            call.name(),
                            reason.clone(),
                            ToolResultState::Denied,
                        )
                        .map_err(|error| AgentError::InvalidModelResponse(error.to_string()))?,
                    );
                }
                None => unreachable!("confirmation_map validates every pending call"),
            }
        }
        finish_checkpoint_calls(&mut history, &checkpoint);
        memory.replace(history.clone()).await?;
        let observation = self
            .record_tool_results(checkpoint.step(), &results)
            .await?;
        history.push(observation);
        *lock(&self.pending_tool_calls) = None;
        if interrupted {
            return Err(AgentError::Interrupted);
        }
        self.continue_reply(history, interrupt, checkpoint.step())
            .await
    }

    async fn observe_without_state_store(&self, message: Msg) -> AgentResult<()> {
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
    }

    async fn begin_state_operation(&self) -> AgentResult<Option<StateOperation>> {
        let Some(binding) = self.state_binding.clone() else {
            return Ok(None);
        };
        let memory = self
            .memory
            .as_ref()
            .ok_or(AgentError::MemoryNotConfigured)?;
        let guard = binding.operation.clone().lock_owned().await;
        let record = binding.store.load(&binding.key).await?;
        let expected_revision = record.as_ref().map(super::StateRecord::revision);
        if let Some(record) = record {
            let state = record.into_state();
            self.validate_state(&state)?;
            let (messages, pending) = state.into_parts();
            memory.replace(messages).await?;
            *lock(&self.pending_tool_calls) = pending;
        }
        Ok(Some(StateOperation {
            binding,
            expected_revision,
            _guard: guard,
        }))
    }

    async fn finish_state_operation(&self, operation: Option<StateOperation>) -> AgentResult<()> {
        let Some(operation) = operation else {
            return Ok(());
        };
        let state = self.snapshot_memory().await?;
        operation
            .binding
            .store
            .save(
                operation.binding.key.clone(),
                operation.expected_revision,
                state,
            )
            .await?;
        Ok(())
    }

    async fn snapshot_memory(&self) -> AgentResult<AgentState> {
        let memory = self
            .memory
            .as_ref()
            .ok_or(AgentError::MemoryNotConfigured)?;
        Ok(AgentState::new(self.name.clone(), memory.messages().await?)
            .with_pending_tool_calls(lock(&self.pending_tool_calls).clone()))
    }

    async fn restore_memory(&self, state: AgentState) -> AgentResult<()> {
        self.validate_state(&state)?;
        let memory = self
            .memory
            .as_ref()
            .ok_or(AgentError::MemoryNotConfigured)?;
        let (messages, pending) = state.into_parts();
        memory.replace(messages).await?;
        *lock(&self.pending_tool_calls) = pending;
        Ok(())
    }

    fn validate_state(&self, state: &AgentState) -> AgentResult<()> {
        if !(1..=AGENT_STATE_VERSION).contains(&state.format_version()) {
            return Err(AgentError::UnsupportedStateVersion {
                found: state.format_version(),
                supported: AGENT_STATE_VERSION,
            });
        }
        if state.agent_name() != self.name {
            return Err(AgentError::StateAgentMismatch {
                expected: self.name.clone(),
                found: state.agent_name().to_owned(),
            });
        }
        if let Some(pending) = state.pending_tool_calls() {
            pending
                .validate()
                .map_err(AgentError::InvalidToolConfirmation)?;
            if pending.step() >= self.max_steps {
                return Err(AgentError::InvalidToolConfirmation(format!(
                    "pending tool step {} leaves no model step to resume",
                    pending.step()
                )));
            }
            validate_checkpoint_message(state.messages(), pending)?;
        }
        Ok(())
    }

    fn requires_confirmation(&self, calls: &[ToolCallBlock]) -> bool {
        calls
            .iter()
            .any(|call| self.confirmation_tools.contains(call.name()))
    }

    async fn pause_for_confirmation(
        &self,
        step: usize,
        mut assistant: Msg,
    ) -> AgentResult<PendingToolCalls> {
        let memory = self
            .memory
            .as_ref()
            .ok_or(AgentError::MemoryNotConfigured)?;
        let mut calls = Vec::new();
        for content in &mut assistant.content {
            if let ContentBlock::ToolCall(call) = content {
                call.state = ToolCallState::Asking;
                calls.push(call.clone());
            }
        }
        let checkpoint = PendingToolCalls::new(assistant.id.clone(), step, calls);
        memory.append(vec![assistant]).await?;
        *lock(&self.pending_tool_calls) = Some(checkpoint.clone());
        Ok(checkpoint)
    }

    pub(super) fn notify_hooks<'a>(&'a self, event: &'a AgentHookEvent) -> AgentFuture<'a, ()> {
        Box::pin(async move {
            for hook in &self.hooks {
                hook.on_event(event).await?;
            }
            Ok(())
        })
    }

    pub(super) fn record_tool_results<'a>(
        &'a self,
        step: usize,
        results: &'a [ToolResultBlock],
    ) -> AgentFuture<'a, Msg> {
        Box::pin(async move {
            let observation = Msg::new(
                "tool",
                Role::Assistant,
                results.iter().cloned().map(ContentBlock::from),
            );
            if let Some(memory) = &self.memory {
                memory.append(vec![observation.clone()]).await?;
            }
            for result in results {
                self.notify_hooks(&AgentHookEvent::AfterToolCall {
                    step,
                    result: result.clone(),
                })
                .await?;
            }
            Ok(observation)
        })
    }

    pub(super) fn notify_before_tool_calls<'a>(
        &'a self,
        step: usize,
        calls: &'a [ToolCallBlock],
    ) -> AgentFuture<'a, ()> {
        Box::pin(async move {
            for call in calls {
                self.notify_hooks(&AgentHookEvent::BeforeToolCall {
                    step,
                    call: call.clone(),
                })
                .await?;
            }
            Ok(())
        })
    }

    fn generate_response<'a>(
        &'a self,
        request: ChatRequest,
        interrupt: &'a mut super::interrupt::AgentInterruptToken,
    ) -> AgentFuture<'a, crate::ChatResponse> {
        Box::pin(async move {
            tokio::select! {
                biased;
                () = interrupt.cancelled() => Err(AgentError::Interrupted),
                response = self.model.generate(request) => response.map_err(AgentError::Model),
            }
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

    fn snapshot(&self) -> AgentFuture<'_, AgentState> {
        Self::snapshot(self)
    }

    fn restore(&self, state: AgentState) -> AgentFuture<'_, ()> {
        Self::restore(self, state)
    }

    fn resume_tool_calls(
        &self,
        reply_id: String,
        confirmations: Vec<ToolConfirmation>,
    ) -> AgentFuture<'_, Msg> {
        Self::resume_tool_calls(self, reply_id, confirmations)
    }

    fn interrupt_handle(&self) -> AgentInterruptHandle {
        Self::interrupt_handle(self)
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
            .field("interrupt", &self.interrupt)
            .field("state_binding", &self.state_key())
            .field("confirmation_tools", &self.confirmation_tools)
            .field(
                "pending_tool_calls",
                &lock(&self.pending_tool_calls).as_ref(),
            )
            .finish()
    }
}

impl StateBinding {
    fn new(key: StateKey, store: Arc<dyn StateStore>) -> Self {
        Self {
            key,
            store,
            operation: Arc::new(AsyncMutex::new(())),
        }
    }
}

fn ensure_not_interrupted(token: &super::interrupt::AgentInterruptToken) -> AgentResult<()> {
    if token.is_interrupted() {
        Err(AgentError::Interrupted)
    } else {
        Ok(())
    }
}

fn confirmation_map(
    checkpoint: &PendingToolCalls,
    confirmations: Vec<ToolConfirmation>,
) -> AgentResult<BTreeMap<String, ToolConfirmationDecision>> {
    let pending_ids = checkpoint
        .calls()
        .iter()
        .map(ToolCallBlock::id)
        .collect::<BTreeSet<_>>();
    let mut decisions = BTreeMap::new();
    for confirmation in confirmations {
        let (call_id, decision) = confirmation.into_parts();
        if !pending_ids.contains(call_id.as_str()) {
            return Err(AgentError::InvalidToolConfirmation(format!(
                "tool call `{call_id}` is not pending"
            )));
        }
        if decisions.insert(call_id.clone(), decision).is_some() {
            return Err(AgentError::InvalidToolConfirmation(format!(
                "tool call `{call_id}` has more than one decision"
            )));
        }
    }
    let missing = checkpoint
        .calls()
        .iter()
        .filter(|call| !decisions.contains_key(call.id()))
        .map(ToolCallBlock::id)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(AgentError::InvalidToolConfirmation(format!(
            "missing decisions for tool calls: {}",
            missing.join(", ")
        )));
    }
    Ok(decisions)
}

fn validate_checkpoint_message(history: &[Msg], checkpoint: &PendingToolCalls) -> AgentResult<()> {
    let message = history
        .iter()
        .find(|message| message.id == checkpoint.reply_id())
        .ok_or_else(|| {
            AgentError::InvalidToolConfirmation(format!(
                "pending reply `{}` is missing from memory",
                checkpoint.reply_id()
            ))
        })?;
    let stored_calls = message
        .content
        .iter()
        .filter_map(|content| match content {
            ContentBlock::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    if stored_calls.len() != checkpoint.calls().len()
        || stored_calls
            .iter()
            .zip(checkpoint.calls())
            .any(|(stored, pending)| *stored != pending)
    {
        return Err(AgentError::InvalidToolConfirmation(
            "pending calls do not match the stored assistant reply".to_owned(),
        ));
    }
    Ok(())
}

fn finish_checkpoint_calls(history: &mut [Msg], checkpoint: &PendingToolCalls) {
    if let Some(message) = history
        .iter_mut()
        .find(|message| message.id == checkpoint.reply_id())
    {
        for content in &mut message.content {
            if let ContentBlock::ToolCall(call) = content {
                call.state = ToolCallState::Finished;
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(super) fn interrupted_tool_results(
    calls: &[ToolCallBlock],
) -> AgentResult<Vec<ToolResultBlock>> {
    calls
        .iter()
        .map(|call| {
            ToolResultBlock::finished(
                call.id(),
                call.name(),
                "agent operation was interrupted",
                ToolResultState::Interrupted,
            )
            .map_err(|error| AgentError::InvalidModelResponse(error.to_string()))
        })
        .collect()
}
