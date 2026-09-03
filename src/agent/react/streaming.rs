//! Streaming `ReAct` execution state machine.

use std::{future::Future, pin::Pin};

use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;

use crate::{
    AgentError, AgentEvent, AgentEventStream, AgentHookEvent, Msg, ToolCallBlock,
    model::{ChatRequest, ChatResponse, ChatResponseAccumulator, FinishReason},
    tool::ToolContext,
};

use super::{ReActAgent, ensure_not_interrupted, interrupted_tool_results};
use crate::agent::interrupt::AgentInterruptToken;
use crate::agent::{AgentFuture, AgentResult};

type AgentOperation<'a, T> = Pin<Box<dyn Future<Output = AgentResult<T>> + Send + 'a>>;
type ModelStepStream<'a> = Pin<Box<dyn Stream<Item = ModelStepItem> + Send + 'a>>;
type ToolStepStream<'a> = Pin<Box<dyn Stream<Item = ToolStepItem> + Send + 'a>>;

enum ModelStepItem {
    Event(AgentEvent),
    Response(ChatResponse),
}

enum ToolStepItem {
    Event(AgentEvent),
    Observation(Msg),
}

impl ReActAgent {
    /// Streams one reply through the complete `ReAct` loop.
    ///
    /// Runtime failures after the stream is created are emitted once as a
    /// terminal [`AgentEvent::Error`]. Only complete messages are persisted to
    /// configured memory. When a state store is bound, callers must poll the
    /// stream through its terminal event for its final state to be saved.
    #[must_use]
    pub fn stream(&self, message: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
        Box::pin(async move {
            let operation = self.begin_state_operation().await?;
            let events = match self.stream_without_state_store(message).await {
                Ok(events) => events,
                Err(error) => {
                    self.finish_state_operation(operation).await?;
                    return Err(error);
                }
            };
            let Some(operation) = operation else {
                return Ok(events);
            };
            Ok(Box::pin(stream! {
                let mut events = events;
                let mut operation = Some(operation);
                while let Some(event) = events.next().await {
                    let terminal = event.as_ref().map_or(true, |event| matches!(
                        event,
                        AgentEvent::Finished { .. } | AgentEvent::Error { .. }
                    ));
                    if terminal {
                        match self.finish_state_operation(operation.take()).await {
                            Ok(()) => yield event,
                            Err(error) => yield Ok(AgentEvent::Error { step: None, error }),
                        }
                        return;
                    }
                    yield event;
                }
                if let Err(error) = self.finish_state_operation(operation).await {
                    yield Ok(AgentEvent::Error { step: None, error });
                }
            }) as AgentEventStream<'_>)
        })
    }

    fn stream_without_state_store(&self, message: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
        Box::pin(async move {
            let interrupt = self.interrupt.token();
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
            let system_prompt = self.system_prompt.as_ref().map(Msg::system);
            Ok(self.agent_event_stream(history, system_prompt, interrupt))
        })
    }

    fn agent_event_stream(
        &self,
        mut history: Vec<Msg>,
        system_prompt: Option<Msg>,
        interrupt: AgentInterruptToken,
    ) -> AgentEventStream<'_> {
        Box::pin(stream! {
            for step_index in 0..self.max_steps {
                let step = step_index + 1;
                if let Err(error) = ensure_not_interrupted(&interrupt) {
                    yield Ok(error_event(step, error));
                    return;
                }
                let request = self.chat_request(&history, system_prompt.as_ref());
                if let Err(error) = self.notify_hooks(&AgentHookEvent::BeforeModelCall {
                    step,
                    request: request.clone(),
                }).await {
                    yield Ok(error_event(step, error));
                    return;
                }
                if let Err(error) = ensure_not_interrupted(&interrupt) {
                    yield Ok(error_event(step, error));
                    return;
                }
                let mut model_step = self.model_step(step, request, interrupt.clone());
                let mut completed_response = None;
                while let Some(item) = model_step.next().await {
                    match item {
                        ModelStepItem::Event(event) => {
                            let terminal = is_error(&event);
                            yield Ok(event);
                            if terminal {
                                return;
                            }
                        }
                        ModelStepItem::Response(response) => completed_response = Some(response),
                    }
                }
                let Some(response) = completed_response else {
                    return;
                };
                if let Err(error) = self.notify_hooks(&AgentHookEvent::AfterModelCall {
                    step,
                    response: response.clone(),
                }).await {
                    yield Ok(error_event(step, error));
                    return;
                }
                if let Err(error) = ensure_not_interrupted(&interrupt) {
                    yield Ok(error_event(step, error));
                    return;
                }
                let calls = response.tool_calls().cloned().collect::<Vec<_>>();
                let finish_reason = response.finish_reason;
                let assistant = response.into_assistant_msg(&self.name);

                if calls.is_empty() {
                    match self.finish_streamed_reply(
                        step,
                        finish_reason,
                        assistant,
                        &interrupt,
                    ).await {
                        Ok(event) => yield Ok(event),
                        Err(error) => yield Ok(error_event(step, error)),
                    }
                    return;
                }

                if step == self.max_steps {
                    yield Ok(self.max_steps_event(step, assistant).await);
                    return;
                }

                if let Err(error) = self.notify_before_tool_calls(step, &calls).await {
                    yield Ok(error_event(step, error));
                    return;
                }
                if let Err(error) = ensure_not_interrupted(&interrupt) {
                    yield Ok(error_event(step, error));
                    return;
                }
                if let Err(error) = self.remember(assistant.clone()).await {
                    yield Ok(error_event(step, error));
                    return;
                }
                history.push(assistant);

                let mut tool_step = self.tool_step(step, calls, interrupt.clone());
                while let Some(item) = tool_step.next().await {
                    match item {
                        ToolStepItem::Event(event) => {
                            let terminal = is_error(&event);
                            yield Ok(event);
                            if terminal {
                                return;
                            }
                        }
                        ToolStepItem::Observation(observation) => {
                            history.push(observation);
                        }
                    }
                }
            }
        })
    }

    fn model_step(
        &self,
        step: usize,
        request: ChatRequest,
        mut interrupt: AgentInterruptToken,
    ) -> ModelStepStream<'_> {
        Box::pin(stream! {
            let started = tokio::select! {
                biased;
                () = interrupt.cancelled() => {
                    yield ModelStepItem::Event(error_event(step, AgentError::Interrupted));
                    return;
                }
                started = self.model.stream(request) => started,
            };
            let mut events = match started {
                Ok(events) => events,
                Err(error) => {
                    yield ModelStepItem::Event(error_event(step, AgentError::Model(error)));
                    return;
                }
            };
            let mut accumulator = ChatResponseAccumulator::new();
            loop {
                let next = tokio::select! {
                    biased;
                    () = interrupt.cancelled() => {
                        yield ModelStepItem::Event(error_event(step, AgentError::Interrupted));
                        return;
                    }
                    next = events.next() => next,
                };
                let Some(event) = next else {
                    break;
                };
                let event = match event {
                    Ok(event) => event,
                    Err(error) => {
                        yield ModelStepItem::Event(error_event(step, AgentError::Model(error)));
                        return;
                    }
                };
                let agent_event = AgentEvent::from_chat_event(step, event.clone());
                if let Err(error) = accumulator.apply(event) {
                    yield ModelStepItem::Event(error_event(step, error.into()));
                    return;
                }
                let terminal = is_error(&agent_event);
                yield ModelStepItem::Event(agent_event);
                if terminal {
                    return;
                }
            }
            match accumulator.into_response() {
                Ok(response) => yield ModelStepItem::Response(response),
                Err(error) => yield ModelStepItem::Event(error_event(step, error.into())),
            }
        })
    }

    fn finish_streamed_reply<'a>(
        &'a self,
        step: usize,
        finish_reason: Option<FinishReason>,
        assistant: Msg,
        interrupt: &'a AgentInterruptToken,
    ) -> AgentOperation<'a, AgentEvent> {
        Box::pin(async move {
            if finish_reason == Some(FinishReason::ToolCalls) {
                return Err(AgentError::InvalidModelResponse(
                    "finish reason requested tools, but no tool calls were present".to_owned(),
                ));
            }
            ensure_not_interrupted(interrupt)?;
            self.remember(assistant.clone()).await?;
            self.notify_hooks(&AgentHookEvent::AfterReply {
                steps: step,
                message: assistant.clone(),
            })
            .await?;
            Ok(AgentEvent::Finished {
                steps: step,
                message: assistant,
            })
        })
    }

    async fn max_steps_event(&self, step: usize, assistant: Msg) -> AgentEvent {
        match self.remember(assistant).await {
            Ok(()) => error_event(
                step,
                AgentError::MaxStepsExceeded {
                    max_steps: self.max_steps,
                },
            ),
            Err(error) => error_event(step, error),
        }
    }

    fn tool_step(
        &self,
        step: usize,
        calls: Vec<ToolCallBlock>,
        mut interrupt: AgentInterruptToken,
    ) -> ToolStepStream<'_> {
        Box::pin(stream! {
            for call in &calls {
                yield ToolStepItem::Event(AgentEvent::ToolStarted {
                    step,
                    call: call.clone(),
                });
            }
            let execution = tokio::select! {
                biased;
                () = interrupt.cancelled() => {
                    match interrupted_tool_results(&calls) {
                        Ok(results) => (results, true),
                        Err(error) => {
                            yield ToolStepItem::Event(error_event(step, error));
                            return;
                        }
                    }
                }
                results = self.tools.execute_all(&calls, ToolContext::new()) => {
                    match results {
                        Ok(results) => (results, false),
                        Err(error) => {
                            yield ToolStepItem::Event(error_event(step, AgentError::Tool(error)));
                            return;
                        }
                    }
                }
            };
            let (results, interrupted) = execution;
            let observation = match self.record_tool_results(step, &results).await {
                Ok(observation) => observation,
                Err(error) => {
                yield ToolStepItem::Event(error_event(step, error));
                return;
                }
            };
            for result in &results {
                yield ToolStepItem::Event(AgentEvent::ToolFinished {
                    step,
                    result: result.clone(),
                });
            }
            if interrupted {
                yield ToolStepItem::Event(error_event(step, AgentError::Interrupted));
                return;
            }
            yield ToolStepItem::Observation(observation);
        })
    }

    fn remember(&self, message: Msg) -> AgentOperation<'_, ()> {
        Box::pin(async move {
            match &self.memory {
                Some(memory) => memory
                    .append(vec![message])
                    .await
                    .map_err(AgentError::Memory),
                None => Ok(()),
            }
        })
    }

    fn chat_request(&self, history: &[Msg], system_prompt: Option<&Msg>) -> ChatRequest {
        ChatRequest::new(
            system_prompt
                .into_iter()
                .cloned()
                .chain(history.iter().cloned()),
        )
        .with_options(self.options.clone())
        .with_tools(self.tools.registry().definitions())
    }
}

fn error_event(step: usize, error: AgentError) -> AgentEvent {
    AgentEvent::Error {
        step: Some(step),
        error,
    }
}

fn is_error(event: &AgentEvent) -> bool {
    matches!(event, AgentEvent::Error { .. })
}
