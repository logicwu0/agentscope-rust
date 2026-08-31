//! Streaming `ReAct` execution state machine.

use std::{future::Future, pin::Pin};

use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;

use crate::{
    AgentError, AgentEvent, AgentEventStream, ContentBlock, Msg, Role, ToolCallBlock,
    model::{ChatRequest, ChatResponse, ChatResponseAccumulator, FinishReason},
    tool::ToolContext,
};

use super::ReActAgent;
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
    /// configured memory.
    #[must_use]
    pub fn stream(&self, message: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
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
            Ok(self.agent_event_stream(history, system_prompt))
        })
    }

    fn agent_event_stream(
        &self,
        mut history: Vec<Msg>,
        system_prompt: Option<Msg>,
    ) -> AgentEventStream<'_> {
        Box::pin(stream! {
            for step_index in 0..self.max_steps {
                let step = step_index + 1;
                let request = self.chat_request(&history, system_prompt.as_ref());
                let mut model_step = self.model_step(step, request);
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
                let calls = response.tool_calls().cloned().collect::<Vec<_>>();
                let finish_reason = response.finish_reason;
                let assistant = response.into_assistant_msg(&self.name);

                if calls.is_empty() {
                    if finish_reason == Some(FinishReason::ToolCalls) {
                        yield Ok(error_event(step, AgentError::InvalidModelResponse(
                            "finish reason requested tools, but no tool calls were present".to_owned(),
                        )));
                        return;
                    }
                    if let Err(error) = self.remember(assistant.clone()).await {
                        yield Ok(error_event(step, error));
                        return;
                    }
                    yield Ok(AgentEvent::Finished { steps: step, message: assistant });
                    return;
                }

                if let Err(error) = self.remember(assistant.clone()).await {
                    yield Ok(error_event(step, error));
                    return;
                }
                history.push(assistant);
                if step == self.max_steps {
                    yield Ok(error_event(step, AgentError::MaxStepsExceeded {
                        max_steps: self.max_steps,
                    }));
                    return;
                }

                let mut tool_step = self.tool_step(step, calls);
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
                            if let Err(error) = self.remember(observation.clone()).await {
                                yield Ok(error_event(step, error));
                                return;
                            }
                            history.push(observation);
                        }
                    }
                }
            }
        })
    }

    fn model_step(&self, step: usize, request: ChatRequest) -> ModelStepStream<'_> {
        Box::pin(stream! {
            let mut events = match self.model.stream(request).await {
                Ok(events) => events,
                Err(error) => {
                    yield ModelStepItem::Event(error_event(step, AgentError::Model(error)));
                    return;
                }
            };
            let mut accumulator = ChatResponseAccumulator::new();
            while let Some(event) = events.next().await {
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

    fn tool_step(&self, step: usize, calls: Vec<ToolCallBlock>) -> ToolStepStream<'_> {
        Box::pin(stream! {
            for call in &calls {
                yield ToolStepItem::Event(AgentEvent::ToolStarted {
                    step,
                    call: call.clone(),
                });
            }
            let results = match self.tools.execute_all(&calls, ToolContext::new()).await {
                Ok(results) => results,
                Err(error) => {
                    yield ToolStepItem::Event(error_event(step, AgentError::Tool(error)));
                    return;
                }
            };
            for result in &results {
                yield ToolStepItem::Event(AgentEvent::ToolFinished {
                    step,
                    result: result.clone(),
                });
            }
            yield ToolStepItem::Observation(Msg::new(
                "tool",
                Role::Assistant,
                results.into_iter().map(ContentBlock::from),
            ));
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
