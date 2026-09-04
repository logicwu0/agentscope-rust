use std::{
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use futures_core::Stream;
use futures_executor::block_on;
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::Barrier;

use crate::{
    AGENT_STATE_VERSION, Agent, AgentError, AgentEvent, AgentHook, AgentHookError, AgentHookEvent,
    AgentHookFuture, AgentInterruptHandle, AgentState, ChatEvent, ChatEventStream, ChatModel,
    ChatRequest, ChatResponse, ContentBlock, FinishReason, GenerateOptions, InMemoryMemory,
    InMemoryStateStore, Memory, MockChatModel, MockTool, ModelCapabilities, ModelError,
    ModelFuture, ModelResult, Msg, ReActAgent, Role, StateKey, StateStore, Tool, ToolCallBlock,
    ToolCallState, ToolConfirmation, ToolContext, ToolDefinition, ToolExecutor, ToolFuture,
    ToolRegistry, ToolResultOutput, ToolResultState, Usage,
};

fn calculator_definition() -> ToolDefinition {
    ToolDefinition::new(
        "calculator",
        "Evaluate an arithmetic expression",
        json!({
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }),
    )
    .unwrap()
}

fn calculator_call_stream() -> [ModelResult<ChatEvent>; 4] {
    [
        Ok(ChatEvent::ToolCallDelta {
            tool_call_id: "call-stream-1".to_owned(),
            tool_name: "calculator".to_owned(),
            delta: "{\"expression\":\"".to_owned(),
        }),
        Ok(ChatEvent::ToolCallDelta {
            tool_call_id: "call-stream-1".to_owned(),
            tool_name: "calculator".to_owned(),
            delta: "6*7\"}".to_owned(),
        }),
        Ok(ChatEvent::Usage {
            usage: Usage::new(10, 4),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::ToolCalls,
        }),
    ]
}

fn final_answer_stream() -> [ModelResult<ChatEvent>; 4] {
    [
        Ok(ChatEvent::TextDelta {
            block_id: "text-final".to_owned(),
            delta: "The answer ".to_owned(),
        }),
        Ok(ChatEvent::TextDelta {
            block_id: "text-final".to_owned(),
            delta: "is 42.".to_owned(),
        }),
        Ok(ChatEvent::Usage {
            usage: Usage::new(18, 6),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]
}

struct OrderedHook {
    name: &'static str,
    events: Arc<Mutex<Vec<String>>>,
}

impl AgentHook for OrderedHook {
    fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a> {
        Box::pin(async move {
            let kind = match event {
                AgentHookEvent::BeforeObserve { .. } => "before_observe",
                AgentHookEvent::AfterObserve { .. } => "after_observe",
                AgentHookEvent::BeforeReply { .. } => "before_reply",
                AgentHookEvent::BeforeModelCall { .. } => "before_model_call",
                AgentHookEvent::AfterModelCall { .. } => "after_model_call",
                AgentHookEvent::BeforeToolCall { .. } => "before_tool_call",
                AgentHookEvent::AfterToolCall { .. } => "after_tool_call",
                AgentHookEvent::AfterReply { .. } => "after_reply",
            };
            self.events
                .lock()
                .unwrap()
                .push(format!("{}:{kind}", self.name));
            Ok(())
        })
    }
}

struct RejectToolHook;

impl AgentHook for RejectToolHook {
    fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a> {
        Box::pin(async move {
            if matches!(event, AgentHookEvent::BeforeToolCall { .. }) {
                Err(AgentHookError::new("tool observation rejected").with_code("hook_rejected"))
            } else {
                Ok(())
            }
        })
    }
}

struct InterruptingStreamModel {
    handle: Arc<Mutex<Option<AgentInterruptHandle>>>,
    requests: AtomicUsize,
}

struct BarrierModel {
    barrier: Arc<Barrier>,
}

impl ChatModel for BarrierModel {
    fn name(&self) -> &'static str {
        "barrier-model"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }

    fn generate(&self, _request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        let barrier = self.barrier.clone();
        Box::pin(async move {
            barrier.wait().await;
            Ok(ChatResponse::completed([ContentBlock::from("done")]))
        })
    }

    fn stream(&self, _request: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async { Err(ModelError::new("streaming is not used by this test")) })
    }
}

impl ChatModel for InterruptingStreamModel {
    fn name(&self) -> &'static str {
        "interrupting-stream-model"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }

    fn generate(&self, _request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let handle = self.handle.clone();
        Box::pin(async move {
            handle.lock().unwrap().clone().unwrap().interrupt();
            std::future::pending::<ModelResult<ChatResponse>>().await
        })
    }

    fn stream(&self, _request: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let handle = self.handle.clone();
        Box::pin(async move {
            Ok(Box::pin(InterruptingEventStream {
                handle,
                interrupted: false,
            }) as ChatEventStream<'_>)
        })
    }
}

struct InterruptingEventStream {
    handle: Arc<Mutex<Option<AgentInterruptHandle>>>,
    interrupted: bool,
}

impl Stream for InterruptingEventStream {
    type Item = ModelResult<ChatEvent>;

    fn poll_next(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.interrupted {
            let handle = self.handle.lock().unwrap().clone().unwrap();
            self.interrupted = true;
            handle.interrupt();
        }
        Poll::Pending
    }
}

struct InterruptingTool {
    definition: ToolDefinition,
    handle: Arc<Mutex<Option<AgentInterruptHandle>>>,
    invocations: AtomicUsize,
}

impl Tool for InterruptingTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn execute(
        &self,
        _input: serde_json::Value,
        _context: ToolContext,
    ) -> ToolFuture<'_, ToolResultOutput> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        let handle = self.handle.lock().unwrap().clone().unwrap();
        Box::pin(async move {
            handle.interrupt();
            std::future::pending::<crate::ToolResult<ToolResultOutput>>().await
        })
    }
}

#[test]
fn react_agent_completes_a_model_tool_model_loop() {
    let call = ToolCallBlock::complete("call-1", "calculator", r#"{"expression":"6*7"}"#).unwrap();
    let model = Arc::new(
        MockChatModel::new("mock-model")
            .with_response(ChatResponse::finished(
                [ContentBlock::from(call)],
                FinishReason::ToolCalls,
            ))
            .with_response(ChatResponse::completed([ContentBlock::from(
                "The answer is 42.",
            )])),
    );
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let memory = Arc::new(InMemoryMemory::new());
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::from_shared("Friday", shared_model, ToolExecutor::new(registry))
        .unwrap()
        .with_shared_memory(shared_memory)
        .with_system_prompt("Use tools when helpful.")
        .with_options(GenerateOptions::new().with_temperature(0.0));

    let reply = block_on(agent.reply(Msg::user("What is 6 * 7?"))).unwrap();

    assert_eq!(reply.name, "Friday");
    assert_eq!(reply.role, Role::Assistant);
    assert_eq!(reply.text_content(""), Some("The answer is 42.".to_owned()));
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert_eq!(
        tool.recorded_invocations()[0].input,
        json!({"expression": "6*7"})
    );

    let requests = model.recorded_requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].messages.len(), 2);
    assert_eq!(requests[0].messages[0].role, Role::System);
    assert_eq!(requests[0].tools, [calculator_definition()]);
    assert_eq!(requests[0].options.temperature, Some(0.0));
    assert_eq!(requests[1].messages.len(), 4);
    assert!(matches!(
        &requests[1].messages[2].content[0],
        ContentBlock::ToolCall(_)
    ));
    let ContentBlock::ToolResult(result) = &requests[1].messages[3].content[0] else {
        panic!("the observation message should contain a tool result")
    };
    assert_eq!(result.state(), ToolResultState::Success);
    assert_eq!(result.output(), &ToolResultOutput::Text("42".to_owned()));

    let remembered = block_on(memory.messages()).unwrap();
    assert_eq!(remembered.len(), 4);
    assert_eq!(remembered[0].role, Role::User);
    assert!(matches!(
        remembered[1].content[0],
        ContentBlock::ToolCall(_)
    ));
    assert!(matches!(
        remembered[2].content[0],
        ContentBlock::ToolResult(_)
    ));
    assert_eq!(remembered[3], reply);
}

#[test]
fn react_agent_pauses_and_resumes_an_approved_tool_call() {
    let call =
        ToolCallBlock::complete("call-confirm-1", "calculator", r#"{"expression":"6*7"}"#).unwrap();
    let model = Arc::new(
        MockChatModel::new("mock-model")
            .with_response(ChatResponse::finished(
                [ContentBlock::from(call)],
                FinishReason::ToolCalls,
            ))
            .with_response(ChatResponse::completed([ContentBlock::from("42")])),
    );
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let memory = Arc::new(InMemoryMemory::new());
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let agent = ReActAgent::from_shared("Friday", shared_model, ToolExecutor::new(registry))
        .unwrap()
        .with_shared_memory(shared_memory)
        .with_tool_confirmation_required("calculator");

    let paused = block_on(agent.reply(Msg::user("What is 6 * 7?"))).unwrap_err();
    let AgentError::ToolConfirmationRequired { checkpoint } = paused else {
        panic!("tool call should pause for confirmation")
    };
    assert_eq!(checkpoint.calls().len(), 1);
    assert_eq!(checkpoint.calls()[0].state, ToolCallState::Asking);
    assert!(tool.recorded_invocations().is_empty());

    let unrelated = block_on(agent.reply(Msg::user("Ignore that and say hello"))).unwrap_err();
    assert!(matches!(
        unrelated,
        AgentError::ToolConfirmationRequired { .. }
    ));
    assert_eq!(model.recorded_requests().len(), 1);

    let reply = block_on(agent.resume_tool_calls(
        checkpoint.reply_id(),
        vec![ToolConfirmation::approve("call-confirm-1")],
    ))
    .unwrap();

    assert_eq!(reply.text_content(""), Some("42".to_owned()));
    assert_eq!(tool.recorded_invocations().len(), 1);
    let state = block_on(agent.snapshot()).unwrap();
    assert!(state.pending_tool_calls().is_none());
    assert_eq!(state.messages().len(), 4);
    let ContentBlock::ToolCall(stored_call) = &state.messages()[1].content[0] else {
        panic!("assistant reply should contain the tool call")
    };
    assert_eq!(stored_call.state, ToolCallState::Finished);
}

#[test]
fn react_agent_denies_a_tool_without_executing_it() {
    let call =
        ToolCallBlock::complete("call-confirm-2", "calculator", r#"{"expression":"6*7"}"#).unwrap();
    let model = MockChatModel::new("mock-model")
        .with_response(ChatResponse::finished(
            [ContentBlock::from(call)],
            FinishReason::ToolCalls,
        ))
        .with_response(ChatResponse::completed([ContentBlock::from(
            "I cannot run that tool.",
        )]));
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let memory = Arc::new(InMemoryMemory::new());
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))
        .unwrap()
        .with_shared_memory(shared_memory)
        .with_tool_confirmation_required("calculator");
    let paused = block_on(agent.reply(Msg::user("Run the calculator"))).unwrap_err();
    let AgentError::ToolConfirmationRequired { checkpoint } = paused else {
        panic!("tool call should pause for confirmation")
    };

    let reply = block_on(agent.resume_tool_calls(
        checkpoint.reply_id(),
        vec![ToolConfirmation::deny(
            "call-confirm-2",
            "User denied execution",
        )],
    ))
    .unwrap();

    assert_eq!(
        reply.text_content(""),
        Some("I cannot run that tool.".to_owned())
    );
    assert!(tool.recorded_invocations().is_empty());
    let remembered = block_on(memory.messages()).unwrap();
    let ContentBlock::ToolResult(result) = &remembered[2].content[0] else {
        panic!("denial should create a tool result")
    };
    assert_eq!(result.state(), ToolResultState::Denied);
    assert_eq!(
        result.output(),
        &ToolResultOutput::Text("User denied execution".to_owned())
    );
}

#[test]
fn react_agent_rejects_invalid_confirmations_without_losing_the_checkpoint() {
    let call =
        ToolCallBlock::complete("call-confirm-3", "calculator", r#"{"expression":"1+1"}"#).unwrap();
    let model = MockChatModel::new("mock-model")
        .with_response(ChatResponse::finished(
            [ContentBlock::from(call)],
            FinishReason::ToolCalls,
        ))
        .with_response(ChatResponse::completed([ContentBlock::from("done")]));
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("ok"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("calculator");
    let paused = block_on(agent.reply(Msg::user("Run it"))).unwrap_err();
    let AgentError::ToolConfirmationRequired { checkpoint } = paused else {
        panic!("tool call should pause for confirmation")
    };

    let wrong_reply = block_on(agent.resume_tool_calls(
        "wrong-reply",
        vec![ToolConfirmation::approve("call-confirm-3")],
    ))
    .unwrap_err();
    assert!(matches!(
        wrong_reply,
        AgentError::InvalidToolConfirmation(_)
    ));
    let missing = block_on(agent.resume_tool_calls(checkpoint.reply_id(), Vec::new())).unwrap_err();
    assert!(matches!(missing, AgentError::InvalidToolConfirmation(_)));
    assert!(tool.recorded_invocations().is_empty());

    block_on(agent.resume_tool_calls(
        checkpoint.reply_id(),
        vec![ToolConfirmation::approve("call-confirm-3")],
    ))
    .unwrap();
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert_eq!(
        block_on(agent.resume_tool_calls(checkpoint.reply_id(), Vec::new())).unwrap_err(),
        AgentError::NoPendingToolConfirmation
    );
}

#[test]
fn state_store_restores_a_pending_tool_call_in_a_new_agent() {
    let key = StateKey::new("user-1", "hitl-restart").unwrap();
    let store = Arc::new(InMemoryStateStore::new());
    let first_store: Arc<dyn StateStore> = store.clone();
    let call =
        ToolCallBlock::complete("call-confirm-4", "calculator", r#"{"expression":"1+1"}"#).unwrap();
    let first_tool = Arc::new(MockTool::new(calculator_definition()).with_output("unused"));
    let mut first_registry = ToolRegistry::new();
    first_registry.register_shared(first_tool.clone()).unwrap();
    let first = ReActAgent::new(
        "Friday",
        MockChatModel::new("first-model").with_response(ChatResponse::finished(
            [ContentBlock::from(call)],
            FinishReason::ToolCalls,
        )),
        ToolExecutor::new(first_registry),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), first_store)
    .with_tool_confirmation_required("calculator");
    let paused = block_on(first.reply(Msg::user("Run after restart"))).unwrap_err();
    let AgentError::ToolConfirmationRequired { checkpoint } = paused else {
        panic!("tool call should pause for confirmation")
    };
    assert!(first_tool.recorded_invocations().is_empty());
    let saved = block_on(store.load(&key)).unwrap().unwrap();
    assert_eq!(saved.revision(), 1);
    assert!(saved.state().pending_tool_calls().is_some());

    let resumed_tool = Arc::new(MockTool::new(calculator_definition()).with_output("resumed"));
    let mut resumed_registry = ToolRegistry::new();
    resumed_registry
        .register_shared(resumed_tool.clone())
        .unwrap();
    let resumed_store: Arc<dyn StateStore> = store.clone();
    let resumed = ReActAgent::new(
        "Friday",
        MockChatModel::new("resumed-model")
            .with_response(ChatResponse::completed([ContentBlock::from("finished")])),
        ToolExecutor::new(resumed_registry),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), resumed_store)
    .with_tool_confirmation_required("calculator");

    let reply = block_on(resumed.resume_tool_calls(
        checkpoint.reply_id(),
        vec![ToolConfirmation::approve("call-confirm-4")],
    ))
    .unwrap();

    assert_eq!(reply.text_content(""), Some("finished".to_owned()));
    assert_eq!(resumed_tool.recorded_invocations().len(), 1);
    let saved = block_on(store.load(&key)).unwrap().unwrap();
    assert_eq!(saved.revision(), 2);
    assert!(saved.state().pending_tool_calls().is_none());
}

#[test]
fn react_agent_runs_lifecycle_hooks_in_registration_order() {
    let call =
        ToolCallBlock::complete("call-hook-1", "calculator", r#"{"expression":"6*7"}"#).unwrap();
    let model = MockChatModel::new("mock-model")
        .with_response(ChatResponse::finished(
            [ContentBlock::from(call)],
            FinishReason::ToolCalls,
        ))
        .with_response(ChatResponse::completed([ContentBlock::from("42")]));
    let tool = MockTool::new(calculator_definition()).with_output("42");
    let mut registry = ToolRegistry::new();
    registry.register(tool).unwrap();
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))
        .unwrap()
        .with_hook(OrderedHook {
            name: "first",
            events: recorded.clone(),
        })
        .with_hook(OrderedHook {
            name: "second",
            events: recorded.clone(),
        });

    let reply = block_on(agent.reply(Msg::user("What is 6 * 7?"))).unwrap();

    assert_eq!(reply.text_content(""), Some("42".to_owned()));
    assert_eq!(agent.hooks().len(), 2);
    assert_eq!(
        *recorded.lock().unwrap(),
        [
            "first:before_reply",
            "second:before_reply",
            "first:before_model_call",
            "second:before_model_call",
            "first:after_model_call",
            "second:after_model_call",
            "first:before_tool_call",
            "second:before_tool_call",
            "first:after_tool_call",
            "second:after_tool_call",
            "first:before_model_call",
            "second:before_model_call",
            "first:after_model_call",
            "second:after_model_call",
            "first:after_reply",
            "second:after_reply",
        ]
    );
}

#[test]
fn react_agent_uses_memory_across_replies() {
    let model = Arc::new(
        MockChatModel::new("mock-model")
            .with_response(ChatResponse::completed([ContentBlock::from(
                "I will remember that your favorite color is green.",
            )]))
            .with_response(ChatResponse::completed([ContentBlock::from(
                "Your favorite color is green.",
            )])),
    );
    let memory = Arc::new(InMemoryMemory::new());
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::from_shared(
        "Friday",
        shared_model,
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_shared_memory(shared_memory);

    block_on(agent.reply(Msg::user("My favorite color is green."))).unwrap();
    let reply = block_on(agent.reply(Msg::user("What is my favorite color?"))).unwrap();

    assert_eq!(
        reply.text_content(""),
        Some("Your favorite color is green.".to_owned())
    );
    let requests = model.recorded_requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].messages.len(), 1);
    assert_eq!(requests[1].messages.len(), 3);
    assert_eq!(
        requests[1].messages[0].text_content(""),
        Some("My favorite color is green.".to_owned())
    );
    assert_eq!(
        requests[1].messages[1].text_content(""),
        Some("I will remember that your favorite color is green.".to_owned())
    );
    assert_eq!(block_on(memory.messages()).unwrap().len(), 4);
}

#[test]
fn react_agent_observes_external_messages_without_calling_the_model() {
    let model = Arc::new(
        MockChatModel::new("mock-model").with_response(ChatResponse::completed([
            ContentBlock::from("I used the observation."),
        ])),
    );
    let memory = Arc::new(InMemoryMemory::new());
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let agent: Arc<dyn Agent> = Arc::new(
        ReActAgent::from_shared(
            "Friday",
            shared_model,
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_shared_memory(shared_memory)
        .with_hook(OrderedHook {
            name: "first",
            events: recorded.clone(),
        })
        .with_hook(OrderedHook {
            name: "second",
            events: recorded.clone(),
        }),
    );
    let observation = Msg::assistant("planner", "Use the verified result 42.");

    block_on(agent.observe(observation.clone())).unwrap();

    assert!(model.recorded_requests().is_empty());
    assert_eq!(block_on(memory.messages()).unwrap(), [observation.clone()]);
    assert_eq!(
        *recorded.lock().unwrap(),
        [
            "first:before_observe",
            "second:before_observe",
            "first:after_observe",
            "second:after_observe",
        ]
    );

    let reply = block_on(agent.reply(Msg::user("What result should I use?"))).unwrap();
    assert_eq!(
        reply.text_content(""),
        Some("I used the observation.".to_owned())
    );
    let requests = model.recorded_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].messages.len(), 2);
    assert_eq!(requests[0].messages[0], observation);
    assert_eq!(requests[0].messages[1].role, Role::User);
}

#[test]
fn react_agent_rejects_observation_without_memory() {
    let model = Arc::new(MockChatModel::new("mock-model"));
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let agent = ReActAgent::from_shared(
        "Friday",
        shared_model,
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap();

    let error = block_on(agent.observe(Msg::assistant("planner", "Remember this."))).unwrap_err();

    assert_eq!(error, AgentError::MemoryNotConfigured);
    assert!(model.recorded_requests().is_empty());
}

#[test]
fn react_agent_snapshots_and_restores_complete_conversation_state() {
    let source_memory = Arc::new(InMemoryMemory::from_messages([
        Msg::system("Be concise."),
        Msg::user("Remember this conversation."),
    ]));
    let source_shared: Arc<dyn Memory> = source_memory.clone();
    let source = ReActAgent::new(
        "Friday",
        MockChatModel::new("source-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_shared_memory(source_shared);
    let target_memory = Arc::new(InMemoryMemory::from_messages([Msg::user("replace me")]));
    let target_shared: Arc<dyn Memory> = target_memory.clone();
    let target = ReActAgent::new(
        "Friday",
        MockChatModel::new("target-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_shared_memory(target_shared);

    let snapshot = block_on(source.snapshot()).unwrap();
    let serialized = serde_json::to_string(&snapshot).unwrap();
    let transferred: AgentState = serde_json::from_str(&serialized).unwrap();
    block_on(target.restore(transferred)).unwrap();

    assert_eq!(snapshot.format_version(), AGENT_STATE_VERSION);
    assert_eq!(snapshot.agent_name(), "Friday");
    assert_eq!(block_on(target.snapshot()).unwrap(), snapshot);
    assert_eq!(block_on(target_memory.messages()).unwrap().len(), 2);
}

#[test]
fn react_agent_rejects_incompatible_state_without_changing_memory() {
    let original = vec![Msg::user("keep me")];
    let memory = Arc::new(InMemoryMemory::from_messages(original.clone()));
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_shared_memory(shared_memory);

    let mismatch = block_on(agent.restore(AgentState::new("Saturday", Vec::new()))).unwrap_err();
    assert_eq!(
        mismatch,
        AgentError::StateAgentMismatch {
            expected: "Friday".to_owned(),
            found: "Saturday".to_owned(),
        }
    );
    let future_state: AgentState = serde_json::from_value(json!({
        "format_version": AGENT_STATE_VERSION + 1,
        "agent_name": "Friday",
        "messages": [],
    }))
    .unwrap();
    let unsupported = block_on(agent.restore(future_state)).unwrap_err();
    assert_eq!(
        unsupported,
        AgentError::UnsupportedStateVersion {
            found: AGENT_STATE_VERSION + 1,
            supported: AGENT_STATE_VERSION,
        }
    );
    assert_eq!(block_on(memory.messages()).unwrap(), original);
}

#[test]
fn react_agent_requires_memory_for_state_operations() {
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap();

    assert_eq!(
        block_on(agent.snapshot()).unwrap_err(),
        AgentError::MemoryNotConfigured
    );
    assert_eq!(
        block_on(agent.restore(AgentState::new("Friday", Vec::new()))).unwrap_err(),
        AgentError::MemoryNotConfigured
    );
}

#[test]
fn react_agent_restores_legacy_version_one_state() {
    let message = Msg::user("legacy history");
    let legacy: AgentState = serde_json::from_value(json!({
        "format_version": 1,
        "agent_name": "Friday",
        "messages": [message.clone()],
    }))
    .unwrap();
    let memory = Arc::new(InMemoryMemory::new());
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_shared_memory(shared_memory);

    block_on(agent.restore(legacy)).unwrap();

    assert_eq!(block_on(memory.messages()).unwrap(), [message]);
}

#[test]
fn agent_state_operations_are_object_safe() {
    let memory = Arc::new(InMemoryMemory::from_messages([Msg::user("Hello")]));
    let shared_memory: Arc<dyn Memory> = memory;
    let agent: Arc<dyn Agent> = Arc::new(
        ReActAgent::new(
            "Friday",
            MockChatModel::new("mock-model"),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_shared_memory(shared_memory),
    );

    let state = block_on(agent.snapshot()).unwrap();
    block_on(agent.restore(state.clone())).unwrap();

    assert_eq!(block_on(agent.snapshot()).unwrap(), state);
}

#[test]
fn react_agent_automatically_restores_and_saves_a_bound_session() {
    let key = StateKey::new("user-1", "session-1").unwrap();
    let store = Arc::new(InMemoryStateStore::new());
    let first_store: Arc<dyn StateStore> = store.clone();
    let first = ReActAgent::new(
        "Friday",
        MockChatModel::new("first-model")
            .with_response(ChatResponse::completed([ContentBlock::from("First reply")])),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), first_store);

    block_on(first.reply(Msg::user("First question"))).unwrap();
    let first_record = block_on(store.load(&key)).unwrap().unwrap();
    assert_eq!(first_record.revision(), 1);
    assert_eq!(first_record.state().messages().len(), 2);

    let second_model = Arc::new(MockChatModel::new("second-model").with_response(
        ChatResponse::completed([ContentBlock::from("Second reply")]),
    ));
    let shared_model: Arc<dyn ChatModel> = second_model.clone();
    let second_store: Arc<dyn StateStore> = store.clone();
    let second = ReActAgent::from_shared(
        "Friday",
        shared_model,
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), second_store);

    block_on(second.reply(Msg::user("Second question"))).unwrap();

    let requests = second_model.recorded_requests();
    assert_eq!(requests[0].messages.len(), 3);
    assert_eq!(
        requests[0].messages[0].text_content(""),
        Some("First question".to_owned())
    );
    let second_record = block_on(store.load(&key)).unwrap().unwrap();
    assert_eq!(second_record.revision(), 2);
    assert_eq!(second_record.state().messages().len(), 4);
}

#[test]
fn react_agent_automatically_persists_observations() {
    let key = StateKey::new("user-1", "observations").unwrap();
    let store = Arc::new(InMemoryStateStore::new());
    let shared_store: Arc<dyn StateStore> = store.clone();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), shared_store);

    block_on(agent.observe(Msg::assistant("planner", "Use exact arithmetic."))).unwrap();

    let record = block_on(store.load(&key)).unwrap().unwrap();
    assert_eq!(record.revision(), 1);
    assert_eq!(record.state().messages().len(), 1);
}

#[test]
fn cloned_agents_serialize_updates_to_their_bound_session() {
    let key = StateKey::new("user-1", "shared-clones").unwrap();
    let store = Arc::new(InMemoryStateStore::new());
    let shared_store: Arc<dyn StateStore> = store.clone();
    let model = Arc::new(
        MockChatModel::new("mock-model")
            .with_response(ChatResponse::completed([ContentBlock::from("First")]))
            .with_response(ChatResponse::completed([ContentBlock::from("Second")])),
    );
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let agent = ReActAgent::from_shared(
        "Friday",
        shared_model,
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), shared_store);
    let cloned = agent.clone();

    let (first, second) = block_on(futures_util::future::join(
        agent.reply(Msg::user("First question")),
        cloned.reply(Msg::user("Second question")),
    ));

    first.unwrap();
    second.unwrap();
    let request_lengths = model
        .recorded_requests()
        .iter()
        .map(|request| request.messages.len())
        .collect::<Vec<_>>();
    assert_eq!(request_lengths, [1, 3]);
    let record = block_on(store.load(&key)).unwrap().unwrap();
    assert_eq!(record.revision(), 2);
    assert_eq!(record.state().messages().len(), 4);
}

#[tokio::test]
async fn independent_agents_report_concurrent_state_conflicts() {
    let key = StateKey::new("user-1", "concurrent").unwrap();
    let store = Arc::new(InMemoryStateStore::new());
    let barrier = Arc::new(Barrier::new(2));
    let first_store: Arc<dyn StateStore> = store.clone();
    let second_store: Arc<dyn StateStore> = store.clone();
    let first = ReActAgent::new(
        "Friday",
        BarrierModel {
            barrier: barrier.clone(),
        },
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), first_store);
    let second = ReActAgent::new(
        "Friday",
        BarrierModel { barrier },
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), second_store);

    let (first_result, second_result) = tokio::join!(
        first.reply(Msg::user("first")),
        second.reply(Msg::user("second"))
    );

    assert_ne!(first_result.is_ok(), second_result.is_ok());
    let conflict = first_result.err().or_else(|| second_result.err()).unwrap();
    assert!(matches!(
        conflict,
        AgentError::StateStore(error)
            if error.code.as_deref() == Some("revision_conflict") && error.retryable
    ));
    let record = store.load(&key).await.unwrap().unwrap();
    assert_eq!(record.revision(), 1);
    assert_eq!(record.state().messages().len(), 2);
}

#[test]
fn react_agent_persists_state_before_yielding_the_terminal_event() {
    let key = StateKey::new("user-1", "stream").unwrap();
    let store = Arc::new(InMemoryStateStore::new());
    let shared_store: Arc<dyn StateStore> = store.clone();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model").with_stream(final_answer_stream()),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), shared_store);

    block_on(async {
        let mut events = agent.stream(Msg::user("Stream a reply")).await.unwrap();
        while let Some(event) = events.next().await {
            if matches!(event.unwrap(), AgentEvent::Finished { .. }) {
                break;
            }
        }
    });

    let record = block_on(store.load(&key)).unwrap().unwrap();
    assert_eq!(record.revision(), 1);
    assert_eq!(record.state().messages().len(), 2);
}

#[test]
fn state_store_binding_requires_conversation_memory() {
    let key = StateKey::new("user-1", "session-1").unwrap();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_state_store(key, InMemoryStateStore::new());

    let error = block_on(agent.reply(Msg::user("Hello"))).unwrap_err();

    assert_eq!(error, AgentError::MemoryNotConfigured);
}

#[test]
fn react_agent_is_object_safe() {
    let model = MockChatModel::new("mock-model")
        .with_response(ChatResponse::completed([ContentBlock::from(
            "Hello from the agent.",
        )]))
        .with_stream([
            Ok(ChatEvent::TextDelta {
                block_id: "text-1".to_owned(),
                delta: "Streamed hello.".to_owned(),
            }),
            Ok(ChatEvent::Finished {
                reason: FinishReason::Completed,
            }),
        ]);
    let agent: Arc<dyn Agent> =
        Arc::new(ReActAgent::new("Friday", model, ToolExecutor::new(ToolRegistry::new())).unwrap());

    let reply = block_on(agent.reply(Msg::user("Hello"))).unwrap();

    assert_eq!(agent.name(), "Friday");
    drop(agent.interrupt_handle());
    assert_eq!(
        reply.text_content(""),
        Some("Hello from the agent.".to_owned())
    );

    let events = block_on(async {
        agent
            .stream(Msg::user("Hello again"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    });
    assert!(matches!(
        events.last().unwrap().as_ref().unwrap(),
        AgentEvent::Finished { steps: 1, .. }
    ));
    assert_eq!(
        block_on(agent.resume_tool_calls("missing".to_owned(), Vec::new())).unwrap_err(),
        AgentError::NoPendingToolConfirmation
    );
}

#[test]
fn react_agent_streams_a_complete_model_tool_model_loop() {
    let model = Arc::new(
        MockChatModel::new("mock-model")
            .with_stream(calculator_call_stream())
            .with_stream(final_answer_stream()),
    );
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let memory = Arc::new(InMemoryMemory::new());
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::from_shared("Friday", shared_model, ToolExecutor::new(registry))
        .unwrap()
        .with_shared_memory(shared_memory);

    let events = block_on(async {
        agent
            .stream(Msg::user("What is 6 * 7?"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    });

    assert_eq!(events.len(), 11);
    assert!(matches!(
        events[0],
        AgentEvent::ToolCallDelta { step: 1, .. }
    ));
    assert!(matches!(
        events[1],
        AgentEvent::ToolCallDelta { step: 1, .. }
    ));
    assert!(matches!(events[2], AgentEvent::Usage { step: 1, .. }));
    assert!(matches!(
        events[3],
        AgentEvent::StepFinished {
            step: 1,
            reason: FinishReason::ToolCalls
        }
    ));
    assert!(matches!(events[4], AgentEvent::ToolStarted { step: 1, .. }));
    assert!(matches!(
        events[5],
        AgentEvent::ToolFinished { step: 1, .. }
    ));
    assert!(matches!(events[6], AgentEvent::TextDelta { step: 2, .. }));
    assert!(matches!(events[7], AgentEvent::TextDelta { step: 2, .. }));
    assert!(matches!(events[8], AgentEvent::Usage { step: 2, .. }));
    assert!(matches!(
        events[9],
        AgentEvent::StepFinished {
            step: 2,
            reason: FinishReason::Completed
        }
    ));
    let AgentEvent::Finished { steps, message } = &events[10] else {
        panic!("the stream should end with the final message")
    };
    assert_eq!(*steps, 2);
    assert_eq!(
        message.text_content(""),
        Some("The answer is 42.".to_owned())
    );
    assert_eq!(message.usage, Some(Usage::new(18, 6)));
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert_eq!(model.recorded_requests().len(), 2);
    assert_eq!(model.recorded_requests()[1].messages.len(), 3);
    assert_eq!(block_on(memory.messages()).unwrap().len(), 4);
}

#[test]
fn react_agent_stream_emits_and_persists_a_confirmation_checkpoint() {
    let key = StateKey::new("user-1", "hitl-stream").unwrap();
    let store = Arc::new(InMemoryStateStore::new());
    let shared_store: Arc<dyn StateStore> = store.clone();
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model").with_stream(calculator_call_stream()),
        ToolExecutor::new(registry),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), shared_store)
    .with_tool_confirmation_required("calculator");

    let checkpoint = block_on(async {
        let mut events = agent.stream(Msg::user("What is 6 * 7?")).await.unwrap();
        loop {
            match events.next().await.unwrap().unwrap() {
                AgentEvent::ToolConfirmationRequired { checkpoint } => break checkpoint,
                _ => continue,
            }
        }
    });

    assert_eq!(checkpoint.calls()[0].id(), "call-stream-1");
    assert!(tool.recorded_invocations().is_empty());
    let saved = block_on(store.load(&key)).unwrap().unwrap();
    assert_eq!(saved.revision(), 1);
    assert_eq!(saved.state().pending_tool_calls(), Some(&checkpoint));
}

#[test]
fn react_agent_stream_does_not_persist_partial_model_output() {
    let model = MockChatModel::new("mock-model").with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "text-1".to_owned(),
            delta: "partial".to_owned(),
        }),
        Err(ModelError::new("connection lost")
            .with_code("transport")
            .with_retryable(true)),
    ]);
    let memory = Arc::new(InMemoryMemory::new());
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(ToolRegistry::new()))
        .unwrap()
        .with_shared_memory(shared_memory);

    let events = block_on(async {
        agent
            .stream(Msg::user("Hello"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    });

    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], AgentEvent::TextDelta { .. }));
    assert!(matches!(
        &events[1],
        AgentEvent::Error {
            step: Some(1),
            error: AgentError::Model(error),
        } if error.code.as_deref() == Some("transport") && error.retryable
    ));
    let remembered = block_on(memory.messages()).unwrap();
    assert_eq!(remembered.len(), 1);
    assert_eq!(remembered[0].role, Role::User);
}

#[test]
fn react_agent_stream_converts_start_failures_to_terminal_events() {
    let model = MockChatModel::new("mock-model").with_stream_error(
        ModelError::new("provider unavailable")
            .with_code("unavailable")
            .with_retryable(true),
    );
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(ToolRegistry::new())).unwrap();

    let events = block_on(async {
        agent
            .stream(Msg::user("Hello"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    });

    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0].as_ref().unwrap(),
        AgentEvent::Error {
            step: Some(1),
            error: AgentError::Model(error),
        } if error.code.as_deref() == Some("unavailable") && error.retryable
    ));
}

#[test]
fn react_agent_interrupts_an_inflight_model_stream() {
    let handle_slot = Arc::new(Mutex::new(None));
    let model = Arc::new(InterruptingStreamModel {
        handle: handle_slot.clone(),
        requests: AtomicUsize::new(0),
    });
    let memory = Arc::new(InMemoryMemory::new());
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::from_shared(
        "Friday",
        shared_model,
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_shared_memory(shared_memory);
    *handle_slot.lock().unwrap() = Some(agent.interrupt_handle());

    let events = block_on(async {
        agent
            .stream(Msg::user("Start a long response"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    });

    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0],
        AgentEvent::Error {
            step: Some(1),
            error: AgentError::Interrupted
        }
    ));
    assert_eq!(model.requests.load(Ordering::SeqCst), 1);
    let remembered = block_on(memory.messages()).unwrap();
    assert_eq!(remembered.len(), 1);
    assert_eq!(remembered[0].role, Role::User);
}

#[test]
fn react_agent_interrupts_an_inflight_complete_model_call() {
    let handle_slot = Arc::new(Mutex::new(None));
    let model = Arc::new(InterruptingStreamModel {
        handle: handle_slot.clone(),
        requests: AtomicUsize::new(0),
    });
    let memory = Arc::new(InMemoryMemory::new());
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::from_shared(
        "Friday",
        shared_model,
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_shared_memory(shared_memory);
    *handle_slot.lock().unwrap() = Some(agent.interrupt_handle());

    let error = block_on(agent.reply(Msg::user("Start a long response"))).unwrap_err();

    assert_eq!(error, AgentError::Interrupted);
    assert_eq!(model.requests.load(Ordering::SeqCst), 1);
    let remembered = block_on(memory.messages()).unwrap();
    assert_eq!(remembered.len(), 1);
    assert_eq!(remembered[0].role, Role::User);
}

#[test]
fn react_agent_closes_interrupted_tool_calls_in_memory() {
    let handle_slot = Arc::new(Mutex::new(None));
    let tool = Arc::new(InterruptingTool {
        definition: calculator_definition(),
        handle: handle_slot.clone(),
        invocations: AtomicUsize::new(0),
    });
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let memory = Arc::new(InMemoryMemory::new());
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model").with_stream(calculator_call_stream()),
        ToolExecutor::new(registry),
    )
    .unwrap()
    .with_shared_memory(shared_memory);
    *handle_slot.lock().unwrap() = Some(agent.interrupt_handle());

    let events = block_on(async {
        agent
            .stream(Msg::user("What is 6 * 7?"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    });

    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolFinished { result, .. }
            if result.state() == ToolResultState::Interrupted
    )));
    assert!(matches!(
        events.last().unwrap(),
        AgentEvent::Error {
            step: Some(1),
            error: AgentError::Interrupted
        }
    ));
    assert_eq!(tool.invocations.load(Ordering::SeqCst), 1);
    let remembered = block_on(memory.messages()).unwrap();
    assert_eq!(remembered.len(), 3);
    let ContentBlock::ToolResult(result) = &remembered[2].content[0] else {
        panic!("interruption should persist a closing tool result")
    };
    assert_eq!(result.state(), ToolResultState::Interrupted);
}

#[test]
fn react_agent_stream_stops_before_unobservable_tool_execution() {
    let model = MockChatModel::new("mock-model").with_stream(calculator_call_stream());
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))
        .unwrap()
        .with_max_steps(1)
        .unwrap();

    let events = block_on(async {
        agent
            .stream(Msg::user("What is 6 * 7?"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    });

    assert!(matches!(
        events.last().unwrap(),
        AgentEvent::Error {
            step: Some(1),
            error: AgentError::MaxStepsExceeded { max_steps: 1 }
        }
    ));
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolStarted { .. } | AgentEvent::ToolFinished { .. }
    )));
    assert!(tool.recorded_invocations().is_empty());
}

#[test]
fn react_agent_stream_converts_hook_failures_to_terminal_events() {
    let model = MockChatModel::new("mock-model").with_stream(calculator_call_stream());
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let memory = Arc::new(InMemoryMemory::new());
    let shared_memory: Arc<dyn Memory> = memory.clone();
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))
        .unwrap()
        .with_hook(RejectToolHook)
        .with_shared_memory(shared_memory);

    let events = block_on(async {
        agent
            .stream(Msg::user("What is 6 * 7?"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    });

    assert!(matches!(
        events.last().unwrap(),
        AgentEvent::Error {
            step: Some(1),
            error: AgentError::Hook(error),
        } if error.code.as_deref() == Some("hook_rejected")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolStarted { .. }))
    );
    assert!(tool.recorded_invocations().is_empty());
    let remembered = block_on(memory.messages()).unwrap();
    assert_eq!(remembered.len(), 1);
    assert_eq!(remembered[0].role, Role::User);
}

#[test]
fn react_agent_stops_before_executing_unobservable_tool_calls() {
    let call = ToolCallBlock::complete("call-2", "calculator", r#"{"expression":"1+1"}"#).unwrap();
    let model = Arc::new(
        MockChatModel::new("mock-model").with_response(ChatResponse::finished(
            [ContentBlock::from(call)],
            FinishReason::ToolCalls,
        )),
    );
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("2"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let shared_model: Arc<dyn ChatModel> = model.clone();
    let agent = ReActAgent::from_shared("Friday", shared_model, ToolExecutor::new(registry))
        .unwrap()
        .with_max_steps(1)
        .unwrap();

    let error = block_on(agent.reply(Msg::user("What is 1 + 1?"))).unwrap_err();

    assert!(matches!(
        error,
        AgentError::MaxStepsExceeded { max_steps: 1 }
    ));
    assert!(tool.recorded_invocations().is_empty());
    assert_eq!(model.recorded_requests().len(), 1);
}

#[test]
fn react_agent_rejects_invalid_configuration_and_partial_responses() {
    let empty_name = ReActAgent::new(
        "  ",
        MockChatModel::new("mock-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap_err();
    assert!(matches!(empty_name, AgentError::EmptyName));

    let zero_steps = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_max_steps(0)
    .unwrap_err();
    assert!(matches!(zero_steps, AgentError::ZeroMaxSteps));

    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock-model")
            .with_response(ChatResponse::partial([ContentBlock::from("partial")])),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap();
    let partial = block_on(agent.reply(Msg::user("Hello"))).unwrap_err();
    assert!(matches!(partial, AgentError::InvalidModelResponse(_)));
}
