use std::sync::Arc;

use futures_executor::block_on;
use futures_util::StreamExt;
use serde_json::json;

use crate::{
    Agent, AgentError, AgentEvent, ChatEvent, ChatModel, ChatResponse, ContentBlock, FinishReason,
    GenerateOptions, InMemoryMemory, Memory, MockChatModel, MockTool, ModelError, ModelResult, Msg,
    ReActAgent, Role, ToolCallBlock, ToolDefinition, ToolExecutor, ToolRegistry, ToolResultOutput,
    ToolResultState, Usage,
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
