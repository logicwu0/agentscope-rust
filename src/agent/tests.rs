use std::sync::Arc;

use futures_executor::block_on;
use serde_json::json;

use crate::{
    Agent, AgentError, ChatModel, ChatResponse, ContentBlock, FinishReason, GenerateOptions,
    InMemoryMemory, Memory, MockChatModel, MockTool, Msg, ReActAgent, Role, ToolCallBlock,
    ToolDefinition, ToolExecutor, ToolRegistry, ToolResultOutput, ToolResultState,
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
    let model = MockChatModel::new("mock-model").with_response(ChatResponse::completed([
        ContentBlock::from("Hello from the agent."),
    ]));
    let agent: Arc<dyn Agent> =
        Arc::new(ReActAgent::new("Friday", model, ToolExecutor::new(ToolRegistry::new())).unwrap());

    let reply = block_on(agent.reply(Msg::user("Hello"))).unwrap();

    assert_eq!(agent.name(), "Friday");
    assert_eq!(
        reply.text_content(""),
        Some("Hello from the agent.".to_owned())
    );
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
