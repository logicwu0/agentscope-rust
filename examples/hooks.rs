//! Observes a complete `ReActAgent` lifecycle with a read-only hook.

use std::error::Error;

use agentscope::{
    AgentHook, AgentHookEvent, AgentHookFuture, ChatResponse, ContentBlock, FinishReason,
    InMemoryMemory, MockChatModel, MockTool, Msg, ReActAgent, ToolCallBlock, ToolDefinition,
    ToolExecutor, ToolRegistry,
};
use serde_json::json;

struct LifecycleLogger;

impl AgentHook for LifecycleLogger {
    fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a> {
        Box::pin(async move {
            match event {
                AgentHookEvent::BeforeObserve { message } => {
                    println!("observing message from {}", message.name);
                }
                AgentHookEvent::AfterObserve { .. } => println!("observation stored"),
                AgentHookEvent::BeforeReply { .. } => println!("reply started"),
                AgentHookEvent::BeforeModelCall { step, .. } => {
                    println!("model call {step} started");
                }
                AgentHookEvent::AfterModelCall { step, .. } => {
                    println!("model call {step} finished");
                }
                AgentHookEvent::BeforeToolCall { step, call } => {
                    println!("tool {} started at step {step}", call.name());
                }
                AgentHookEvent::AfterToolCall { step, result } => {
                    println!("tool {} finished at step {step}", result.name());
                }
                AgentHookEvent::AfterReply { steps, .. } => {
                    println!("reply finished after {steps} model calls");
                }
            }
            Ok(())
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let call = ToolCallBlock::complete("call-1", "multiply", r#"{"a":6,"b":7}"#)?;
    let model = MockChatModel::new("mock-model")
        .with_response(ChatResponse::finished(
            [ContentBlock::from(call)],
            FinishReason::ToolCalls,
        ))
        .with_response(ChatResponse::completed([ContentBlock::from("42")]));
    let tool = MockTool::new(ToolDefinition::new(
        "multiply",
        "Multiply two integers",
        json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }),
    )?)
    .with_output("42");
    let mut registry = ToolRegistry::new();
    registry.register(tool)?;
    let agent = ReActAgent::new("calculator", model, ToolExecutor::new(registry))?
        .with_memory(InMemoryMemory::new())
        .with_hook(LifecycleLogger);

    futures_executor::block_on(agent.observe(Msg::assistant("planner", "Use exact arithmetic.")))?;
    let reply = futures_executor::block_on(agent.reply(Msg::user("What is 6 * 7?")))?;
    println!("answer: {}", reply.text_content("").unwrap_or_default());
    Ok(())
}
