use std::{error::Error, sync::Arc};

use agentscope::{
    AgentError, ChatResponse, ContentBlock, FinishReason, InMemoryMemory, MockChatModel, MockTool,
    Msg, ReActAgent, ToolCallBlock, ToolConfirmation, ToolDefinition, ToolExecutor, ToolRegistry,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let call = ToolCallBlock::complete("call-1", "calculator", r#"{"expression":"6*7"}"#)?;
    let model = MockChatModel::new("offline-model")
        .with_response(ChatResponse::finished(
            [ContentBlock::from(call)],
            FinishReason::ToolCalls,
        ))
        .with_response(ChatResponse::completed([ContentBlock::from(
            "The result is 42.",
        )]));
    let tool = Arc::new(
        MockTool::new(ToolDefinition::new(
            "calculator",
            "Evaluate an expression",
            json!({
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }),
        )?)
        .with_output("42"),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone())?;
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))?
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("calculator");

    let checkpoint = match agent.reply(Msg::user("What is 6 * 7?")).await {
        Err(AgentError::ToolConfirmationRequired { checkpoint }) => checkpoint,
        result => return Err(format!("expected confirmation checkpoint, got {result:?}").into()),
    };
    println!("approval required for: {}", checkpoint.calls()[0].name());
    assert!(tool.recorded_invocations().is_empty());

    let reply = agent
        .resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve(checkpoint.calls()[0].id())],
        )
        .await?;
    println!("{}", reply.text_content("").unwrap_or_default());
    Ok(())
}
