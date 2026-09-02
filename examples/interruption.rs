//! Cooperatively interrupts an in-flight tool call without corrupting memory.

use std::{error::Error, sync::Arc, time::Duration};

use agentscope::{
    AgentError, ChatResponse, ContentBlock, FinishReason, InMemoryMemory, Memory, MockChatModel,
    Msg, ReActAgent, Tool, ToolCallBlock, ToolContext, ToolDefinition, ToolExecutor, ToolFuture,
    ToolRegistry, ToolResultOutput, ToolResultState,
};
use serde_json::{Value, json};

struct SlowTool {
    definition: ToolDefinition,
}

impl SlowTool {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            definition: ToolDefinition::new(
                "slow_task",
                "Simulate a long-running task",
                json!({
                    "type": "object",
                    "properties": {"seconds": {"type": "integer"}},
                    "required": ["seconds"]
                }),
            )?,
        })
    }
}

impl Tool for SlowTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn execute(&self, _input: Value, _context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok("completed".into())
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let call = ToolCallBlock::complete("call-1", "slow_task", r#"{"seconds":30}"#)?;
    let model = MockChatModel::new("mock-model").with_response(ChatResponse::finished(
        [ContentBlock::from(call)],
        FinishReason::ToolCalls,
    ));
    let mut registry = ToolRegistry::new();
    registry.register(SlowTool::new()?)?;
    let memory = Arc::new(InMemoryMemory::new());
    let agent = ReActAgent::new("worker", model, ToolExecutor::new(registry))?
        .with_shared_memory(memory.clone());
    let interrupt = agent.interrupt_handle();

    let requester = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(25)).await;
        interrupt.interrupt();
    });
    let error = agent
        .reply(Msg::user("Run the slow task"))
        .await
        .unwrap_err();
    requester.await?;
    assert_eq!(error, AgentError::Interrupted);

    let messages = memory.messages().await?;
    let ContentBlock::ToolResult(result) = &messages[2].content[0] else {
        return Err("missing closing tool result".into());
    };
    assert_eq!(result.state(), ToolResultState::Interrupted);
    println!("interrupted safely; {} messages preserved", messages.len());
    Ok(())
}
