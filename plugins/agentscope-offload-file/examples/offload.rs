//! Offline round trip: large tool output -> reference -> bounded read -> answer.
use agentscope::{
    ChatResponse, ContentBlock, FinishReason, InMemoryMemory, MockChatModel, MockTool, Msg,
    ReActAgent, TokenBudget, ToolCallBlock, ToolDefinition, ToolExecutor, ToolRegistry,
    ToolResultOffload,
};
use agentscope_offload_file::FileOffloadStore;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{error::Error, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Temporary demo data only. Real applications must keep a private per-session
    // directory alive for as long as model-visible references may be used.
    let dir = tempfile::tempdir()?;
    let store = Arc::new(FileOffloadStore::new(dir.path())?);
    let text = "Order 42 is ready.\n".repeat(2000);
    // A deterministic mock predicts the reference. Real models read it from output.
    let id = format!("{:x}", Sha256::digest(text.as_bytes()));
    let model = Arc::new(
        MockChatModel::new("offline")
            .with_response(ChatResponse::finished(
                [ToolCallBlock::complete("fetch", "fetch", "{}")?.into()],
                FinishReason::ToolCalls,
            ))
            .with_response(ChatResponse::finished(
                [ToolCallBlock::complete(
                    "read",
                    "read_offloaded_text",
                    json!({"id":id,"offset":0,"max_bytes":128}).to_string(),
                )?
                .into()],
                FinishReason::ToolCalls,
            ))
            .with_response(ChatResponse::completed([ContentBlock::from(
                "Order 42 is ready.",
            )])),
    );
    let mut registry = ToolRegistry::new();
    registry.register(
        MockTool::new(ToolDefinition::new(
            "fetch",
            "Fetch report",
            json!({"type":"object"}),
        )?)
        .with_output(text.clone()),
    )?;
    let agent = ReActAgent::from_shared("Friday", model.clone(), ToolExecutor::new(registry))?
        .with_memory(InMemoryMemory::new())
        .with_tool_result_offload(ToolResultOffload::new(store, 1024, 64, 128)?)?
        .with_token_budget(TokenBudget::new(4000, 128)?);
    let answer = agent
        .reply(Msg::user("Read the report and tell me the order status."))
        .await?;
    println!("{}", answer.text_content("").unwrap_or_default());
    println!(
        "Original: {} bytes; model calls: {}; final call includes a bounded read page.",
        text.len(),
        model.recorded_requests().len()
    );
    assert_eq!(model.recorded_requests().len(), 3);
    Ok(())
}
