//! Offline example using a local fixture server and a deterministic model.
use agentscope::{
    AgentError, ChatResponse, ContentBlock, FinishReason, InMemoryMemory, MockChatModel, Msg,
    ReActAgent, ToolCallBlock, ToolConfirmation, ToolExecutor,
};
use agentscope_mcp::{McpClient, StdioConfig};
use std::{error::Error, path::PathBuf};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut config = StdioConfig::new("/usr/bin/python3");
    config.args.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/server.py")
            .into_os_string(),
    );
    let client = McpClient::connect(config).await?;
    let registry = client.registry("demo").await?;
    println!(
        "Negotiated {}; discovered {} tool(s)",
        client.protocol_version(),
        registry.len()
    );
    let model = MockChatModel::new("offline")
        .with_response(ChatResponse::finished(
            [ToolCallBlock::complete("echo-1", "demo__echo", r#"{"text":"Hello MCP"}"#)?.into()],
            FinishReason::ToolCalls,
        ))
        .with_response(ChatResponse::completed([ContentBlock::from(
            "The local MCP tool returned Hello MCP.",
        )]));
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))?
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("demo__echo");
    let Err(AgentError::ToolConfirmationRequired { checkpoint }) =
        agent.reply(Msg::user("Try the local echo tool.")).await
    else {
        return Err("expected confirmation checkpoint".into());
    };
    // Only this deterministic, side-effect-free fixture is approved automatically.
    // Real applications must collect an actual user/application authorization.
    let reply = agent
        .resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("echo-1")],
        )
        .await?;
    println!("{}", reply.text_content("").unwrap_or_default());
    client.close().await?;
    Ok(())
}
