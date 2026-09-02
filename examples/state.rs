use std::error::Error;

use agentscope::{
    AgentState, InMemoryMemory, MockChatModel, Msg, ReActAgent, ToolExecutor, ToolRegistry,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let source = ReActAgent::new(
        "Friday",
        MockChatModel::new("offline-source"),
        ToolExecutor::new(ToolRegistry::new()),
    )?
    .with_memory(InMemoryMemory::from_messages([
        Msg::system("Be concise."),
        Msg::user("Remember this conversation."),
    ]));

    let json = serde_json::to_string_pretty(&source.snapshot().await?)?;
    println!("{json}");

    let restored: AgentState = serde_json::from_str(&json)?;
    let target = ReActAgent::new(
        "Friday",
        MockChatModel::new("offline-target"),
        ToolExecutor::new(ToolRegistry::new()),
    )?
    .with_memory(InMemoryMemory::new());
    target.restore(restored).await?;

    assert_eq!(target.snapshot().await?.messages().len(), 2);
    Ok(())
}
