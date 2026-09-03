use std::{error::Error, sync::Arc};

use agentscope::{
    ChatResponse, ContentBlock, InMemoryMemory, InMemoryStateStore, MockChatModel, Msg, ReActAgent,
    StateKey, StateStore, ToolExecutor, ToolRegistry,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let key = StateKey::new("demo-user", "demo-session")?;
    let store = Arc::new(InMemoryStateStore::new());
    let first_store: Arc<dyn StateStore> = store.clone();
    let first = ReActAgent::new(
        "Friday",
        MockChatModel::new("offline-first").with_response(ChatResponse::completed([
            ContentBlock::from("I will remember that."),
        ])),
        ToolExecutor::new(ToolRegistry::new()),
    )?
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), first_store);
    first
        .reply(Msg::user("My favorite color is green."))
        .await?;

    // A new agent instance starts with empty memory and restores the session
    // automatically before handling the next message.
    let restarted_store: Arc<dyn StateStore> = store.clone();
    let restarted = ReActAgent::new(
        "Friday",
        MockChatModel::new("offline-restarted")
            .with_response(ChatResponse::completed([ContentBlock::from("Green.")])),
        ToolExecutor::new(ToolRegistry::new()),
    )?
    .with_memory(InMemoryMemory::new())
    .with_shared_state_store(key.clone(), restarted_store);
    let reply = restarted
        .reply(Msg::user("What is my favorite color?"))
        .await?;

    println!("{}", reply.text_content("").unwrap_or_default());
    let record = store.load(&key).await?.expect("session was saved");
    println!("saved revision: {}", record.revision());
    Ok(())
}
