//! Run `pause` and `resume` in separate processes with the same database path.
use agentscope::{
    AgentError, ChatResponse, ContentBlock, FinishReason, InMemoryMemory, MockChatModel, MockTool,
    Msg, ReActAgent, StateKey, StateStore, ToolCallBlock, ToolConfirmation, ToolDefinition,
    ToolExecutor, ToolRegistry,
};
use agentscope_state_sqlite::SQLiteStateStore;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 || !matches!(args[1].as_str(), "pause" | "resume") {
        return Err("usage: restart <pause|resume> <database-path>".into());
    }
    let store = SQLiteStateStore::open(&args[2]).await?;
    let key = StateKey::new("demo-user", "demo-session")?;
    let saved = store.load(&key).await?;
    let model = if args[1] == "pause" {
        if saved.is_some() {
            return Err("session already exists; use resume or a new database path".into());
        }
        MockChatModel::new("offline").with_response(ChatResponse::finished(
            [ContentBlock::from(ToolCallBlock::complete(
                "call-1",
                "calculator",
                "{}",
            )?)],
            FinishReason::ToolCalls,
        ))
    } else {
        MockChatModel::new("offline").with_response(ChatResponse::completed([ContentBlock::from(
            "The answer is 42.",
        )]))
    };
    let mut registry = ToolRegistry::new();
    registry.register(
        MockTool::new(ToolDefinition::new(
            "calculator",
            "Calculate",
            serde_json::json!({"type":"object"}),
        )?)
        .with_output("42"),
    )?;
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))?
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("calculator")
        .with_state_store(key, store);
    if args[1] == "pause" {
        match agent.reply(Msg::user("Calculate 6 * 7")).await {
            Err(AgentError::ToolConfirmationRequired { .. }) => {
                println!("Confirmation saved. This process can now exit; run resume next.");
            }
            result => return Err(format!("expected confirmation, got {result:?}").into()),
        }
    } else {
        let pending = saved
            .as_ref()
            .and_then(|record| record.state().pending_tool_calls())
            .ok_or("no pending confirmation")?;
        let decisions = pending
            .calls()
            .iter()
            .map(|call| ToolConfirmation::approve(call.id()))
            .collect();
        let reply = agent
            .resume_tool_calls(pending.reply_id(), decisions)
            .await?;
        println!("{}", reply.text_content("").unwrap_or_default());
    }
    Ok(())
}
