//! Run `first` then `replay` in separate processes to verify durable deduplication.
use agentscope::{MockTool, PersistentIdempotentTool, Tool, ToolContext, ToolDefinition};
use agentscope_idempotency_sqlite::SQLiteIdempotencyStore;
use serde_json::json;
use std::{error::Error, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 || !matches!(args[1].as_str(), "first" | "replay") {
        return Err("usage: restart <first|replay> <database-path>".into());
    }
    let definition = ToolDefinition::new("save_note", "Save note", json!({"type":"object"}))?;
    let mock = if args[1] == "first" {
        MockTool::new(definition).with_output("note-123")
    } else {
        MockTool::new(definition)
    };
    let inner = Arc::new(mock);
    let store = Arc::new(SQLiteIdempotencyStore::open(&args[2]).await?);
    let tool = PersistentIdempotentTool::from_shared("demo-user:notes:v1", inner.clone(), store)?;
    let result = tool
        .execute(
            json!({"text":"hello"}),
            ToolContext::new().with_idempotency_key("note-request-1"),
        )
        .await?;
    let expected_calls = usize::from(args[1] == "first");
    assert_eq!(
        inner.recorded_invocations().len(),
        expected_calls,
        "use a fresh database for the first run"
    );
    println!("Result: {result:?}; tool executions in this process: {expected_calls}");
    Ok(())
}
