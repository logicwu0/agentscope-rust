use std::{error::Error, sync::Arc};

use agentscope::{IdempotentTool, MockTool, Tool, ToolContext, ToolDefinition};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let inner = Arc::new(
        MockTool::new(ToolDefinition::new(
            "save_note",
            "Save a note",
            json!({"type":"object"}),
        )?)
        .with_output("note-123"),
    );
    let tool = IdempotentTool::from_shared(inner.clone());
    let context = ToolContext::new().with_idempotency_key("save-note-request-1");
    let first = tool
        .execute(json!({"text":"hello"}), context.clone())
        .await?;
    let second = tool.execute(json!({"text":"hello"}), context).await?;
    assert_eq!(first, second);
    assert_eq!(inner.recorded_invocations().len(), 1);
    println!("Two requests, one execution; reused result: {second:?}");
    Ok(())
}
