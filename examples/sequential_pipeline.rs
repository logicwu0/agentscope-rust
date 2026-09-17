//! Offline writer -> reviewer pipeline with independent memories.
use agentscope::{
    ChatResponse, ContentBlock, InMemoryMemory, MockChatModel, Msg, ReActAgent, SequentialPipeline,
    ToolExecutor, ToolRegistry,
};
use std::{error::Error, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let writer_model = Arc::new(MockChatModel::new("offline-writer").with_response(
        ChatResponse::completed([ContentBlock::from(
            "Draft: store API credentials outside source control.",
        )]),
    ));
    let reviewer_model = Arc::new(MockChatModel::new("offline-reviewer")
        .with_response(ChatResponse::completed([ContentBlock::from("Reviewed: use environment variables or an ignored local config; never commit credentials.")])));
    let writer = Arc::new(
        ReActAgent::from_shared(
            "writer",
            writer_model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )?
        .with_system_prompt("Write a concise draft addressing the user request.")
        .with_memory(InMemoryMemory::new()),
    );
    let reviewer = Arc::new(
        ReActAgent::from_shared(
            "reviewer",
            reviewer_model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )?
        .with_system_prompt("Review the supplied draft. Return the corrected final text.")
        .with_memory(InMemoryMemory::new()),
    );
    let pipeline = SequentialPipeline::new(vec![writer.clone(), reviewer.clone()])?;
    let output = pipeline
        .run(Msg::user("Explain how to keep API credentials private."))
        .await?;
    for step in &output.steps {
        println!(
            "{}. {}: {}",
            step.step,
            step.agent_name,
            step.message.text_content("").unwrap_or_default()
        );
    }
    assert_eq!(writer_model.recorded_requests().len(), 1);
    assert_eq!(reviewer_model.recorded_requests().len(), 1);
    assert_eq!(writer.snapshot().await?.messages().len(), 2);
    assert_eq!(reviewer.snapshot().await?.messages().len(), 2);
    println!("Two stages complete; each agent retains its own two-message history.");
    Ok(())
}
