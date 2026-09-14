use agentscope::{
    ChatModelSummarizer, ChatResponse, ContentBlock, InMemoryMemory, MockChatModel, Msg,
    ReActAgent, TokenBudget, ToolExecutor, ToolRegistry,
};
use std::{error::Error, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let summary_model = Arc::new(MockChatModel::new("offline-summary").with_response(
        ChatResponse::completed([ContentBlock::from(
            "The user is building a Rust agent framework; original history must remain stored.",
        )]),
    ));
    let reply_model = Arc::new(MockChatModel::new("offline-reply").with_response(
        ChatResponse::completed([ContentBlock::from(
            "We can continue from the saved summary.",
        )]),
    ));
    let agent = ReActAgent::from_shared(
        "Friday",
        reply_model.clone(),
        ToolExecutor::new(ToolRegistry::new()),
    )?
    .with_memory(InMemoryMemory::from_messages([
        Msg::user("Build a Rust agent framework. ".repeat(100)),
        Msg::assistant("Friday", "We will preserve original history. ".repeat(100)),
        Msg::user("Keep the latest turn verbatim."),
        Msg::assistant("Friday", "Understood."),
    ]))
    .with_summarizer(ChatModelSummarizer::from_shared(
        summary_model.clone(),
        TokenBudget::new(10000, 500)?,
    ));
    assert!(summary_model.recorded_requests().is_empty());
    let summary = agent.compact_context(1).await?.expect("old completed turn");
    assert_eq!(summary_model.recorded_requests().len(), 1);
    assert_eq!(summary.covered_messages(), 2);
    let snapshot = agent.snapshot().await?;
    assert_eq!(snapshot.messages().len(), 4);
    assert!(snapshot.context_summary().is_some());
    assert!(agent.compact_context(1).await?.is_none());
    agent.reply(Msg::user("Continue.")).await?;
    assert!(
        reply_model.recorded_requests()[0]
            .messages
            .iter()
            .any(|m| m.name == "context_summary")
    );
    println!("Summary: {}", summary.text());
    println!(
        "Summary calls: 1; original messages after reply: {}",
        agent.snapshot().await?.messages().len()
    );
    println!(
        "State format: {}; summary and original history are stored separately.",
        snapshot.format_version()
    );
    Ok(())
}
