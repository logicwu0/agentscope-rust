use agentscope::{
    AgentEvent, ChatEvent, ChatModelSummarizer, ChatResponse, ContentBlock, FinishReason,
    InMemoryMemory, MockChatModel, Msg, ReActAgent, TokenBudget, ToolExecutor, ToolRegistry,
};
use futures_util::StreamExt;
use std::{error::Error, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let summary_model = Arc::new(MockChatModel::new("offline-summary").with_response(
        ChatResponse::completed([ContentBlock::from(
            "The old task completed; preserve all original messages.",
        )]),
    ));
    let main_model = Arc::new(MockChatModel::new("offline-main").with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "answer".into(),
            delta: "Continue from the summary.".into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]));
    let history = vec![
        Msg::user("Old task details. ".repeat(300)),
        Msg::assistant("Friday", "Old task result. ".repeat(300)),
        Msg::user("Keep this recent turn."),
        Msg::assistant("Friday", "Understood."),
    ];
    let agent = ReActAgent::from_shared(
        "Friday",
        main_model.clone(),
        ToolExecutor::new(ToolRegistry::new()),
    )?
    .with_memory(InMemoryMemory::from_messages(history.clone()))
    .with_summarizer(ChatModelSummarizer::from_shared(
        summary_model.clone(),
        TokenBudget::new(16000, 512)?,
    ))
    .with_token_budget(TokenBudget::new(1000, 128)?)
    .with_auto_compaction(1)?;
    let mut stream = agent.stream(Msg::user("Continue.")).await?;
    let mut completed = false;
    while let Some(event) = stream.next().await {
        match event? {
            AgentEvent::ContextCompactionStarted { .. } => println!("Automatic compaction started"),
            AgentEvent::ContextCompactionCompleted { covered_messages } => {
                println!("Committed summary covering {covered_messages} original messages");
            }
            AgentEvent::TextDelta { delta, .. } => println!("{delta}"),
            AgentEvent::Finished { .. } => completed = true,
            AgentEvent::Error { error, .. } => return Err(error.into()),
            _ => {}
        }
    }
    drop(stream);
    assert!(completed);
    assert_eq!(summary_model.recorded_requests().len(), 1);
    assert_eq!(main_model.recorded_requests().len(), 1);
    let state = agent.snapshot().await?;
    assert_eq!(&state.messages()[..4], history);
    assert!(state.context_summary().is_some());
    println!(
        "Summary calls: 1; main calls: 1; saved original messages: {}",
        state.messages().len()
    );
    Ok(())
}
