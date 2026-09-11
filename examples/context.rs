use std::{error::Error, sync::Arc};

use agentscope::{
    ChatResponse, ContentBlock, FinishReason, InMemoryMemory, MockChatModel, Msg, ReActAgent,
    RecentTurns, ToolExecutor, ToolRegistry,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut model = MockChatModel::new("offline-context");
    for answer in ["first answer", "second answer", "third answer"] {
        model = model.with_response(ChatResponse::finished(
            [ContentBlock::from(answer)],
            FinishReason::Completed,
        ));
    }
    let model = Arc::new(model);
    let agent = ReActAgent::from_shared(
        "Friday",
        model.clone(),
        ToolExecutor::new(ToolRegistry::new()),
    )?
    .with_memory(InMemoryMemory::new())
    .with_system_prompt("Be concise.")
    .with_context_policy(RecentTurns::new(1)?);

    for question in ["first question", "second question", "third question"] {
        agent.reply(Msg::user(question)).await?;
    }
    let requests = model.recorded_requests();
    let latest = &requests.last().expect("three model calls").messages;
    assert_eq!(latest.len(), 2); // configured system prompt + current question
    assert_eq!(latest[1].text_content(""), Some("third question".into()));
    let state = agent.snapshot().await?;
    assert_eq!(state.messages().len(), 6);
    println!(
        "Latest model input: {} messages (system + current turn)",
        latest.len()
    );
    println!(
        "Full saved history: {} messages (all 3 turns)",
        state.messages().len()
    );
    Ok(())
}
