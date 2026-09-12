use std::{error::Error, sync::Arc};

use agentscope::{
    ChatResponse, ContentBlock, FinishReason, HeuristicTokenCounter, InMemoryMemory, MockChatModel,
    Msg, ReActAgent, TokenBudget, TokenCounter, ToolExecutor, ToolRegistry,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let model = Arc::new(MockChatModel::new("offline-budget").with_response(
        ChatResponse::finished([ContentBlock::from("42")], FinishReason::Completed),
    ));
    let budget = TokenBudget::new(512, 64)?;
    let input_limit = budget.input_limit();
    let agent = ReActAgent::from_shared(
        "Friday",
        model.clone(),
        ToolExecutor::new(ToolRegistry::new()),
    )?
    .with_memory(InMemoryMemory::from_messages([
        Msg::user("An old question"),
        Msg::assistant("Friday", "Long historical answer. ".repeat(300)),
    ]))
    .with_system_prompt("Be concise.")
    .with_token_budget(budget);

    agent.reply(Msg::user("What is 6 times 7?")).await?;
    let requests = model.recorded_requests();
    let request = &requests[0];
    let count = HeuristicTokenCounter.count(request)?;
    assert!(count.tokens <= input_limit);
    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.options.max_tokens, Some(64));
    let state = agent.snapshot().await?;
    assert_eq!(state.messages().len(), 4);
    assert!(state.messages()[1].text_content("").unwrap().len() > 6000);
    println!(
        "Input: {} tokens ({:?}), allowance: {input_limit}",
        count.tokens, count.accuracy
    );
    println!("Output reservation/cap: 64 tokens");
    println!("Model sees 2 messages; saved history retains all 4 messages.");
    Ok(())
}
