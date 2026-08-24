//! Streams a `DeepSeek` response through the OpenAI-compatible model.

use std::{error::Error, io::Write};

use agentscope::{
    ChatEvent, ChatModel, ChatRequest, ChatResponseAccumulator, Msg, OpenAIChatModel,
};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let model = OpenAIChatModel::builder()
        .model("deepseek-chat")
        .api_key_from_env("DEEPSEEK_API_KEY")?
        .base_url("https://api.deepseek.com")
        .build()?;
    let mut stream = model
        .stream(ChatRequest::new([Msg::user(
            "Reply with exactly: AgentScope Rust streaming works",
        )]))
        .await?;
    let mut accumulator = ChatResponseAccumulator::new();

    while let Some(event) = stream.next().await {
        let event = event?;
        match &event {
            ChatEvent::TextDelta { delta, .. } => {
                print!("{delta}");
                std::io::stdout().flush()?;
            }
            ChatEvent::ThinkingDelta { .. }
            | ChatEvent::ToolCallDelta { .. }
            | ChatEvent::StructuredOutputDelta { .. }
            | ChatEvent::Usage { .. }
            | ChatEvent::Finished { .. }
            | ChatEvent::Error { .. } => {}
        }
        accumulator.apply(event)?;
    }

    let response = accumulator.into_response()?;
    println!();
    if let Some(usage) = response.usage {
        println!("tokens: {}", usage.total_tokens());
    }
    Ok(())
}
