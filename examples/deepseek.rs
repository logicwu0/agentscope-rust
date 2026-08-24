//! Calls `DeepSeek` through its OpenAI-compatible chat-completions API.

use std::error::Error;

use agentscope::{ChatModel, ChatRequest, Msg, OpenAIChatModel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let model = OpenAIChatModel::builder()
        .model("deepseek-chat")
        .api_key_from_env("DEEPSEEK_API_KEY")?
        .base_url("https://api.deepseek.com")
        .build()?;
    let response = model
        .generate(ChatRequest::new([Msg::user(
            "Reply with exactly: AgentScope Rust works",
        )]))
        .await?;

    println!("{}", response.text_content("").unwrap_or_default());
    Ok(())
}
