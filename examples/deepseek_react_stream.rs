//! Streams a complete `DeepSeek` -> Rust tool -> `DeepSeek` `ReAct` loop.

use std::{error::Error, io::Write};

use agentscope::{
    AgentEvent, Msg, OpenAIChatModel, ReActAgent, Tool, ToolContext, ToolDefinition, ToolError,
    ToolExecutor, ToolFuture, ToolRegistry, ToolResultOutput,
};
use futures_util::StreamExt;
use serde_json::{Value, json};

struct MultiplyTool {
    definition: ToolDefinition,
}

impl MultiplyTool {
    fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            definition: ToolDefinition::new(
                "multiply",
                "Multiply two integers exactly",
                json!({
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"],
                    "additionalProperties": false
                }),
            )?,
        })
    }
}

impl Tool for MultiplyTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn execute(&self, input: Value, _context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        Box::pin(async move {
            let a = input
                .get("a")
                .and_then(Value::as_i64)
                .ok_or_else(|| ToolError::new("`a` must be an integer").with_code("invalid_a"))?;
            let b = input
                .get("b")
                .and_then(Value::as_i64)
                .ok_or_else(|| ToolError::new("`b` must be an integer").with_code("invalid_b"))?;
            let product = a.checked_mul(b).ok_or_else(|| {
                ToolError::new("integer multiplication overflow").with_code("overflow")
            })?;
            Ok(product.to_string().into())
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let model = OpenAIChatModel::builder()
        .model("deepseek-chat")
        .api_key_from_env("DEEPSEEK_API_KEY")?
        .base_url("https://api.deepseek.com")
        .build()?;
    let mut registry = ToolRegistry::new();
    registry.register(MultiplyTool::new()?)?;
    let agent = ReActAgent::new("calculator", model, ToolExecutor::new(registry))?
        .with_max_steps(4)?
        .with_system_prompt(
            "Use the multiply tool for multiplication. Return only the final integer.",
        );

    let mut events = agent
        .stream(Msg::user("What is 123 multiplied by 456?"))
        .await?;
    while let Some(event) = events.next().await {
        match event? {
            AgentEvent::TextDelta { delta, .. } => {
                print!("{delta}");
                std::io::stdout().flush()?;
            }
            AgentEvent::ToolStarted { call, .. } => {
                eprintln!("\ntool started: {} {}", call.name(), call.input());
            }
            AgentEvent::ToolFinished { result, .. } => {
                eprintln!("tool finished: {:?}", result.state());
            }
            AgentEvent::Finished { .. } => println!(),
            AgentEvent::Error { error, .. } => return Err(error.into()),
            _ => {}
        }
    }
    Ok(())
}
