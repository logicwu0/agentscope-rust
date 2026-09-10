mod model;
mod session;

use agentscope::{
    ChatModel, InMemoryMemory, OpenAIChatModel, PersistentIdempotentTool, ReActAgent, StateKey,
    ToolExecutor, ToolRegistry,
};
use agentscope_idempotency_sqlite::SQLiteIdempotencyStore;
use agentscope_state_sqlite::SQLiteStateStore;
use std::{
    error::Error,
    io::{self, Write},
    sync::Arc,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Options {
    db: String,
    user: String,
    session: String,
    offline: bool,
}

fn options() -> Result<Option<Options>> {
    let mut result = Options {
        db: "agentscope-chat.db".into(),
        user: "local".into(),
        session: "default".into(),
        offline: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "agentscope-chat [--offline] [--db PATH] [--user USER] [--session SESSION]\nDeepSeek reads DEEPSEEK_API_KEY from the environment. Offline tool prompt: multiply 6 7"
                );
                return Ok(None);
            }
            "--offline" => result.offline = true,
            "--db" | "--user" | "--session" => {
                let value = args.next().ok_or("missing option value")?;
                if value.trim().is_empty() {
                    return Err("option value cannot be blank".into());
                }
                match arg.as_str() {
                    "--db" => result.db = value,
                    "--user" => result.user = value,
                    _ => result.session = value,
                }
            }
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    Ok(Some(result))
}

#[tokio::main]
async fn main() -> Result<()> {
    let Some(options) = options()? else {
        return Ok(());
    };
    let model: Arc<dyn ChatModel> = if options.offline {
        Arc::new(model::OfflineModel)
    } else {
        Arc::new(
            OpenAIChatModel::builder()
                .model("deepseek-chat")
                .api_key_from_env("DEEPSEEK_API_KEY")?
                .base_url("https://api.deepseek.com")
                .build()?,
        )
    };
    let key = StateKey::new(&options.user, &options.session)?;
    let store = SQLiteStateStore::open(&options.db).await?;
    let idempotency = Arc::new(SQLiteIdempotencyStore::open(&options.db).await?);
    let namespace = serde_json::to_string(&(&options.user, &options.session, "multiply:v1"))?;
    let mut registry = ToolRegistry::new();
    registry.register(PersistentIdempotentTool::from_shared(
        &namespace,
        Arc::new(model::Multiply::new()),
        idempotency.clone(),
    )?)?;
    let agent = ReActAgent::from_shared("chat", model, ToolExecutor::new(registry))?
        .with_memory(InMemoryMemory::new()).with_state_store(key, store)
        .with_tool_confirmation_required("multiply")
        .with_system_prompt("You are a helpful assistant. Use multiply for integer multiplication. Tool calls require human approval.");
    let session = session::Session {
        agent,
        idempotency,
        namespace,
    };
    println!(
        "Session {}/{} [{}]. /help for commands.",
        options.user,
        options.session,
        if options.offline {
            "offline"
        } else {
            "DeepSeek"
        }
    );
    session.status().await?;
    loop {
        print!("you> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line == "/quit" {
            break;
        }
        if line.is_empty() {
            continue;
        }
        if let Err(error) = session.handle(line).await {
            eprintln!("Error: {error}");
        }
    }
    println!("Goodbye. Saved checkpoints remain available on restart.");
    Ok(())
}
