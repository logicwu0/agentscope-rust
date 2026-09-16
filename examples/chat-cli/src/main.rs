mod model;
mod session;

use agentscope::{
    ChatModel, ChatModelSummarizer, ContextSummarizer, InMemoryMemory, OpenAIChatModel,
    PersistentIdempotentTool, ReActAgent, StateKey, TokenBudget, ToolExecutor, ToolRegistry,
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
    auto_compact: Option<usize>,
    context_window: u64,
    output_reserve: u32,
}

fn options() -> Result<Option<Options>> {
    let mut result = Options {
        db: "agentscope-chat.db".into(),
        user: "local".into(),
        session: "default".into(),
        offline: false,
        auto_compact: None,
        context_window: 8192,
        output_reserve: 1024,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "agentscope-chat [--offline] [--db PATH] [--user USER] [--session SESSION]\nOptional: --auto-compact KEEP [--context-window TOKENS] [--output-reserve TOKENS]\nAuto compaction is OFF by default and may add paid model calls. Budget defaults: 8192/1024.\nDeepSeek reads DEEPSEEK_API_KEY from the environment. Offline tool prompt: multiply 6 7"
                );
                return Ok(None);
            }
            "--offline" => result.offline = true,
            "--auto-compact" => {
                let keep = args
                    .next()
                    .ok_or("missing retained turn count")?
                    .parse::<usize>()?;
                if keep == 0 {
                    return Err("retained turn count must be positive".into());
                }
                result.auto_compact = Some(keep);
            }
            "--context-window" => {
                result.context_window = args.next().ok_or("missing context window")?.parse()?;
            }
            "--output-reserve" => {
                result.output_reserve = args.next().ok_or("missing output reserve")?.parse()?;
            }
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
    if result.auto_compact.is_some() {
        TokenBudget::new(result.context_window, result.output_reserve)?;
    } else if result.context_window != 8192 || result.output_reserve != 1024 {
        return Err("budget options require --auto-compact".into());
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
    let mut agent = ReActAgent::from_shared("chat", model.clone(), ToolExecutor::new(registry))?
        .with_memory(InMemoryMemory::new()).with_state_store(key, store)
        .with_tool_confirmation_required("multiply")
        .with_system_prompt("You are a helpful assistant. Use multiply for integer multiplication. Tool calls require human approval.");
    if let Some(keep) = options.auto_compact {
        let summarizer: Arc<dyn ContextSummarizer> = if options.offline {
            Arc::new(model::OfflineSummarizer)
        } else {
            Arc::new(ChatModelSummarizer::from_shared(
                model,
                TokenBudget::new(
                    options.context_window.saturating_mul(2),
                    options.output_reserve,
                )?,
            ))
        };
        agent = agent
            .with_shared_summarizer(summarizer)
            .with_token_budget(TokenBudget::new(
                options.context_window,
                options.output_reserve,
            )?)
            .with_auto_compaction(keep)?;
        println!(
            "Automatic compaction enabled: keep {keep} existing turns, input/output window {}/{}.",
            options.context_window, options.output_reserve
        );
    }
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
