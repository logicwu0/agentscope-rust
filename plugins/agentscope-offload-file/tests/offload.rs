use agentscope::*;
use agentscope_offload_file::FileOffloadStore;
use futures_util::StreamExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

fn config(store: Arc<dyn OffloadStore>) -> ToolResultOffload {
    ToolResultOffload::new(store, 1024, 64, 128).unwrap()
}

fn output(messages: &[Msg], name: &str) -> String {
    messages
        .iter()
        .flat_map(|m| &m.content)
        .find_map(|b| match b {
            ContentBlock::ToolResult(r) if r.name() == name => match r.output() {
                ToolResultOutput::Text(t) => Some(t.clone()),
                ToolResultOutput::Blocks(_) => None,
            },
            _ => None,
        })
        .unwrap()
}

fn fixture(model: Arc<MockChatModel>, text: &str) -> (ReActAgent, Arc<MockTool>) {
    let tool = Arc::new(
        MockTool::new(
            ToolDefinition::new("large", "Return text", json!({"type":"object"})).unwrap(),
        )
        .with_output(text),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let agent = ReActAgent::from_shared("agent", model, ToolExecutor::new(registry))
        .unwrap()
        .with_memory(InMemoryMemory::new());
    (agent, tool)
}

#[allow(clippy::needless_pass_by_value)]
fn call(name: &str, input: Value) -> ChatResponse {
    ChatResponse::finished(
        [ToolCallBlock::complete(name, name, input.to_string())
            .unwrap()
            .into()],
        FinishReason::ToolCalls,
    )
}
fn done() -> ChatResponse {
    ChatResponse::completed([ContentBlock::from("done")])
}

#[tokio::test]
async fn offload_precedes_auto_compaction_budget_check() {
    let dir = tempfile::tempdir().unwrap();
    let text = "x".repeat(20000);
    let history = vec![
        Msg::user("old"),
        Msg::new(
            "agent",
            Role::Assistant,
            [ToolCallBlock::complete("large", "large", "{}")
                .unwrap()
                .into()],
        ),
        Msg::new(
            "tool",
            Role::User,
            [ToolResultBlock::success("large", "large", text.clone())
                .unwrap()
                .into()],
        ),
        Msg::assistant("agent", "done"),
    ];
    let model = Arc::new(MockChatModel::new("main").with_response(done()));
    let summary = Arc::new(MockChatModel::new("summary"));
    let (agent, _) = fixture(model.clone(), "unused");
    let agent = agent
        .with_memory(InMemoryMemory::from_messages(history))
        .with_tool_result_offload(config(Arc::new(FileOffloadStore::new(dir.path()).unwrap())))
        .unwrap()
        .with_token_budget(TokenBudget::new(4000, 128).unwrap())
        .with_summarizer(ChatModelSummarizer::from_shared(
            summary.clone(),
            TokenBudget::new(4000, 128).unwrap(),
        ))
        .with_auto_compaction(1)
        .unwrap();
    agent.reply(Msg::user("continue")).await.unwrap();
    assert!(summary.recorded_requests().is_empty());
    assert!(output(&model.recorded_requests()[0].messages, "large").len() <= 1024);
    assert_eq!(
        output(agent.snapshot().await.unwrap().messages(), "large"),
        text
    );
}

#[tokio::test]
async fn store_roundtrip_reopen_unicode_and_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileOffloadStore::new(dir.path()).unwrap();
    let text = "a你好🙂end";
    let id = store.put(text).await.unwrap();
    assert_eq!(store.put(text).await.unwrap(), id);
    let store = FileOffloadStore::new(dir.path()).unwrap();
    let mut offset = 0;
    let mut all = String::new();
    loop {
        let chunk = store.read(&id, offset, 4).await.unwrap();
        assert!(chunk.text.len() <= 4);
        all.push_str(&chunk.text);
        offset = chunk.next_offset;
        if offset == chunk.total_bytes {
            break;
        }
    }
    assert_eq!(all, text);
    assert!(store.read(&id, 2, 4).await.is_err());
    assert!(store.read(&id, text.len() + 1, 4).await.is_err());
    assert!(store.read(&id, 0, 0).await.is_err());
    assert!(store.read("../secret", 0, 4).await.is_err());
    assert!(store.read(&"0".repeat(64), 0, 4).await.is_err());
    assert_eq!(store.read(&id, text.len(), 4).await.unwrap().text, "");
    let other = tempfile::tempdir().unwrap();
    assert!(
        FileOffloadStore::new(other.path())
            .unwrap()
            .read(&id, 0, 4)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn concurrent_puts_are_atomic_and_deduplicated() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FileOffloadStore::new(dir.path()).unwrap());
    let text = "x".repeat(9000);
    let (a, b) = tokio::join!(store.put(&text), store.put(&text));
    assert_eq!(a.unwrap(), b.unwrap());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[cfg(unix)]
#[tokio::test]
async fn rejects_symlinks_and_existing_corrupt_content() {
    let dir = tempfile::tempdir().unwrap();
    let store = FileOffloadStore::new(dir.path()).unwrap();
    let id = format!("{:x}", Sha256::digest(b"hello"));
    let other = tempfile::NamedTempFile::new().unwrap();
    std::os::unix::fs::symlink(other.path(), dir.path().join(&id)).unwrap();
    assert!(store.read(&id, 0, 4).await.is_err());
    assert!(store.put("hello").await.is_err());
    let corrupt_id = format!("{:x}", Sha256::digest(b"world"));
    std::fs::write(dir.path().join(corrupt_id), b"wrong").unwrap();
    assert!(store.put("world").await.is_err());
}

#[tokio::test]
async fn agent_offloads_reads_and_preserves_raw_state() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FileOffloadStore::new(dir.path()).unwrap());
    let text = "你好🙂".repeat(2000);
    let id = format!("{:x}", Sha256::digest(text.as_bytes()));
    let model = Arc::new(
        MockChatModel::new("mock")
            .with_response(call("large", json!({})))
            .with_response(call(
                "read_offloaded_text",
                json!({"id":id,"offset":0,"max_bytes":128}),
            ))
            .with_response(done()),
    );
    let (agent, tool) = fixture(model.clone(), &text);
    let agent = agent
        .with_tool_result_offload(config(store))
        .unwrap()
        .with_token_budget(TokenBudget::new(4000, 128).unwrap());
    agent.reply(Msg::user("fetch and read")).await.unwrap();
    let requests = model.recorded_requests();
    assert_eq!(requests.len(), 3);
    let preview: Value = serde_json::from_str(&output(&requests[1].messages, "large")).unwrap();
    assert_eq!(preview["offloaded_text"]["id"], id);
    let read: Value =
        serde_json::from_str(&output(&requests[2].messages, "read_offloaded_text")).unwrap();
    assert!(read["text"].as_str().unwrap().len() <= 128);
    assert_eq!(read["total_bytes"], text.len());
    assert_eq!(
        output(agent.snapshot().await.unwrap().messages(), "large"),
        text
    );
    assert_eq!(tool.recorded_invocations().len(), 1);
}

#[tokio::test]
async fn disabled_and_small_results_unchanged() {
    for enabled in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let model = Arc::new(
            MockChatModel::new("mock")
                .with_response(call("large", json!({})))
                .with_response(done()),
        );
        let text = if enabled {
            "small".to_owned()
        } else {
            "x".repeat(9000)
        };
        let (mut agent, _) = fixture(model.clone(), &text);
        if enabled {
            agent = agent
                .with_tool_result_offload(config(Arc::new(
                    FileOffloadStore::new(dir.path()).unwrap(),
                )))
                .unwrap();
        }
        agent.reply(Msg::user("run")).await.unwrap();
        assert_eq!(
            output(&model.recorded_requests()[1].messages, "large"),
            text
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}

struct FailingStore;
impl OffloadStore for FailingStore {
    fn put<'a>(&'a self, _: &'a str) -> ToolFuture<'a, String> {
        Box::pin(async { Err(ToolError::new("disk full")) })
    }
    fn read<'a>(&'a self, _: &'a str, _: usize, _: usize) -> ToolFuture<'a, OffloadedTextChunk> {
        Box::pin(async { Err(ToolError::new("missing")) })
    }
}

#[tokio::test]
async fn storage_failure_preserves_successful_tool_and_durable_history() {
    let text = "x".repeat(9000);
    let model = Arc::new(MockChatModel::new("mock").with_response(call("large", json!({}))));
    let (agent, tool) = fixture(model.clone(), &text);
    let state_store = Arc::new(InMemoryStateStore::new());
    let key = StateKey::new("user", "offload").unwrap();
    let agent = agent
        .with_shared_state_store(key.clone(), state_store.clone())
        .with_tool_result_offload(config(Arc::new(FailingStore)))
        .unwrap();
    assert!(agent.reply(Msg::user("run")).await.is_err());
    let state = agent.snapshot().await.unwrap();
    assert_eq!(output(state.messages(), "large"), text);
    assert!(state.pending_tool_execution().is_none());
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert_eq!(model.recorded_requests().len(), 1);
    assert_eq!(
        state_store.load(&key).await.unwrap().unwrap().state(),
        &state
    );
    // A fresh instance with working storage can project the saved raw result,
    // without rerunning the successful original tool.
    let dir = tempfile::tempdir().unwrap();
    let fresh_model = Arc::new(MockChatModel::new("fresh").with_response(done()));
    let (fresh, fresh_tool) = fixture(fresh_model.clone(), "unused");
    let fresh = fresh
        .with_shared_state_store(key, state_store)
        .with_tool_result_offload(config(Arc::new(FileOffloadStore::new(dir.path()).unwrap())))
        .unwrap();
    fresh.reply(Msg::user("continue")).await.unwrap();
    assert!(fresh_tool.recorded_invocations().is_empty());
    assert!(output(&fresh_model.recorded_requests()[0].messages, "large").len() <= 1024);
}

struct WaitingStore(tokio::sync::Notify);
impl OffloadStore for WaitingStore {
    fn put<'a>(&'a self, _: &'a str) -> ToolFuture<'a, String> {
        Box::pin(async move {
            self.0.notify_one();
            std::future::pending().await
        })
    }
    fn read<'a>(&'a self, _: &'a str, _: usize, _: usize) -> ToolFuture<'a, OffloadedTextChunk> {
        Box::pin(async { Err(ToolError::new("missing")) })
    }
}

#[tokio::test]
async fn interrupt_waiting_offload_preserves_raw_result() {
    let text = "x".repeat(9000);
    let model = Arc::new(MockChatModel::new("mock").with_response(call("large", json!({}))));
    let (agent, tool) = fixture(model.clone(), &text);
    let store = Arc::new(WaitingStore(tokio::sync::Notify::new()));
    let agent = agent
        .with_tool_result_offload(config(store.clone()))
        .unwrap();
    let handle = agent.interrupt_handle();
    let (reply, ()) = tokio::join!(agent.reply(Msg::user("run")), async {
        store.0.notified().await;
        handle.interrupt();
    });
    assert!(matches!(reply, Err(AgentError::Interrupted)));
    assert_eq!(
        output(agent.snapshot().await.unwrap().messages(), "large"),
        text
    );
    assert_eq!(tool.recorded_invocations().len(), 1);
}

#[tokio::test]
async fn offload_does_not_replace_idempotency_cache() {
    let dir = tempfile::tempdir().unwrap();
    let text = "x".repeat(9000);
    let raw = Arc::new(
        MockTool::new(
            ToolDefinition::new("large", "Return text", json!({"type":"object"})).unwrap(),
        )
        .with_output(text.clone()),
    );
    let wrapped = IdempotentTool::from_shared(raw.clone());
    let mut registry = ToolRegistry::new();
    registry.register(wrapped.clone()).unwrap();
    let agent = ReActAgent::new(
        "agent",
        MockChatModel::new("mock")
            .with_response(call("large", json!({})))
            .with_response(done()),
        ToolExecutor::new(registry),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_state_store(
        StateKey::new("user", "idem").unwrap(),
        InMemoryStateStore::new(),
    )
    .with_tool_confirmation_required("large")
    .with_tool_result_offload(config(Arc::new(FileOffloadStore::new(dir.path()).unwrap())))
    .unwrap();
    let Err(AgentError::ToolConfirmationRequired { checkpoint }) =
        agent.reply(Msg::user("run")).await
    else {
        panic!("expected approval")
    };
    agent
        .resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("large")],
        )
        .await
        .unwrap();
    let invocation = raw.recorded_invocations()[0].clone();
    assert!(invocation.context.idempotency_key().is_some());
    let cached = wrapped
        .execute(invocation.input, invocation.context)
        .await
        .unwrap();
    assert_eq!(cached, ToolResultOutput::Text(text));
    assert_eq!(raw.recorded_invocations().len(), 1);
}

#[tokio::test]
async fn streaming_confirmation_resume_uses_same_projection() {
    let dir = tempfile::tempdir().unwrap();
    let text = "x".repeat(9000);
    let model = Arc::new(
        MockChatModel::new("mock")
            .with_response(call("large", json!({})))
            .with_stream([
                Ok(ChatEvent::TextDelta {
                    block_id: "a".into(),
                    delta: "done".into(),
                }),
                Ok(ChatEvent::Finished {
                    reason: FinishReason::Completed,
                }),
            ]),
    );
    let (agent, tool) = fixture(model.clone(), &text);
    let agent = agent
        .with_tool_result_offload(config(Arc::new(FileOffloadStore::new(dir.path()).unwrap())))
        .unwrap()
        .with_tool_confirmation_required("large");
    let Err(AgentError::ToolConfirmationRequired { checkpoint }) =
        agent.reply(Msg::user("run")).await
    else {
        panic!("expected checkpoint")
    };
    assert!(tool.recorded_invocations().is_empty());
    let mut stream = agent
        .stream_resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("large")],
        )
        .await
        .unwrap();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AgentEvent::Finished { .. } => finished = true,
            AgentEvent::Error { error, .. } => panic!("{error}"),
            _ => {}
        }
    }
    drop(stream);
    assert!(finished);
    assert!(output(&model.recorded_requests()[1].messages, "large").len() < 1024);
    assert_eq!(
        output(agent.snapshot().await.unwrap().messages(), "large"),
        text
    );
}

#[tokio::test]
async fn oversized_read_is_rejected_and_config_collision_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FileOffloadStore::new(dir.path()).unwrap());
    let id = store.put("sample").await.unwrap();
    let model = Arc::new(
        MockChatModel::new("mock")
            .with_response(call(
                "read_offloaded_text",
                json!({"id":id,"offset":0,"max_bytes":1_000_000}),
            ))
            .with_response(done()),
    );
    let (agent, _) = fixture(model.clone(), "unused");
    let agent = agent
        .with_tool_result_offload(config(store.clone()))
        .unwrap();
    assert!(
        agent
            .clone()
            .with_tool_result_offload(config(store.clone()))
            .is_err()
    );
    assert!(ToolResultOffload::new(store, 100, 99, 1).is_err());
    agent.reply(Msg::user("read")).await.unwrap();
    assert!(
        model.recorded_requests()[1]
            .messages
            .iter()
            .flat_map(|m| &m.content)
            .any(
                |b| matches!(b, ContentBlock::ToolResult(r) if r.state() == ToolResultState::Error)
            )
    );
}
