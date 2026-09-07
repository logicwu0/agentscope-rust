use agentscope::{AgentState, Msg, StateKey, StateStore};
use agentscope_state_sqlite::SQLiteStateStore;

#[tokio::test]
async fn reopens_isolated_sessions_and_rejects_stale_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.db");
    let store = SQLiteStateStore::open(&path).await.unwrap();
    let keys =
        [("a", "one"), ("b", "one"), ("a", "two")].map(|(u, s)| StateKey::new(u, s).unwrap());
    for (index, key) in keys.iter().enumerate() {
        assert!(store.load(key).await.unwrap().is_none());
        store
            .save(
                key.clone(),
                None,
                AgentState::new("agent", vec![Msg::user(index.to_string())]),
            )
            .await
            .unwrap();
    }
    drop(store);
    let store = SQLiteStateStore::open(&path).await.unwrap();
    for (index, key) in keys.iter().enumerate() {
        let record = store.load(key).await.unwrap().unwrap();
        assert_eq!(record.revision(), 1);
        assert_eq!(
            record.state().messages()[0].text_content(""),
            Some(index.to_string())
        );
    }
    let other = SQLiteStateStore::open(&path).await.unwrap();
    let state = AgentState::new("agent", vec![Msg::user("winner")]);
    let (a, b) = tokio::join!(
        store.save(keys[0].clone(), Some(1), state.clone()),
        other.save(keys[0].clone(), Some(1), state)
    );
    assert_ne!(a.is_ok(), b.is_ok());
    let error = a.err().or_else(|| b.err()).unwrap();
    assert_eq!(error.code.as_deref(), Some("revision_conflict"));
    assert!(error.retryable);
    assert_eq!(store.load(&keys[0]).await.unwrap().unwrap().revision(), 2);
}

#[tokio::test]
async fn restart_restores_confirmation_and_continues_the_original_reply() {
    use agentscope::{
        AgentError, ChatResponse, ContentBlock, FinishReason, InMemoryMemory, MockChatModel,
        MockTool, ReActAgent, ToolCallBlock, ToolConfirmation, ToolDefinition, ToolExecutor,
        ToolRegistry,
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.db");
    let key = StateKey::new("user", "session").unwrap();
    let first = ReActAgent::new(
        "agent",
        MockChatModel::new("mock").with_response(ChatResponse::finished(
            [ContentBlock::from(
                ToolCallBlock::complete("call", "tool", "{}").unwrap(),
            )],
            FinishReason::ToolCalls,
        )),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_tool_confirmation_required("tool")
    .with_state_store(key.clone(), SQLiteStateStore::open(&path).await.unwrap());
    let AgentError::ToolConfirmationRequired { checkpoint } =
        first.reply(Msg::user("go")).await.unwrap_err()
    else {
        panic!("expected confirmation")
    };
    drop(first);
    let store = SQLiteStateStore::open(&path).await.unwrap();
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_calls(),
        Some(&checkpoint)
    );
    let mut registry = ToolRegistry::new();
    registry
        .register(
            MockTool::new(
                ToolDefinition::new("tool", "test", serde_json::json!({"type":"object"})).unwrap(),
            )
            .with_output("ok"),
        )
        .unwrap();
    let resumed = ReActAgent::new(
        "agent",
        MockChatModel::new("mock")
            .with_response(ChatResponse::completed([ContentBlock::from("done")])),
        ToolExecutor::new(registry),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_state_store(key.clone(), store.clone());
    assert_eq!(
        resumed
            .resume_tool_calls(
                checkpoint.reply_id(),
                vec![ToolConfirmation::approve("call")]
            )
            .await
            .unwrap()
            .text_content(""),
        Some("done".into())
    );
    let record = store.load(&key).await.unwrap().unwrap();
    assert_eq!(record.revision(), 3);
    assert!(record.state().pending_tool_calls().is_none());
    assert!(record.state().pending_tool_execution().is_none());
}

#[tokio::test]
async fn rejects_unknown_schema_and_corrupt_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.db");
    let store = SQLiteStateStore::open(&path).await.unwrap();
    let key = StateKey::new("u", "s").unwrap();
    store
        .save(key.clone(), None, AgentState::new("agent", vec![]))
        .await
        .unwrap();
    let raw = tokio_rusqlite::Connection::open(&path).await.unwrap();
    raw.call(|db| db.execute("UPDATE agentscope_states SET state_json = 'invalid'", []))
        .await
        .unwrap();
    assert_eq!(
        store.load(&key).await.unwrap_err().code.as_deref(),
        Some("invalid_state_json")
    );
    raw.call(|db| db.execute("UPDATE agentscope_state_schema SET version = 999", []))
        .await
        .unwrap();
    let error = SQLiteStateStore::open(&path).await.err().unwrap();
    assert_eq!(error.code.as_deref(), Some("unsupported_schema_version"));
}
