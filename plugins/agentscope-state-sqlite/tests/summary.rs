use agentscope::{
    ChatModelSummarizer, ChatResponse, ContentBlock, InMemoryMemory, MockChatModel, Msg,
    ReActAgent, StateKey, TokenBudget, ToolExecutor, ToolRegistry,
};
use agentscope_state_sqlite::SQLiteStateStore;
use std::sync::Arc;

#[tokio::test]
async fn summary_survives_database_close_and_reopen_without_removing_raw_messages() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("summary.db");
    let key = StateKey::new("user", "summary").unwrap();
    let raw = vec![
        Msg::user("old question".repeat(100)),
        Msg::assistant("Friday", "old answer".repeat(100)),
        Msg::user("recent"),
        Msg::assistant("Friday", "answer"),
    ];
    let summarizer = ChatModelSummarizer::new(
        MockChatModel::new("summary").with_response(ChatResponse::completed([ContentBlock::from(
            "Old goal completed.",
        )])),
        TokenBudget::new(10000, 500).unwrap(),
    );
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("main"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::from_messages(raw.clone()))
    .with_state_store(key.clone(), SQLiteStateStore::open(&path).await.unwrap())
    .with_summarizer(summarizer);
    let summary = agent.compact_context(1).await.unwrap().unwrap();
    drop(agent);
    let model = Arc::new(
        MockChatModel::new("main")
            .with_response(ChatResponse::completed([ContentBlock::from("done")])),
    );
    let agent = ReActAgent::from_shared(
        "Friday",
        model.clone(),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_state_store(key, SQLiteStateStore::open(&path).await.unwrap());
    let state = agent.snapshot().await.unwrap();
    assert_eq!(state.messages(), raw);
    assert_eq!(state.context_summary(), Some(&summary));
    agent.reply(Msg::user("continue")).await.unwrap();
    assert!(
        model.recorded_requests()[0]
            .messages
            .iter()
            .any(|m| m.name == "context_summary")
    );
    assert_eq!(agent.snapshot().await.unwrap().messages().len(), 6);
}
