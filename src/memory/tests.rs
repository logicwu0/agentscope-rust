use std::sync::Arc;

use futures_executor::block_on;

#[cfg(feature = "sqlite")]
use crate::{ContentBlock, Role, SQLiteMemory, ToolCallBlock, ToolResultBlock};
use crate::{InMemoryMemory, Memory, Msg};

#[test]
fn in_memory_memory_preserves_order_and_returns_snapshots() {
    let memory = InMemoryMemory::from_messages([Msg::system("Be concise")]);
    block_on(memory.append(vec![Msg::user("Hello"), Msg::assistant("bot", "Hi")])).unwrap();

    let mut snapshot = block_on(memory.messages()).unwrap();
    snapshot.push(Msg::user("not persisted"));
    let stored = block_on(memory.messages()).unwrap();

    assert_eq!(stored.len(), 3);
    assert_eq!(stored[0].text_content(""), Some("Be concise".to_owned()));
    assert_eq!(stored[1].text_content(""), Some("Hello".to_owned()));
    assert_eq!(stored[2].text_content(""), Some("Hi".to_owned()));
}

#[test]
fn memory_is_object_safe_and_can_be_cleared() {
    let memory: Arc<dyn Memory> = Arc::new(InMemoryMemory::new());

    block_on(memory.append(vec![Msg::user("Hello")])).unwrap();
    assert_eq!(block_on(memory.messages()).unwrap().len(), 1);
    block_on(memory.clear()).unwrap();

    assert!(block_on(memory.messages()).unwrap().is_empty());
}

#[tokio::test]
#[cfg(feature = "sqlite")]
async fn sqlite_memory_persists_complete_messages_across_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let database_path = directory.path().join("memory.db");
    let call = ToolCallBlock::complete("call-1", "calculator", r#"{"expression":"6*7"}"#).unwrap();
    let result = ToolResultBlock::success("call-1", "calculator", "42").unwrap();
    let expected = vec![
        Msg::user("What is 6 * 7?"),
        Msg::new("Friday", Role::Assistant, [ContentBlock::from(call)]),
        Msg::new("tool", Role::Assistant, [ContentBlock::from(result)]),
        Msg::assistant("Friday", "The answer is 42."),
    ];

    let memory = SQLiteMemory::open(&database_path, "session-1")
        .await
        .unwrap();
    memory.append(expected.clone()).await.unwrap();
    drop(memory);

    let reopened = SQLiteMemory::open(&database_path, "session-1")
        .await
        .unwrap();
    assert_eq!(reopened.session_id(), "session-1");
    assert_eq!(reopened.messages().await.unwrap(), expected);
}

#[tokio::test]
#[cfg(feature = "sqlite")]
async fn sqlite_memory_isolates_sessions_and_clears_only_one() {
    let directory = tempfile::tempdir().unwrap();
    let database_path = directory.path().join("sessions.db");
    let first = SQLiteMemory::open(&database_path, "first").await.unwrap();
    let second = SQLiteMemory::open(&database_path, "second").await.unwrap();

    first
        .append(vec![Msg::user("first message")])
        .await
        .unwrap();
    second
        .append(vec![Msg::user("second message")])
        .await
        .unwrap();

    assert_eq!(first.messages().await.unwrap().len(), 1);
    assert_eq!(second.messages().await.unwrap().len(), 1);
    first.clear().await.unwrap();
    assert!(first.messages().await.unwrap().is_empty());
    assert_eq!(
        second.messages().await.unwrap()[0].text_content(""),
        Some("second message".to_owned())
    );
}

#[tokio::test]
#[cfg(feature = "sqlite")]
async fn sqlite_memory_rolls_back_an_invalid_batch() {
    let memory = SQLiteMemory::open_in_memory("session-1").await.unwrap();
    let duplicate = Msg::user("store me once");

    let error = memory
        .append(vec![duplicate.clone(), duplicate])
        .await
        .unwrap_err();

    assert_eq!(error.code.as_deref(), Some("sqlite_write"));
    assert!(memory.messages().await.unwrap().is_empty());
}

#[tokio::test]
#[cfg(feature = "sqlite")]
async fn sqlite_memory_rejects_blank_session_ids() {
    let error = SQLiteMemory::open_in_memory("  ").await.unwrap_err();

    assert_eq!(error.code.as_deref(), Some("invalid_session_id"));
}

#[tokio::test]
#[cfg(feature = "sqlite")]
async fn sqlite_memory_rejects_unknown_schema_versions() {
    let directory = tempfile::tempdir().unwrap();
    let database_path = directory.path().join("future.db");
    let connection = tokio_rusqlite::Connection::open(&database_path)
        .await
        .unwrap();
    connection
        .call(|database| {
            database.execute_batch(
                "CREATE TABLE agentscope_memory_schema (
                    singleton INTEGER PRIMARY KEY,
                    version INTEGER NOT NULL
                );
                INSERT INTO agentscope_memory_schema (singleton, version) VALUES (1, 999);",
            )
        })
        .await
        .unwrap();
    drop(connection);

    let error = SQLiteMemory::open(&database_path, "session-1")
        .await
        .unwrap_err();

    assert_eq!(error.code.as_deref(), Some("unsupported_schema_version"));
}
