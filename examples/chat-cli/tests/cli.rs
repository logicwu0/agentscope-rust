use agentscope::{StateKey, StateStore};
use agentscope_state_sqlite::SQLiteStateStore;
use std::{
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

fn run(db: &Path, session: &str, input: &str) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_agentscope-chat"))
        .args(["--offline", "--session", session, "--db"])
        .arg(db)
        .env_remove("DEEPSEEK_API_KEY")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[tokio::test]
async fn separate_processes_restore_chat_and_confirmation() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("chat.db");
    let first = run(&db, "one", "hello\nmultiply 6 7\n/quit\n");
    assert!(first.contains("Offline turn 1: hello"));
    assert!(first.contains("Approval required"));
    let store = SQLiteStateStore::open(&db).await.unwrap();
    let key = StateKey::new("local", "one").unwrap();
    assert!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_calls()
            .is_some()
    );
    let second = run(&db, "one", "/approve\nagain\n/history\n/quit\n");
    assert!(second.contains("Approval required"));
    assert!(second.contains("42"));
    assert!(second.contains("Offline turn 3: again"));
    assert!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_calls()
            .is_none()
    );
    let isolated = run(&db, "two", "hello\n/quit\n");
    assert!(isolated.contains("Offline turn 1: hello"));
    let denied = run(&db, "two", "multiply 2 3\n/deny no thanks\n/quit\n");
    assert!(denied.contains("no thanks"));
}

#[tokio::test]
async fn cli_reconciles_an_uncertain_execution_and_replays_result() {
    use agentscope::{AgentState, IdempotencyRequest, IdempotencyStore, ToolContext};
    use agentscope_idempotency_sqlite::SQLiteIdempotencyStore;
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("chat.db");
    run(&db, "one", "multiply 6 7\n/quit\n");
    let store = SQLiteStateStore::open(&db).await.unwrap();
    let key = StateKey::new("local", "one").unwrap();
    let record = store.load(&key).await.unwrap().unwrap();
    let pending = record.state().pending_tool_calls().unwrap();
    let call = &pending.calls()[0];
    let call_id = call.id().to_owned();
    let mut wire = serde_json::to_value(record.state()).unwrap();
    wire["pending_tool_execution"] = json!({"execution_id":"interrupted", "confirmation": pending, "decisions":[{"tool_call_id":call_id,"decision":{"decision":"approve"}}]});
    wire.as_object_mut().unwrap().remove("pending_tool_calls");
    let state: AgentState = serde_json::from_value(wire).unwrap();
    store
        .save(key, Some(record.revision()), state)
        .await
        .unwrap();
    let cache = SQLiteIdempotencyStore::open(&db).await.unwrap();
    let namespace = serde_json::to_string(&("local", "one", "multiply:v1")).unwrap();
    let request = IdempotencyRequest::new(
        namespace,
        "multiply",
        call.parsed_input().unwrap(),
        ToolContext::new().with_idempotency_key(format!("interrupted:{call_id}")),
    )
    .unwrap();
    cache.claim(request).await.unwrap();
    let output = run(
        &db,
        "one",
        &format!("/retry\n/resolve {call_id} 42\n/retry\n/quit\n"),
    );
    assert!(output.contains("Uncertain execution"));
    assert!(output.contains("Verified result saved"));
    assert!(output.contains("Tool result: Text(\"42\")"));
}
