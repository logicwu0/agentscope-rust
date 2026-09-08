use std::sync::Arc;

use agentscope::{
    IdempotencyClaim, IdempotencyRequest, IdempotencyStore, MockTool, PersistentIdempotentTool,
    Tool, ToolContext, ToolDefinition, ToolError, ToolResultOutput,
};
use agentscope_idempotency_sqlite::SQLiteIdempotencyStore;
use serde_json::{Value, json};

fn definition() -> ToolDefinition {
    ToolDefinition::new("write", "Write a value", json!({"type":"object"})).unwrap()
}

fn context(key: &str) -> ToolContext {
    ToolContext::new().with_idempotency_key(key)
}

fn request(namespace: &str, tool: &str, key: &str) -> IdempotencyRequest {
    IdempotencyRequest::new(namespace, tool, json!({"a":1}), context(key)).unwrap()
}

#[tokio::test]
async fn reopened_wrapper_reuses_success_and_error_without_invoking_tool() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.db");
    let inner = Arc::new(
        MockTool::new(definition())
            .with_output("saved")
            .with_error(ToolError::new("known failure").with_retryable(true)),
    );
    let tool = PersistentIdempotentTool::from_shared(
        "tenant:v1",
        inner.clone(),
        Arc::new(SQLiteIdempotencyStore::open(&path).await.unwrap()),
    )
    .unwrap();
    let success = tool
        .execute(json!({"a":1}), context("success"))
        .await
        .unwrap();
    let failure = tool
        .execute(json!({"a":1}), context("failure"))
        .await
        .unwrap_err();
    assert_eq!(inner.recorded_invocations().len(), 2);
    drop(tool);
    let inner = Arc::new(MockTool::new(definition()));
    let tool = PersistentIdempotentTool::from_shared(
        "tenant:v1",
        inner.clone(),
        Arc::new(SQLiteIdempotencyStore::open(&path).await.unwrap()),
    )
    .unwrap();
    assert_eq!(
        tool.execute(json!({"a":1}), context("success"))
            .await
            .unwrap(),
        success
    );
    assert_eq!(
        tool.execute(json!({"a":1}), context("failure"))
            .await
            .unwrap_err(),
        failure
    );
    assert_eq!(
        tool.execute(json!({"a":2}), context("success"))
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_conflict")
    );
    let mut changed_context = context("success");
    changed_context
        .metadata
        .insert("tenant".into(), json!("someone else"));
    assert_eq!(
        tool.execute(json!({"a":1}), changed_context)
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_conflict")
    );
    assert!(inner.recorded_invocations().is_empty());
}

#[tokio::test]
async fn claims_are_atomic_isolated_and_completion_is_immutable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.db");
    let first = SQLiteIdempotencyStore::open(&path).await.unwrap();
    let second = SQLiteIdempotencyStore::open(&path).await.unwrap();
    let request = request("tenant", "write", "key");
    let (a, b) = tokio::join!(first.claim(request.clone()), second.claim(request.clone()));
    let (a, b) = (a.unwrap(), b.unwrap());
    let token = match (a, b) {
        (IdempotencyClaim::Acquired { token }, IdempotencyClaim::InDoubt)
        | (IdempotencyClaim::InDoubt, IdempotencyClaim::Acquired { token }) => token,
        other => panic!("expected exactly one owner: {other:?}"),
    };
    assert_eq!(
        first
            .complete(request.clone(), "wrong".into(), Ok("bad".into()))
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_stale_owner")
    );
    assert_eq!(
        first.claim(request.clone()).await.unwrap(),
        IdempotencyClaim::InDoubt
    );
    first
        .reconcile(request.clone(), Ok("verified".into()))
        .await
        .unwrap();
    first
        .reconcile(request.clone(), Ok("verified".into()))
        .await
        .unwrap();
    assert_eq!(
        first
            .complete(request.clone(), token, Ok("late worker".into()))
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_conflict")
    );
    assert_eq!(
        first.claim(request).await.unwrap(),
        IdempotencyClaim::Completed {
            result: Ok("verified".into())
        }
    );
    for (namespace, tool) in [("other", "write"), ("tenant", "other")] {
        let distinct =
            IdempotencyRequest::new(namespace, tool, json!({"a":1}), context("key")).unwrap();
        assert!(matches!(
            first.claim(distinct).await.unwrap(),
            IdempotencyClaim::Acquired { .. }
        ));
    }
}

#[tokio::test]
async fn failed_result_write_leaves_durable_uncertain_claim() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.db");
    let store = Arc::new(SQLiteIdempotencyStore::open(&path).await.unwrap());
    let raw = tokio_rusqlite::Connection::open(&path).await.unwrap();
    raw.call(|db| db.execute_batch("CREATE TRIGGER fail_result BEFORE UPDATE ON agentscope_idempotency BEGIN SELECT RAISE(ABORT, 'injected write failure'); END;")).await.unwrap();
    let inner = Arc::new(MockTool::new(definition()).with_output("side effect done"));
    let wrapper =
        PersistentIdempotentTool::from_shared("tenant", inner.clone(), store.clone()).unwrap();
    assert_eq!(
        wrapper
            .execute(json!({"a":1}), context("key"))
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_in_doubt")
    );
    assert_eq!(
        wrapper
            .execute(json!({"a":1}), context("key"))
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_in_doubt")
    );
    assert_eq!(inner.recorded_invocations().len(), 1);
    assert_eq!(
        SQLiteIdempotencyStore::open(&path)
            .await
            .unwrap()
            .claim(request("tenant", "write", "key"))
            .await
            .unwrap(),
        IdempotencyClaim::InDoubt
    );
    raw.call(|db| db.execute_batch("DROP TRIGGER fail_result;"))
        .await
        .unwrap();
    store
        .reconcile(
            request("tenant", "write", "key"),
            Ok("verified effect".into()),
        )
        .await
        .unwrap();
    assert_eq!(
        wrapper
            .execute(json!({"a":1}), context("key"))
            .await
            .unwrap(),
        ToolResultOutput::from("verified effect")
    );
    assert_eq!(inner.recorded_invocations().len(), 1);
}

#[tokio::test]
async fn invalid_inputs_unknown_records_and_schema_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.db");
    let store = Arc::new(SQLiteIdempotencyStore::open(&path).await.unwrap());
    assert_eq!(
        store
            .reconcile(request("tenant", "write", "missing"), Ok("result".into()))
            .await
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_not_found")
    );
    let inner = Arc::new(MockTool::new(definition()).with_output("unkeyed"));
    let wrapper =
        PersistentIdempotentTool::from_shared("tenant", inner.clone(), store.clone()).unwrap();
    for key in [Value::Null, json!(42), json!(" ")] {
        let mut context = ToolContext::new();
        context
            .metadata
            .insert(agentscope::TOOL_IDEMPOTENCY_KEY.into(), key);
        assert_eq!(
            wrapper
                .execute(json!({}), context)
                .await
                .unwrap_err()
                .code
                .as_deref(),
            Some("invalid_idempotency_key")
        );
    }
    assert!(inner.recorded_invocations().is_empty());
    wrapper
        .execute(json!({}), ToolContext::new())
        .await
        .unwrap();
    assert_eq!(inner.recorded_invocations().len(), 1);
    let raw = tokio_rusqlite::Connection::open(&path).await.unwrap();
    let pending = request("tenant", "write", "key");
    store.claim(pending.clone()).await.unwrap();
    raw.call(|db| {
        db.execute(
            "UPDATE agentscope_idempotency SET request_json = 'broken'",
            [],
        )
    })
    .await
    .unwrap();
    assert_eq!(
        store.claim(pending).await.unwrap_err().code.as_deref(),
        Some("invalid_idempotency_json")
    );
    raw.call(|db| db.execute("UPDATE agentscope_idempotency_schema SET version = 999", []))
        .await
        .unwrap();
    assert_eq!(
        SQLiteIdempotencyStore::open(&path)
            .await
            .err()
            .unwrap()
            .code
            .as_deref(),
        Some("unsupported_schema_version")
    );
}
