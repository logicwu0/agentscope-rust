use futures_executor::block_on;
use futures_util::FutureExt;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Notify;

use super::super::*;
use crate::{MockTool, ToolResultOutput};

fn definition() -> ToolDefinition {
    ToolDefinition::new("write", "Write a value", json!({"type":"object"})).unwrap()
}

#[test]
fn duplicate_reuses_output_and_rejects_conflicting_input_and_context() {
    let inner = Arc::new(MockTool::new(definition()).with_output("saved"));
    let tool = IdempotentTool::from_shared(inner.clone());
    let context = ToolContext::new().with_idempotency_key("key");
    let first = block_on(tool.execute(json!({"a":1}), context.clone())).unwrap();
    assert_eq!(
        block_on(tool.clone().execute(json!({"a":1}), context.clone())).unwrap(),
        first
    );
    let error = block_on(tool.execute(json!({"a":2}), context.clone())).unwrap_err();
    assert_eq!(error.code.as_deref(), Some("idempotency_conflict"));
    let mut changed = context;
    changed.metadata.insert("tenant".into(), json!("different"));
    assert_eq!(
        block_on(tool.execute(json!({"a":1}), changed))
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_conflict")
    );
    assert_eq!(inner.recorded_invocations().len(), 1);
}

#[test]
fn errors_are_cached_and_capacity_never_evicts_old_keys() {
    let inner = Arc::new(
        MockTool::new(definition()).with_error(ToolError::new("failed").with_retryable(true)),
    );
    let tool = IdempotentTool::with_capacity(inner.clone(), 1);
    let context = ToolContext::new().with_idempotency_key("key");
    let first = block_on(tool.execute(json!({}), context.clone())).unwrap_err();
    assert_eq!(
        block_on(tool.execute(json!({}), context.clone())).unwrap_err(),
        first
    );
    assert_eq!(
        block_on(tool.execute(json!({}), ToolContext::new().with_idempotency_key("new")))
            .unwrap_err()
            .code
            .as_deref(),
        Some("idempotency_capacity_exceeded")
    );
    assert_eq!(
        block_on(tool.execute(json!({}), context)).unwrap_err(),
        first
    );
    assert_eq!(inner.recorded_invocations().len(), 1);
}

#[test]
fn missing_keys_pass_through_and_invalid_keys_fail_before_execution() {
    let inner = Arc::new(
        MockTool::new(definition())
            .with_output("one")
            .with_output("two"),
    );
    let tool = IdempotentTool::from_shared(inner.clone());
    block_on(tool.execute(json!({}), ToolContext::new())).unwrap();
    block_on(tool.execute(json!({}), ToolContext::new())).unwrap();
    for value in [json!(" "), json!(3), Value::Null] {
        let mut context = ToolContext::new();
        context.metadata.insert(TOOL_IDEMPOTENCY_KEY.into(), value);
        assert_eq!(
            block_on(tool.execute(json!({}), context))
                .unwrap_err()
                .code
                .as_deref(),
            Some("invalid_idempotency_key")
        );
    }
    assert_eq!(inner.recorded_invocations().len(), 2);
}

struct BlockingTool {
    calls: AtomicUsize,
    started: Notify,
    release: Notify,
}

// Keep the definition owned alongside the blocking control for a normal Tool implementation.
struct ControlledTool {
    definition: ToolDefinition,
    control: Arc<BlockingTool>,
}

impl Tool for ControlledTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    fn execute(&self, _input: Value, _context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        Box::pin(async move {
            self.control.calls.fetch_add(1, Ordering::SeqCst);
            self.control.started.notify_one();
            self.control.release.notified().await;
            Ok("saved".into())
        })
    }
}

#[tokio::test]
async fn concurrent_duplicates_wait_and_cancelled_execution_stays_uncertain() {
    let control = Arc::new(BlockingTool {
        calls: AtomicUsize::new(0),
        started: Notify::new(),
        release: Notify::new(),
    });
    let tool = IdempotentTool::new(ControlledTool {
        definition: definition(),
        control: control.clone(),
    });
    let context = ToolContext::new().with_idempotency_key("key");
    let owner = tool.clone();
    let owner_context = context.clone();
    let task = tokio::spawn(async move { owner.execute(json!({}), owner_context).await });
    control.started.notified().await;
    let mut duplicate = tool.execute(json!({}), context.clone());
    assert!(duplicate.as_mut().now_or_never().is_none());
    control.release.notify_one();
    assert_eq!(task.await.unwrap().unwrap(), duplicate.await.unwrap());
    assert_eq!(control.calls.load(Ordering::SeqCst), 1);

    let owner = tool.clone();
    let task = tokio::spawn(async move {
        owner
            .execute(
                json!({}),
                ToolContext::new().with_idempotency_key("cancelled"),
            )
            .await
    });
    control.started.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    let error = tool
        .execute(
            json!({}),
            ToolContext::new().with_idempotency_key("cancelled"),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code.as_deref(), Some("idempotency_in_doubt"));
    assert_eq!(control.calls.load(Ordering::SeqCst), 2);
}
