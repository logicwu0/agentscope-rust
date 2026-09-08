use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use agentscope::{
    AgentError, ChatResponse, ContentBlock, FinishReason, IdempotencyRequest, IdempotencyStore,
    InMemoryMemory, MockChatModel, MockTool, Msg, PersistentIdempotentTool, ReActAgent, StateKey,
    StateStore, Tool, ToolCallBlock, ToolConfirmation, ToolContext, ToolDefinition, ToolExecutor,
    ToolFuture, ToolRegistry, ToolResultOutput,
};
use agentscope_idempotency_sqlite::SQLiteIdempotencyStore;
use agentscope_state_sqlite::SQLiteStateStore;
use serde_json::{Value, json};
use tokio::sync::Notify;

struct InterruptedEffect {
    definition: ToolDefinition,
    calls: AtomicUsize,
    started: Notify,
}

impl Tool for InterruptedEffect {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    fn execute(&self, _input: Value, _context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            std::future::pending().await
        })
    }
}

fn executor(tool: Arc<dyn Tool>, store: Arc<SQLiteIdempotencyStore>) -> ToolExecutor {
    let mut registry = ToolRegistry::new();
    registry
        .register(PersistentIdempotentTool::from_shared("tenant:v1", tool, store).unwrap())
        .unwrap();
    ToolExecutor::new(registry)
}

#[tokio::test]
async fn aborted_agent_restores_both_stores_and_resumes_after_reconciliation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("agent.db");
    let key = StateKey::new("tenant", "session").unwrap();
    let effect = Arc::new(InterruptedEffect {
        definition: ToolDefinition::new("write", "Write", json!({"type":"object"})).unwrap(),
        calls: AtomicUsize::new(0),
        started: Notify::new(),
    });
    let idempotency = Arc::new(SQLiteIdempotencyStore::open(&path).await.unwrap());
    let agent = Arc::new(
        ReActAgent::new(
            "agent",
            MockChatModel::new("mock").with_response(ChatResponse::finished(
                [ContentBlock::from(
                    ToolCallBlock::complete("call", "write", "{}").unwrap(),
                )],
                FinishReason::ToolCalls,
            )),
            executor(effect.clone(), idempotency),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("write")
        .with_state_store(key.clone(), SQLiteStateStore::open(&path).await.unwrap()),
    );
    let AgentError::ToolConfirmationRequired { checkpoint } =
        agent.reply(Msg::user("write")).await.unwrap_err()
    else {
        panic!("expected confirmation")
    };
    let reply_id = checkpoint.reply_id().to_owned();
    let owner = agent.clone();
    let running_id = reply_id.clone();
    let task = tokio::spawn(async move {
        owner
            .resume_tool_calls(running_id, vec![ToolConfirmation::approve("call")])
            .await
    });
    effect.started.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    drop(agent);

    let state_store = SQLiteStateStore::open(&path).await.unwrap();
    let state = state_store.load(&key).await.unwrap().unwrap();
    let execution = state.state().pending_tool_execution().unwrap();
    let context =
        ToolContext::new().with_idempotency_key(execution.idempotency_key("call").unwrap());
    let request = IdempotencyRequest::new("tenant:v1", "write", json!({}), context).unwrap();
    let reopened = Arc::new(SQLiteIdempotencyStore::open(&path).await.unwrap());
    let replacement = Arc::new(MockTool::new(effect.definition.clone()));
    let recovered = ReActAgent::new(
        "agent",
        MockChatModel::new("mock")
            .with_response(ChatResponse::completed([ContentBlock::from("done")])),
        executor(replacement.clone(), reopened.clone()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_state_store(key.clone(), state_store.clone());
    assert!(matches!(
        recovered.retry_tool_execution(&reply_id).await.unwrap_err(),
        AgentError::ToolExecutionInDoubt { .. }
    ));
    assert!(
        state_store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_execution()
            .is_some()
    );
    assert!(replacement.recorded_invocations().is_empty());
    // Simulates checking the external system after the original worker has stopped.
    reopened
        .reconcile(request, Ok("verified external result".into()))
        .await
        .unwrap();
    let reply = recovered.retry_tool_execution(&reply_id).await.unwrap();
    assert_eq!(reply.text_content(""), Some("done".into()));
    assert!(replacement.recorded_invocations().is_empty());
    assert_eq!(effect.calls.load(Ordering::SeqCst), 1);
    assert!(
        state_store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_execution()
            .is_none()
    );
}
