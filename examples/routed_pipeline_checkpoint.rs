//! Offline routed checkpoints: read committed replies and reconcile a local gate.
use agentscope::{
    ChatEventStream, ChatModel, ChatRequest, ChatResponse, ContentBlock, InMemoryMemory,
    InMemoryRoutedStore, MockChatModel, ModelCapabilities, ModelError, ModelFuture, Msg,
    ReActAgent, RoutedCheckpointStatus, RoutedFailure, RoutedPipeline, RoutedStore, StateKey,
    ToolExecutor, ToolRegistry,
};
use std::{
    error::Error,
    sync::{Arc, Mutex},
};
use tokio::sync::Notify;

fn agent(name: &str, model: Arc<dyn ChatModel>) -> Result<Arc<ReActAgent>, Box<dyn Error>> {
    Ok(Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))?
            .with_memory(InMemoryMemory::new()),
    ))
}

async fn committed_reply_demo() -> Result<(), Box<dyn Error>> {
    let selected_model = Arc::new(MockChatModel::new("offline-code").with_response(
        ChatResponse::completed([ContentBlock::from("Use Result for recoverable failures.")]),
    ));
    let unused_model = Arc::new(MockChatModel::new("must-not-run"));
    let pipeline = RoutedPipeline::new(vec![
        ("code".into(), agent("coder", selected_model.clone())?),
        ("writing".into(), agent("writer", unused_model.clone())?),
    ])?;
    let store = InMemoryRoutedStore::new();
    let key = StateKey::new("offline-user", "committed-route")?;
    let input = Msg::user("How should this Rust library report errors?").with_metadata(
        [("request_id".into(), serde_json::json!("committed-demo"))]
            .into_iter()
            .collect(),
    );
    let output = pipeline
        .run_checkpointed(&store, key.clone(), "code", input.clone())
        .await?;
    assert_eq!(output.route, "code");
    assert_eq!(output.agent_name, "coder");
    assert_eq!(selected_model.recorded_requests().len(), 1);
    assert_eq!(
        selected_model.recorded_requests()[0].messages.last(),
        Some(&input)
    );
    assert!(unused_model.recorded_requests().is_empty());
    let record = store
        .load(&key)
        .await?
        .ok_or("committed checkpoint is missing")?;
    let committed = record
        .checkpoint
        .finished_result()
        .ok_or("checkpoint is not terminal")??;
    assert_eq!(committed, output);
    let replay = pipeline
        .resume_checkpointed(&store, key)
        .await
        .expect_err("terminal records cannot be replayed");
    assert!(matches!(replay.cause, RoutedFailure::UnsafeResume(_)));
    assert_eq!(selected_model.recorded_requests().len(), 1);
    println!(
        "Committed {} reply: {}",
        output.route,
        output.message.text_content("").unwrap_or_default()
    );
    println!("Reading the terminal checkpoint invoked no agent; resume was rejected.");
    Ok(())
}

/// A deterministic local model. No network or real tools are involved; only
/// delivery of its configured fixture is gated. This lets the demo independently
/// inspect the known fixture after cancellation, rather than guess real effects.
struct GatedModel {
    response: ChatResponse,
    requests: Mutex<Vec<ChatRequest>>,
    entered: Notify,
    release: Notify,
}

impl GatedModel {
    fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl ChatModel for GatedModel {
    fn name(&self) -> &'static str {
        "offline-gate"
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }
    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async move {
            self.requests.lock().unwrap().push(request);
            self.entered.notify_one();
            self.release.notified().await;
            Ok(self.response.clone())
        })
    }
    fn stream(&self, _: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async { Err(ModelError::new("this local demo does not stream")) })
    }
}

async fn reconcile_local_gate_demo() -> Result<(), Box<dyn Error>> {
    let gate = Arc::new(GatedModel {
        response: ChatResponse::completed([ContentBlock::from("Verified local fixture result.")]),
        requests: Mutex::new(Vec::new()),
        entered: Notify::new(),
        release: Notify::new(),
    });
    let worker = agent("worker", gate.clone())?;
    let pipeline = RoutedPipeline::new(vec![("local".into(), worker.clone())])?;
    let store = InMemoryRoutedStore::new();
    let key = StateKey::new("offline-user", "interrupted-route")?;
    let input = Msg::user("Run the deterministic local fixture.");
    let mut running = pipeline.run_checkpointed(&store, key.clone(), "local", input.clone());
    tokio::select! {
        result = &mut running => return Err(format!("gated run unexpectedly returned {result:?}").into()),
        () = gate.entered.notified() => {}
    }
    pipeline.interrupt_handle().interrupt();
    let interrupted = running
        .await
        .expect_err("the delivery gate was interrupted");
    assert!(matches!(interrupted.cause, RoutedFailure::Interrupted));
    let record = store
        .load(&key)
        .await?
        .ok_or("in-flight checkpoint is missing")?;
    assert_eq!(record.checkpoint.status, RoutedCheckpointStatus::InFlight);
    let unsafe_resume = pipeline
        .resume_checkpointed(&store, key.clone())
        .await
        .expect_err("in-flight work cannot be replayed");
    assert!(matches!(
        unsafe_resume.cause,
        RoutedFailure::UnsafeResume(_)
    ));
    assert_eq!(gate.requests().len(), 1);
    assert_eq!(gate.requests()[0].messages.last(), Some(&input));

    // This evidence is verified against a configured LOCAL fixture, not inferred
    // from an error. Real applications must verify effects and resolve any tool
    // approval/in-doubt checkpoints through the original agent APIs first.
    let verified = gate.response.clone().into_assistant_msg("worker");
    assert_eq!(verified.content, gate.response.content);
    assert_eq!(worker.snapshot().await?.messages(), &[input.clone()]);
    // Reconciliation does not repair agent memory; update that separately here.
    worker.observe(verified.clone()).await?;
    let resolved = pipeline
        .reconcile_checkpointed(&store, key.clone(), record.revision, verified.clone())
        .await?;
    assert_eq!(resolved.revision, record.revision + 1);
    let committed = resolved
        .checkpoint
        .finished_result()
        .ok_or("reconciliation did not commit a terminal result")??;
    assert_eq!(committed.route, "local");
    assert_eq!(committed.message, verified);
    assert_eq!(gate.requests().len(), 1);
    assert_eq!(
        worker.snapshot().await?.messages(),
        &[input, committed.message]
    );
    let replay = pipeline
        .resume_checkpointed(&store, key)
        .await
        .expect_err("reconciled terminal work cannot be replayed");
    assert!(matches!(replay.cause, RoutedFailure::UnsafeResume(_)));
    assert_eq!(gate.requests().len(), 1);
    println!(
        "Interrupted local delivery was reconciled from its inspected fixture; no replay occurred."
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    committed_reply_demo().await?;
    reconcile_local_gate_demo().await?;
    println!("This store is process-local. SQLiteRoutedStore supports recovery after restart.");
    Ok(())
}
