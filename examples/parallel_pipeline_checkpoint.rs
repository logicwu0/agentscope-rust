//! Offline committed-boundary recovery without replaying a completed branch.
use agentscope::{
    AgentInterruptHandle, ChatResponse, ContentBlock, InMemoryMemory, InMemoryParallelStore, Msg,
    ParallelBranchCheckpoint, ParallelBranchOutcome, ParallelCheckpoint, ParallelFailure,
    ParallelPipeline, ParallelRecord, ParallelStore, PipelineStoreFuture, ReActAgent, StateKey,
    ToolExecutor, ToolRegistry,
};
use std::{
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

struct StopAfterFirstCompletion {
    inner: InMemoryParallelStore,
    handle: AgentInterruptHandle,
    stopped: AtomicBool,
}

impl ParallelStore for StopAfterFirstCompletion {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<ParallelRecord>> {
        self.inner.load(key)
    }

    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        checkpoint: ParallelCheckpoint,
    ) -> PipelineStoreFuture<'_, ParallelRecord> {
        Box::pin(async move {
            let record = self.inner.save(key, expected_revision, checkpoint).await?;
            if matches!(
                record.checkpoint.branches.first(),
                Some(ParallelBranchCheckpoint::Completed(_))
            ) && !self.stopped.swap(true, Ordering::SeqCst)
            {
                // Stop only after the first reply is safely committed.
                self.handle.interrupt();
            }
            Ok(record)
        })
    }
}

fn model(name: &str, text: &str) -> Arc<agentscope::MockChatModel> {
    Arc::new(
        agentscope::MockChatModel::new(name)
            .with_response(ChatResponse::completed([ContentBlock::from(text)])),
    )
}

fn agent(
    name: &str,
    model: Arc<agentscope::MockChatModel>,
) -> Result<Arc<ReActAgent>, Box<dyn Error>> {
    Ok(Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))?
            .with_memory(InMemoryMemory::new()),
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let first_model = model("offline-security", "Security: keep credentials private.");
    let second_model = model(
        "offline-reliability",
        "Reliability: verify uncertain effects.",
    );
    let first = agent("security", first_model.clone())?;
    let second = agent("reliability", second_model.clone())?;
    // A bound of one makes the next branch remain Ready at our deliberate stop.
    let pipeline = ParallelPipeline::new(vec![first.clone(), second.clone()], 1)?;
    let store = StopAfterFirstCompletion {
        inner: InMemoryParallelStore::new(),
        handle: pipeline.interrupt_handle(),
        stopped: AtomicBool::new(false),
    };
    let key = StateKey::new("offline-user", "parallel-checkpoint")?;
    let question = Msg::user("What should we consider when building an agent service?");
    let error = pipeline
        .run_checkpointed(&store, key.clone(), question.clone())
        .await
        .expect_err("the demo deliberately interrupts after the first committed reply");
    assert!(matches!(error.cause, ParallelFailure::Interrupted));
    let record = store.load(&key).await?.ok_or("checkpoint is missing")?;
    assert!(matches!(
        record.checkpoint.branches[0],
        ParallelBranchCheckpoint::Completed(_)
    ));
    assert!(matches!(
        record.checkpoint.branches[1],
        ParallelBranchCheckpoint::Ready
    ));
    assert_eq!(first_model.recorded_requests().len(), 1);
    assert!(second_model.recorded_requests().is_empty());
    println!("First reply committed; second branch is Ready. Resuming the checkpoint...");

    let output = pipeline.resume_checkpointed(&store, key.clone()).await?;
    for branch in &output.branches {
        let ParallelBranchOutcome::Completed(message) = &branch.outcome else {
            unreachable!("both scripted branches complete successfully");
        };
        println!(
            "{}. {}: {}",
            branch.branch,
            branch.agent_name,
            message.text_content("").unwrap_or_default()
        );
    }
    assert_eq!(first_model.recorded_requests().len(), 1);
    assert_eq!(second_model.recorded_requests().len(), 1);
    for model in [&first_model, &second_model] {
        assert_eq!(
            model.recorded_requests()[0].messages.last(),
            Some(&question)
        );
    }
    assert_eq!(first.snapshot().await?.messages().len(), 2);
    assert_eq!(second.snapshot().await?.messages().len(), 2);
    let final_record = store
        .load(&key)
        .await?
        .ok_or("final checkpoint is missing")?;
    let committed = final_record
        .checkpoint
        .finished_result()
        .ok_or("checkpoint is not terminal")??;
    assert_eq!(committed.branches, output.branches);
    println!("Both replies committed; the completed first branch was invoked only once.");
    println!("This store is process-local. Use SQLiteParallelStore for recovery after restart.");
    Ok(())
}
