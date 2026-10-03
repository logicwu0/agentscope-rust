//! Offline checkpointed streams: stop after a committed branch, then resume.
use agentscope::{
    AgentEvent, AgentInterruptHandle, ChatEvent, FinishReason, InMemoryMemory,
    InMemoryParallelStore, MockChatModel, Msg, ParallelBranchCheckpoint, ParallelError,
    ParallelEvent, ParallelEventStream, ParallelFailure, ParallelOutput, ParallelPipeline,
    ParallelStore, ReActAgent, StateKey, ToolExecutor, ToolRegistry,
};
use futures_util::StreamExt;
use std::{error::Error, sync::Arc};

fn model(name: &str, text: &str) -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new(name).with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: text.into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]))
}

fn agent(name: &str, model: Arc<MockChatModel>) -> Result<Arc<ReActAgent>, Box<dyn Error>> {
    Ok(Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))?
            .with_memory(InMemoryMemory::new()),
    ))
}

async fn consume(
    mut events: ParallelEventStream<'_>,
    stop_after_first: Option<AgentInterruptHandle>,
) -> Result<Result<ParallelOutput, ParallelError>, Box<dyn Error>> {
    let mut terminal = None;
    while let Some(event) = events.next().await {
        match event {
            ParallelEvent::BranchStarted { branch, agent_name } => {
                println!("branch {branch} started: {agent_name}");
            }
            ParallelEvent::Agent {
                branch,
                agent_name,
                event: AgentEvent::TextDelta { delta, .. },
            } => println!("branch {branch} {agent_name}: {delta}"),
            ParallelEvent::BranchFinished { result } => {
                println!("branch {} committed: {}", result.branch, result.agent_name);
                if let (1, Some(handle)) = (result.branch, &stop_after_first) {
                    // BranchFinished confirms the reply's checkpoint write succeeded.
                    handle.interrupt();
                }
            }
            ParallelEvent::Finished { output } => terminal = Some(Ok(output)),
            ParallelEvent::Error { error } => terminal = Some(Err(error)),
            ParallelEvent::Agent { .. } => {}
        }
    }
    terminal.ok_or_else(|| "parallel stream ended without a terminal event".into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let first_model = model("offline-security", "Security: isolate credentials.");
    let second_model = model(
        "offline-reliability",
        "Reliability: verify uncertain effects.",
    );
    let first = agent("security", first_model.clone())?;
    let second = agent("reliability", second_model.clone())?;
    // Keep the second branch Ready when stopping after the first committed reply.
    let pipeline = ParallelPipeline::new(vec![first.clone(), second.clone()], 1)?;
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("offline-user", "parallel-checkpoint-stream")?;
    let question = Msg::user("What should we consider when building an agent service?");
    let events = pipeline
        .stream_checkpointed(&store, key.clone(), question.clone())
        .await?;
    let initial = store
        .load(&key)
        .await?
        .ok_or("initial checkpoint missing")?;
    assert!(
        initial
            .checkpoint
            .branches
            .iter()
            .all(|branch| matches!(branch, ParallelBranchCheckpoint::Ready))
    );
    assert!(first_model.recorded_requests().is_empty());
    assert!(second_model.recorded_requests().is_empty());

    let error = consume(events, Some(pipeline.interrupt_handle()))
        .await?
        .expect_err("the demo stops after the first committed branch");
    assert!(matches!(error.cause, ParallelFailure::Interrupted));
    let paused = store.load(&key).await?.ok_or("paused checkpoint missing")?;
    assert!(matches!(
        paused.checkpoint.branches[0],
        ParallelBranchCheckpoint::Completed(_)
    ));
    assert!(matches!(
        paused.checkpoint.branches[1],
        ParallelBranchCheckpoint::Ready
    ));
    assert_eq!(first_model.recorded_requests().len(), 1);
    assert!(second_model.recorded_requests().is_empty());
    println!("Resuming only the Ready branch; the committed first reply will not be replayed.");

    let events = pipeline
        .resume_checkpointed_stream(&store, key.clone())
        .await?;
    let output = consume(events, None).await??;
    assert_eq!(output.branches.len(), 2);
    for model in [&first_model, &second_model] {
        let requests = model.recorded_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.last(), Some(&question));
    }
    assert_eq!(first.snapshot().await?.messages().len(), 2);
    assert_eq!(second.snapshot().await?.messages().len(), 2);
    let final_record = store.load(&key).await?.ok_or("final checkpoint missing")?;
    let committed = final_record
        .checkpoint
        .finished_result()
        .ok_or("checkpoint is not terminal")??;
    assert_eq!(committed.branches, output.branches);
    println!("All replies committed. Each model was called once with the original input.");
    println!("This store is process-local; SQLiteParallelStore provides restart persistence.");
    Ok(())
}
