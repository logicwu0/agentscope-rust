//! Offline durable routed streaming: preparation, dispatch fences and acknowledgements.
use agentscope::{
    AgentEvent, ChatEvent, ContentBlock, FinishReason, InMemoryMemory, InMemoryRoutedStore,
    MockChatModel, Msg, ReActAgent, RoutedCheckpointStatus, RoutedEvent, RoutedFailure,
    RoutedOutput, RoutedPipeline, RoutedStore, StateKey, ToolExecutor, ToolRegistry,
};
use futures_util::StreamExt;
use std::{error::Error, sync::Arc};

fn model() -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new("offline-code").with_stream([
        Ok(ChatEvent::ThinkingDelta {
            block_id: "thinking".into(),
            delta: "private-analysis".into(),
        }),
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: "Code: bound execution; ".into(),
        }),
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: "verify uncertain effects.".into(),
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

async fn demonstrate_fence(
    pipeline: &RoutedPipeline,
    store: &InMemoryRoutedStore,
) -> Result<(), Box<dyn Error>> {
    let key = StateKey::new("offline-user", "drop-after-selection")?;
    let mut events = pipeline
        .stream_checkpointed(
            store,
            key.clone(),
            "code",
            Msg::user("Do not execute this input."),
        )
        .await?;
    assert!(matches!(
        events.next().await,
        Some(RoutedEvent::RouteStarted { .. })
    ));
    drop(events); // Selection was fenced, but the agent was not invoked yet.
    assert_eq!(
        store
            .load(&key)
            .await?
            .ok_or("missing fence")?
            .checkpoint
            .status,
        RoutedCheckpointStatus::InFlight,
    );
    let error = match pipeline.resume_checkpointed_stream(store, key).await {
        Ok(_) => return Err("fenced progress must not resume".into()),
        Err(error) => error,
    };
    assert!(matches!(error.cause, RoutedFailure::UnsafeResume(_)));
    println!("Dropped after selection: InFlight refuses automatic replay.");
    Ok(())
}

async fn consume(
    pipeline: &RoutedPipeline,
    store: &InMemoryRoutedStore,
    key: &StateKey,
    selected: &MockChatModel,
) -> Result<RoutedOutput, Box<dyn Error>> {
    let mut events = pipeline
        .resume_checkpointed_stream(store, key.clone())
        .await?;
    let mut text = String::new();
    let mut original_reply = None;
    let mut finished = None;
    while let Some(event) = events.next().await {
        match event {
            RoutedEvent::RouteStarted { route, agent_name } => {
                assert!(selected.recorded_requests().is_empty());
                assert_eq!(
                    store
                        .load(key)
                        .await?
                        .ok_or("missing selection")?
                        .checkpoint
                        .status,
                    RoutedCheckpointStatus::InFlight,
                );
                println!("{route} selected: {agent_name}; dispatch fence committed.");
            }
            RoutedEvent::Agent {
                event: AgentEvent::TextDelta { delta, .. },
                ..
            } => {
                print!("{delta}"); // Print visible text only, never private thinking or metadata.
                text.push_str(&delta);
            }
            RoutedEvent::Agent {
                event: AgentEvent::Finished { message, .. },
                ..
            } => {
                assert_eq!(
                    store
                        .load(key)
                        .await?
                        .ok_or("missing progress")?
                        .checkpoint
                        .status,
                    RoutedCheckpointStatus::InFlight,
                );
                original_reply = Some(message);
                println!("\nAgent finished; route write still pending.");
            }
            RoutedEvent::Finished { output } => {
                let record = store.load(key).await?.ok_or("missing committed result")?;
                assert_eq!(
                    record.checkpoint.status,
                    RoutedCheckpointStatus::Completed(output.message.clone())
                );
                assert_eq!(
                    record.checkpoint.finished_result(),
                    Some(Ok(output.clone()))
                );
                assert_eq!(Some(&output.message), original_reply.as_ref());
                assert_eq!(output.message.text_content(""), Some(text.clone()));
                finished = Some(output);
            }
            RoutedEvent::Error { error } => return Err(error.into()),
            RoutedEvent::Agent { .. } => {}
        }
    }
    finished.ok_or_else(|| "stream ended without a durable routed acknowledgement".into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let code_model = model();
    let writing_model = Arc::new(MockChatModel::new("must-not-run"));
    let coder = agent("coder", code_model.clone())?;
    let writer = agent("writer", writing_model.clone())?;
    let pipeline = RoutedPipeline::new(vec![
        ("code".into(), coder.clone()),
        ("writing".into(), writer.clone()),
    ])?;
    let store = InMemoryRoutedStore::new();
    demonstrate_fence(&pipeline, &store).await?;
    assert!(code_model.recorded_requests().is_empty());
    assert!(writing_model.recorded_requests().is_empty());

    let key = StateKey::new("offline-user", "resume-ready")?;
    let input = Msg::user("Explain safe agent execution.").with_metadata(
        [("request_id".into(), serde_json::json!("checkpoint-demo"))]
            .into_iter()
            .collect(),
    );
    let prepared = pipeline
        .stream_checkpointed(&store, key.clone(), "code", input.clone())
        .await?;
    assert_eq!(
        store
            .load(&key)
            .await?
            .ok_or("missing preparation")?
            .checkpoint
            .status,
        RoutedCheckpointStatus::Ready
    );
    drop(prepared); // No poll: no dispatch fence or agent call, so Ready can resume.
    assert!(code_model.recorded_requests().is_empty());

    let output = consume(&pipeline, &store, &key, &code_model).await?;
    let requests = code_model.recorded_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].messages.last(), Some(&input));
    assert_eq!(
        coder.snapshot().await?.messages(),
        &[input, output.message.clone()]
    );
    assert!(
        output
            .message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Thinking(_)))
    );
    assert!(writing_model.recorded_requests().is_empty());
    assert!(writer.snapshot().await?.messages().is_empty());
    let error = match pipeline.resume_checkpointed_stream(&store, key).await {
        Ok(_) => return Err("terminal progress must not resume".into()),
        Err(error) => error,
    };
    assert!(matches!(error.cause, RoutedFailure::UnsafeResume(_)));
    assert_eq!(code_model.recorded_requests().len(), 1);
    println!("Routed Finished confirmed the full reply; terminal resume made no new calls.");
    println!("This store is process-local; use SQLiteRoutedStore for restart persistence.");
    Ok(())
}
