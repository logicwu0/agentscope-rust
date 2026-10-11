//! Offline automatic selection followed by a single target's streamed reply.
use agentscope::{
    AgentEvent, ChatEvent, ChatResponse, ContentBlock, FinishReason, InMemoryMemory, MockChatModel,
    ModelRouter, Msg, ReActAgent, RouteSelectionError, RoutedEvent, RoutedFailure, RoutedPipeline,
    StructuredOutputBlock, ToolExecutor, ToolRegistry,
};
use futures_util::StreamExt;
use serde_json::json;
use std::{error::Error, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let choice = |route| -> Result<ChatResponse, Box<dyn Error>> {
        Ok(ChatResponse::completed([ContentBlock::from(
            StructuredOutputBlock::complete(json!({}), json!({"route": route}))?,
        )]))
    };
    let selector = Arc::new(
        MockChatModel::new("offline-selector")
            .with_response(choice(json!("code"))?)
            .with_response(choice(json!(null))?),
    );
    let worker_model = Arc::new(MockChatModel::new("offline-worker").with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: "Use Result ".into(),
        }),
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: "for recoverable errors.".into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]));
    let worker = Arc::new(
        ReActAgent::from_shared(
            "coder",
            worker_model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )?
        .with_memory(InMemoryMemory::new()),
    );
    let pipeline = RoutedPipeline::new(vec![("code".into(), worker.clone())])?;
    let router = ModelRouter::from_shared(
        pipeline,
        selector.clone(),
        vec![("code".into(), "Rust code and debugging".into())],
    )?;
    let input = Msg::user("Explain recoverable errors in Rust.");
    let mut events = router.stream(input.clone()).await?;
    assert!(selector.recorded_requests().is_empty());
    assert!(worker_model.recorded_requests().is_empty());
    assert!(matches!(events.next().await,
        Some(RoutedEvent::RouteStarted { route, agent_name }) if route == "code" && agent_name == "coder"));
    assert_eq!(selector.recorded_requests().len(), 1);
    assert!(worker_model.recorded_requests().is_empty());
    let mut original = None;
    let mut completed = None;
    while let Some(event) = events.next().await {
        match event {
            RoutedEvent::Agent {
                event: AgentEvent::TextDelta { delta, .. },
                ..
            } => print!("{delta}"),
            RoutedEvent::Agent {
                event: AgentEvent::Finished { message, .. },
                ..
            } => original = Some(message),
            RoutedEvent::Finished { output } => {
                assert_eq!(original.as_ref(), Some(&output.message));
                completed = Some(output.message);
            }
            RoutedEvent::Error { error } => return Err(error.into()),
            _ => {}
        }
    }
    let completed = completed.ok_or("missing routed terminal")?;
    assert_eq!(worker_model.recorded_requests().len(), 1);
    assert_eq!(
        worker_model.recorded_requests()[0].messages.last(),
        Some(&input)
    );
    assert_eq!(worker.snapshot().await?.messages(), &[input, completed]);
    let unmatched = router
        .stream(Msg::user("No suitable route."))
        .await?
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(unmatched.as_slice(), [RoutedEvent::Error { error }]
        if error.cause == RoutedFailure::Selection(RouteSelectionError::NoMatch))
    );
    assert_eq!(worker_model.recorded_requests().len(), 1);
    println!("\nNo-match selection made no additional worker call.");
    Ok(())
}
