//! Offline routed streams: exact selection, lazy execution and original replies.
use agentscope::{
    AgentEvent, ChatEvent, ContentBlock, FinishReason, InMemoryMemory, MockChatModel, Msg,
    ReActAgent, RoutedEvent, RoutedFailure, RoutedOutput, RoutedPipeline, ToolExecutor,
    ToolRegistry,
};
use futures_util::StreamExt;
use std::{error::Error, sync::Arc};

fn model(name: &str, first: &str, second: &str) -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new(name).with_stream([
        Ok(ChatEvent::ThinkingDelta {
            block_id: "thinking".into(),
            delta: "private-analysis".into(),
        }),
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: first.into(),
        }),
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: second.into(),
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

fn input(text: &str, request_id: &str) -> Msg {
    Msg::user(text).with_metadata(
        [("request_id".into(), serde_json::json!(request_id))]
            .into_iter()
            .collect(),
    )
}

async fn reply(
    pipeline: &RoutedPipeline,
    route: &str,
    input: Msg,
    selected: &MockChatModel,
    other: &MockChatModel,
) -> Result<RoutedOutput, Box<dyn Error>> {
    let selected_calls = selected.recorded_requests().len();
    let other_calls = other.recorded_requests().len();
    let mut events = pipeline.stream(route, input).await?;
    assert_eq!(selected.recorded_requests().len(), selected_calls);
    let Some(RoutedEvent::RouteStarted {
        route: started_route,
        agent_name,
    }) = events.next().await
    else {
        return Err("the first routed event must announce the selection".into());
    };
    assert_eq!(started_route, route);
    assert_eq!(selected.recorded_requests().len(), selected_calls);
    println!("route {started_route} started: {agent_name}");

    let mut text = String::new();
    let mut original_reply = None;
    let mut finished = None;
    while let Some(event) = events.next().await {
        match event {
            RoutedEvent::Agent {
                route,
                event: AgentEvent::TextDelta { delta, .. },
                ..
            } => {
                println!("{route}: {delta}");
                text.push_str(&delta);
            }
            RoutedEvent::Agent {
                event: AgentEvent::Finished { message, .. },
                ..
            } => original_reply = Some(message),
            RoutedEvent::Finished { output } => {
                assert_eq!(output.message.text_content(""), Some(text.clone()));
                assert_eq!(Some(&output.message), original_reply.as_ref());
                finished = Some(output);
            }
            RoutedEvent::Error { error } => return Err(error.into()),
            RoutedEvent::Agent { .. } | RoutedEvent::RouteStarted { .. } => {}
        }
    }
    assert_eq!(other.recorded_requests().len(), other_calls);
    finished.ok_or_else(|| "routed stream ended without its final reply".into())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let code_model = model(
        "offline-code",
        "Code: return Result; ",
        "handle recoverable failures.",
    );
    let writing_model = model(
        "offline-writing",
        "Writing: state the problem; ",
        "explain the next action.",
    );
    let coder = agent("coder", code_model.clone())?;
    let writer = agent("writer", writing_model.clone())?;
    let pipeline = RoutedPipeline::new(vec![
        ("code".into(), coder.clone()),
        ("writing".into(), writer.clone()),
    ])?;
    let code_input = input("How should a Rust library report failures?", "demo-code");
    let code = reply(
        &pipeline,
        "code",
        code_input.clone(),
        &code_model,
        &writing_model,
    )
    .await?;
    assert!(writing_model.recorded_requests().is_empty());
    assert!(writer.snapshot().await?.messages().is_empty());
    assert!(
        code.message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Thinking(_)))
    );
    let code_history = coder.snapshot().await?;
    assert_eq!(code_history.messages(), &[code_input.clone(), code.message]);

    let writing_input = input("Explain clear error messages.", "demo-writing");
    let writing = reply(
        &pipeline,
        "writing",
        writing_input.clone(),
        &writing_model,
        &code_model,
    )
    .await?;
    let writing_history = writer.snapshot().await?;
    assert_eq!(
        writing_history.messages(),
        &[writing_input.clone(), writing.message]
    );
    for (model, input) in [(&code_model, &code_input), (&writing_model, &writing_input)] {
        let requests = model.recorded_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.last(), Some(input));
    }

    let error = match pipeline
        .stream("Code", Msg::user("No matching route."))
        .await
    {
        Ok(_) => return Err("unknown route unexpectedly created a stream".into()),
        Err(error) => error,
    };
    assert!(matches!(error.cause, RoutedFailure::UnknownRoute));
    assert!(error.agent_name.is_none());
    assert_eq!(code_model.recorded_requests().len(), 1);
    assert_eq!(writing_model.recorded_requests().len(), 1);
    assert_eq!(coder.snapshot().await?, code_history);
    assert_eq!(writer.snapshot().await?, writing_history);
    println!(
        "Each routed stream invoked one agent; original metadata and separate histories survived."
    );
    Ok(())
}
