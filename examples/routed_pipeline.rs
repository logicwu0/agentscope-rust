//! Offline explicit routing: invoke one agent and preserve its original messages.
use agentscope::{
    ChatResponse, ContentBlock, InMemoryMemory, MockChatModel, Msg, ReActAgent, RoutedFailure,
    RoutedPipeline, ThinkingBlock, ToolExecutor, ToolRegistry,
};
use std::{error::Error, sync::Arc};

fn agent(name: &str, model: Arc<MockChatModel>) -> Result<Arc<ReActAgent>, Box<dyn Error>> {
    Ok(Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))?
            .with_memory(InMemoryMemory::new()),
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let code_response = ChatResponse::completed([
        ContentBlock::Thinking(ThinkingBlock::new("private-code-analysis")),
        ContentBlock::from("Code: use a Result type to report recoverable failures."),
    ])
    .with_metadata(
        [(
            "internal_note".into(),
            serde_json::json!("private-code-metadata"),
        )]
        .into_iter()
        .collect(),
    );
    let code_model =
        Arc::new(MockChatModel::new("offline-code").with_response(code_response.clone()));
    let writing_model = Arc::new(MockChatModel::new("offline-writing").with_response(
        ChatResponse::completed([ContentBlock::from(
            "Writing: explain what happened and what the reader can do next.",
        )]),
    ));
    let coder = agent("coder", code_model.clone())?;
    let writer = agent("writer", writing_model.clone())?;
    let pipeline = RoutedPipeline::new(vec![
        ("code".into(), coder.clone()),
        ("writing".into(), writer.clone()),
    ])?;

    let code_input = Msg::user("How should a Rust library report recoverable failures?")
        .with_metadata(
            [("request_id".into(), serde_json::json!("demo-code"))]
                .into_iter()
                .collect(),
        );
    let code = pipeline.run("code", code_input.clone()).await?;
    assert_eq!(code.route, "code");
    assert_eq!(code.agent_name, "coder");
    assert_eq!(code.message.content, code_response.content);
    assert_eq!(code.message.metadata, code_response.metadata);
    assert_eq!(code_model.recorded_requests().len(), 1);
    assert_eq!(
        code_model.recorded_requests()[0].messages.last(),
        Some(&code_input)
    );
    assert!(writing_model.recorded_requests().is_empty());
    assert!(writer.snapshot().await?.messages().is_empty());
    let code_history = coder.snapshot().await?;
    assert_eq!(code_history.messages(), &[code_input, code.message.clone()]);
    // The routed output retains private thinking/metadata; select text for display.
    println!(
        "code -> coder: {}",
        code.message.text_content("").unwrap_or_default()
    );

    let writing_input = Msg::user("Write a brief explanation of clear error messages.")
        .with_metadata(
            [("request_id".into(), serde_json::json!("demo-writing"))]
                .into_iter()
                .collect(),
        );
    let writing = pipeline.run("writing", writing_input.clone()).await?;
    assert_eq!(writing.route, "writing");
    assert_eq!(writing.agent_name, "writer");
    assert_eq!(writing_model.recorded_requests().len(), 1);
    assert_eq!(
        writing_model.recorded_requests()[0].messages.last(),
        Some(&writing_input)
    );
    assert_eq!(code_model.recorded_requests().len(), 1);
    let writing_history = writer.snapshot().await?;
    assert_eq!(
        writing_history.messages(),
        &[writing_input, writing.message.clone()]
    );
    println!(
        "writing -> writer: {}",
        writing.message.text_content("").unwrap_or_default()
    );

    for route in ["unknown", "Code", " code"] {
        let error = pipeline
            .run(route, Msg::user("This must not reach either agent."))
            .await
            .expect_err("route lookup is exact; unknown routes never fall back");
        assert_eq!(error.route, route);
        assert!(error.agent_name.is_none());
        assert!(matches!(error.cause, RoutedFailure::UnknownRoute));
    }
    assert_eq!(code_model.recorded_requests().len(), 1);
    assert_eq!(writing_model.recorded_requests().len(), 1);
    assert_eq!(coder.snapshot().await?, code_history);
    assert_eq!(writer.snapshot().await?, writing_history);
    println!("Each route called only its selected agent; unknown routes called neither.");
    Ok(())
}
