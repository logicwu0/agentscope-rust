//! Offline model-assisted routing: strict selections, unchanged worker input and safe composition.
use agentscope::{
    ChatResponse, ContentBlock, DataBlock, InMemoryMemory, InMemoryRoutedStore, MockChatModel,
    ModelRouter, Msg, ReActAgent, Role, RouteSelectionError, RoutedFailure, RoutedPipeline,
    RoutedStore, StateKey, StructuredOutputBlock, ThinkingBlock, ToolExecutor, ToolRegistry, Usage,
};
use serde_json::{Value, json};
use std::{error::Error, sync::Arc};

fn selection(route: &Value) -> Result<ChatResponse, Box<dyn Error>> {
    // A provider's echoed schema is not an authority; the router validates locally.
    let block =
        StructuredOutputBlock::complete(json!({"type": "object"}), json!({"route": route}))?;
    Ok(ChatResponse::completed([ContentBlock::from(block)]).with_usage(Usage::new(10, 2)))
}

fn selector() -> Result<Arc<MockChatModel>, Box<dyn Error>> {
    Ok(Arc::new(
        MockChatModel::new("offline-selector")
            .with_response(selection(&json!("code"))?)
            .with_response(selection(&json!("writing"))?)
            .with_response(selection(&json!("Code"))?)
            .with_response(selection(&Value::Null)?),
    ))
}

fn agent(name: &str, model: Arc<MockChatModel>) -> Result<Arc<ReActAgent>, Box<dyn Error>> {
    Ok(Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))?
            .with_memory(InMemoryMemory::new()),
    ))
}

fn input() -> Result<Msg, Box<dyn Error>> {
    Ok(Msg::new(
        "private-user-label",
        Role::User,
        [
            ContentBlock::from("How should Rust report recoverable failures?"),
            ContentBlock::Thinking(ThinkingBlock::new("private-input-analysis")),
            ContentBlock::Data(DataBlock::base64(
                "cHJpdmF0ZQ==",
                "application/octet-stream",
            )?),
            ContentBlock::from("Suggest a short implementation."),
        ],
    )
    .with_metadata(
        [("private_note".into(), json!("local-input-metadata"))]
            .into_iter()
            .collect(),
    ))
}

fn check_selector_requests(selector: &MockChatModel, original: &Msg) {
    let requests = selector.recorded_requests();
    assert_eq!(requests.len(), 4);
    assert_eq!(
        requests[0]
            .messages
            .last()
            .and_then(|message| message.text_content("")),
        Some(json!({"input": original.text_content("\n")}).to_string())
    );
    for request in requests {
        assert!(request.tools.is_empty());
        assert!(request.structured_output_schema.is_some());
        for message in request.messages {
            assert_ne!(message.id, original.id);
            assert_ne!(message.name, original.name);
            assert!(message.metadata.is_empty());
            assert!(
                message
                    .content
                    .iter()
                    .all(|block| matches!(block, ContentBlock::Text(_)))
            );
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let selector = selector()?;
    let code_response = ChatResponse::completed([
        ContentBlock::Thinking(ThinkingBlock::new("private-code-analysis")),
        ContentBlock::from("Use Result for recoverable failures."),
    ])
    .with_metadata(
        [("private_note".into(), json!("worker-metadata"))]
            .into_iter()
            .collect(),
    );
    let code_model =
        Arc::new(MockChatModel::new("offline-code").with_response(code_response.clone()));
    let writing_model = Arc::new(
        MockChatModel::new("offline-writing").with_response(
            ChatResponse::completed([ContentBlock::from(
                "Explain the problem and the next action.",
            )])
            .with_usage(Usage::new(5, 4)),
        ),
    );
    let coder = agent("coder", code_model.clone())?;
    let writer = agent("writer", writing_model.clone())?;
    let pipeline = RoutedPipeline::new(vec![
        ("code".into(), coder.clone()),
        ("writing".into(), writer.clone()),
    ])?;
    let router = ModelRouter::from_shared(
        pipeline.clone(),
        selector.clone(),
        vec![
            (
                "code".into(),
                "Rust code, debugging and implementation".into(),
            ),
            ("writing".into(), "Drafting and editing prose".into()),
        ],
    )?;

    let code_input = input()?;
    let selected = router.select(code_input.clone()).await?;
    assert_eq!(selected.route, "code");
    assert_eq!(selected.agent_name, "coder");
    assert_eq!(selected.usage, Some(Usage::new(10, 2)));
    assert!(code_model.recorded_requests().is_empty());
    assert!(writing_model.recorded_requests().is_empty());

    // Selection and checkpointed execution are separate operations, not atomic.
    let store = InMemoryRoutedStore::new();
    let key = StateKey::new("offline-user", "selected-route")?;
    let code = pipeline
        .run_checkpointed(&store, key.clone(), selected.route, code_input.clone())
        .await?;
    let record = store.load(&key).await?.ok_or("missing routed checkpoint")?;
    assert_eq!(record.checkpoint.route, "code");
    assert_eq!(record.checkpoint.input, code_input);
    assert_eq!(record.checkpoint.finished_result(), Some(Ok(code.clone())));
    assert_eq!(code.message.content, code_response.content);
    assert_eq!(code.message.metadata, code_response.metadata);
    assert!(writer.snapshot().await?.messages().is_empty());
    println!(
        "code -> coder: {}",
        code.message.text_content("").unwrap_or_default()
    );

    // run holds the shared pipeline lock across selection and one worker reply.
    let writing_input = Msg::user("Write a sentence about clear error messages.");
    let writing = router.run(writing_input.clone()).await?;
    assert_eq!(writing.route, "writing");
    assert_eq!(writing.agent_name, "writer");
    assert_eq!(writing.message.usage, Some(Usage::new(5, 4))); // No selector tokens added.
    println!(
        "writing -> writer: {}",
        writing.message.text_content("").unwrap_or_default()
    );

    for failure in [
        RouteSelectionError::InvalidResponse,
        RouteSelectionError::NoMatch,
    ] {
        let error = router
            .run(Msg::user("Do not dispatch this request."))
            .await
            .unwrap_err();
        assert_eq!(error.cause, RoutedFailure::Selection(failure));
        assert!(error.route.is_empty());
        assert!(error.agent_name.is_none());
    }
    for (model, original) in [(&code_model, &code_input), (&writing_model, &writing_input)] {
        let requests = model.recorded_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.last(), Some(original));
    }
    assert_eq!(
        coder.snapshot().await?.messages(),
        &[code_input.clone(), code.message]
    );
    assert_eq!(
        writer.snapshot().await?.messages(),
        &[writing_input, writing.message]
    );
    check_selector_requests(&selector, &code_input);
    println!("Invalid and null selections called no workers; classifier saw text only.");
    Ok(())
}
