use super::*;
use crate::*;
use futures_util::FutureExt;
use serde_json::json;
use std::{
    error::Error as _,
    sync::{Arc, atomic::Ordering},
};
#[path = "tests_support.rs"]
mod support;
use support::*;

#[test]
fn configuration_requires_nonempty_exact_known_unique_allowlist_and_descriptions() {
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "unused")));
    let other = ScriptAgent::new("other", Ok(reply("other", "unused")));
    let pipeline = pipeline(&selected, &other);
    let model = Arc::new(MockChatModel::new("selector"));
    assert!(matches!(
        ModelRouter::from_shared(pipeline.clone(), model.clone(), vec![]),
        Err(ModelRouterConfigError::Empty)
    ));
    for route in ["Selected", "selected ", " selected", "unknown", ""] {
        assert!(
            matches!(ModelRouter::from_shared(pipeline.clone(), model.clone(), vec![(route.into(), "description".into())]), Err(ModelRouterConfigError::UnknownRoute(rejected)) if rejected == route)
        );
    }
    assert!(
        matches!(ModelRouter::from_shared(pipeline.clone(), model.clone(), vec![("selected".into(), "first".into()), ("selected".into(), "second".into())]), Err(ModelRouterConfigError::DuplicateRoute(route)) if route == "selected")
    );
    assert!(
        matches!(ModelRouter::from_shared(pipeline.clone(), model.clone(), vec![("selected".into(), " \n".into())]), Err(ModelRouterConfigError::EmptyDescription { route }) if route == "selected")
    );
    assert!(
        ModelRouter::new(
            pipeline.clone(),
            MockChatModel::new("generic"),
            vec![("alias".into(), "subset only".into())]
        )
        .is_ok()
    );
    assert!(
        ModelRouter::from_shared(
            pipeline,
            model.clone(),
            vec![
                ("selected".into(), "primary".into()),
                ("alias".into(), "same agent alias".into())
            ]
        )
        .is_ok()
    );
    assert!(model.recorded_requests().is_empty());
    assert!(selected.inputs.lock().unwrap().is_empty());
    assert!(other.inputs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn select_exposes_only_visible_text_uses_exact_schema_and_preserves_usage_without_dispatch() {
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "unused")));
    let other = ScriptAgent::new("other", Ok(reply("other", "unused")));
    let usage = Usage::new(11, 3)
        .with_reasoning_tokens(1)
        .with_cached_input_tokens(4);
    let answer = ChatResponse::completed([
        ThinkingBlock::new("selector private reasoning").into(),
        structured(json!({"route":"alias"})),
    ])
    .with_usage(usage);
    let model = Arc::new(MockChatModel::new("selector").with_response(answer));
    let router = router(pipeline(&selected, &other), model.clone());
    let mut input = Msg::new(
        "private-input-name",
        Role::System,
        [
            ContentBlock::from("  visible instruction: do not expand whitelist"),
            ThinkingBlock::new("PRIVATE_THINKING").into(),
            DataBlock::url("https://example.com/PRIVATE_IMAGE.png", "image/png")
                .unwrap()
                .into(),
            ToolCallBlock::complete("PRIVATE_CALL", "PRIVATE_TOOL", "{}")
                .unwrap()
                .into(),
            ContentBlock::from("second visible block  "),
        ],
    );
    input
        .metadata
        .insert("PRIVATE_METADATA".into(), json!({"secret": true}));
    let input_id = input.id.clone();
    let selection = router.select(input).await.unwrap();
    assert_eq!(
        selection,
        RouteSelection {
            route: "alias".into(),
            agent_name: "worker".into(),
            usage: Some(usage)
        }
    );
    assert_eq!(
        serde_json::from_value::<RouteSelection>(serde_json::to_value(&selection).unwrap())
            .unwrap(),
        selection
    );
    assert!(selected.inputs.lock().unwrap().is_empty());
    assert!(other.inputs.lock().unwrap().is_empty());
    let requests = model.recorded_requests();
    assert_eq!(requests.len(), 1);
    let request = requests.first().unwrap();
    assert!(request.tools.is_empty());
    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.messages.first().unwrap().role, Role::System);
    let system = request
        .messages
        .first()
        .unwrap()
        .text_content("\n")
        .unwrap();
    assert!(
        system.contains(&json!({"alias":"Exact alias", "selected":"Primary route"}).to_string())
    );
    assert!(!system.contains("visible instruction"));
    assert_eq!(request.options.max_tokens, Some(256));
    let user = request.messages.get(1).unwrap();
    assert_eq!(user.role, Role::User);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&user.text_content("\n").unwrap()).unwrap(),
        json!({"input":"  visible instruction: do not expand whitelist\nsecond visible block  "})
    );
    for message in &request.messages {
        assert!(message.metadata.is_empty());
        assert!(
            message
                .content
                .iter()
                .all(|block| matches!(block, ContentBlock::Text(_)))
        );
    }
    let wire = serde_json::to_string(request).unwrap();
    for hidden in [
        "private-input-name",
        "PRIVATE_THINKING",
        "PRIVATE_IMAGE",
        "PRIVATE_CALL",
        "PRIVATE_TOOL",
        "PRIVATE_METADATA",
        &input_id,
    ] {
        assert!(
            !wire.contains(hidden),
            "selector request must not contain {hidden}"
        );
    }
    assert_exact_schema(request);
}

#[tokio::test]
async fn run_preserves_full_input_and_original_reply_and_invokes_only_selected_allowlisted_agent() {
    let mut original = Msg::new(
        "original-author",
        Role::User,
        [
            ThinkingBlock::new("private reply").into(),
            DataBlock::base64("YWJj", "application/octet-stream")
                .unwrap()
                .into(),
        ],
    );
    original
        .metadata
        .insert("reply-private".into(), json!({"value": 7}));
    original.usage = Some(Usage::new(2, 1));
    let selected = ScriptAgent::new("worker", Ok(original.clone()));
    let other = ScriptAgent::new("other", Ok(reply("other", "unused")));
    let model = Arc::new(
        MockChatModel::new("selector")
            .with_response(response("selected").with_usage(Usage::new(100, 5))),
    );
    let router = router(pipeline(&selected, &other), model.clone());
    let mut input = Msg::new(
        "input-author",
        Role::System,
        [
            ContentBlock::from("visible task"),
            ThinkingBlock::new("private input").into(),
            DataBlock::url("https://example.com/input.png", "image/png")
                .unwrap()
                .into(),
        ],
    );
    input.metadata.insert("input-private".into(), json!(true));
    let output = router.run(input.clone()).await.unwrap();
    assert_eq!(output.route, "selected");
    assert_eq!(output.agent_name, "worker");
    assert_eq!(output.message, original);
    assert_eq!(*selected.inputs.lock().unwrap(), vec![input]);
    assert!(other.inputs.lock().unwrap().is_empty());
    assert_eq!(model.recorded_requests().len(), 1);
    assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    assert_eq!(
        serde_json::from_value::<RoutedOutput>(serde_json::to_value(&output).unwrap()).unwrap(),
        output
    );
}

#[tokio::test]
async fn raw_choice_json_is_strict_duplicate_safe_and_never_normalizes_or_falls_back() {
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "must not be called")));
    let other = ScriptAgent::new("other", Ok(reply("other", "configured but excluded")));
    let pipeline = pipeline(&selected, &other);
    for raw in [
        "{}",
        r#"{"route":"selected","extra":"must not be ignored"}"#,
        r#"{"route":"selected","route":"selected"}"#,
        r#"{"route":"selected","route":"alias"}"#,
        r#"{"route":"Selected"}"#,
        r#"{"route":"selected "}"#,
        r#"{"route":" selected"}"#,
        r#"{"route":"unknown"}"#,
        r#"{"route":"other"}"#,
        r#"{"route":""}"#,
        r#"{"route":7}"#,
        r#"{"route":true}"#,
        r#"{"route":["selected"]}"#,
        r#"{"route":{"key":"selected"}}"#,
        r#"["selected"]"#,
    ] {
        let model = Arc::new(MockChatModel::new("selector").with_response(raw_response(raw)));
        let router = router(pipeline.clone(), model.clone());
        let Err(error) = router.run(Msg::user("visible task")).await else {
            panic!("invalid selection was accepted: {raw}")
        };
        assert_selection_error(&error, RouteSelectionError::InvalidResponse);
        assert_eq!(
            model.recorded_requests().len(),
            1,
            "no implicit fallback or retry"
        );
    }
    let model =
        Arc::new(MockChatModel::new("no match").with_response(raw_response(r#"{"route":null}"#)));
    let router = router(pipeline, model.clone());
    assert_selection_error(
        &router.run(Msg::user("no match")).await.unwrap_err(),
        RouteSelectionError::NoMatch,
    );
    assert_eq!(model.recorded_requests().len(), 1);
    assert!(selected.inputs.lock().unwrap().is_empty());
    assert!(other.inputs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn only_one_complete_structured_selection_in_a_completed_final_response_is_accepted() {
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "must not be called")));
    let other = ScriptAgent::new("other", Ok(reply("other", "unused")));
    let pipeline = pipeline(&selected, &other);
    let valid = structured(json!({"route":"selected"}));
    let mut unfinished = StructuredOutputBlock::streaming(json!({})).unwrap();
    unfinished
        .append_output_delta(r#"{"route":"selected"}"#)
        .unwrap();
    let mut missing_reason = response("selected");
    missing_reason.finish_reason = None;
    let mut cases = vec![
        ChatResponse::completed(Vec::<ContentBlock>::new()),
        ChatResponse::completed([ThinkingBlock::new("only reasoning").into()]),
        ChatResponse::completed([ContentBlock::from(r#"{"route":"selected"}"#)]),
        ChatResponse::completed([valid.clone(), ContentBlock::from("extra text")]),
        ChatResponse::completed([valid.clone(), valid.clone()]),
        ChatResponse::completed([unfinished.into()]),
        ChatResponse::partial([valid.clone()]),
        ChatResponse::completed([
            valid.clone(),
            ToolCallBlock::complete("call", "tool", "{}")
                .unwrap()
                .into(),
        ]),
        ChatResponse::completed([
            valid.clone(),
            ToolResultBlock::success("call", "tool", "result")
                .unwrap()
                .into(),
        ]),
        ChatResponse::completed([
            valid.clone(),
            DataBlock::url("https://example.com/image.png", "image/png")
                .unwrap()
                .into(),
        ]),
        missing_reason,
    ];
    cases.extend(
        [
            FinishReason::Length,
            FinishReason::ToolCalls,
            FinishReason::ContentFilter,
            FinishReason::Interrupted,
        ]
        .into_iter()
        .map(|reason| ChatResponse::finished([valid.clone()], reason)),
    );
    for invalid in cases {
        let model = Arc::new(MockChatModel::new("selector").with_response(invalid));
        let router = router(pipeline.clone(), model.clone());
        assert_selection_error(
            &router.run(Msg::user("visible task")).await.unwrap_err(),
            RouteSelectionError::InvalidResponse,
        );
        assert_eq!(model.recorded_requests().len(), 1);
    }
    assert!(selected.inputs.lock().unwrap().is_empty());
    assert!(other.inputs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn unsupported_model_empty_visible_input_and_original_model_errors_are_unattributed() {
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "must not be called")));
    let other = ScriptAgent::new("other", Ok(reply("other", "unused")));
    let pipeline = pipeline(&selected, &other);
    let unsupported = Arc::new(
        MockChatModel::new("unsupported")
            .with_capabilities(ModelCapabilities::new())
            .with_response(response("selected")),
    );
    let router = router(pipeline.clone(), unsupported.clone());
    assert_selection_error(
        &router.run(Msg::user("visible")).await.unwrap_err(),
        RouteSelectionError::UnsupportedModel,
    );
    assert!(unsupported.recorded_requests().is_empty());
    for input in [
        Msg::user(" \n\t "),
        Msg::new(
            "private",
            Role::System,
            [ThinkingBlock::new("not visible").into()],
        ),
        Msg::new(
            "private",
            Role::User,
            [
                DataBlock::url("https://example.com/private.png", "image/png")
                    .unwrap()
                    .into(),
            ],
        ),
        Msg::new(
            "private",
            Role::User,
            [structured(json!({"input":"not a Text block"}))],
        ),
    ] {
        let model = Arc::new(MockChatModel::new("selector").with_response(response("selected")));
        let router = support::router(pipeline.clone(), model.clone());
        assert_selection_error(
            &router.run(input).await.unwrap_err(),
            RouteSelectionError::EmptyInput,
        );
        assert!(model.recorded_requests().is_empty());
    }
    let original = ModelError::new("provider unavailable")
        .with_code("private-code")
        .with_retryable(true);
    let model = Arc::new(MockChatModel::new("selector").with_error(original.clone()));
    let router = support::router(pipeline, model.clone());
    let error = router.run(Msg::user("visible")).await.unwrap_err();
    assert_selection_error(&error, RouteSelectionError::Model(original.clone()));
    let source = error
        .source()
        .unwrap()
        .downcast_ref::<RouteSelectionError>()
        .unwrap();
    assert_eq!(
        source.source().unwrap().downcast_ref::<ModelError>(),
        Some(&original)
    );
    assert_eq!(
        model.recorded_requests().len(),
        1,
        "retryable is not automatic retry authorization"
    );
    assert!(selected.inputs.lock().unwrap().is_empty());
    assert!(other.inputs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn lazy_selection_and_shared_busy_never_construct_model_or_child_and_drop_stops_polling() {
    for cancel in [false, true] {
        let gate = Arc::new(Gate::default());
        let model = GateModel::gated(&gate, response("selected"));
        let selected = ScriptAgent::new("worker", Ok(reply("worker", "unused")));
        let other = ScriptAgent::new("other", Ok(reply("other", "done")));
        let pipeline = pipeline(&selected, &other);
        let router = router(pipeline.clone(), model.clone());
        let clone = router.clone();
        drop(router.select(Msg::user("unpolled select")));
        drop(router.run(Msg::user("unpolled run")));
        assert!(model.requests.lock().unwrap().is_empty());
        let mut selection = router.select(Msg::user("active selection"));
        assert!(selection.as_mut().now_or_never().is_none());
        assert_eq!(model.requests.lock().unwrap().len(), 1);
        assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
        assert_eq!(
            clone.select(Msg::user("busy")).await.unwrap_err().cause,
            RoutedFailure::Busy
        );
        assert_eq!(
            clone.run(Msg::user("busy")).await.unwrap_err().cause,
            RoutedFailure::Busy
        );
        assert_eq!(
            pipeline
                .clone()
                .run("other", Msg::user("busy"))
                .await
                .unwrap_err()
                .cause,
            RoutedFailure::Busy
        );
        let Err(error) = pipeline.clone().stream("other", Msg::user("busy")).await else {
            panic!("expected Busy")
        };
        assert_eq!(error.cause, RoutedFailure::Busy);
        assert_eq!(model.requests.lock().unwrap().len(), 1);
        assert!(selected.inputs.lock().unwrap().is_empty());
        assert!(other.inputs.lock().unwrap().is_empty());
        gate.release();
        if cancel {
            router.clone().interrupt_handle().interrupt();
            let error = selection.await.unwrap_err();
            assert_eq!(error.cause, RoutedFailure::Interrupted);
            assert_eq!(error.route, "");
            assert_eq!(error.agent_name, None);
        } else {
            drop(selection);
        }
        assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
        pipeline
            .clone()
            .run("other", Msg::user("released by drop"))
            .await
            .unwrap();
        assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
        assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
        let fresh = clone
            .select(Msg::user("explicit selection after drop"))
            .await
            .unwrap();
        assert_eq!(fresh.route, "selected");
        assert_eq!(model.requests.lock().unwrap().len(), 2);
        assert!(selected.inputs.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn selected_execution_holds_same_lock_and_resumed_cancel_or_drop_never_repolls_child() {
    for cancel in [true, false] {
        let gate = Arc::new(Gate::default());
        let original = reply("original-author", "original reply");
        let selected = ScriptAgent::gated("worker", original, &gate);
        let other = ScriptAgent::new("other", Ok(reply("other", "done")));
        let pipeline = pipeline(&selected, &other);
        let model = Arc::new(MockChatModel::new("selector").with_response(response("alias")));
        let router = router(pipeline.clone(), model.clone());
        let mut run = router.run(Msg::user("task"));
        assert!(run.as_mut().now_or_never().is_none());
        assert_eq!(model.recorded_requests().len(), 1);
        assert_eq!(selected.inputs.lock().unwrap().len(), 1);
        assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
        assert_eq!(
            router
                .clone()
                .select(Msg::user("busy"))
                .await
                .unwrap_err()
                .cause,
            RoutedFailure::Busy
        );
        assert_eq!(
            pipeline
                .run("other", Msg::user("busy"))
                .await
                .unwrap_err()
                .cause,
            RoutedFailure::Busy
        );
        assert_eq!(model.recorded_requests().len(), 1);
        gate.release();
        if cancel {
            router.clone().interrupt_handle().interrupt();
            let error = run.await.unwrap_err();
            assert_eq!(error.cause, RoutedFailure::Interrupted);
            assert_eq!(error.route, "alias");
            assert_eq!(error.agent_name.as_deref(), Some("worker"));
        } else {
            drop(run);
        }
        assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
        assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
        assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
        pipeline
            .clone()
            .run("other", Msg::user("explicit independent next work"))
            .await
            .unwrap();
        assert_eq!(selected.inputs.lock().unwrap().len(), 1);
        assert_eq!(model.recorded_requests().len(), 1);
    }
}

#[tokio::test]
async fn model_reply_that_signals_interrupt_in_same_poll_must_never_dispatch_target() {
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "must not be called")));
    let other = ScriptAgent::new("other", Ok(reply("other", "unused")));
    let pipeline = pipeline(&selected, &other);
    let model = GateModel::interrupting(pipeline.interrupt_handle(), response("selected"));
    let router = router(pipeline.clone(), model.clone());
    let error = router.run(Msg::user("task")).await.unwrap_err();
    assert_eq!(error.cause, RoutedFailure::Interrupted);
    assert_eq!(error.route, "");
    assert_eq!(error.agent_name, None);
    assert_eq!(model.requests.lock().unwrap().len(), 1);
    assert!(selected.inputs.lock().unwrap().is_empty());
    assert!(other.inputs.lock().unwrap().is_empty());
    let fresh_model = Arc::new(MockChatModel::new("fresh").with_response(response("selected")));
    let fresh = support::router(pipeline.clone(), fresh_model.clone());
    pipeline.interrupt_handle().interrupt();
    fresh
        .run(Msg::user("old signal ignored by fresh baseline"))
        .await
        .unwrap();
    assert_eq!(fresh_model.recorded_requests().len(), 1);
    assert_eq!(selected.inputs.lock().unwrap().len(), 1);
    assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn original_confirmation_and_uncertain_execution_errors_remain_attributed_without_retry() {
    let (worker, agent_model, tool) = uncertain_agent();
    let selector = Arc::new(
        MockChatModel::new("selector")
            .with_response(response("selected"))
            .with_response(response("selected")),
    );
    let pipeline = RoutedPipeline::new(vec![("selected".into(), worker.clone())]).unwrap();
    let router = ModelRouter::from_shared(
        pipeline,
        selector.clone(),
        vec![("selected".into(), "Only selected route".into())],
    )
    .unwrap();
    let confirmation = router.run(Msg::user("write")).await.unwrap_err();
    assert_eq!(confirmation.route, "selected");
    assert_eq!(confirmation.agent_name.as_deref(), Some("worker"));
    let RoutedFailure::Agent(original) = &confirmation.cause else {
        panic!("expected original agent failure")
    };
    let AgentError::ToolConfirmationRequired { checkpoint } = original.as_ref() else {
        panic!("expected confirmation")
    };
    assert!(tool.recorded_invocations().is_empty());
    assert_eq!(
        confirmation.source().unwrap().downcast_ref::<AgentError>(),
        Some(original.as_ref())
    );
    assert_eq!(
        serde_json::from_value::<RoutedError>(serde_json::to_value(&confirmation).unwrap())
            .unwrap(),
        confirmation
    );
    let uncertain = worker
        .resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("call")],
        )
        .await
        .unwrap_err();
    assert!(matches!(uncertain, AgentError::ToolExecutionInDoubt { .. }));
    let observed = router
        .run(Msg::user("do not automatically retry"))
        .await
        .unwrap_err();
    assert_eq!(observed.route, "selected");
    assert_eq!(observed.agent_name.as_deref(), Some("worker"));
    assert_eq!(observed.cause, RoutedFailure::Agent(Box::new(uncertain)));
    assert_eq!(selector.recorded_requests().len(), 2);
    assert_eq!(agent_model.recorded_requests().len(), 1);
    assert_eq!(tool.recorded_invocations().len(), 1);
}
