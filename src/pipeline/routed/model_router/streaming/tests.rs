use super::super::super::streaming_tests_support::{
    Gate, ScriptAgent, assert_roundtrips, confirming_agent, reply, terminal_count,
};
use super::super::tests_support::{Gate as ModelGate, GateModel, raw_response, response};
use crate::*;
use futures_util::{FutureExt, StreamExt};
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};

fn router(agent: Arc<dyn Agent>, model: Arc<dyn ChatModel>) -> (ModelRouter, RoutedPipeline) {
    let pipeline = RoutedPipeline::new(vec![("chosen".into(), agent)]).unwrap();
    let router = ModelRouter::from_shared(
        pipeline.clone(),
        model,
        vec![("chosen".into(), "Allowed target".into())],
    )
    .unwrap();
    (router, pipeline)
}

#[tokio::test]
async fn preparation_is_lazy_and_shares_lock_with_all_original_operations() {
    let selected = ScriptAgent::finished("worker");
    let model = Arc::new(MockChatModel::new("selector").with_response(response("chosen")));
    let (router, pipeline) = router(selected.clone(), model.clone());
    drop(router.stream(Msg::user("unpolled")));
    let mut stream = router.stream(Msg::user("task")).await.unwrap();
    assert!(model.recorded_requests().is_empty());
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    let clone = router.clone();
    assert!(
        matches!(clone.stream(Msg::user("busy")).await, Err(error) if error.cause == RoutedFailure::Busy)
    );
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
            .run("chosen", Msg::user("busy"))
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Busy
    );
    assert!(
        matches!(pipeline.stream("chosen", Msg::user("busy")).await, Err(error) if error.cause == RoutedFailure::Busy)
    );
    let store = InMemoryRoutedStore::new();
    assert_eq!(
        pipeline
            .run_checkpointed(
                &store,
                StateKey::new("user", "busy").unwrap(),
                "chosen",
                Msg::user("busy")
            )
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Busy
    );
    assert!(model.recorded_requests().is_empty());
    assert!(
        matches!(stream.next().await, Some(RoutedEvent::RouteStarted { route, agent_name }) if route == "chosen" && agent_name == "worker")
    );
    assert_eq!(model.recorded_requests().len(), 1);
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    drop(stream);
    pipeline.run("chosen", Msg::user("released")).await.unwrap();
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn original_events_and_message_survive_terminal_gap_and_release_lock_before_eof() {
    let mut original = reply("original-author", "complete reply");
    original
        .content
        .push(ThinkingBlock::new("private reply").into());
    original.metadata.insert("private".into(), json!(true));
    original.usage = Some(Usage::new(7, 3));
    let events = vec![
        AgentEvent::TextDelta {
            step: 9,
            block_id: "text".into(),
            delta: "visible".into(),
        },
        AgentEvent::Finished {
            steps: 11,
            message: original.clone(),
        },
    ];
    let mut scripted = events.iter().cloned().map(Ok).collect::<Vec<_>>();
    scripted.push(Err(AgentError::Interrupted));
    let selected = ScriptAgent::new("worker", scripted);
    let other = ScriptAgent::finished("other");
    let pipeline = RoutedPipeline::new(vec![
        ("chosen".into(), selected.clone()),
        ("excluded".into(), other.clone()),
    ])
    .unwrap();
    let model = Arc::new(
        MockChatModel::new("selector")
            .with_response(response("chosen").with_usage(Usage::new(99, 99))),
    );
    let router = ModelRouter::from_shared(
        pipeline.clone(),
        model.clone(),
        vec![("chosen".into(), "Allowed target".into())],
    )
    .unwrap();
    let mut input = Msg::new(
        "original-input",
        Role::System,
        [
            ContentBlock::from("visible task"),
            ThinkingBlock::new("private input").into(),
        ],
    );
    input.metadata.insert("private".into(), json!(true));
    let mut stream = router.stream(input.clone()).await.unwrap();
    let mut observed = vec![stream.next().await.unwrap()];
    for event in &events {
        let wrapped = stream.next().await.unwrap();
        assert!(
            matches!(&wrapped, RoutedEvent::Agent { route, agent_name, event: raw } if route == "chosen" && agent_name == "worker" && raw == event)
        );
        observed.push(wrapped);
    }
    assert_eq!(*selected.inputs.lock().unwrap(), vec![input]);
    assert_eq!(selected.events_dropped.load(Ordering::SeqCst), 1);
    router.interrupt_handle().interrupt();
    let terminal = stream.next().await.unwrap();
    assert!(matches!(&terminal, RoutedEvent::Finished { output } if output.message == original));
    observed.push(terminal);
    pipeline
        .run("excluded", Msg::user("terminal releases lock"))
        .await
        .unwrap();
    assert_eq!(other.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(selected.item_polls.load(Ordering::SeqCst), events.len());
    assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    assert_eq!(terminal_count(&observed), 1);
    assert_roundtrips(&observed);
    assert!(stream.next().await.is_none());
    assert_eq!(model.recorded_requests().len(), 1);
    let request = &model.recorded_requests()[0];
    assert!(request.tools.is_empty());
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&request.messages[1].text_content("").unwrap())
            .unwrap(),
        json!({"input":"visible task"})
    );
}

#[tokio::test]
async fn selection_failures_are_single_unattributed_events_without_start_or_target_calls() {
    let model_error = ModelError::new("provider failure").with_retryable(true);
    let cases = vec![
        (
            MockChatModel::new("invalid").with_response(raw_response(r#"["chosen"]"#)),
            Msg::user("task"),
            RouteSelectionError::InvalidResponse,
        ),
        (
            MockChatModel::new("unknown").with_response(response("unknown")),
            Msg::user("task"),
            RouteSelectionError::InvalidResponse,
        ),
        (
            MockChatModel::new("null").with_response(raw_response(r#"{"route":null}"#)),
            Msg::user("task"),
            RouteSelectionError::NoMatch,
        ),
        (
            MockChatModel::new("error").with_error(model_error.clone()),
            Msg::user("task"),
            RouteSelectionError::Model(model_error),
        ),
        (
            MockChatModel::new("unsupported").with_capabilities(ModelCapabilities::new()),
            Msg::user("task"),
            RouteSelectionError::UnsupportedModel,
        ),
        (
            MockChatModel::new("empty"),
            Msg::user(" \n"),
            RouteSelectionError::EmptyInput,
        ),
    ];
    for (model, input, expected) in cases {
        let selected = ScriptAgent::finished("worker");
        let (router, pipeline) = router(selected.clone(), Arc::new(model));
        let mut stream = router.stream(input).await.unwrap();
        let Some(RoutedEvent::Error { error }) = stream.next().await else {
            panic!("expected selection error")
        };
        assert_eq!(error.cause, RoutedFailure::Selection(expected));
        assert_eq!(error.route, "");
        assert_eq!(error.agent_name, None);
        pipeline
            .run("chosen", Msg::user("lock released before EOF"))
            .await
            .unwrap();
        assert!(stream.next().await.is_none());
        assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn classifier_cancel_and_drop_prevent_dispatch_on_resumed_polls() {
    for cancel in [false, true] {
        let gate = Arc::new(ModelGate::default());
        let model = GateModel::gated(&gate, response("chosen"));
        let selected = ScriptAgent::finished("worker");
        let (router, pipeline) = router(selected.clone(), model.clone());
        let mut stream = router.stream(Msg::user("task")).await.unwrap();
        assert!(stream.next().now_or_never().is_none());
        assert_eq!(model.requests.lock().unwrap().len(), 1);
        gate.release();
        if cancel {
            pipeline.interrupt_handle().interrupt();
            assert!(
                matches!(stream.next().await, Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Interrupted && error.route.is_empty() && error.agent_name.is_none())
            );
        }
        drop(stream);
        assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
        assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
        assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
        let fresh = router
            .stream(Msg::user("fresh baseline"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(fresh.last(), Some(RoutedEvent::Finished { .. })));
    }
    let selected = ScriptAgent::finished("worker");
    let pipeline = RoutedPipeline::new(vec![("chosen".into(), selected.clone())]).unwrap();
    let model = GateModel::interrupting(pipeline.interrupt_handle(), response("chosen"));
    let router =
        ModelRouter::from_shared(pipeline, model, vec![("chosen".into(), "Allowed".into())])
            .unwrap();
    let events = router
        .stream(Msg::user("same poll interrupt"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(events.as_slice(), [RoutedEvent::Error { error }] if error.cause == RoutedFailure::Interrupted)
    );
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancellation_before_classification_and_after_start_never_constructs_child() {
    for started in [false, true] {
        let selected = ScriptAgent::finished("worker");
        let model = Arc::new(MockChatModel::new("selector").with_response(response("chosen")));
        let (router, _) = router(selected.clone(), model.clone());
        let mut stream = router.stream(Msg::user("task")).await.unwrap();
        if started {
            assert!(matches!(
                stream.next().await,
                Some(RoutedEvent::RouteStarted { .. })
            ));
        }
        router.interrupt_handle().interrupt();
        assert!(
            matches!(stream.next().await, Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Interrupted)
        );
        assert_eq!(model.recorded_requests().len(), usize::from(started));
        assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
        assert!(stream.next().await.is_none());
    }
}

#[tokio::test]
async fn pending_child_startup_or_event_cancel_and_drop_never_repoll_ready_work() {
    for startup in [false, true] {
        for cancel in [false, true] {
            let gate = Arc::new(Gate::default());
            let selected = if startup {
                ScriptAgent::startup_gated("worker", &gate)
            } else {
                ScriptAgent::event_gated("worker", &gate)
            };
            let model = Arc::new(MockChatModel::new("selector").with_response(response("chosen")));
            let (router, pipeline) = router(selected.clone(), model);
            let mut stream = router.stream(Msg::user("task")).await.unwrap();
            assert!(matches!(
                stream.next().await,
                Some(RoutedEvent::RouteStarted { .. })
            ));
            assert!(stream.next().now_or_never().is_none());
            assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 1);
            gate.release();
            if cancel {
                router.interrupt_handle().interrupt();
                assert!(
                    matches!(stream.next().await, Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Interrupted && error.route == "chosen")
                );
            }
            drop(stream);
            assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
            assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
            assert_eq!(gate.dropped.load(Ordering::SeqCst), usize::from(startup));
            assert_eq!(
                selected.events_dropped.load(Ordering::SeqCst),
                usize::from(!startup)
            );
            assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
            pipeline
                .run("chosen", Msg::user("lock released"))
                .await
                .unwrap();
        }
    }
}

#[tokio::test]
async fn child_failures_and_confirmation_preserve_original_outcomes() {
    let original = AgentError::Model(ModelError::new("original failure"));
    let cases = [
        ScriptAgent::startup_error("worker", original.clone()),
        ScriptAgent::new("worker", vec![Err(original.clone())]),
        ScriptAgent::new(
            "worker",
            vec![Ok(AgentEvent::Error {
                step: Some(7),
                error: original.clone(),
            })],
        ),
    ];
    for selected in cases {
        let (router, _) = router(
            selected,
            Arc::new(MockChatModel::new("selector").with_response(response("chosen"))),
        );
        let events = router
            .stream(Msg::user("task"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(events.last(), Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Agent(Box::new(original.clone())))
        );
        assert_eq!(terminal_count(&events), 1);
    }
    let store = Arc::new(InMemoryStateStore::new());
    let key = StateKey::new("user", "confirmation").unwrap();
    let (agent, tool, model) = confirming_agent(store.clone(), key.clone());
    let (router, _) = router(
        agent,
        Arc::new(MockChatModel::new("selector").with_response(response("chosen"))),
    );
    let events = router
        .stream(Msg::user("write"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let pending = events
        .iter()
        .find_map(|event| match event {
            RoutedEvent::Agent {
                event: AgentEvent::ToolConfirmationRequired { checkpoint },
                ..
            } => Some(checkpoint),
            _ => None,
        })
        .unwrap();
    assert!(
        matches!(events.last(), Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Agent(Box::new(AgentError::ToolConfirmationRequired { checkpoint: pending.clone() })))
    );
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_calls(),
        Some(pending)
    );
    assert!(tool.recorded_invocations().is_empty());
    assert_eq!(model.recorded_requests().len(), 1);
}
