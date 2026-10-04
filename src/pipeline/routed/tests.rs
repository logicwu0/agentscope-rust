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
fn configuration_rejects_only_invalid_route_entries_and_allows_aliases_and_same_names() {
    assert!(matches!(
        RoutedPipeline::new(vec![]),
        Err(RoutedConfigError::Empty)
    ));
    let a = ScriptAgent::returning("same", Ok(reply("same", "unused")));
    assert!(matches!(
        RoutedPipeline::new(vec![("valid".into(), a.clone()), (" \t".into(), a.clone())]),
        Err(RoutedConfigError::EmptyRoute { entry: 2 })
    ));
    assert!(
        matches!(RoutedPipeline::new(vec![("same".into(), a.clone()), ("same".into(), a.clone())]), Err(RoutedConfigError::DuplicateRoute(route)) if route == "same")
    );
    let blank = ScriptAgent::returning(" \n", Ok(reply("blank", "unused")));
    assert!(
        matches!(RoutedPipeline::new(vec![("selected".into(), blank)]), Err(RoutedConfigError::EmptyAgentName { route }) if route == "selected")
    );
    let another = ScriptAgent::returning("same", Ok(reply("same", "unused")));
    assert!(
        RoutedPipeline::new(vec![
            ("route".into(), a.clone()),
            ("Route".into(), a.clone()),
            (" route ".into(), another)
        ])
        .is_ok()
    );
    assert!(a.inputs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn routes_are_exact_and_unknown_keys_fail_closed_without_calling_any_agent() {
    let a = ScriptAgent::returning("a", Ok(reply("a", "done")));
    let b = ScriptAgent::returning("b", Ok(reply("b", "done")));
    let pipeline = RoutedPipeline::new(vec![
        ("write".into(), a.clone()),
        ("review".into(), b.clone()),
    ])
    .unwrap();
    for route in ["Write", "write ", " write", "", "missing"] {
        let error = pipeline
            .run(route, Msg::user("must not dispatch"))
            .await
            .unwrap_err();
        assert_eq!(error.route, route);
        assert_eq!(error.agent_name, None);
        assert_eq!(error.cause, RoutedFailure::UnknownRoute);
    }
    assert!(a.inputs.lock().unwrap().is_empty());
    assert!(b.inputs.lock().unwrap().is_empty());
    let output = pipeline.run("write", Msg::user("selected")).await.unwrap();
    assert_eq!(output.route, "write");
    assert_eq!(output.agent_name, "a");
    assert_eq!(a.inputs.lock().unwrap().len(), 1);
    assert!(b.inputs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn selected_input_and_reply_are_preserved_without_handoff_or_name_reinterpretation() {
    let mut input = Msg::new(
        "original-input",
        Role::System,
        [
            ThinkingBlock::new("input reasoning").into(),
            ContentBlock::from("input text"),
            DataBlock::url("https://example.com/input.png", "image/png")
                .unwrap()
                .with_name("input.png")
                .into(),
        ],
    );
    input.metadata.insert("private".into(), json!({"value": 7}));
    let mut original = Msg::new(
        "actual-author",
        Role::User,
        [
            ThinkingBlock::new("reply reasoning").into(),
            ContentBlock::from("original reply"),
            DataBlock::base64("YWJj", "application/octet-stream")
                .unwrap()
                .with_name("result.bin")
                .into(),
        ],
    );
    original.metadata.insert("original".into(), json!(true));
    let selected = ScriptAgent::returning("configured-agent", Ok(original.clone()));
    let untouched = ScriptAgent::returning("untouched", Ok(reply("untouched", "unused")));
    let pipeline = RoutedPipeline::new(vec![
        ("exact route ".into(), selected.clone()),
        ("other".into(), untouched.clone()),
    ])
    .unwrap();
    selected.renamed.store(true, Ordering::SeqCst);
    let output = pipeline.run("exact route ", input.clone()).await.unwrap();
    assert_eq!(output.route, "exact route ");
    assert_eq!(output.agent_name, "configured-agent");
    assert_eq!(output.message, original);
    assert_eq!(*selected.inputs.lock().unwrap(), vec![input]);
    assert!(untouched.inputs.lock().unwrap().is_empty());
    let encoded = serde_json::to_string(&output).unwrap();
    assert_eq!(
        serde_json::from_str::<RoutedOutput>(&encoded).unwrap(),
        output
    );
}

#[tokio::test]
async fn aliases_share_the_selected_agent_while_distinct_same_named_agents_keep_independent_memory()
{
    let left_model = Arc::new(
        MockChatModel::new("left")
            .with_response(ChatResponse::completed([ContentBlock::from("left reply")]))
            .with_response(ChatResponse::completed([ContentBlock::from("left again")])),
    );
    let right_model = model("right reply");
    let left = agent("same", left_model.clone());
    let right = agent("same", right_model.clone());
    left.observe(Msg::user("left private history"))
        .await
        .unwrap();
    right
        .observe(Msg::user("right private history"))
        .await
        .unwrap();
    let pipeline = RoutedPipeline::new(vec![
        ("left".into(), left.clone()),
        ("alias".into(), left.clone()),
        ("right".into(), right.clone()),
    ])
    .unwrap();
    let input = Msg::user("original task");
    pipeline.run("left", input.clone()).await.unwrap();
    assert!(right_model.recorded_requests().is_empty());
    pipeline.run("right", input.clone()).await.unwrap();
    let alias = pipeline
        .run("alias", Msg::user("continue left"))
        .await
        .unwrap();
    assert_eq!(alias.route, "alias");
    assert_eq!(alias.agent_name, "same");
    assert_eq!(left_model.recorded_requests().len(), 2);
    assert_eq!(right_model.recorded_requests().len(), 1);
    assert_eq!(
        left_model.recorded_requests()[0].messages.last(),
        Some(&input)
    );
    assert_eq!(
        right_model.recorded_requests()[0].messages.last(),
        Some(&input)
    );
    assert_eq!(left_model.recorded_requests()[1].messages.len(), 4);
    assert_eq!(left.snapshot().await.unwrap().messages().len(), 5);
    assert_eq!(right.snapshot().await.unwrap().messages().len(), 3);
    assert_eq!(
        right_model.recorded_requests()[0].messages[0]
            .text_content("")
            .as_deref(),
        Some("right private history")
    );
}

#[tokio::test]
async fn run_is_lazy_and_clones_share_busy_lock_across_routes_but_unknown_route_wins() {
    let gate = Arc::new(ReplyGate::default());
    let gated = ScriptAgent::gated("first", &gate);
    let other = ScriptAgent::returning("second", Ok(reply("second", "done")));
    let pipeline = RoutedPipeline::new(vec![
        ("first-route".into(), gated.clone()),
        ("second-route".into(), other.clone()),
    ])
    .unwrap();
    let clone = pipeline.clone();
    let unpolled = pipeline.run("first-route", Msg::user("unpolled"));
    clone
        .run("second-route", Msg::user("unreserved"))
        .await
        .unwrap();
    assert!(gated.inputs.lock().unwrap().is_empty());
    drop(unpolled);
    let mut active = pipeline.run("first-route", Msg::user("active"));
    assert!(active.as_mut().now_or_never().is_none());
    let busy = clone
        .run("second-route", Msg::user("busy"))
        .await
        .unwrap_err();
    assert_eq!(busy.route, "second-route");
    assert_eq!(busy.agent_name.as_deref(), Some("second"));
    assert_eq!(busy.cause, RoutedFailure::Busy);
    let unknown = clone
        .run("missing", Msg::user("unknown while busy"))
        .await
        .unwrap_err();
    assert_eq!(unknown.cause, RoutedFailure::UnknownRoute);
    assert_eq!(unknown.agent_name, None);
    assert_eq!(other.inputs.lock().unwrap().len(), 1);
    gate.release();
    active.await.unwrap();
}

#[tokio::test]
async fn drop_cancels_selected_future_and_releases_lock_without_rolling_back_or_retrying() {
    let gate = Arc::new(ReplyGate::default());
    let gated = ScriptAgent::gated("gated", &gate);
    let untouched = ScriptAgent::returning("other", Ok(reply("other", "done")));
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), gated.clone()),
        ("other".into(), untouched.clone()),
    ])
    .unwrap();
    let input = Msg::user("invocation already observed");
    let mut active = pipeline.run("selected", input.clone());
    assert!(active.as_mut().now_or_never().is_none());
    drop(active);
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(*gated.inputs.lock().unwrap(), vec![input]);
    assert!(untouched.inputs.lock().unwrap().is_empty());
    let output = pipeline
        .clone()
        .run("other", Msg::user("explicit new run"))
        .await
        .unwrap();
    assert_eq!(output.agent_name, "other");
    assert_eq!(gated.inputs.lock().unwrap().len(), 1);
    assert_eq!(untouched.inputs.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn interruption_wins_over_ready_reply_on_resumed_poll_without_touching_child_handle() {
    let gate = Arc::new(ReplyGate::default());
    let selected = ScriptAgent::gated("worker", &gate);
    let other = ScriptAgent::returning("other", Ok(reply("other", "done")));
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let mut active = pipeline.run("selected", Msg::user("active"));
    assert!(active.as_mut().now_or_never().is_none());
    assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
    gate.release();
    pipeline.clone().interrupt_handle().interrupt();
    let error = active.await.unwrap_err();
    assert_eq!(error.route, "selected");
    assert_eq!(error.agent_name.as_deref(), Some("worker"));
    assert_eq!(error.cause, RoutedFailure::Interrupted);
    assert_eq!(
        gate.polls.load(Ordering::SeqCst),
        1,
        "cancellation must prevent a ready child from being re-polled"
    );
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(selected.interrupt_handle_reads.load(Ordering::SeqCst), 0);
    pipeline
        .clone()
        .run("other", Msg::user("clone after interruption"))
        .await
        .unwrap();
    assert_eq!(other.inputs.lock().unwrap().len(), 1);
    assert_eq!(selected.inputs.lock().unwrap().len(), 1);
    pipeline.interrupt_handle().interrupt();
    let fresh = pipeline
        .run("selected", Msg::user("fresh baseline"))
        .await
        .unwrap();
    assert_eq!(fresh.route, "selected");
    assert_eq!(gate.effects.load(Ordering::SeqCst), 1);
    assert_eq!(selected.inputs.lock().unwrap().len(), 2);
    assert_eq!(selected.interrupt_handle_reads.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn confirmation_and_uncertain_execution_errors_are_preserved_without_approval_or_retry() {
    let (worker, model, tool) = uncertain_agent();
    let other = ScriptAgent::returning("other", Ok(reply("other", "unused")));
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), worker.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let error = pipeline
        .run("selected", Msg::user("task"))
        .await
        .unwrap_err();
    let RoutedFailure::Agent(cause) = &error.cause else {
        panic!("expected original confirmation failure")
    };
    let AgentError::ToolConfirmationRequired { checkpoint } = cause.as_ref() else {
        panic!("checkpoint must be retained")
    };
    let before = worker.snapshot().await.unwrap();
    assert_eq!(before.pending_tool_calls(), Some(checkpoint));
    assert!(tool.recorded_invocations().is_empty());
    assert!(other.inputs.lock().unwrap().is_empty());
    let encoded = serde_json::to_string(&error).unwrap();
    assert_eq!(
        serde_json::from_str::<RoutedError>(&encoded).unwrap(),
        error
    );
    let uncertain = worker
        .resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("call")],
        )
        .await
        .unwrap_err();
    assert!(matches!(uncertain, AgentError::ToolExecutionInDoubt { .. }));
    let before = worker.snapshot().await.unwrap();
    let error = pipeline
        .run("selected", Msg::user("must not implicitly retry"))
        .await
        .unwrap_err();
    assert_eq!(error.cause, RoutedFailure::Agent(Box::new(uncertain)));
    assert_eq!(worker.snapshot().await.unwrap(), before);
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert_eq!(model.recorded_requests().len(), 1);
    assert!(other.inputs.lock().unwrap().is_empty());
    let encoded = serde_json::to_string(&error).unwrap();
    assert_eq!(
        serde_json::from_str::<RoutedError>(&encoded).unwrap(),
        error
    );
}

#[test]
fn runtime_errors_roundtrip_all_tagged_causes_and_expose_only_agent_error_source() {
    let agent_error = AgentError::Model(ModelError::new("original model failure"));
    for (kind, cause) in [
        ("unknown_route", RoutedFailure::UnknownRoute),
        ("busy", RoutedFailure::Busy),
        ("interrupted", RoutedFailure::Interrupted),
        ("agent", RoutedFailure::Agent(Box::new(agent_error.clone()))),
    ] {
        let error = RoutedError {
            route: "exact route ".into(),
            agent_name: if cause == RoutedFailure::UnknownRoute {
                None
            } else {
                Some("worker".into())
            },
            cause,
        };
        let encoded = serde_json::to_value(&error).unwrap();
        assert_eq!(encoded["cause"]["kind"], kind);
        assert_eq!(
            serde_json::from_value::<RoutedError>(encoded).unwrap(),
            error
        );
        if kind == "agent" {
            let source = error.source().unwrap();
            assert_eq!(source.downcast_ref::<AgentError>(), Some(&agent_error));
            assert_eq!(source.to_string(), agent_error.to_string());
        } else {
            assert!(error.source().is_none());
        }
        assert!(!error.to_string().is_empty());
    }
}
