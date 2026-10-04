use super::*;
use crate::*;
use futures_util::{FutureExt, StreamExt};
use serde_json::json;
use std::{
    error::Error as _,
    sync::{Arc, atomic::Ordering},
};
#[path = "tests_support.rs"]
mod support;
use support::*;

#[tokio::test]
async fn preparation_is_exact_lazy_and_shares_run_stream_and_clone_lock() {
    let gate = Arc::new(Gate::default());
    let selected = ScriptAgent::startup_gated("selected", &gate);
    let other = ScriptAgent::finished("other");
    let pipeline = RoutedPipeline::new(vec![
        ("exact".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let clone = pipeline.clone();
    drop(pipeline.stream("exact", Msg::user("unpolled preparation")));
    let reserved = pipeline
        .stream("exact", Msg::user("reserved"))
        .await
        .unwrap();
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    for route in ["Exact", "exact ", " exact", "", "missing"] {
        let Err(error) = clone.stream(route, Msg::user("unknown while busy")).await else {
            panic!("unknown route must win")
        };
        assert_eq!(error.route, route);
        assert_eq!(error.agent_name, None);
        assert_eq!(error.cause, RoutedFailure::UnknownRoute);
    }
    assert!(
        matches!(clone.stream("other", Msg::user("busy stream")).await, Err(RoutedError { cause: RoutedFailure::Busy, agent_name: Some(name), .. }) if name == "other")
    );
    assert!(matches!(
        clone.run("other", Msg::user("busy run")).await,
        Err(RoutedError {
            cause: RoutedFailure::Busy,
            ..
        })
    ));
    assert_eq!(other.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(other.reply_calls.load(Ordering::SeqCst), 0);
    drop(reserved);
    let mut run = pipeline.run("exact", Msg::user("active run"));
    assert!(run.as_mut().now_or_never().is_none());
    assert!(matches!(
        clone.stream("other", Msg::user("run holds lock")).await,
        Err(RoutedError {
            cause: RoutedFailure::Busy,
            ..
        })
    ));
    drop(run);
    drop(clone.stream("other", Msg::user("released")).await.unwrap());
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn dropping_reserved_or_announced_route_stops_before_synchronous_agent_invocation() {
    for announce in [false, true] {
        let selected = ScriptAgent::finished("selected");
        let other = ScriptAgent::finished("other");
        let pipeline = RoutedPipeline::new(vec![
            ("selected".into(), selected.clone()),
            ("other".into(), other.clone()),
        ])
        .unwrap();
        let mut stream = pipeline
            .stream("selected", Msg::user("task"))
            .await
            .unwrap();
        if announce {
            assert!(
                matches!(stream.next().await, Some(RoutedEvent::RouteStarted { route, agent_name }) if route == "selected" && agent_name == "selected")
            );
        }
        assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
        drop(stream);
        assert!(selected.inputs.lock().unwrap().is_empty());
        assert_eq!(selected.events_dropped.load(Ordering::SeqCst), 0);
        pipeline
            .clone()
            .run("other", Msg::user("explicit new work"))
            .await
            .unwrap();
        assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
        assert_eq!(other.reply_calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn rich_messages_and_nonterminal_agent_events_are_forwarded_unchanged_and_serializable() {
    let mut input = Msg::new(
        "input-author",
        Role::System,
        [
            ThinkingBlock::new("private input").into(),
            DataBlock::url("https://example.com/image.png", "image/png")
                .unwrap()
                .into(),
        ],
    );
    input.metadata.insert("private".into(), json!({"value": 7}));
    let mut original = Msg::new(
        "reply-author",
        Role::User,
        [
            ThinkingBlock::new("private reply").into(),
            DataBlock::base64("YWJj", "application/octet-stream")
                .unwrap()
                .into(),
        ],
    );
    original.metadata.insert("private".into(), json!(true));
    let events = vec![
        AgentEvent::ContextCompactionStarted {
            keep_recent_turns: 2,
        },
        AgentEvent::ContextCompactionCompleted {
            covered_messages: 3,
        },
        AgentEvent::ContextCompactionFailed {
            error: AgentError::Model(ModelError::new("nonterminal compaction report")),
        },
        AgentEvent::ThinkingDelta {
            step: 17,
            block_id: "same-id".into(),
            delta: "reasoning".into(),
        },
        AgentEvent::TextDelta {
            step: 17,
            block_id: "same-id".into(),
            delta: "visible".into(),
        },
        AgentEvent::StepFinished {
            step: 17,
            reason: FinishReason::Completed,
        },
        AgentEvent::Finished {
            steps: 42,
            message: original.clone(),
        },
    ];
    let selected = ScriptAgent::new("configured-name", events.iter().cloned().map(Ok).collect());
    let other = ScriptAgent::finished("other");
    let pipeline = RoutedPipeline::new(vec![
        (" exact ".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let observed = pipeline
        .stream(" exact ", input.clone())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let forwarded: Vec<_> = observed
        .iter()
        .filter_map(|event| {
            if let RoutedEvent::Agent {
                route,
                agent_name,
                event,
            } = event
            {
                assert_eq!(route, " exact ");
                assert_eq!(agent_name, "configured-name");
                Some(event.clone())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(forwarded, events);
    let Some(RoutedEvent::Finished { output }) = observed.last() else {
        panic!("compaction failure itself must not terminate the stream")
    };
    assert_eq!(output.route, " exact ");
    assert_eq!(output.agent_name, "configured-name");
    assert_eq!(output.message, original);
    assert_eq!(*selected.inputs.lock().unwrap(), vec![input]);
    assert_eq!(other.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(selected.item_polls.load(Ordering::SeqCst), events.len());
    assert_eq!(terminal_count(&observed), 1);
    assert_roundtrips(&observed);
}

#[tokio::test]
async fn startup_item_eof_and_agent_error_failures_preserve_causes_and_clean_up() {
    let startup_error = AgentError::Model(ModelError::new("startup"));
    let item_error = AgentError::Model(ModelError::new("item"));
    let child_error = AgentError::Interrupted;
    let cases = [
        (
            ScriptAgent::startup_error("worker", startup_error.clone()),
            Some(startup_error),
            0,
        ),
        (
            ScriptAgent::new("worker", vec![Err(item_error.clone())]),
            Some(item_error),
            1,
        ),
        (
            ScriptAgent::new(
                "worker",
                vec![Ok(AgentEvent::TextDelta {
                    step: 7,
                    block_id: "partial".into(),
                    delta: "incomplete".into(),
                })],
            ),
            None,
            1,
        ),
        (
            ScriptAgent::new(
                "worker",
                vec![Ok(AgentEvent::Error {
                    step: Some(9),
                    error: child_error.clone(),
                })],
            ),
            Some(child_error),
            1,
        ),
    ];
    for (selected, expected, dropped) in cases {
        let other = ScriptAgent::finished("other");
        let pipeline = RoutedPipeline::new(vec![
            ("selected".into(), selected.clone()),
            ("other".into(), other),
        ])
        .unwrap();
        let mut stream = pipeline
            .stream("selected", Msg::user("task"))
            .await
            .unwrap();
        let mut observed = Vec::new();
        while let Some(event) = stream.next().await {
            let terminal = matches!(event, RoutedEvent::Error { .. });
            observed.push(event);
            if terminal {
                break;
            }
        }
        let Some(RoutedEvent::Error { error }) = observed.last() else {
            panic!("expected routed error")
        };
        assert_eq!(error.route, "selected");
        assert_eq!(error.agent_name.as_deref(), Some("worker"));
        let RoutedFailure::Agent(cause) = &error.cause else {
            panic!("child failures must remain agent failures")
        };
        if let Some(expected) = expected {
            assert_eq!(cause.as_ref(), &expected);
        } else {
            assert!(
                matches!(cause.as_ref(), AgentError::InvalidModelResponse(reason) if reason.contains("terminal"))
            );
        }
        assert_eq!(
            error.source().unwrap().downcast_ref::<AgentError>(),
            Some(cause.as_ref())
        );
        assert_eq!(selected.events_dropped.load(Ordering::SeqCst), dropped);
        pipeline
            .clone()
            .run("other", Msg::user("error before EOF"))
            .await
            .unwrap();
        assert!(stream.next().await.is_none());
        assert_eq!(terminal_count(&observed), 1);
        assert_roundtrips(&observed);
    }
}

#[tokio::test]
async fn observed_terminal_wins_late_interrupt_drops_child_and_releases_lock_before_final_yield() {
    let message = reply("original-author", "original reply");
    for terminal in [
        AgentEvent::Finished {
            steps: 7,
            message: message.clone(),
        },
        AgentEvent::Error {
            step: Some(8),
            error: AgentError::Model(ModelError::new("original error")),
        },
    ] {
        let selected = ScriptAgent::new(
            "worker",
            vec![
                Ok(terminal.clone()),
                Err(AgentError::Model(ModelError::new(
                    "tail must never be read",
                ))),
            ],
        );
        let other = ScriptAgent::finished("other");
        let pipeline = RoutedPipeline::new(vec![
            ("selected".into(), selected.clone()),
            ("other".into(), other.clone()),
        ])
        .unwrap();
        let mut stream = pipeline
            .stream("selected", Msg::user("task"))
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await,
            Some(RoutedEvent::RouteStarted { .. })
        ));
        let wrapped = stream.next().await.unwrap();
        assert!(matches!(&wrapped, RoutedEvent::Agent { event, .. } if event == &terminal));
        assert_eq!(
            selected.events_dropped.load(Ordering::SeqCst),
            1,
            "child must be closed before its terminal event is exposed"
        );
        pipeline.interrupt_handle().interrupt();
        let final_event = stream.next().await.unwrap();
        match terminal {
            AgentEvent::Finished { message, .. } => assert!(
                matches!(&final_event, RoutedEvent::Finished { output } if output.message == message)
            ),
            AgentEvent::Error { error, .. } => assert!(
                matches!(&final_event, RoutedEvent::Error { error: routed } if routed.cause == RoutedFailure::Agent(Box::new(error)))
            ),
            _ => unreachable!(),
        }
        assert_eq!(selected.item_polls.load(Ordering::SeqCst), 1);
        pipeline
            .clone()
            .run("other", Msg::user("stream is still retained"))
            .await
            .unwrap();
        drop(
            pipeline
                .clone()
                .stream("other", Msg::user("another reserved stream"))
                .await
                .unwrap(),
        );
        assert_eq!(other.reply_calls.load(Ordering::SeqCst), 1);
        assert_eq!(other.stream_calls.load(Ordering::SeqCst), 0);
        assert!(stream.next().await.is_none());
        assert_roundtrips(&[wrapped, final_event]);
    }
}

#[tokio::test]
async fn confirmation_is_finalized_before_forwarding_and_uncertain_execution_is_not_retried() {
    let store = Arc::new(InMemoryStateStore::new());
    let key = StateKey::new("test", "worker").unwrap();
    let (worker, tool, model) = confirming_agent(store.clone(), key.clone());
    let pipeline = RoutedPipeline::new(vec![("selected".into(), worker.clone())]).unwrap();
    let mut stream = pipeline
        .stream("selected", Msg::user("task"))
        .await
        .unwrap();
    let checkpoint = loop {
        let event = stream.next().await.unwrap();
        if let RoutedEvent::Agent {
            event: AgentEvent::ToolConfirmationRequired { checkpoint },
            ..
        } = event
        {
            break checkpoint;
        }
    };
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_calls(),
        Some(&checkpoint)
    );
    assert!(tool.recorded_invocations().is_empty());
    pipeline.interrupt_handle().interrupt();
    let terminal = stream.next().await.unwrap();
    assert!(
        matches!(&terminal, RoutedEvent::Error { error } if error.cause == RoutedFailure::Agent(Box::new(AgentError::ToolConfirmationRequired { checkpoint: checkpoint.clone() })))
    );
    let reserved = pipeline.clone();
    drop(
        reserved
            .stream("selected", Msg::user("lock already released"))
            .await
            .unwrap(),
    );
    assert!(stream.next().await.is_none());
    let uncertain = worker
        .resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("call")],
        )
        .await
        .unwrap_err();
    assert!(matches!(uncertain, AgentError::ToolExecutionInDoubt { .. }));
    let before = store.load(&key).await.unwrap().unwrap();
    let observed = pipeline
        .stream("selected", Msg::user("no automatic retry"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(observed.last(), Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Agent(Box::new(uncertain)))
    );
    assert_eq!(
        store.load(&key).await.unwrap().unwrap().state(),
        before.state()
    );
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert_eq!(model.recorded_requests().len(), 1);
    assert_eq!(terminal_count(&observed), 1);
    assert_roundtrips(&observed);
}

#[tokio::test]
async fn interruption_before_dispatch_prevents_sync_calls_and_fresh_preparation_ignores_old_signal()
{
    for announce in [false, true] {
        let selected = ScriptAgent::finished("worker");
        let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
        let mut stream = pipeline
            .stream("selected", Msg::user("task"))
            .await
            .unwrap();
        if announce {
            assert!(matches!(
                stream.next().await,
                Some(RoutedEvent::RouteStarted { .. })
            ));
        }
        pipeline.interrupt_handle().interrupt();
        let observed = stream.collect::<Vec<_>>().await;
        assert!(
            matches!(observed.as_slice(), [RoutedEvent::Error { error }] if error.cause == RoutedFailure::Interrupted)
        );
        assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
        assert!(selected.inputs.lock().unwrap().is_empty());
        pipeline.interrupt_handle().interrupt();
        let fresh = pipeline
            .stream("selected", Msg::user("fresh baseline"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(fresh.last(), Some(RoutedEvent::Finished { .. })));
        assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 1);
        assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn resumed_startup_cancellation_wins_before_underlying_ready_future_is_polled_again() {
    let gate = Arc::new(Gate::default());
    let selected = ScriptAgent::startup_gated("worker", &gate);
    let other = ScriptAgent::finished("other");
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let mut stream = pipeline
        .stream("selected", Msg::user("task"))
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(RoutedEvent::RouteStarted { .. })
    ));
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    assert!(stream.next().now_or_never().is_none());
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
    gate.release();
    pipeline.interrupt_handle().interrupt();
    assert!(
        matches!(stream.next().await, Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Interrupted)
    );
    assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    pipeline
        .clone()
        .run("other", Msg::user("Error visible before EOF"))
        .await
        .unwrap();
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn resumed_event_cancellation_drops_child_before_error_without_another_poll_or_effect() {
    let gate = Arc::new(Gate::default());
    let selected = ScriptAgent::event_gated("worker", &gate);
    let other = ScriptAgent::finished("other");
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let mut stream = pipeline
        .stream("selected", Msg::user("task"))
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(RoutedEvent::RouteStarted { .. })
    ));
    assert!(stream.next().now_or_never().is_none());
    assert_eq!(selected.item_polls.load(Ordering::SeqCst), 1);
    gate.release();
    pipeline.interrupt_handle().interrupt();
    assert!(
        matches!(stream.next().await, Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Interrupted)
    );
    assert_eq!(selected.item_polls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(selected.events_dropped.load(Ordering::SeqCst), 1);
    assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    pipeline
        .clone()
        .run("other", Msg::user("Error visible before EOF"))
        .await
        .unwrap();
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn dropping_active_stream_releases_child_and_lock_without_background_polling() {
    let gate = Arc::new(Gate::default());
    let selected = ScriptAgent::event_gated("worker", &gate);
    let other = ScriptAgent::finished("other");
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let mut stream = pipeline
        .stream("selected", Msg::user("task"))
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(RoutedEvent::RouteStarted { .. })
    ));
    assert!(stream.next().now_or_never().is_none());
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(selected.item_polls.load(Ordering::SeqCst), 1);
    drop(stream);
    assert_eq!(selected.events_dropped.load(Ordering::SeqCst), 1);
    gate.release();
    let fresh = pipeline
        .clone()
        .stream("other", Msg::user("explicit next route"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(fresh.last(), Some(RoutedEvent::Finished { .. })));
    assert_eq!(selected.item_polls.load(Ordering::SeqCst), 1);
    assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
    assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    assert_eq!(other.stream_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn selected_alias_streams_finalize_independent_same_named_agent_sessions() {
    let store = Arc::new(InMemoryStateStore::new());
    let left_key = StateKey::new("test", "left").unwrap();
    let right_key = StateKey::new("test", "right").unwrap();
    let left_model = model("left done");
    left_model.push_stream(Ok(vec![
        Ok(ChatEvent::TextDelta {
            block_id: "again".into(),
            delta: "left again".into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]));
    let right_model = model("right done");
    let left = Arc::new(
        ReActAgent::from_shared(
            "same",
            left_model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_shared_state_store(left_key.clone(), store.clone()),
    );
    let right = Arc::new(
        ReActAgent::from_shared(
            "same",
            right_model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_shared_state_store(right_key.clone(), store.clone()),
    );
    let pipeline = RoutedPipeline::new(vec![
        ("left".into(), left.clone()),
        ("alias".into(), left.clone()),
        ("right".into(), right.clone()),
    ])
    .unwrap();
    let first = pipeline
        .stream("left", Msg::user("first left"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(first.last(), Some(RoutedEvent::Finished { .. })));
    assert!(store.load(&right_key).await.unwrap().is_none());
    pipeline
        .stream("right", Msg::user("first right"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let alias = pipeline
        .stream("alias", Msg::user("continue left"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(alias.last(), Some(RoutedEvent::Finished { output }) if output.route == "alias" && output.agent_name == "same")
    );
    assert_eq!(
        store
            .load(&left_key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .messages()
            .len(),
        4
    );
    assert_eq!(
        store
            .load(&right_key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .messages()
            .len(),
        2
    );
    assert_eq!(left_model.recorded_requests().len(), 2);
    assert_eq!(left_model.recorded_requests()[1].messages.len(), 3);
    assert_eq!(right_model.recorded_requests().len(), 1);
}
