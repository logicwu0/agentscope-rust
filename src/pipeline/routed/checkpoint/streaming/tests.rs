use super::super::super::streaming_tests_support::*;
use super::*;
use crate::*;
use futures_util::{FutureExt, StreamExt};
use std::sync::{Arc, atomic::Ordering};
#[path = "tests_support.rs"]
mod support;
use support::*;

#[tokio::test]
async fn preparation_is_lazy_exact_fail_closed_and_shares_all_operation_locks() {
    let store = ControlledStore::plain();
    let untouched = ControlledStore::plain();
    let selected = ScriptAgent::finished("worker");
    let other = ScriptAgent::finished("other");
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let clone = pipeline.clone();
    let state_key = key("prepared");
    drop(pipeline.stream_checkpointed(
        &store,
        state_key.clone(),
        "selected",
        Msg::user("not polled"),
    ));
    assert_eq!(store.saves.load(Ordering::SeqCst), 0);
    let stream = pipeline
        .stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("saved"))
        .await
        .unwrap();
    let prepared = assert_status(&store.inner, &state_key, RoutedCheckpointStatus::Ready).await;
    assert_eq!(prepared.revision, 1);
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    for route in ["Selected", "selected ", " selected", "", "missing"] {
        let Err(error) = clone
            .stream_checkpointed(
                &untouched,
                key("unknown"),
                route,
                Msg::user("unknown wins even busy"),
            )
            .await
        else {
            panic!("expected unknown route")
        };
        assert_eq!(error.cause, RoutedFailure::UnknownRoute);
        assert_eq!(error.route, route);
        assert_eq!(error.agent_name, None);
    }
    assert_eq!(untouched.loads.load(Ordering::SeqCst), 0);
    assert_eq!(untouched.saves.load(Ordering::SeqCst), 0);
    assert!(matches!(
        clone
            .run("other", Msg::user("busy"))
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Busy
    ));
    let Err(error) = clone.stream("other", Msg::user("busy")).await else {
        panic!("expected busy")
    };
    assert_eq!(error.cause, RoutedFailure::Busy);
    let Err(error) = clone
        .stream_checkpointed(&untouched, key("busy"), "other", Msg::user("busy"))
        .await
    else {
        panic!("expected busy")
    };
    assert_eq!(error.cause, RoutedFailure::Busy);
    assert_eq!(error.agent_name.as_deref(), Some("other"));
    assert!(matches!(
        clone
            .run_checkpointed(&untouched, key("busy"), "other", Msg::user("busy"))
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Busy
    ));
    let Err(error) = clone
        .resume_checkpointed_stream(&untouched, key("busy"))
        .await
    else {
        panic!("expected busy")
    };
    assert_eq!(error.cause, RoutedFailure::Busy);
    assert!(matches!(
        clone
            .resume_checkpointed(&untouched, key("busy"))
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Busy
    ));
    assert_eq!(untouched.loads.load(Ordering::SeqCst), 0);
    assert_eq!(untouched.saves.load(Ordering::SeqCst), 0);
    drop(stream);
    assert_eq!(store.inner.load(&state_key).await.unwrap(), Some(prepared));
    clone.resume_checkpointed(&store, state_key).await.unwrap();
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(selected.reply_calls.load(Ordering::SeqCst), 1);
    assert_eq!(other.stream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn interruption_at_first_poll_or_after_started_never_invokes_selected_agent() {
    for started in [false, true] {
        let store = InMemoryRoutedStore::new();
        let selected = ScriptAgent::finished("worker");
        let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
        let state_key = key("cancel-before-child");
        let mut stream = pipeline
            .stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
            .await
            .unwrap();
        if started {
            assert!(matches!(
                stream.next().await,
                Some(RoutedEvent::RouteStarted { .. })
            ));
            assert_status(&store, &state_key, RoutedCheckpointStatus::InFlight).await;
            assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
        }
        pipeline.interrupt_handle().interrupt();
        let observed = collect_to_terminal(&mut stream).await;
        assert!(
            matches!(observed.as_slice(), [RoutedEvent::Error { error }] if error.cause == RoutedFailure::Interrupted)
        );
        let expected = if started {
            RoutedCheckpointStatus::InFlight
        } else {
            RoutedCheckpointStatus::Ready
        };
        assert_status(&store, &state_key, expected).await;
        assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
        assert!(selected.inputs.lock().unwrap().is_empty());
        if started {
            assert_unsafe_stream_resume(&pipeline, &store, state_key).await;
        } else {
            pipeline
                .resume_checkpointed_stream(&store, state_key)
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 1);
        }
        assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
        assert!(stream.next().await.is_none());
    }
}

#[tokio::test]
async fn interruption_during_preparation_or_marker_does_not_cancel_store_acknowledgement() {
    let gate = Arc::new(StoreGate::default());
    let store = ControlledStore::new(SavePoint::Initial, SaveAction::Gate(gate.clone()));
    let selected = ScriptAgent::finished("worker");
    let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
    let state_key = key("pending-ready");
    let mut preparation =
        pipeline.stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"));
    assert!(preparation.as_mut().now_or_never().is_none());
    assert_status(&store.inner, &state_key, RoutedCheckpointStatus::Ready).await;
    pipeline.interrupt_handle().interrupt();
    assert!(preparation.as_mut().now_or_never().is_none());
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 0);
    gate.release();
    let mut stream = preparation.await.unwrap();
    assert!(
        matches!(stream.next().await, Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Interrupted)
    );
    assert_status(&store.inner, &state_key, RoutedCheckpointStatus::Ready).await;
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    assert!(stream.next().await.is_none());

    let gate = Arc::new(StoreGate::default());
    let store = ControlledStore::new(SavePoint::Marker, SaveAction::Gate(gate.clone()));
    let state_key = key("pending-marker");
    let mut stream = pipeline
        .stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
        .await
        .unwrap();
    assert!(stream.next().now_or_never().is_none());
    assert_status(&store.inner, &state_key, RoutedCheckpointStatus::InFlight).await;
    pipeline.interrupt_handle().interrupt();
    assert!(stream.next().now_or_never().is_none());
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 0);
    gate.release();
    assert!(
        matches!(stream.next().await, Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Interrupted)
    );
    assert_status(&store.inner, &state_key, RoutedCheckpointStatus::InFlight).await;
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    assert_unsafe_stream_resume(&pipeline, &store, state_key).await;
    assert_eq!(gate.effects.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn rich_terminal_event_is_not_acknowledgement_and_late_interrupt_cannot_cancel_commit() {
    let (input, original, events) = rich_fixture();
    let mut scripted = events.iter().cloned().map(Ok).collect::<Vec<_>>();
    scripted.push(Err(AgentError::Model(ModelError::new(
        "tail must not be polled",
    ))));
    let selected = ScriptAgent::new("worker", scripted);
    let other = ScriptAgent::finished("other");
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("other".into(), other),
    ])
    .unwrap();
    let gate = Arc::new(StoreGate::default());
    let store = ControlledStore::new(SavePoint::Terminal, SaveAction::Gate(gate.clone()));
    let state_key = key("rich-terminal");
    let mut stream = pipeline
        .stream_checkpointed(&store, state_key.clone(), "selected", input.clone())
        .await
        .unwrap();
    let mut observed = vec![stream.next().await.unwrap()];
    assert!(matches!(
        observed.first(),
        Some(RoutedEvent::RouteStarted { .. })
    ));
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    for event in &events {
        let wrapped = stream.next().await.unwrap();
        assert!(
            matches!(&wrapped, RoutedEvent::Agent { route, agent_name, event: inner } if route == "selected" && agent_name == "worker" && inner == event)
        );
        observed.push(wrapped);
    }
    assert_eq!(*selected.inputs.lock().unwrap(), vec![input.clone()]);
    assert_eq!(selected.events_dropped.load(Ordering::SeqCst), 1);
    let inflight = assert_status(&store.inner, &state_key, RoutedCheckpointStatus::InFlight).await;
    assert_eq!(inflight.revision, 2);
    pipeline.interrupt_handle().interrupt();
    assert!(stream.next().now_or_never().is_none());
    let committed = assert_status(
        &store.inner,
        &state_key,
        RoutedCheckpointStatus::Completed(original.clone()),
    )
    .await;
    assert_eq!(committed.revision, 3);
    pipeline.interrupt_handle().interrupt();
    assert!(stream.next().now_or_never().is_none());
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 0);
    gate.release();
    let final_event = stream.next().await.unwrap();
    assert!(matches!(&final_event, RoutedEvent::Finished { output } if output.message == original));
    assert_eq!(
        committed.checkpoint.finished_result().unwrap(),
        match &final_event {
            RoutedEvent::Finished { output } => Ok(output.clone()),
            _ => unreachable!(),
        }
    );
    observed.push(final_event);
    pipeline
        .clone()
        .run("other", Msg::user("terminal releases lock before EOF"))
        .await
        .unwrap();
    assert_eq!(selected.item_polls.load(Ordering::SeqCst), events.len());
    assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    assert_eq!(terminal_count(&observed), 1);
    assert_roundtrips(&observed);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn startup_item_eof_and_agent_error_failures_are_saved_before_routed_error() {
    let startup = AgentError::Model(ModelError::new("startup"));
    let item = AgentError::Model(ModelError::new("raw item"));
    let terminal = AgentError::Model(ModelError::new("original terminal"));
    let cases = [
        (
            ScriptAgent::startup_error("worker", startup.clone()),
            Some(startup),
            0,
        ),
        (
            ScriptAgent::new("worker", vec![Err(item.clone())]),
            Some(item),
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
                vec![
                    Ok(AgentEvent::ContextCompactionFailed {
                        error: AgentError::Model(ModelError::new("nonterminal compaction")),
                    }),
                    Ok(AgentEvent::Error {
                        step: Some(9),
                        error: terminal.clone(),
                    }),
                    Err(AgentError::Interrupted),
                ],
            ),
            Some(terminal),
            1,
        ),
    ];
    for (selected, expected, drops) in cases {
        let other = ScriptAgent::finished("other");
        let pipeline = RoutedPipeline::new(vec![
            ("selected".into(), selected.clone()),
            ("other".into(), other),
        ])
        .unwrap();
        let store = InMemoryRoutedStore::new();
        let state_key = key("failed");
        let mut stream = pipeline
            .stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
            .await
            .unwrap();
        let observed = collect_to_terminal(&mut stream).await;
        let Some(RoutedEvent::Error { error }) = observed.last() else {
            panic!("expected routed error")
        };
        let RoutedFailure::Agent(cause) = &error.cause else {
            panic!("expected original agent failure")
        };
        if let Some(expected) = expected {
            assert_eq!(cause.as_ref(), &expected);
        } else {
            assert!(
                matches!(cause.as_ref(), AgentError::InvalidModelResponse(reason) if reason.contains("terminal"))
            );
        }
        let committed = assert_status(
            &store,
            &state_key,
            RoutedCheckpointStatus::Failed(cause.clone()),
        )
        .await;
        assert_eq!(committed.revision, 3);
        assert_eq!(
            committed.checkpoint.finished_result().unwrap().unwrap_err(),
            *error
        );
        assert_eq!(selected.events_dropped.load(Ordering::SeqCst), drops);
        pipeline
            .clone()
            .run("other", Msg::user("terminal retains stream without EOF"))
            .await
            .unwrap();
        assert_unsafe_stream_resume(&pipeline, &store, state_key).await;
        assert_eq!(terminal_count(&observed), 1);
        assert_roundtrips(&observed);
        assert!(stream.next().await.is_none());
    }
}

#[tokio::test]
async fn save_rejection_ambiguous_commit_and_bad_ack_never_report_completion_or_dispatch_early() {
    for point in [SavePoint::Initial, SavePoint::Marker, SavePoint::Terminal] {
        for action in [
            SaveAction::Reject,
            SaveAction::Ambiguous,
            SaveAction::BadAck,
        ] {
            let rejected = matches!(action, SaveAction::Reject);
            let store = ControlledStore::new(point, action);
            let selected = ScriptAgent::finished("worker");
            let other = ScriptAgent::finished("other");
            let pipeline = RoutedPipeline::new(vec![
                ("selected".into(), selected.clone()),
                ("other".into(), other),
            ])
            .unwrap();
            let state_key = key("store-failure");
            let prepared = pipeline
                .stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
                .await;
            let observed = if matches!(point, SavePoint::Initial) {
                let Err(error) = prepared else {
                    panic!("bad initial save must reject preparation")
                };
                assert!(matches!(error.cause, RoutedFailure::Store(_)));
                Vec::new()
            } else {
                let mut stream = prepared.unwrap();
                let observed = collect_to_terminal(&mut stream).await;
                assert!(
                    matches!(observed.last(), Some(RoutedEvent::Error { error }) if matches!(error.cause, RoutedFailure::Store(_)))
                );
                assert_eq!(terminal_count(&observed), 1);
                assert!(
                    !observed
                        .iter()
                        .any(|event| matches!(event, RoutedEvent::Finished { .. }))
                );
                pipeline
                    .clone()
                    .run("other", Msg::user("Store error releases before EOF"))
                    .await
                    .unwrap();
                assert!(stream.next().await.is_none());
                observed
            };
            let record = store.inner.load(&state_key).await.unwrap();
            match point {
                SavePoint::Initial if rejected => assert!(record.is_none()),
                SavePoint::Initial => {
                    assert_eq!(
                        record.unwrap().checkpoint.status,
                        RoutedCheckpointStatus::Ready
                    );
                }
                SavePoint::Marker => {
                    let expected = if rejected {
                        RoutedCheckpointStatus::Ready
                    } else {
                        RoutedCheckpointStatus::InFlight
                    };
                    assert_eq!(record.unwrap().checkpoint.status, expected);
                    assert_eq!(
                        observed.len(),
                        1,
                        "no Started until the marker ack is valid"
                    );
                }
                SavePoint::Terminal => {
                    let record = record.unwrap();
                    if rejected {
                        assert_eq!(record.checkpoint.status, RoutedCheckpointStatus::InFlight);
                    } else {
                        let message = observed
                            .iter()
                            .find_map(|event| match event {
                                RoutedEvent::Agent {
                                    event: AgentEvent::Finished { message, .. },
                                    ..
                                } => Some(message.clone()),
                                _ => None,
                            })
                            .unwrap();
                        assert_eq!(
                            record.checkpoint.status,
                            RoutedCheckpointStatus::Completed(message)
                        );
                    }
                    assert_unsafe_stream_resume(&pipeline, &store.inner, state_key).await;
                }
            }
            assert_eq!(
                selected.stream_calls.load(Ordering::SeqCst),
                usize::from(matches!(point, SavePoint::Terminal))
            );
            assert_roundtrips(&observed);
        }
    }
}

#[tokio::test]
async fn drop_after_started_pending_child_or_wrapped_terminal_keeps_inflight_and_never_replays() {
    for stage in 0..3 {
        let gate = Arc::new(Gate::default());
        let selected = if stage == 2 {
            ScriptAgent::event_gated("worker", &gate)
        } else {
            ScriptAgent::finished("worker")
        };
        let other = ScriptAgent::finished("other");
        let pipeline = RoutedPipeline::new(vec![
            ("selected".into(), selected.clone()),
            ("other".into(), other),
        ])
        .unwrap();
        let store = InMemoryRoutedStore::new();
        let state_key = key("dropped");
        let mut stream = pipeline
            .stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await,
            Some(RoutedEvent::RouteStarted { .. })
        ));
        if stage == 1 {
            assert!(matches!(
                stream.next().await,
                Some(RoutedEvent::Agent {
                    event: AgentEvent::Finished { .. },
                    ..
                })
            ));
            assert_eq!(selected.events_dropped.load(Ordering::SeqCst), 1);
        } else if stage == 2 {
            assert!(stream.next().now_or_never().is_none());
            assert_eq!(selected.item_polls.load(Ordering::SeqCst), 1);
        }
        drop(stream);
        let record = assert_status(&store, &state_key, RoutedCheckpointStatus::InFlight).await;
        assert_eq!(record.revision, 2);
        assert_eq!(
            selected.events_dropped.load(Ordering::SeqCst),
            usize::from(stage != 0)
        );
        assert_unsafe_stream_resume(&pipeline, &store, state_key.clone()).await;
        assert!(matches!(
            pipeline
                .resume_checkpointed(&store, state_key)
                .await
                .unwrap_err()
                .cause,
            RoutedFailure::UnsafeResume(_)
        ));
        pipeline
            .clone()
            .run("other", Msg::user("explicit independent next work"))
            .await
            .unwrap();
        gate.release();
        assert_eq!(
            selected.stream_calls.load(Ordering::SeqCst),
            usize::from(stage != 0)
        );
        assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
        assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn every_resumed_startup_and_event_poll_prioritizes_cancel_without_saving_failed() {
    for startup in [true, false] {
        let gate = Arc::new(Gate::default());
        let selected = if startup {
            ScriptAgent::startup_gated("worker", &gate)
        } else {
            ScriptAgent::event_gated("worker", &gate)
        };
        let other = ScriptAgent::finished("other");
        let pipeline = RoutedPipeline::new(vec![
            ("selected".into(), selected.clone()),
            ("other".into(), other),
        ])
        .unwrap();
        let store = InMemoryRoutedStore::new();
        let state_key = key("cancel-child");
        let mut stream = pipeline
            .stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await,
            Some(RoutedEvent::RouteStarted { .. })
        ));
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
        assert_eq!(gate.dropped.load(Ordering::SeqCst), usize::from(startup));
        assert_eq!(
            selected.item_polls.load(Ordering::SeqCst),
            usize::from(!startup)
        );
        assert_eq!(
            selected.events_dropped.load(Ordering::SeqCst),
            usize::from(!startup)
        );
        assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
        let record = assert_status(&store, &state_key, RoutedCheckpointStatus::InFlight).await;
        assert_eq!(record.revision, 2);
        let fresh = pipeline
            .clone()
            .stream_checkpointed(&store, key("fresh"), "other", Msg::user("fresh baseline"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(fresh.last(), Some(RoutedEvent::Finished { .. })));
        assert!(stream.next().await.is_none());
    }
}

#[tokio::test]
async fn ready_resume_validates_saved_binding_and_finalizes_bound_agent_state() {
    let selected_model = model("resumed answer");
    let state_store = Arc::new(InMemoryStateStore::new());
    let agent_key = key("agent-state");
    let selected = Arc::new(
        ReActAgent::from_shared(
            "worker",
            selected_model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_shared_state_store(agent_key.clone(), state_store.clone()),
    );
    let pipeline = RoutedPipeline::new(vec![
        (
            "changed-unrelated-route".into(),
            ScriptAgent::finished("new-other-name") as Arc<dyn Agent>,
        ),
        ("selected".into(), selected),
    ])
    .unwrap();
    let store = InMemoryRoutedStore::new();
    for (index, saved) in [
        RoutedCheckpoint {
            version: ROUTED_CHECKPOINT_VERSION + 1,
            ..checkpoint(RoutedCheckpointStatus::Ready)
        },
        RoutedCheckpoint {
            agent_name: "changed-name".into(),
            ..checkpoint(RoutedCheckpointStatus::Ready)
        },
        RoutedCheckpoint {
            route: "Selected".into(),
            ..checkpoint(RoutedCheckpointStatus::Ready)
        },
        checkpoint(RoutedCheckpointStatus::InFlight),
        checkpoint(RoutedCheckpointStatus::Completed(reply(
            "worker",
            "already done",
        ))),
        checkpoint(RoutedCheckpointStatus::Failed(Box::new(
            AgentError::Interrupted,
        ))),
    ]
    .into_iter()
    .enumerate()
    {
        let state_key = key(&format!("invalid-{index}"));
        let record = store.save(state_key.clone(), None, saved).await.unwrap();
        assert_unsafe_stream_resume(&pipeline, &store, state_key.clone()).await;
        assert_eq!(store.load(&state_key).await.unwrap(), Some(record));
    }
    assert_unsafe_stream_resume(&pipeline, &store, key("absent")).await;
    assert!(selected_model.recorded_requests().is_empty());
    assert!(state_store.load(&agent_key).await.unwrap().is_none());
    let state_key = key("ready-resume");
    let ready = checkpoint(RoutedCheckpointStatus::Ready);
    store
        .save(state_key.clone(), None, ready.clone())
        .await
        .unwrap();
    pipeline.interrupt_handle().interrupt();
    let mut stream = pipeline
        .resume_checkpointed_stream(&store, state_key.clone())
        .await
        .unwrap();
    assert_status(&store, &state_key, RoutedCheckpointStatus::Ready).await;
    assert!(selected_model.recorded_requests().is_empty());
    let observed = collect_to_terminal(&mut stream).await;
    let Some(RoutedEvent::Finished { output }) = observed.last() else {
        panic!("expected resumed completion")
    };
    assert_eq!(output.route, "selected");
    assert_eq!(output.agent_name, "worker");
    let record = assert_status(
        &store,
        &state_key,
        RoutedCheckpointStatus::Completed(output.message.clone()),
    )
    .await;
    assert_eq!(record.revision, 3);
    assert_eq!(record.checkpoint.input, ready.input);
    assert_eq!(selected_model.recorded_requests().len(), 1);
    let state = state_store.load(&agent_key).await.unwrap().unwrap();
    assert_eq!(state.state().messages().len(), 2);
    assert_eq!(state.state().messages().first(), Some(&ready.input));
    assert_eq!(state.state().messages().last(), Some(&output.message));
    assert_unsafe_stream_resume(&pipeline, &store, state_key).await;
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn confirmation_agent_state_is_finalized_while_route_remains_inflight_until_ack() {
    let state_store = Arc::new(InMemoryStateStore::new());
    let agent_key = key("confirmation-agent");
    let (worker, tool, selected_model) = confirming_agent(state_store.clone(), agent_key.clone());
    let pipeline = RoutedPipeline::new(vec![("selected".into(), worker.clone())]).unwrap();
    let store = InMemoryRoutedStore::new();
    let state_key = key("confirmation-route");
    let mut stream = pipeline
        .stream_checkpointed(&store, state_key.clone(), "selected", Msg::user("write"))
        .await
        .unwrap();
    let pending = loop {
        let event = stream.next().await.unwrap();
        if let RoutedEvent::Agent {
            event: AgentEvent::ToolConfirmationRequired { checkpoint },
            ..
        } = event
        {
            break checkpoint;
        }
    };
    let state = state_store.load(&agent_key).await.unwrap().unwrap();
    assert_eq!(state.state().pending_tool_calls(), Some(&pending));
    assert!(tool.recorded_invocations().is_empty());
    let inflight = assert_status(&store, &state_key, RoutedCheckpointStatus::InFlight).await;
    assert_eq!(inflight.revision, 2);
    pipeline.interrupt_handle().interrupt();
    let routed = stream.next().await.unwrap();
    let original = AgentError::ToolConfirmationRequired {
        checkpoint: pending.clone(),
    };
    assert!(
        matches!(&routed, RoutedEvent::Error { error } if error.cause == RoutedFailure::Agent(Box::new(original.clone())))
    );
    let failed = assert_status(
        &store,
        &state_key,
        RoutedCheckpointStatus::Failed(Box::new(original)),
    )
    .await;
    assert_eq!(failed.revision, 3);
    assert_unsafe_stream_resume(&pipeline, &store, state_key).await;
    assert!(stream.next().await.is_none());
    let uncertain = worker
        .resume_tool_calls(pending.reply_id(), vec![ToolConfirmation::approve("call")])
        .await
        .unwrap_err();
    assert!(matches!(uncertain, AgentError::ToolExecutionInDoubt { .. }));
    let uncertain_key = key("uncertain-route");
    let mut stream = pipeline
        .stream_checkpointed(
            &store,
            uncertain_key.clone(),
            "selected",
            Msg::user("no retry"),
        )
        .await
        .unwrap();
    let observed = collect_to_terminal(&mut stream).await;
    assert!(
        matches!(observed.last(), Some(RoutedEvent::Error { error }) if error.cause == RoutedFailure::Agent(Box::new(uncertain.clone())))
    );
    assert_status(
        &store,
        &uncertain_key,
        RoutedCheckpointStatus::Failed(Box::new(uncertain)),
    )
    .await;
    assert_unsafe_stream_resume(&pipeline, &store, uncertain_key).await;
    assert_eq!(selected_model.recorded_requests().len(), 1);
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert_roundtrips(&observed);
}

#[tokio::test]
async fn independent_prepared_resumes_race_at_marker_and_only_cas_winner_dispatches() {
    let selected = ScriptAgent::finished("worker");
    let first = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
    let second = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
    let store = InMemoryRoutedStore::new();
    let state_key = key("competing-resumes");
    let ready = store
        .save(
            state_key.clone(),
            None,
            checkpoint(RoutedCheckpointStatus::Ready),
        )
        .await
        .unwrap();
    let mut winner = first
        .resume_checkpointed_stream(&store, state_key.clone())
        .await
        .unwrap();
    let mut loser = second
        .resume_checkpointed_stream(&store, state_key.clone())
        .await
        .unwrap();
    assert_eq!(store.load(&state_key).await.unwrap(), Some(ready));
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    assert!(matches!(
        winner.next().await,
        Some(RoutedEvent::RouteStarted { .. })
    ));
    assert_status(&store, &state_key, RoutedCheckpointStatus::InFlight).await;
    let lost = loser.next().await.unwrap();
    assert!(
        matches!(lost, RoutedEvent::Error { error } if matches!(&error.cause, RoutedFailure::Store(reason) if reason.contains("conflict")))
    );
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 0);
    assert!(loser.next().await.is_none());
    let observed = collect_to_terminal(&mut winner).await;
    let Some(RoutedEvent::Finished { output }) = observed.last() else {
        panic!("CAS winner must finish")
    };
    let committed = assert_status(
        &store,
        &state_key,
        RoutedCheckpointStatus::Completed(output.message.clone()),
    )
    .await;
    assert_eq!(committed.revision, 3);
    assert_eq!(selected.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(selected.inputs.lock().unwrap().len(), 1);
    assert_eq!(selected.events_dropped.load(Ordering::SeqCst), 1);
    assert!(winner.next().await.is_none());
}
