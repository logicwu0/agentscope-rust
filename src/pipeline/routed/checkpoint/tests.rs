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
fn checkpoint_terminal_reading_and_serialization_preserve_original_values() {
    let mut message = Msg::new(
        "original-author",
        Role::System,
        [
            ThinkingBlock::new("private").into(),
            DataBlock::base64("YWJj", "application/octet-stream")
                .unwrap()
                .into(),
        ],
    );
    message.metadata.insert("opaque".into(), json!({"value":7}));
    let error = AgentError::Model(ModelError::new("original error"));
    for status in [
        RoutedCheckpointStatus::Ready,
        RoutedCheckpointStatus::InFlight,
        RoutedCheckpointStatus::Completed(message.clone()),
        RoutedCheckpointStatus::Failed(Box::new(error.clone())),
    ] {
        let checkpoint = checkpoint(status.clone());
        let value = serde_json::to_value(&checkpoint).unwrap();
        assert!(matches!(
            value["status"]["kind"].as_str(),
            Some("ready" | "in_flight" | "completed" | "failed")
        ));
        assert_eq!(
            serde_json::from_value::<RoutedCheckpoint>(value).unwrap(),
            checkpoint
        );
        match status {
            RoutedCheckpointStatus::Completed(_) => {
                let output = checkpoint.finished_result().unwrap().unwrap();
                assert_eq!(output.route, "selected");
                assert_eq!(output.agent_name, "worker");
                assert_eq!(output.message, message);
            }
            RoutedCheckpointStatus::Failed(_) => {
                let failed = checkpoint.finished_result().unwrap().unwrap_err();
                assert_eq!(failed.cause, RoutedFailure::Agent(Box::new(error.clone())));
                assert_eq!(
                    failed.source().unwrap().downcast_ref::<AgentError>(),
                    Some(&error)
                );
            }
            RoutedCheckpointStatus::Ready | RoutedCheckpointStatus::InFlight => {
                assert!(checkpoint.finished_result().is_none());
            }
        }
    }
    for invalid in [
        RoutedCheckpoint {
            version: ROUTED_CHECKPOINT_VERSION + 1,
            ..checkpoint(RoutedCheckpointStatus::Completed(message.clone()))
        },
        RoutedCheckpoint {
            route: " \n".into(),
            ..checkpoint(RoutedCheckpointStatus::Completed(message.clone()))
        },
        RoutedCheckpoint {
            agent_name: " \n".into(),
            ..checkpoint(RoutedCheckpointStatus::Completed(message))
        },
    ] {
        assert!(invalid.finished_result().is_none());
    }
}

#[tokio::test]
async fn memory_store_compare_and_swap_revisions_are_isolated_and_recoverable() {
    let store = InMemoryRoutedStore::new();
    let one = key("one");
    let two = key("two");
    assert!(store.load(&one).await.unwrap().is_none());
    assert!(
        store
            .save(
                one.clone(),
                Some(0),
                checkpoint(RoutedCheckpointStatus::Ready)
            )
            .await
            .is_err()
    );
    let created = store
        .save(one.clone(), None, checkpoint(RoutedCheckpointStatus::Ready))
        .await
        .unwrap();
    assert_eq!(created.revision, 1);
    assert!(
        store
            .save(
                one.clone(),
                None,
                checkpoint(RoutedCheckpointStatus::InFlight)
            )
            .await
            .is_err()
    );
    assert!(
        store
            .save(
                one.clone(),
                Some(0),
                checkpoint(RoutedCheckpointStatus::InFlight)
            )
            .await
            .is_err()
    );
    assert_eq!(store.load(&one).await.unwrap(), Some(created.clone()));
    let committed = store
        .save(
            one.clone(),
            Some(1),
            checkpoint(RoutedCheckpointStatus::Completed(reply("worker", "done"))),
        )
        .await
        .unwrap();
    assert_eq!(committed.revision, 2);
    assert_eq!(created.checkpoint.status, RoutedCheckpointStatus::Ready);
    assert!(store.load(&two).await.unwrap().is_none());
    assert_eq!(
        store
            .save(two.clone(), None, checkpoint(RoutedCheckpointStatus::Ready))
            .await
            .unwrap()
            .revision,
        1
    );
    assert_eq!(store.load(&one).await.unwrap(), Some(committed.clone()));
    assert_eq!(
        serde_json::from_str::<RoutedRecord>(&serde_json::to_string(&committed).unwrap()).unwrap(),
        committed
    );
    let exhausted_key = key("exhausted");
    let exhausted = RoutedRecord {
        revision: u64::MAX,
        checkpoint: checkpoint(RoutedCheckpointStatus::Ready),
    };
    store
        .records
        .lock()
        .unwrap()
        .insert(exhausted_key.clone(), exhausted.clone());
    let overflow = store
        .save(
            exhausted_key.clone(),
            Some(u64::MAX),
            checkpoint(RoutedCheckpointStatus::InFlight),
        )
        .await
        .unwrap_err();
    assert!(overflow.message.contains("overflow"));
    assert_eq!(store.load(&exhausted_key).await.unwrap(), Some(exhausted));
}

#[tokio::test]
async fn checkpointed_run_is_lazy_fail_closed_and_commits_before_invocation_and_return() {
    let store = ControlledStore::plain();
    let mut input = Msg::new(
        "input-author",
        Role::System,
        [DataBlock::url("https://example.com/input.png", "image/png")
            .unwrap()
            .into()],
    );
    input.metadata.insert("secret".into(), json!("kept local"));
    let mut original = reply("not-the-configured-name", "original reply");
    original
        .metadata
        .insert("private".into(), json!({"value":1}));
    let selected = ScriptAgent::witnessed("worker", Ok(original.clone()), &store.history);
    let other = ScriptAgent::new("other", Ok(reply("other", "unused")));
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let state_key = key("successful");
    drop(pipeline.run_checkpointed(&store, state_key.clone(), "selected", input.clone()));
    assert_eq!(store.saves.load(Ordering::SeqCst), 0);
    for route in ["Selected", "selected ", " selected", "missing", ""] {
        let error = pipeline
            .run_checkpointed(&store, state_key.clone(), route, input.clone())
            .await
            .unwrap_err();
        assert_eq!(error.route, route);
        assert_eq!(error.agent_name, None);
        assert_eq!(error.cause, RoutedFailure::UnknownRoute);
    }
    assert_eq!(store.loads.load(Ordering::SeqCst), 0);
    assert_eq!(store.saves.load(Ordering::SeqCst), 0);
    let output = pipeline
        .run_checkpointed(&store, state_key.clone(), "selected", input.clone())
        .await
        .unwrap();
    assert_eq!(output.message, original);
    assert_eq!(output.agent_name, "worker");
    assert_eq!(*selected.inputs.lock().unwrap(), vec![input]);
    assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
    assert_eq!(other.calls.load(Ordering::SeqCst), 0);
    let records = store.history.lock().unwrap().clone();
    assert_eq!(
        records
            .iter()
            .map(|record| record.revision)
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert_eq!(
        records.first().unwrap().checkpoint.status,
        RoutedCheckpointStatus::Ready
    );
    assert_eq!(
        records.get(1).unwrap().checkpoint.status,
        RoutedCheckpointStatus::InFlight
    );
    let stored = store.inner.load(&state_key).await.unwrap().unwrap();
    assert_eq!(
        stored.checkpoint.finished_result().unwrap().unwrap(),
        output
    );
    assert!(matches!(
        pipeline
            .run_checkpointed(&store, state_key, "selected", Msg::user("existing key"))
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Store(_)
    ));
    assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_waits_for_store_ack_and_respects_ready_and_inflight_boundaries() {
    for (point, expected_status) in [
        (SavePoint::Initial, RoutedCheckpointStatus::Ready),
        (SavePoint::Marker, RoutedCheckpointStatus::InFlight),
    ] {
        let gate = Arc::new(Gate::default());
        let store = ControlledStore::gated(point, &gate);
        let original = reply("worker", "done");
        let selected = ScriptAgent::new("worker", Ok(original.clone()));
        let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
        let state_key = key("boundary");
        let mut run =
            pipeline.run_checkpointed(&store, state_key.clone(), "selected", Msg::user("original"));
        assert!(run.as_mut().now_or_never().is_none());
        let pending = store.inner.load(&state_key).await.unwrap().unwrap();
        assert_eq!(pending.checkpoint.status, expected_status);
        assert_eq!(selected.calls.load(Ordering::SeqCst), 0);
        pipeline.interrupt_handle().interrupt();
        assert!(
            run.as_mut().now_or_never().is_none(),
            "store await must not be cancelled"
        );
        assert_eq!(gate.dropped.load(Ordering::SeqCst), 0);
        gate.release();
        let error = run.await.unwrap_err();
        assert_eq!(error.cause, RoutedFailure::Interrupted);
        assert_eq!(error.route, "selected");
        assert_eq!(error.agent_name.as_deref(), Some("worker"));
        assert_eq!(selected.calls.load(Ordering::SeqCst), 0);
        assert_eq!(store.inner.load(&state_key).await.unwrap(), Some(pending));
        let resumed = pipeline
            .resume_checkpointed(&store, state_key.clone())
            .await;
        if expected_status == RoutedCheckpointStatus::Ready {
            assert_eq!(resumed.unwrap().message, original);
            assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
        } else {
            assert!(matches!(
                resumed.unwrap_err().cause,
                RoutedFailure::UnsafeResume(_)
            ));
            assert_eq!(selected.calls.load(Ordering::SeqCst), 0);
        }
        assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn checkpointed_and_plain_operations_share_lock_before_any_store_io() {
    let gate = Arc::new(Gate::default());
    let store = ControlledStore::gated(SavePoint::Initial, &gate);
    let untouched_store = ControlledStore::plain();
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "done")));
    let other = ScriptAgent::new("other", Ok(reply("other", "done")));
    let pipeline = RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap();
    let clone = pipeline.clone();
    let state_key = key("held");
    let mut run =
        pipeline.run_checkpointed(&store, state_key.clone(), "selected", Msg::user("held"));
    assert!(run.as_mut().now_or_never().is_none());
    assert!(matches!(
        clone
            .run("other", Msg::user("busy"))
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Busy
    ));
    let Err(stream_error) = clone.stream("other", Msg::user("busy")).await else {
        panic!("expected Busy")
    };
    assert_eq!(stream_error.cause, RoutedFailure::Busy);
    assert_eq!(stream_error.agent_name.as_deref(), Some("other"));
    assert!(matches!(
        clone
            .run_checkpointed(&untouched_store, key("new"), "other", Msg::user("busy"))
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Busy
    ));
    assert!(matches!(
        clone
            .resume_checkpointed(&untouched_store, key("new"))
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::Busy
    ));
    let unknown = clone
        .run_checkpointed(
            &untouched_store,
            key("new"),
            "unknown",
            Msg::user("unknown wins"),
        )
        .await
        .unwrap_err();
    assert_eq!(unknown.cause, RoutedFailure::UnknownRoute);
    assert_eq!(untouched_store.loads.load(Ordering::SeqCst), 0);
    assert_eq!(untouched_store.saves.load(Ordering::SeqCst), 0);
    assert_eq!(selected.calls.load(Ordering::SeqCst), 0);
    assert_eq!(other.calls.load(Ordering::SeqCst), 0);
    drop(run);
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        store
            .inner
            .load(&state_key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .status,
        RoutedCheckpointStatus::Ready
    );
    gate.release();
    clone.resume_checkpointed(&store, state_key).await.unwrap();
    assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn active_drop_keeps_inflight_and_resumed_poll_cancellation_never_repolls_child() {
    for interrupt in [false, true] {
        let gate = Arc::new(Gate::default());
        let selected = ScriptAgent::gated("worker", &gate);
        let other = ScriptAgent::new("other", Ok(reply("other", "done")));
        let pipeline = RoutedPipeline::new(vec![
            ("selected".into(), selected.clone()),
            ("other".into(), other),
        ])
        .unwrap();
        let store = InMemoryRoutedStore::new();
        let state_key = key("inflight");
        let mut run =
            pipeline.run_checkpointed(&store, state_key.clone(), "selected", Msg::user("original"));
        assert!(run.as_mut().now_or_never().is_none());
        let inflight = store.load(&state_key).await.unwrap().unwrap();
        assert_eq!(inflight.revision, 2);
        assert_eq!(inflight.checkpoint.status, RoutedCheckpointStatus::InFlight);
        assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
        assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
        if interrupt {
            gate.release();
            pipeline.interrupt_handle().interrupt();
            assert_eq!(run.await.unwrap_err().cause, RoutedFailure::Interrupted);
        } else {
            drop(run);
            gate.release();
        }
        assert_eq!(gate.polls.load(Ordering::SeqCst), 1);
        assert_eq!(gate.effects.load(Ordering::SeqCst), 0);
        assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(selected.child_handle_reads.load(Ordering::SeqCst), 0);
        assert_eq!(store.load(&state_key).await.unwrap(), Some(inflight));
        assert!(matches!(
            pipeline
                .resume_checkpointed(&store, state_key.clone())
                .await
                .unwrap_err()
                .cause,
            RoutedFailure::UnsafeResume(_)
        ));
        pipeline
            .clone()
            .run("other", Msg::user("fresh route"))
            .await
            .unwrap();
        assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn marker_and_terminal_save_rejection_never_claims_uncommitted_result_or_replays() {
    for (point, expected_status, calls) in [
        (SavePoint::Initial, None, 0),
        (SavePoint::Marker, Some(RoutedCheckpointStatus::Ready), 0),
        (
            SavePoint::Terminal,
            Some(RoutedCheckpointStatus::InFlight),
            1,
        ),
    ] {
        let store = ControlledStore::reject(point);
        let selected = ScriptAgent::new(
            "worker",
            Ok(reply("worker", "observed but not necessarily committed")),
        );
        let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
        let state_key = key("rejected");
        let error = pipeline
            .run_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
            .await
            .unwrap_err();
        assert!(matches!(error.cause, RoutedFailure::Store(reason) if reason.contains("rejected")));
        assert_eq!(selected.calls.load(Ordering::SeqCst), calls);
        let record = store.inner.load(&state_key).await.unwrap();
        assert_eq!(
            record
                .as_ref()
                .map(|record| record.checkpoint.status.clone()),
            expected_status
        );
        assert!(
            record
                .as_ref()
                .and_then(|record| record.checkpoint.finished_result())
                .is_none()
        );
        if calls == 1 {
            assert!(matches!(
                pipeline
                    .resume_checkpointed(&store, state_key)
                    .await
                    .unwrap_err()
                    .cause,
                RoutedFailure::UnsafeResume(_)
            ));
            assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
        }
    }
    let store = ControlledStore::ambiguous(SavePoint::Terminal);
    let original = reply("worker", "committed despite lost ack");
    let selected = ScriptAgent::new("worker", Ok(original.clone()));
    let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
    let state_key = key("ambiguous");
    assert!(
        matches!(pipeline.run_checkpointed(&store, state_key.clone(), "selected", Msg::user("task")).await.unwrap_err().cause, RoutedFailure::Store(reason) if reason.contains("acknowledgement"))
    );
    let record = store.inner.load(&state_key).await.unwrap().unwrap();
    assert_eq!(record.revision, 3);
    assert_eq!(
        record
            .checkpoint
            .finished_result()
            .unwrap()
            .unwrap()
            .message,
        original
    );
    assert!(matches!(
        pipeline
            .resume_checkpointed(&store, state_key)
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::UnsafeResume(_)
    ));
    assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn observed_agent_result_is_committed_even_when_terminal_ack_interrupts_pipeline() {
    let original_error = AgentError::Model(ModelError::new("original model failure"));
    for result in [
        Ok(reply("original-author", "original message")),
        Err(original_error),
    ] {
        let store = ControlledStore::signal(SavePoint::Terminal);
        let selected = ScriptAgent::new("worker", result.clone());
        let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
        *store.interrupt.lock().unwrap() = Some(pipeline.interrupt_handle());
        let state_key = key("terminal");
        let observed = pipeline
            .run_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
            .await;
        match result {
            Ok(message) => assert_eq!(observed.as_ref().unwrap().message, message),
            Err(error) => assert_eq!(
                observed.as_ref().unwrap_err().cause,
                RoutedFailure::Agent(Box::new(error))
            ),
        }
        let committed = store.inner.load(&state_key).await.unwrap().unwrap();
        assert_eq!(committed.revision, 3);
        assert_eq!(committed.checkpoint.finished_result().unwrap(), observed);
        assert!(matches!(
            pipeline
                .resume_checkpointed(&store, state_key)
                .await
                .unwrap_err()
                .cause,
            RoutedFailure::UnsafeResume(_)
        ));
        assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn resume_rejects_missing_malformed_incompatible_and_terminal_records_without_dispatch() {
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "done")));
    let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
    let valid = RoutedRecord {
        revision: 1,
        checkpoint: checkpoint(RoutedCheckpointStatus::Ready),
    };
    let mut records = vec![
        RoutedRecord {
            revision: 0,
            ..valid.clone()
        },
        RoutedRecord {
            checkpoint: RoutedCheckpoint {
                version: ROUTED_CHECKPOINT_VERSION + 1,
                ..valid.checkpoint.clone()
            },
            ..valid.clone()
        },
        RoutedRecord {
            checkpoint: RoutedCheckpoint {
                route: " \n".into(),
                ..valid.checkpoint.clone()
            },
            ..valid.clone()
        },
        RoutedRecord {
            checkpoint: RoutedCheckpoint {
                agent_name: " \n".into(),
                ..valid.checkpoint.clone()
            },
            ..valid.clone()
        },
        RoutedRecord {
            checkpoint: RoutedCheckpoint {
                route: "Selected".into(),
                ..valid.checkpoint.clone()
            },
            ..valid.clone()
        },
        RoutedRecord {
            checkpoint: RoutedCheckpoint {
                agent_name: "changed-name".into(),
                ..valid.checkpoint.clone()
            },
            ..valid.clone()
        },
    ];
    records.extend(
        [
            RoutedCheckpointStatus::InFlight,
            RoutedCheckpointStatus::Completed(reply("worker", "finished")),
            RoutedCheckpointStatus::Failed(Box::new(AgentError::Interrupted)),
        ]
        .into_iter()
        .map(|status| RoutedRecord {
            revision: 1,
            checkpoint: checkpoint(status),
        }),
    );
    for record in records {
        let store = StaticStore::with_record(record.clone());
        let error = pipeline
            .resume_checkpointed(&store, key("invalid"))
            .await
            .unwrap_err();
        assert!(matches!(error.cause, RoutedFailure::UnsafeResume(_)));
        assert_eq!(error.route, record.checkpoint.route);
        assert_eq!(error.agent_name, Some(record.checkpoint.agent_name));
        assert_eq!(store.saves.load(Ordering::SeqCst), 0);
    }
    let mut store = StaticStore::with_record(valid);
    store.record = None;
    let missing = pipeline
        .resume_checkpointed(&store, key("missing"))
        .await
        .unwrap_err();
    assert!(matches!(missing.cause, RoutedFailure::UnsafeResume(_)));
    assert_eq!(missing.route, "");
    assert_eq!(missing.agent_name, None);
    store.error = Some(PipelineStoreError::new("load unavailable"));
    let unavailable = pipeline
        .resume_checkpointed(&store, key("load-error"))
        .await
        .unwrap_err();
    assert_eq!(
        unavailable.cause,
        RoutedFailure::Store("load unavailable".into())
    );
    assert_eq!(unavailable.route, "");
    assert_eq!(unavailable.agent_name, None);
    assert_eq!(selected.calls.load(Ordering::SeqCst), 0);
    let mut missing_status =
        serde_json::to_value(checkpoint(RoutedCheckpointStatus::Ready)).unwrap();
    missing_status.as_object_mut().unwrap().remove("status");
    assert!(serde_json::from_value::<RoutedCheckpoint>(missing_status).is_err());
}

#[tokio::test]
async fn ready_resume_uses_saved_selection_input_and_allows_only_unrelated_route_changes() {
    let store = InMemoryRoutedStore::new();
    let state_key = key("ready");
    let input = Msg::new(
        "saved-author",
        Role::System,
        [
            ThinkingBlock::new("saved reasoning").into(),
            DataBlock::url("https://example.com/saved.png", "image/png")
                .unwrap()
                .into(),
        ],
    );
    let ready = RoutedCheckpoint {
        input: input.clone(),
        ..checkpoint(RoutedCheckpointStatus::Ready)
    };
    store
        .save(state_key.clone(), None, ready.clone())
        .await
        .unwrap();
    for entries in [
        vec![(
            "other".into(),
            ScriptAgent::new("other", Ok(reply("other", "unused"))) as Arc<dyn Agent>,
        )],
        vec![(
            "selected".into(),
            ScriptAgent::new("changed-name", Ok(reply("changed-name", "unused"))) as Arc<dyn Agent>,
        )],
    ] {
        let incompatible = RoutedPipeline::new(entries).unwrap();
        assert!(matches!(
            incompatible
                .resume_checkpointed(&store, state_key.clone())
                .await
                .unwrap_err()
                .cause,
            RoutedFailure::UnsafeResume(_)
        ));
        assert_eq!(
            store.load(&state_key).await.unwrap().unwrap().checkpoint,
            ready
        );
    }
    let original = reply("original-author", "original answer");
    let selected = ScriptAgent::new("worker", Ok(original.clone()));
    let other = ScriptAgent::new(
        "new-unrelated-name",
        Ok(reply("new-unrelated-name", "unused")),
    );
    let compatible = RoutedPipeline::new(vec![
        ("new-route".into(), other.clone()),
        ("selected".into(), selected.clone()),
        ("alias".into(), selected.clone()),
    ])
    .unwrap();
    compatible.interrupt_handle().interrupt();
    let output = compatible
        .resume_checkpointed(&store, state_key.clone())
        .await
        .unwrap();
    assert_eq!(output.route, "selected");
    assert_eq!(output.agent_name, "worker");
    assert_eq!(output.message, original);
    assert_eq!(*selected.inputs.lock().unwrap(), vec![input]);
    assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
    assert_eq!(other.calls.load(Ordering::SeqCst), 0);
    assert_eq!(store.load(&state_key).await.unwrap().unwrap().revision, 3);
}

#[tokio::test]
async fn explicit_reconciliation_validates_evidence_revision_and_status_without_agent_execution() {
    let selected = ScriptAgent::new("worker", Ok(reply("worker", "must not be invoked")));
    let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
    for initial in [
        RoutedCheckpointStatus::InFlight,
        RoutedCheckpointStatus::Failed(Box::new(AgentError::Interrupted)),
    ] {
        let store = InMemoryRoutedStore::new();
        let state_key = key("reconcile");
        let record = store
            .save(state_key.clone(), None, checkpoint(initial))
            .await
            .unwrap();
        let mut evidence = reply("worker", "externally verified");
        evidence.content.push(
            DataBlock::base64("YWJj", "application/octet-stream")
                .unwrap()
                .into(),
        );
        evidence
            .metadata
            .insert("verification".into(), json!({"external": true}));
        let wrong_role = Msg::new("worker", Role::User, [ContentBlock::from("not assistant")]);
        for (revision, candidate) in [
            (0, evidence.clone()),
            (2, evidence.clone()),
            (1, wrong_role),
            (1, reply("other", "wrong author")),
        ] {
            assert!(matches!(
                pipeline
                    .reconcile_checkpointed(&store, state_key.clone(), revision, candidate)
                    .await
                    .unwrap_err()
                    .cause,
                RoutedFailure::UnsafeResume(_)
            ));
            assert_eq!(store.load(&state_key).await.unwrap(), Some(record.clone()));
        }
        let completed = pipeline
            .reconcile_checkpointed(&store, state_key.clone(), 1, evidence.clone())
            .await
            .unwrap();
        assert_eq!(completed.revision, 2);
        assert_eq!(completed.checkpoint.input, record.checkpoint.input);
        assert_eq!(
            completed
                .checkpoint
                .finished_result()
                .unwrap()
                .unwrap()
                .message,
            evidence
        );
        for revision in [1, 2] {
            assert!(matches!(
                pipeline
                    .reconcile_checkpointed(
                        &store,
                        state_key.clone(),
                        revision,
                        reply("worker", "duplicate")
                    )
                    .await
                    .unwrap_err()
                    .cause,
                RoutedFailure::UnsafeResume(_)
            ));
        }
        assert!(matches!(
            pipeline
                .resume_checkpointed(&store, state_key.clone())
                .await
                .unwrap_err()
                .cause,
            RoutedFailure::UnsafeResume(_)
        ));
        assert_eq!(store.load(&state_key).await.unwrap(), Some(completed));
    }
    assert_reconciliation_fences_ready_and_concurrent_write(&pipeline).await;
    assert_eq!(selected.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn confirmation_and_in_doubt_errors_are_durable_without_approval_retry_or_restoration() {
    let state_store = Arc::new(InMemoryStateStore::new());
    let state_key = key("agent-session");
    let (worker, model, tool) = confirming_agent(state_store.clone(), state_key.clone());
    let pipeline = RoutedPipeline::new(vec![("selected".into(), worker.clone())]).unwrap();
    let store = InMemoryRoutedStore::new();
    let confirmation_key = key("confirmation");
    let error = pipeline
        .run_checkpointed(
            &store,
            confirmation_key.clone(),
            "selected",
            Msg::user("write"),
        )
        .await
        .unwrap_err();
    let RoutedFailure::Agent(original) = error.cause else {
        panic!("expected original agent error")
    };
    let AgentError::ToolConfirmationRequired {
        checkpoint: pending,
    } = original.as_ref()
    else {
        panic!("expected confirmation")
    };
    assert_eq!(
        state_store
            .load(&state_key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_calls(),
        Some(pending)
    );
    assert!(tool.recorded_invocations().is_empty());
    let recorded = store.load(&confirmation_key).await.unwrap().unwrap();
    assert_eq!(
        recorded.checkpoint.status,
        RoutedCheckpointStatus::Failed(original.clone())
    );
    assert_resume_rejected(&pipeline, &store, confirmation_key.clone()).await;
    let uncertain = worker
        .resume_tool_calls(pending.reply_id(), vec![ToolConfirmation::approve("call")])
        .await
        .unwrap_err();
    assert!(matches!(uncertain, AgentError::ToolExecutionInDoubt { .. }));
    let uncertain_key = key("uncertain");
    let observed = pipeline
        .run_checkpointed(
            &store,
            uncertain_key.clone(),
            "selected",
            Msg::user("must not retry"),
        )
        .await
        .unwrap_err();
    assert_eq!(
        observed.cause,
        RoutedFailure::Agent(Box::new(uncertain.clone()))
    );
    let state_before = state_store.load(&state_key).await.unwrap().unwrap();
    let uncertain_record = store.load(&uncertain_key).await.unwrap().unwrap();
    assert_eq!(
        uncertain_record.checkpoint.status,
        RoutedCheckpointStatus::Failed(Box::new(uncertain))
    );
    assert_resume_rejected(&pipeline, &store, uncertain_key.clone()).await;
    pipeline
        .reconcile_checkpointed(
            &store,
            uncertain_key.clone(),
            uncertain_record.revision,
            reply("worker", "externally reconciled only"),
        )
        .await
        .unwrap();
    assert_resume_rejected(&pipeline, &store, uncertain_key).await;
    assert_eq!(
        state_store.load(&state_key).await.unwrap().unwrap().state(),
        state_before.state()
    );
    assert_eq!(model.recorded_requests().len(), 1);
    assert_eq!(tool.recorded_invocations().len(), 1);
}

#[tokio::test]
async fn incompatible_store_acknowledgements_are_fenced_before_dispatch_or_success() {
    let original = reply("worker", "done");
    for (point, alter_checkpoint, expected_status, calls) in [
        (SavePoint::Initial, false, RoutedCheckpointStatus::Ready, 0),
        (SavePoint::Marker, true, RoutedCheckpointStatus::InFlight, 0),
        (
            SavePoint::Terminal,
            false,
            RoutedCheckpointStatus::Completed(original.clone()),
            1,
        ),
        (
            SavePoint::Terminal,
            true,
            RoutedCheckpointStatus::Completed(original.clone()),
            1,
        ),
    ] {
        let store = ControlledStore::bad_ack(point, alter_checkpoint);
        let selected = ScriptAgent::new("worker", Ok(original.clone()));
        let pipeline = RoutedPipeline::new(vec![("selected".into(), selected.clone())]).unwrap();
        let state_key = key("bad-ack");
        let error = pipeline
            .run_checkpointed(&store, state_key.clone(), "selected", Msg::user("task"))
            .await
            .unwrap_err();
        assert!(
            matches!(error.cause, RoutedFailure::Store(reason) if reason.contains("acknowledgement"))
        );
        assert_eq!(selected.calls.load(Ordering::SeqCst), calls);
        let committed = store.inner.load(&state_key).await.unwrap().unwrap();
        assert_eq!(committed.checkpoint.status, expected_status);
        assert!(
            !committed
                .checkpoint
                .input
                .metadata
                .contains_key("corrupted")
        );
        if expected_status == RoutedCheckpointStatus::Ready {
            pipeline
                .resume_checkpointed(&store.inner, state_key)
                .await
                .unwrap();
            assert_eq!(selected.calls.load(Ordering::SeqCst), 1);
        } else {
            assert!(matches!(
                pipeline
                    .resume_checkpointed(&store.inner, state_key)
                    .await
                    .unwrap_err()
                    .cause,
                RoutedFailure::UnsafeResume(_)
            ));
            assert_eq!(selected.calls.load(Ordering::SeqCst), calls);
        }
    }
}
