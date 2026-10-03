use super::*;
use crate::*;
use futures_util::StreamExt;
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};
#[path = "tests_support.rs"]
mod support;
use support::*;

#[tokio::test]
async fn preparation_is_lazy_shares_the_run_lock_and_unpolled_drop_can_resume_ready_work() {
    let a_model = model("completed");
    let a = agent("one", a_model.clone());
    let pipeline = ParallelPipeline::new(vec![a.clone()], 1).unwrap();
    let clone = pipeline.clone();
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "reserved").unwrap();
    drop(pipeline.stream_checkpointed(&store, key.clone(), Msg::user("unpolled preparation")));
    assert!(store.load(&key).await.unwrap().is_none());
    let reserved = pipeline
        .stream_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap();
    let ready = store.load(&key).await.unwrap().unwrap();
    assert_eq!(ready.revision, 1);
    assert_eq!(
        ready.checkpoint.branches,
        vec![ParallelBranchCheckpoint::Ready]
    );
    assert!(a_model.recorded_requests().is_empty());
    assert!(a.snapshot().await.unwrap().messages().is_empty());
    assert!(matches!(
        clone.run(Msg::user("busy run")).await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    assert!(matches!(
        clone.stream(Msg::user("busy stream")).await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    assert!(matches!(
        clone.resume_checkpointed_stream(&store, key.clone()).await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    assert!(matches!(
        clone
            .run_checkpointed(
                &store,
                StateKey::new("test", "busy-key").unwrap(),
                Msg::user("busy durable run")
            )
            .await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    assert_eq!(store.load(&key).await.unwrap().unwrap(), ready);
    drop(reserved);
    let resumed = clone
        .resume_checkpointed_stream(&store, key.clone())
        .await
        .unwrap();
    assert!(a_model.recorded_requests().is_empty());
    let observed = resumed.collect::<Vec<_>>().await;
    let Some(ParallelEvent::Finished { output }) = observed.last() else {
        panic!("expected safe Ready resume")
    };
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .finished_result(),
        Some(Ok(output.clone()))
    );
    assert_eq!(a_model.recorded_requests().len(), 1);
    assert!(matches!(
        pipeline
            .stream_checkpointed(&store, key, Msg::user("duplicate"))
            .await,
        Err(ParallelError {
            cause: ParallelFailure::Store(_),
            ..
        })
    ));
}

#[tokio::test]
async fn marker_precedes_start_and_terminal_save_precedes_branch_completion() {
    let a_model = model("original reply");
    let pipeline = ParallelPipeline::new(vec![agent("one", a_model.clone())], 1).unwrap();
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "boundary-order").unwrap();
    let mut events = pipeline
        .stream_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap();
    let mut observed = vec![events.next().await.unwrap()];
    assert!(
        matches!(observed[0], ParallelEvent::BranchStarted { branch: 1, ref agent_name } if agent_name == "one")
    );
    assert_eq!(
        store.load(&key).await.unwrap().unwrap().checkpoint.branches,
        vec![ParallelBranchCheckpoint::InFlight]
    );
    assert!(a_model.recorded_requests().is_empty());
    let message = loop {
        let event = events.next().await.unwrap();
        let reply = match &event {
            ParallelEvent::Agent {
                event: AgentEvent::Finished { message, .. },
                ..
            } => Some(message.clone()),
            _ => None,
        };
        observed.push(event);
        if let Some(reply) = reply {
            break reply;
        }
    };
    assert_eq!(
        store.load(&key).await.unwrap().unwrap().checkpoint.branches,
        vec![ParallelBranchCheckpoint::InFlight]
    );
    let completed = events.next().await.unwrap();
    assert!(
        matches!(&completed, ParallelEvent::BranchFinished { result } if result.outcome == ParallelBranchOutcome::Completed(message.clone()))
    );
    observed.push(completed);
    let committed = store.load(&key).await.unwrap().unwrap();
    assert_eq!(
        committed.checkpoint.branches,
        vec![ParallelBranchCheckpoint::Completed(message)]
    );
    let terminal = events.next().await.unwrap();
    let ParallelEvent::Finished { output } = &terminal else {
        panic!("expected terminal success")
    };
    assert_eq!(
        committed.checkpoint.finished_result(),
        Some(Ok(output.clone()))
    );
    observed.push(terminal);
    assert!(events.next().await.is_none());
    assert_eq!(terminal_count(&observed), 1);
    assert!(observed.iter().any(|event| matches!(event, ParallelEvent::Agent { branch: 1, event: AgentEvent::TextDelta { step: 1, block_id, delta }, .. } if block_id == "shared-text" && delta == "original reply")));
    assert_roundtrips(&observed);
}

#[tokio::test]
async fn bounded_streams_witness_durable_markers_and_return_committed_results_in_order() {
    let store = Arc::new(InMemoryParallelStore::new());
    let key = StateKey::new("test", "bounded-stream").unwrap();
    let activity = Arc::new(Activity::default());
    let models = ["a", "b", "c"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            GateModel::new(name, &activity, Some((store.clone(), key.clone(), index)))
        })
        .collect::<Vec<_>>();
    let pipeline = ParallelPipeline::new(
        ["a", "b", "c"]
            .into_iter()
            .zip(&models)
            .map(|(name, model)| agent(name, model.clone()) as Arc<dyn Agent>)
            .collect(),
        2,
    )
    .unwrap();
    let mut input = Msg::user("same input");
    input.metadata.insert("shared".into(), json!({"value": 7}));
    let mut events = pipeline
        .stream_checkpointed(store.as_ref(), key.clone(), input.clone())
        .await
        .unwrap();
    let mut observed = Vec::new();
    assert!(!drain_ready(&mut events, &mut observed));
    assert_eq!(activity.active.load(Ordering::SeqCst), 2);
    assert_eq!(
        store.load(&key).await.unwrap().unwrap().checkpoint.branches,
        vec![
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::Ready
        ]
    );
    assert!(models[2].requests().is_empty());
    models[1].release();
    assert!(!drain_ready(&mut events, &mut observed));
    let boundary = store.load(&key).await.unwrap().unwrap();
    assert!(matches!(
        boundary.checkpoint.branches[1],
        ParallelBranchCheckpoint::Completed(_)
    ));
    assert_eq!(
        boundary.checkpoint.branches[2],
        ParallelBranchCheckpoint::InFlight
    );
    assert_eq!(models[2].requests().len(), 1);
    models[2].release();
    assert!(!drain_ready(&mut events, &mut observed));
    models[0].release();
    observed.extend(events.collect::<Vec<_>>().await);
    assert_eq!(activity.peak.load(Ordering::SeqCst), 2);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert_eq!(
        *activity.completion_order.lock().unwrap(),
        vec!["b", "c", "a"]
    );
    let Some(ParallelEvent::Finished { output }) = observed.last() else {
        panic!("expected success")
    };
    for (index, branch) in output.branches.iter().enumerate() {
        assert_eq!(branch.branch, index + 1);
        assert_eq!(branch.agent_name, ["a", "b", "c"][index]);
        assert_eq!(models[index].requests()[0].messages.last(), Some(&input));
    }
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .finished_result(),
        Some(Ok(output.clone()))
    );
    assert_eq!(terminal_count(&observed), 1);
}

#[tokio::test]
async fn preparation_and_marker_save_failures_do_not_invoke_agents_or_announce_uncommitted_start() {
    let a_model = model("must not run");
    let pipeline = ParallelPipeline::new(vec![agent("one", a_model.clone())], 1).unwrap();
    let key = StateKey::new("test", "marker-failure").unwrap();
    let initial = ControlledStore::reject(SavePoint::Initial);
    assert!(matches!(
        pipeline
            .stream_checkpointed(&initial, key.clone(), Msg::user("task"))
            .await,
        Err(ParallelError {
            cause: ParallelFailure::Store(_),
            ..
        })
    ));
    assert!(initial.load(&key).await.unwrap().is_none());
    let marker = ControlledStore::reject(SavePoint::Marker(0));
    let observed = pipeline
        .stream_checkpointed(&marker, key.clone(), Msg::user("task"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let error = error(&observed);
    assert!(matches!(error.cause, ParallelFailure::Store(_)));
    assert_eq!(observed.len(), 1);
    assert_eq!(error.branches[0].outcome, ParallelBranchOutcome::NotStarted);
    assert_eq!(
        marker
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .branches,
        vec![ParallelBranchCheckpoint::Ready]
    );
    assert!(a_model.recorded_requests().is_empty());
    let ambiguous = ControlledStore::lose_acknowledgement(SavePoint::Terminal(0));
    let observed = pipeline
        .stream_checkpointed(&ambiguous, key.clone(), Msg::user("task"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let ambiguous_error = support::error(&observed);
    assert!(matches!(ambiguous_error.cause, ParallelFailure::Store(_)));
    assert_eq!(
        ambiguous_error.branches[0].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert!(
        !observed
            .iter()
            .any(|event| matches!(event, ParallelEvent::BranchFinished { .. }))
    );
    let actual = ambiguous.load(&key).await.unwrap().unwrap();
    assert!(
        matches!(actual.checkpoint.finished_result(), Some(Ok(_))),
        "reload reveals the committed terminal record despite the lost acknowledgement"
    );
    assert!(
        pipeline
            .resume_checkpointed_stream(&ambiguous, key)
            .await
            .is_err()
    );
    assert_eq!(a_model.recorded_requests().len(), 1);
}

#[tokio::test]
async fn rejected_terminal_save_drops_siblings_before_error_and_does_not_claim_completion() {
    let store = ControlledStore::reject(SavePoint::Terminal(1));
    let key = StateKey::new("test", "terminal-failure").unwrap();
    let activity = Arc::new(Activity::default());
    let pending = GateModel::new(
        "pending",
        &activity,
        Some((store.inner.clone(), key.clone(), 0)),
    );
    let completed = model("uncommitted reply");
    let queued = model("must not run");
    let pipeline = ParallelPipeline::new(
        vec![
            agent("pending", pending.clone()),
            agent("completed", completed.clone()),
            agent("queued", queued.clone()),
        ],
        2,
    )
    .unwrap();
    let mut events = pipeline
        .stream_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap();
    let mut observed = Vec::new();
    loop {
        let event = events.next().await.unwrap();
        let terminal = matches!(event, ParallelEvent::Error { .. });
        observed.push(event);
        if terminal {
            break;
        }
    }
    let error = error(&observed);
    assert!(matches!(error.cause, ParallelFailure::Store(_)));
    assert_eq!(
        error.branches[0].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert_eq!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert_eq!(error.branches[2].outcome, ParallelBranchOutcome::NotStarted);
    assert_eq!(
        activity.active.load(Ordering::SeqCst),
        0,
        "sibling must be dropped before Error is visible"
    );
    assert_eq!(pending.dropped.load(Ordering::SeqCst), 1);
    assert!(observed.iter().any(|event| matches!(
        event,
        ParallelEvent::Agent {
            branch: 2,
            event: AgentEvent::Finished { .. },
            ..
        }
    )));
    assert!(
        !observed
            .iter()
            .any(|event| matches!(event, ParallelEvent::BranchFinished { .. }))
    );
    assert_eq!(completed.recorded_requests().len(), 1);
    assert!(queued.recorded_requests().is_empty());
    assert_eq!(
        store.load(&key).await.unwrap().unwrap().checkpoint.branches,
        vec![
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::Ready
        ]
    );
    assert!(events.next().await.is_none());
    assert!(
        pipeline
            .resume_checkpointed_stream(&store, key)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn cancellation_before_poll_or_after_start_reports_only_committed_progress() {
    for announce in [false, true] {
        let a_model = model("must not run");
        let pipeline = ParallelPipeline::new(vec![agent("one", a_model.clone())], 1).unwrap();
        let store = InMemoryParallelStore::new();
        let key = StateKey::new("test", "cancel-boundary").unwrap();
        let mut events = pipeline
            .stream_checkpointed(&store, key.clone(), Msg::user("task"))
            .await
            .unwrap();
        if announce {
            assert!(matches!(
                events.next().await,
                Some(ParallelEvent::BranchStarted { .. })
            ));
        }
        pipeline.interrupt_handle().interrupt();
        let observed = events.collect::<Vec<_>>().await;
        let error = error(&observed);
        assert_eq!(error.cause, ParallelFailure::Interrupted);
        let expected = if announce {
            ParallelBranchCheckpoint::InFlight
        } else {
            ParallelBranchCheckpoint::Ready
        };
        assert_eq!(
            store.load(&key).await.unwrap().unwrap().checkpoint.branches,
            vec![expected]
        );
        assert_eq!(
            error.branches[0].outcome,
            if announce {
                ParallelBranchOutcome::Interrupted
            } else {
                ParallelBranchOutcome::NotStarted
            }
        );
        assert!(a_model.recorded_requests().is_empty());
        assert_eq!(observed.len(), 1);
        if announce {
            assert!(
                pipeline
                    .resume_checkpointed_stream(&store, key)
                    .await
                    .is_err()
            );
        }
    }
}

#[tokio::test]
async fn interruption_triggered_by_store_acknowledgement_prevents_agent_invocation() {
    for point in [SavePoint::Initial, SavePoint::Marker(0)] {
        let a_model = model("must not run");
        let pipeline = ParallelPipeline::new(vec![agent("one", a_model.clone())], 1).unwrap();
        let store = ControlledStore::interrupt_after(point);
        *store.interrupt.lock().unwrap() = Some(pipeline.interrupt_handle());
        let key = StateKey::new("test", "interrupt-after-save").unwrap();
        let observed = pipeline
            .stream_checkpointed(&store, key.clone(), Msg::user("task"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let error = error(&observed);
        assert_eq!(error.cause, ParallelFailure::Interrupted);
        assert!(a_model.recorded_requests().is_empty());
        let stored = store.load(&key).await.unwrap().unwrap();
        assert_eq!(
            error.branches[0].outcome,
            match stored.checkpoint.branches.first().unwrap() {
                ParallelBranchCheckpoint::Ready => ParallelBranchOutcome::NotStarted,
                ParallelBranchCheckpoint::InFlight => ParallelBranchOutcome::Interrupted,
                _ => panic!("no terminal agent result should exist"),
            }
        );
    }
}

#[tokio::test]
async fn cancellation_during_terminal_save_acknowledges_commit_then_resumes_only_ready_work() {
    let a = model("a completed");
    let b = model("b completed");
    let pipeline =
        ParallelPipeline::new(vec![agent("a", a.clone()), agent("b", b.clone())], 1).unwrap();
    let store = ControlledStore::interrupt_after(SavePoint::Terminal(0));
    *store.interrupt.lock().unwrap() = Some(pipeline.interrupt_handle());
    let key = StateKey::new("test", "terminal-save-interrupt").unwrap();
    let observed = pipeline
        .stream_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let interrupted = error(&observed);
    assert_eq!(interrupted.cause, ParallelFailure::Interrupted);
    let ParallelBranchOutcome::Completed(message) = &interrupted.branches[0].outcome else {
        panic!("the acknowledged save must remain completed despite cancellation")
    };
    assert_eq!(
        interrupted.branches[1].outcome,
        ParallelBranchOutcome::NotStarted
    );
    assert!(
        matches!(observed.get(observed.len() - 2), Some(ParallelEvent::BranchFinished { result }) if result.branch == 1 && result.outcome == ParallelBranchOutcome::Completed(message.clone()))
    );
    assert_eq!(a.recorded_requests().len(), 1);
    assert!(b.recorded_requests().is_empty());
    assert_eq!(
        store.load(&key).await.unwrap().unwrap().checkpoint.branches,
        vec![
            ParallelBranchCheckpoint::Completed(message.clone()),
            ParallelBranchCheckpoint::Ready
        ]
    );
    let resumed = pipeline
        .resume_checkpointed_stream(&store, key.clone())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let Some(ParallelEvent::Finished { output }) = resumed.last() else {
        panic!("resume must use a fresh interruption baseline and finish Ready work")
    };
    assert_eq!(
        output.branches[0].outcome,
        ParallelBranchOutcome::Completed(message.clone())
    );
    assert_eq!(a.recorded_requests().len(), 1);
    assert_eq!(b.recorded_requests().len(), 1);
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .finished_result(),
        Some(Ok(output.clone()))
    );
}

#[tokio::test]
async fn interrupt_after_forwarded_agent_terminal_retains_inflight_until_a_commit_is_acknowledged()
{
    let store = Arc::new(InMemoryParallelStore::new());
    let key = StateKey::new("test", "terminal-interrupt").unwrap();
    let activity = Arc::new(Activity::default());
    let a = GateModel::new("a", &activity, Some((store.clone(), key.clone(), 0)));
    let b = GateModel::new("b", &activity, Some((store.clone(), key.clone(), 1)));
    let queued = model("must not run");
    let pipeline = ParallelPipeline::new(
        vec![
            agent("a", a.clone()),
            agent("b", b.clone()),
            agent("queued", queued.clone()),
        ],
        2,
    )
    .unwrap();
    let mut events = pipeline
        .stream_checkpointed(store.as_ref(), key.clone(), Msg::user("task"))
        .await
        .unwrap();
    let mut observed = Vec::new();
    assert!(!drain_ready(&mut events, &mut observed));
    a.release();
    loop {
        let event = events.next().await.unwrap();
        let terminal = matches!(
            event,
            ParallelEvent::Agent {
                branch: 1,
                event: AgentEvent::Finished { .. },
                ..
            }
        );
        observed.push(event);
        if terminal {
            break;
        }
    }
    pipeline.interrupt_handle().interrupt();
    let terminal = events.next().await.unwrap();
    observed.push(terminal);
    let error = error(&observed);
    assert_eq!(error.cause, ParallelFailure::Interrupted);
    assert_eq!(
        error.branches[0].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert_eq!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert_eq!(error.branches[2].outcome, ParallelBranchOutcome::NotStarted);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert_eq!(b.dropped.load(Ordering::SeqCst), 1);
    assert!(queued.recorded_requests().is_empty());
    assert_eq!(
        store.load(&key).await.unwrap().unwrap().checkpoint.branches,
        vec![
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::Ready
        ]
    );
    assert!(
        !observed
            .iter()
            .any(|event| matches!(event, ParallelEvent::BranchFinished { .. }))
    );
    assert!(events.next().await.is_none());
}

#[tokio::test]
async fn active_drop_preserves_multiple_inflight_markers_and_releases_lock_without_replay() {
    let store = Arc::new(InMemoryParallelStore::new());
    let key = StateKey::new("test", "drop-active").unwrap();
    let activity = Arc::new(Activity::default());
    let a = GateModel::new("a", &activity, Some((store.clone(), key.clone(), 0)));
    let b = GateModel::new("b", &activity, Some((store.clone(), key.clone(), 1)));
    let queued = model("must not run");
    let pipeline = ParallelPipeline::new(
        vec![
            agent("a", a.clone()),
            agent("b", b.clone()),
            agent("queued", queued.clone()),
        ],
        2,
    )
    .unwrap();
    let mut events = pipeline
        .stream_checkpointed(store.as_ref(), key.clone(), Msg::user("task"))
        .await
        .unwrap();
    let mut observed = Vec::new();
    assert!(!drain_ready(&mut events, &mut observed));
    let before = store.load(&key).await.unwrap().unwrap();
    drop(events);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert_eq!(a.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(b.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(store.load(&key).await.unwrap().unwrap(), before);
    assert!(matches!(
        pipeline
            .resume_checkpointed_stream(store.as_ref(), key)
            .await,
        Err(ParallelError {
            cause: ParallelFailure::UnsafeResume(_),
            ..
        })
    ));
    assert_eq!(a.requests().len(), 1);
    assert_eq!(b.requests().len(), 1);
    assert!(queued.recorded_requests().is_empty());
    assert_eq!(terminal_count(&observed), 0);
    drop(
        pipeline
            .clone()
            .stream(Msg::user("lock released"))
            .await
            .unwrap(),
    );
}

#[tokio::test]
async fn ready_resume_stream_skips_completed_and_failed_branches_and_preserves_original_input() {
    let a = model("must not run");
    let b = model("must not run");
    let c = model("c completed");
    let pipeline = ParallelPipeline::new(
        vec![
            agent("a", a.clone()),
            agent("b", b.clone()),
            agent("c", c.clone()),
        ],
        1,
    )
    .unwrap();
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "resume-ready").unwrap();
    let mut input = Msg::user("original task");
    input.metadata.insert("original".into(), json!(true));
    let completed = reply("a", "already committed");
    let failed = AgentError::Model(ModelError::new("already failed"));
    store
        .save(
            key.clone(),
            None,
            checkpoint(
                &["a", "b", "c"],
                input.clone(),
                vec![
                    ParallelBranchCheckpoint::Completed(completed.clone()),
                    ParallelBranchCheckpoint::Failed(Box::new(failed.clone())),
                    ParallelBranchCheckpoint::Ready,
                ],
            ),
        )
        .await
        .unwrap();
    let reserved = pipeline
        .resume_checkpointed_stream(&store, key.clone())
        .await
        .unwrap();
    assert!(c.recorded_requests().is_empty());
    let observed = reserved.collect::<Vec<_>>().await;
    let error = error(&observed);
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    assert_eq!(
        error.branches[0].outcome,
        ParallelBranchOutcome::Completed(completed)
    );
    assert_eq!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Failed(Box::new(failed))
    );
    assert!(matches!(
        error.branches[2].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert!(a.recorded_requests().is_empty());
    assert!(b.recorded_requests().is_empty());
    assert_eq!(c.recorded_requests()[0].messages.last(), Some(&input));
    assert!(
        observed
            .iter()
            .filter_map(
                |event| if let ParallelEvent::BranchStarted { branch, .. } = event {
                    Some(branch)
                } else {
                    None
                }
            )
            .all(|branch| *branch == 3)
    );
    assert!(observed.iter().any(|event| matches!(
        event,
        ParallelEvent::Agent {
            branch: 3,
            event: AgentEvent::TextDelta { step: 1, .. },
            ..
        }
    )));
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .finished_result(),
        Some(Err(error.clone()))
    );
}

#[tokio::test]
async fn resume_preparation_rejects_missing_incompatible_inflight_and_finished_records() {
    let a = model("must not run");
    let b = model("must not run");
    let pipeline =
        ParallelPipeline::new(vec![agent("a", a.clone()), agent("b", b.clone())], 2).unwrap();
    let store = InMemoryParallelStore::new();
    let missing = StateKey::new("test", "missing").unwrap();
    assert!(matches!(
        pipeline.resume_checkpointed_stream(&store, missing).await,
        Err(ParallelError {
            cause: ParallelFailure::UnsafeResume(_),
            ..
        })
    ));
    let base = checkpoint(
        &["a", "b"],
        Msg::user("task"),
        vec![
            ParallelBranchCheckpoint::Ready,
            ParallelBranchCheckpoint::Ready,
        ],
    );
    let mut cases = vec![base.clone(); 5];
    cases[0].version += 1;
    cases[1].agent_names.swap(0, 1);
    cases[2].branches.pop();
    cases[3].branches[0] = ParallelBranchCheckpoint::InFlight;
    cases[4].branches = vec![
        ParallelBranchCheckpoint::Completed(reply("a", "done")),
        ParallelBranchCheckpoint::Failed(Box::new(AgentError::Interrupted)),
    ];
    for (index, checkpoint) in cases.into_iter().enumerate() {
        let key = StateKey::new("test", format!("invalid-{index}")).unwrap();
        let record = store.save(key.clone(), None, checkpoint).await.unwrap();
        assert!(matches!(
            pipeline
                .resume_checkpointed_stream(&store, key.clone())
                .await,
            Err(ParallelError {
                cause: ParallelFailure::UnsafeResume(_),
                ..
            })
        ));
        assert_eq!(store.load(&key).await.unwrap().unwrap(), record);
    }
    assert!(a.recorded_requests().is_empty());
    assert!(b.recorded_requests().is_empty());
    drop(
        pipeline
            .stream(Msg::user("validation releases lock"))
            .await
            .unwrap(),
    );
}

#[tokio::test]
async fn mixed_failures_and_confirmation_are_committed_while_other_branches_continue() {
    let startup_error = AgentError::Model(ModelError::new("startup failure"));
    let raw_error = AgentError::Model(ModelError::new("raw failure"));
    let startup = ScriptAgent::startup("startup", startup_error.clone());
    let raw = ScriptAgent::new("raw", vec![Err(raw_error.clone())]);
    let delta = AgentEvent::TextDelta {
        step: 17,
        block_id: "opaque-id".into(),
        delta: "partial".into(),
    };
    let missing = ScriptAgent::new("missing", vec![Ok(delta.clone())]);
    let event_error = AgentError::Model(ModelError::new("terminal error"));
    let error_event = AgentEvent::Error {
        step: Some(23),
        error: event_error.clone(),
    };
    let terminal = ScriptAgent::new("terminal", vec![Ok(error_event.clone())]);
    let agent_store = Arc::new(InMemoryStateStore::new());
    let agent_key = StateKey::new("test", "blocked-agent").unwrap();
    let (blocked, tool) = confirming_agent(agent_store.clone(), agent_key.clone());
    let finished = AgentEvent::Finished {
        steps: 42,
        message: reply("success", "done"),
    };
    let success = ScriptAgent::new("success", vec![Ok(finished.clone())]);
    let pipeline = ParallelPipeline::new(
        vec![
            startup.clone(),
            raw.clone(),
            missing.clone(),
            terminal.clone(),
            blocked,
            success.clone(),
        ],
        2,
    )
    .unwrap();
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "mixed-failures").unwrap();
    let input = Msg::user("shared task");
    let observed = pipeline
        .stream_checkpointed(&store, key.clone(), input.clone())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let error = error(&observed);
    assert_scripted_failures(error, [startup_error, raw_error, event_error]);
    let confirmation = observed
        .iter()
        .find_map(|event| {
            if let ParallelEvent::Agent {
                branch: 5,
                event: AgentEvent::ToolConfirmationRequired { checkpoint },
                ..
            } = event
            {
                Some(checkpoint)
            } else {
                None
            }
        })
        .unwrap();
    assert!(
        matches!(&error.branches[4].outcome, ParallelBranchOutcome::Failed(cause) if matches!(cause.as_ref(), AgentError::ToolConfirmationRequired { checkpoint } if checkpoint == confirmation))
    );
    assert!(tool.recorded_invocations().is_empty());
    assert_eq!(
        agent_store
            .load(&agent_key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_calls(),
        Some(confirmation)
    );
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .finished_result(),
        Some(Err(error.clone()))
    );
    assert_eq!(
        observed
            .iter()
            .filter(|event| matches!(event, ParallelEvent::BranchFinished { .. }))
            .count(),
        6
    );
    for (branch, original) in [(3, delta), (4, error_event), (6, finished)] {
        assert!(observed.iter().any(|event| matches!(event, ParallelEvent::Agent { branch: id, event, .. } if *id == branch && event == &original)));
    }
    for agent in [startup, raw, missing, terminal, success] {
        assert_eq!(*agent.inputs.lock().unwrap(), vec![input.clone()]);
    }
    assert_roundtrips(&observed);
}
