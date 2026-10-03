use super::*;
use crate::*;
use futures_util::FutureExt;
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};
#[path = "tests_support.rs"]
mod support;
use support::*;

#[tokio::test]
async fn bounded_calls_witness_durable_markers_and_committed_results_keep_configured_order() {
    let store = Arc::new(InMemoryParallelStore::new());
    let key = StateKey::new("test", "bounded").unwrap();
    let activity = Arc::new(Activity::default());
    let models: Vec<_> = (0..3)
        .map(|index| GateModel::new(&activity, Some((store.clone(), key.clone(), index))))
        .collect();
    let pipeline = ParallelPipeline::new(
        ["a", "b", "c"]
            .into_iter()
            .zip(&models)
            .map(|(name, model)| agent(name, model.clone()) as Arc<dyn Agent>)
            .collect(),
        2,
    )
    .unwrap();
    drop(pipeline.run_checkpointed(store.as_ref(), key.clone(), Msg::user("unpolled")));
    assert!(store.load(&key).await.unwrap().is_none());
    let mut input = Msg::user("shared task");
    input.metadata.insert("shared".into(), json!({"value": 7}));
    let mut run = pipeline.run_checkpointed(store.as_ref(), key.clone(), input.clone());
    assert!(run.as_mut().now_or_never().is_none());
    let initial = store.load(&key).await.unwrap().unwrap();
    assert_eq!(
        initial.checkpoint.branches,
        vec![
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::Ready
        ]
    );
    assert_eq!(activity.active.load(Ordering::SeqCst), 2);
    assert!(models[2].requests().is_empty());
    models[1].release();
    assert!(run.as_mut().now_or_never().is_none());
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
    assert!(run.as_mut().now_or_never().is_none());
    models[0].release();
    let output = run.await.unwrap();
    assert_eq!(activity.peak.load(Ordering::SeqCst), 2);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    for (index, branch) in output.branches.iter().enumerate() {
        assert_eq!(branch.branch, index + 1);
        assert_eq!(branch.agent_name, ["a", "b", "c"][index]);
        assert_eq!(models[index].requests()[0].messages.last(), Some(&input));
    }
    let finished = store.load(&key).await.unwrap().unwrap();
    assert_eq!(finished.checkpoint.finished_result(), Some(Ok(output)));
    let encoded = serde_json::to_string(&finished).unwrap();
    assert_eq!(
        serde_json::from_str::<ParallelRecord>(&encoded).unwrap(),
        finished
    );
    assert!(matches!(
        pipeline
            .resume_checkpointed(store.as_ref(), key.clone())
            .await,
        Err(ParallelError {
            cause: ParallelFailure::UnsafeResume(_),
            ..
        })
    ));
    assert!(matches!(
        pipeline
            .run_checkpointed(store.as_ref(), key.clone(), Msg::user("duplicate run"))
            .await,
        Err(ParallelError {
            cause: ParallelFailure::Store(_),
            ..
        })
    ));
    assert_eq!(store.load(&key).await.unwrap().unwrap(), finished);
    assert!(models.iter().all(|model| model.requests().len() == 1));
}

#[tokio::test]
async fn ready_resume_skips_committed_success_and_failure_and_retains_original_input() {
    let a = model("must not run a");
    let b = model("must not run b");
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
    let key = StateKey::new("test", "ready").unwrap();
    let mut input = Msg::user("original task");
    input.metadata.insert("original".into(), json!(true));
    let completed = reply("a", "a committed");
    let failed = AgentError::Model(ModelError::new("b committed failure"));
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
    let error = pipeline
        .resume_checkpointed(&store, key.clone())
        .await
        .unwrap_err();
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
    assert_eq!(c.recorded_requests().len(), 1);
    assert_eq!(c.recorded_requests()[0].messages.last(), Some(&input));
    let committed = store.load(&key).await.unwrap().unwrap();
    assert_eq!(committed.checkpoint.finished_result(), Some(Err(error)));
}

#[tokio::test]
async fn dropped_multi_inflight_run_requires_every_uncertain_branch_to_be_reconciled() {
    let store = Arc::new(InMemoryParallelStore::new());
    let key = StateKey::new("test", "dropped").unwrap();
    let activity = Arc::new(Activity::default());
    let a = GateModel::new(&activity, Some((store.clone(), key.clone(), 0)));
    let b = GateModel::new(&activity, Some((store.clone(), key.clone(), 1)));
    let c = model("c completed");
    let pipeline = ParallelPipeline::new(
        vec![
            agent("a", a.clone()),
            agent("b", b.clone()),
            agent("c", c.clone()),
        ],
        2,
    )
    .unwrap();
    let mut run = pipeline.run_checkpointed(store.as_ref(), key.clone(), Msg::user("task"));
    assert!(run.as_mut().now_or_never().is_none());
    drop(run);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert_eq!(a.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(b.dropped.load(Ordering::SeqCst), 1);
    let original = store.load(&key).await.unwrap().unwrap();
    assert_eq!(
        original.checkpoint.branches,
        vec![
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::Ready
        ]
    );
    assert!(matches!(
        pipeline
            .resume_checkpointed(store.as_ref(), key.clone())
            .await,
        Err(ParallelError {
            cause: ParallelFailure::UnsafeResume(_),
            ..
        })
    ));
    let reconciled_a = reply("a", "verified a");
    let one = pipeline
        .reconcile_checkpointed(
            store.as_ref(),
            key.clone(),
            original.revision,
            1,
            reconciled_a.clone(),
        )
        .await
        .unwrap();
    assert_eq!(
        one.checkpoint.branches[0],
        ParallelBranchCheckpoint::Completed(reconciled_a.clone())
    );
    assert_eq!(
        one.checkpoint.branches[1],
        ParallelBranchCheckpoint::InFlight
    );
    assert!(
        pipeline
            .resume_checkpointed(store.as_ref(), key.clone())
            .await
            .is_err()
    );
    let two = pipeline
        .reconcile_checkpointed(
            store.as_ref(),
            key.clone(),
            one.revision,
            2,
            reply("b", "verified b"),
        )
        .await
        .unwrap();
    assert!(two.checkpoint.finished_result().is_none());
    assert!(c.recorded_requests().is_empty());
    let output = pipeline
        .resume_checkpointed(store.as_ref(), key.clone())
        .await
        .unwrap();
    assert_eq!(output.branches.len(), 3);
    assert_eq!(a.requests().len(), 1);
    assert_eq!(b.requests().len(), 1);
    assert_eq!(c.recorded_requests().len(), 1);
    assert_eq!(
        store
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .finished_result(),
        Some(Ok(output))
    );
}

#[tokio::test]
async fn reconciliation_rejects_stale_invalid_or_already_committed_evidence_without_mutation() {
    let pipeline = ParallelPipeline::new(
        vec![
            agent("a", model("unused")),
            agent("b", model("unused")),
            agent("c", model("unused")),
        ],
        2,
    )
    .unwrap();
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "invalid-evidence").unwrap();
    let original = store
        .save(
            key.clone(),
            None,
            checkpoint(
                &["a", "b", "c"],
                Msg::user("task"),
                vec![
                    ParallelBranchCheckpoint::InFlight,
                    ParallelBranchCheckpoint::Ready,
                    ParallelBranchCheckpoint::Completed(reply("c", "committed")),
                ],
            ),
        )
        .await
        .unwrap();
    let attempts = [
        (original.revision + 1, 1, reply("a", "stale")),
        (original.revision, 0, reply("a", "bad index")),
        (original.revision, 4, reply("a", "bad index")),
        (original.revision, 1, Msg::user("wrong role")),
        (original.revision, 1, reply("other", "wrong name")),
        (original.revision, 2, reply("b", "unstarted")),
        (
            original.revision,
            3,
            reply("c", "cannot overwrite completion"),
        ),
    ];
    for (revision, branch, message) in attempts {
        assert!(
            pipeline
                .reconcile_checkpointed(&store, key.clone(), revision, branch, message)
                .await
                .is_err()
        );
        assert_eq!(store.load(&key).await.unwrap().unwrap(), original);
    }
    let reconciled = pipeline
        .reconcile_checkpointed(
            &store,
            key.clone(),
            original.revision,
            1,
            reply("a", "verified"),
        )
        .await
        .unwrap();
    assert!(
        pipeline
            .reconcile_checkpointed(
                &store,
                key.clone(),
                original.revision,
                1,
                reply("a", "stale duplicate")
            )
            .await
            .is_err()
    );
    assert!(
        pipeline
            .reconcile_checkpointed(
                &store,
                key.clone(),
                reconciled.revision,
                1,
                reply("a", "duplicate")
            )
            .await
            .is_err()
    );
    assert_eq!(store.load(&key).await.unwrap().unwrap(), reconciled);
}

#[tokio::test]
async fn checkpoint_write_failures_fence_invocation_and_uncommitted_results() {
    for point in [SavePoint::Initial, SavePoint::Marker, SavePoint::Terminal] {
        let model = model("result may have effects");
        let pipeline = ParallelPipeline::new(vec![agent("worker", model.clone())], 1).unwrap();
        let store = ControlledStore::reject(point);
        let key = StateKey::new("test", "write-failure").unwrap();
        let error = pipeline
            .run_checkpointed(&store, key.clone(), Msg::user("task"))
            .await
            .unwrap_err();
        assert!(matches!(error.cause, ParallelFailure::Store(_)));
        match point {
            SavePoint::Initial => {
                assert!(store.load(&key).await.unwrap().is_none());
                assert!(model.recorded_requests().is_empty());
                assert!(error.branches.is_empty());
            }
            SavePoint::Marker => {
                let committed = store.load(&key).await.unwrap().unwrap();
                assert_eq!(
                    committed.checkpoint.branches,
                    vec![ParallelBranchCheckpoint::Ready]
                );
                assert!(model.recorded_requests().is_empty());
                assert_eq!(error.branches[0].outcome, ParallelBranchOutcome::NotStarted);
            }
            SavePoint::Terminal => {
                let committed = store.load(&key).await.unwrap().unwrap();
                assert_eq!(
                    committed.checkpoint.branches,
                    vec![ParallelBranchCheckpoint::InFlight]
                );
                assert_eq!(model.recorded_requests().len(), 1);
                assert_eq!(
                    error.branches[0].outcome,
                    ParallelBranchOutcome::Interrupted
                );
                assert!(committed.checkpoint.finished_result().is_none());
                assert!(pipeline.resume_checkpointed(&store, key).await.is_err());
                assert_eq!(model.recorded_requests().len(), 1);
            }
        }
    }
}

#[tokio::test]
async fn completion_write_failure_drops_active_siblings_and_never_dispatches_queued_work() {
    let activity = Arc::new(Activity::default());
    let pending = GateModel::new(&activity, None);
    let completed = model("uncommitted completion");
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
    let store = ControlledStore::reject(SavePoint::Terminal);
    let key = StateKey::new("test", "completion-fence").unwrap();
    let error = pipeline
        .run_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap_err();
    assert!(matches!(error.cause, ParallelFailure::Store(_)));
    assert_eq!(pending.requests().len(), 1);
    assert_eq!(completed.recorded_requests().len(), 1);
    assert!(queued.recorded_requests().is_empty());
    assert_eq!(pending.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
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
        store.load(&key).await.unwrap().unwrap().checkpoint.branches,
        vec![
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::Ready
        ]
    );
    assert!(pipeline.resume_checkpointed(&store, key).await.is_err());
    assert!(queued.recorded_requests().is_empty());
}

#[tokio::test]
async fn interruption_after_store_acknowledgement_prevents_dispatch_and_keeps_committed_state() {
    for point in [SavePoint::Initial, SavePoint::Marker] {
        let a = model("must not run a");
        let b = model("must not run b");
        let pipeline =
            ParallelPipeline::new(vec![agent("a", a.clone()), agent("b", b.clone())], 2).unwrap();
        let store = ControlledStore::interrupt_after(point);
        *store.interrupt.lock().unwrap() = Some(pipeline.interrupt_handle());
        let key = StateKey::new("test", "post-save-interrupt").unwrap();
        let error = pipeline
            .run_checkpointed(&store, key.clone(), Msg::user("task"))
            .await
            .unwrap_err();
        assert_eq!(error.cause, ParallelFailure::Interrupted);
        assert!(a.recorded_requests().is_empty());
        assert!(b.recorded_requests().is_empty());
        let committed = store.load(&key).await.unwrap().unwrap();
        for (saved, reported) in committed.checkpoint.branches.iter().zip(&error.branches) {
            match saved {
                ParallelBranchCheckpoint::Ready => {
                    assert_eq!(reported.outcome, ParallelBranchOutcome::NotStarted);
                }
                ParallelBranchCheckpoint::InFlight => {
                    assert_eq!(reported.outcome, ParallelBranchOutcome::Interrupted);
                }
                _ => panic!("no agent result should have been committed"),
            }
        }
    }
}

#[tokio::test]
async fn terminal_agent_failure_preserves_confirmation_and_can_be_explicitly_reconciled() {
    let tool = Arc::new(
        MockTool::new(ToolDefinition::new("write", "write", json!({"type":"object"})).unwrap())
            .with_output("saved"),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let blocked = Arc::new(
        ReActAgent::new(
            "blocked",
            MockChatModel::new("approval").with_response(ChatResponse::finished(
                [ToolCallBlock::complete("call", "write", "{}")
                    .unwrap()
                    .into()],
                FinishReason::ToolCalls,
            )),
            ToolExecutor::new(registry),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("write"),
    );
    let sibling = model("sibling completed");
    let pipeline =
        ParallelPipeline::new(vec![blocked.clone(), agent("sibling", sibling.clone())], 1).unwrap();
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "confirmation").unwrap();
    let error = pipeline
        .run_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap_err();
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    let agent_before = blocked.snapshot().await.unwrap();
    assert!(
        matches!(&error.branches[0].outcome, ParallelBranchOutcome::Failed(cause) if matches!(cause.as_ref(), AgentError::ToolConfirmationRequired { checkpoint } if Some(checkpoint) == agent_before.pending_tool_calls()))
    );
    assert!(matches!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert!(tool.recorded_invocations().is_empty());
    assert_eq!(sibling.recorded_requests().len(), 1);
    let terminal = store.load(&key).await.unwrap().unwrap();
    assert_eq!(terminal.checkpoint.finished_result(), Some(Err(error)));
    let encoded = serde_json::to_string(&terminal).unwrap();
    assert_eq!(
        serde_json::from_str::<ParallelRecord>(&encoded).unwrap(),
        terminal
    );
    assert!(
        pipeline
            .resume_checkpointed(&store, key.clone())
            .await
            .is_err()
    );
    let verified = reply("blocked", "verified independently");
    let reconciled = pipeline
        .reconcile_checkpointed(&store, key.clone(), terminal.revision, 1, verified)
        .await
        .unwrap();
    assert!(matches!(
        reconciled.checkpoint.finished_result(),
        Some(Ok(_))
    ));
    assert_eq!(
        blocked.snapshot().await.unwrap(),
        agent_before,
        "pipeline reconciliation does not restore or resume child state"
    );
    assert!(tool.recorded_invocations().is_empty());
}

#[tokio::test]
async fn resume_validation_rejects_missing_key_changed_shape_version_and_agent_order() {
    let a = model("must not run");
    let b = model("must not run");
    let pipeline =
        ParallelPipeline::new(vec![agent("a", a.clone()), agent("b", b.clone())], 2).unwrap();
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "invalid").unwrap();
    assert!(matches!(
        pipeline.resume_checkpointed(&store, key.clone()).await,
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
    let mut invalid = vec![base.clone(); 4];
    invalid[0].version += 1;
    invalid[1].agent_names.swap(0, 1);
    invalid[2].branches.pop();
    invalid[3].agent_names[0] = "other".into();
    for (index, checkpoint) in invalid.into_iter().enumerate() {
        let key = StateKey::new("test", format!("invalid-{index}")).unwrap();
        let record = store.save(key.clone(), None, checkpoint).await.unwrap();
        assert!(matches!(
            pipeline.resume_checkpointed(&store, key.clone()).await,
            Err(ParallelError {
                cause: ParallelFailure::UnsafeResume(_),
                ..
            })
        ));
        assert!(
            pipeline
                .reconcile_checkpointed(
                    &store,
                    key.clone(),
                    record.revision,
                    1,
                    reply("a", "invalid evidence")
                )
                .await
                .is_err()
        );
        assert_eq!(store.load(&key).await.unwrap().unwrap(), record);
    }
    assert!(a.recorded_requests().is_empty());
    assert!(b.recorded_requests().is_empty());
}

#[tokio::test]
async fn checkpoint_store_compare_and_swap_and_finished_result_validate_terminal_shape() {
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "cas").unwrap();
    let ready = checkpoint(
        &["one"],
        Msg::user("task"),
        vec![ParallelBranchCheckpoint::Ready],
    );
    let first = store.save(key.clone(), None, ready.clone()).await.unwrap();
    assert_eq!(first.revision, 1);
    assert!(store.save(key.clone(), None, ready.clone()).await.is_err());
    assert!(
        store
            .save(key.clone(), Some(0), ready.clone())
            .await
            .is_err()
    );
    assert_eq!(store.load(&key).await.unwrap().unwrap(), first);
    let done = checkpoint(
        &["one"],
        ready.input.clone(),
        vec![ParallelBranchCheckpoint::Completed(reply("one", "done"))],
    );
    let second = store
        .save(key.clone(), Some(first.revision), done.clone())
        .await
        .unwrap();
    assert_eq!(second.revision, 2);
    assert!(
        store
            .save(key.clone(), Some(first.revision), ready.clone())
            .await
            .is_err()
    );
    assert_eq!(store.load(&key).await.unwrap().unwrap(), second);
    assert!(matches!(done.finished_result(), Some(Ok(_))));
    assert!(ready.finished_result().is_none());
    let mut invalid = vec![done.clone(); 5];
    invalid[0].version += 1;
    invalid[1].agent_names.clear();
    invalid[2].branches.clear();
    invalid[3].agent_names[0] = " \t".into();
    invalid[4].agent_names.push("one".into());
    invalid[4]
        .branches
        .push(ParallelBranchCheckpoint::Completed(reply(
            "one",
            "duplicate name",
        )));
    for checkpoint in invalid {
        assert!(checkpoint.finished_result().is_none());
    }
}

#[tokio::test]
async fn checkpoint_operations_share_busy_lock_with_plain_runs_streams_and_clones() {
    let store = InMemoryParallelStore::new();
    let key = StateKey::new("test", "busy").unwrap();
    let activity = Arc::new(Activity::default());
    let gated = GateModel::new(&activity, None);
    let pipeline = ParallelPipeline::new(vec![agent("one", gated.clone())], 1).unwrap();
    let cloned = pipeline.clone();
    let reserved = pipeline.stream(Msg::user("reserved")).await.unwrap();
    assert!(matches!(
        cloned
            .run_checkpointed(&store, key.clone(), Msg::user("busy"))
            .await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    assert!(store.load(&key).await.unwrap().is_none());
    drop(reserved);
    let mut active = pipeline.run_checkpointed(&store, key.clone(), Msg::user("active"));
    assert!(active.as_mut().now_or_never().is_none());
    assert!(matches!(
        cloned.run(Msg::user("busy plain run")).await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    assert!(matches!(
        cloned.stream(Msg::user("busy stream")).await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    assert!(matches!(
        cloned.resume_checkpointed(&store, key.clone()).await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    let revision = store.load(&key).await.unwrap().unwrap().revision;
    assert!(matches!(
        cloned
            .reconcile_checkpointed(&store, key, revision, 1, reply("one", "must not mutate"))
            .await,
        Err(ParallelError {
            cause: ParallelFailure::Busy,
            ..
        })
    ));
    drop(active);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    drop(cloned.stream(Msg::user("released")).await.unwrap());
}
