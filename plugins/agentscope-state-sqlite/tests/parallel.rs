use agentscope::{
    AgentError, AgentInterruptHandle, AgentState, ChatEventStream, ChatModel, ChatRequest,
    ChatResponse, ContentBlock, InMemoryMemory, MockChatModel, ModelCapabilities, ModelError,
    ModelFuture, Msg, PARALLEL_CHECKPOINT_VERSION, PIPELINE_CHECKPOINT_VERSION,
    ParallelBranchCheckpoint, ParallelBranchOutcome, ParallelCheckpoint, ParallelFailure,
    ParallelPipeline, ParallelRecord, ParallelStore, PipelineCheckpoint, PipelineCheckpointStatus,
    PipelineStore, PipelineStoreFuture, ReActAgent, Role, StateKey, StateStore, ToolExecutor,
    ToolRegistry,
};
use agentscope_state_sqlite::{SQLiteParallelStore, SQLitePipelineStore, SQLiteStateStore};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::Notify;

fn agent(name: &str, model: Arc<dyn ChatModel>) -> Arc<ReActAgent> {
    Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))
            .unwrap()
            .with_memory(InMemoryMemory::new()),
    )
}

fn model(text: &str) -> Arc<MockChatModel> {
    Arc::new(
        MockChatModel::new("offline")
            .with_response(ChatResponse::completed([ContentBlock::from(text)])),
    )
}

fn checkpoint(input: &str) -> ParallelCheckpoint {
    ParallelCheckpoint {
        version: PARALLEL_CHECKPOINT_VERSION,
        agent_names: vec!["worker".into()],
        input: Msg::user(input),
        branches: vec![ParallelBranchCheckpoint::Ready],
    }
}

fn verified(name: &str, text: &str) -> Msg {
    Msg::new(name, Role::Assistant, [ContentBlock::from(text)])
}

struct StopAfterTwo {
    inner: SQLiteParallelStore,
    interrupt: AgentInterruptHandle,
}

impl ParallelStore for StopAfterTwo {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<ParallelRecord>> {
        self.inner.load(key)
    }

    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        checkpoint: ParallelCheckpoint,
    ) -> PipelineStoreFuture<'_, ParallelRecord> {
        Box::pin(async move {
            let at_boundary = matches!(
                checkpoint.branches.as_slice(),
                [
                    ParallelBranchCheckpoint::Completed(_),
                    ParallelBranchCheckpoint::Failed(_),
                    ParallelBranchCheckpoint::Ready,
                ]
            );
            let record = self.inner.save(key, expected_revision, checkpoint).await?;
            if at_boundary {
                self.interrupt.interrupt();
            }
            Ok(record)
        })
    }
}

#[tokio::test]
async fn reopens_partial_success_and_failure_then_resumes_only_ready_branches() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("parallel.db");
    let key = StateKey::new("user", "ready-resume").unwrap();
    let first = model("completed first");
    let failed =
        Arc::new(MockChatModel::new("failed").with_error(ModelError::new("second failed")));
    let third = Arc::new(MockChatModel::new("not-yet-called"));
    let original = ParallelPipeline::new(
        vec![
            agent("first", first.clone()),
            agent("second", failed.clone()),
            agent("third", third.clone()),
        ],
        1,
    )
    .unwrap();
    let store = StopAfterTwo {
        inner: SQLiteParallelStore::open(&path).await.unwrap(),
        interrupt: original.interrupt_handle(),
    };
    let input = Msg::user("one unchanged task");
    let error = original
        .run_checkpointed(&store, key.clone(), input.clone())
        .await
        .unwrap_err();
    assert_eq!(error.cause, ParallelFailure::Interrupted);
    assert_eq!(first.recorded_requests().len(), 1);
    assert_eq!(failed.recorded_requests().len(), 1);
    assert!(third.recorded_requests().is_empty());
    drop(store);

    let reopened = SQLiteParallelStore::open(&path).await.unwrap();
    let record = reopened.load(&key).await.unwrap().unwrap();
    assert_eq!(record.checkpoint.input, input);
    assert!(matches!(
        record.checkpoint.branches[0],
        ParallelBranchCheckpoint::Completed(_)
    ));
    assert!(matches!(
        record.checkpoint.branches[1],
        ParallelBranchCheckpoint::Failed(_)
    ));
    assert_eq!(
        record.checkpoint.branches[2],
        ParallelBranchCheckpoint::Ready
    );
    assert!(record.checkpoint.finished_result().is_none());
    let skipped_first = Arc::new(MockChatModel::new("must-not-repeat-first"));
    let skipped_second = Arc::new(MockChatModel::new("must-not-repeat-second"));
    let resumed_third = model("completed third");
    let restarted = ParallelPipeline::new(
        vec![
            agent("first", skipped_first.clone()),
            agent("second", skipped_second.clone()),
            agent("third", resumed_third.clone()),
        ],
        2,
    )
    .unwrap();
    let finished_error = restarted
        .resume_checkpointed(&reopened, key.clone())
        .await
        .unwrap_err();
    assert_eq!(finished_error.cause, ParallelFailure::AgentFailures);
    assert!(matches!(
        finished_error.branches[0].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert!(matches!(
        finished_error.branches[1].outcome,
        ParallelBranchOutcome::Failed(_)
    ));
    assert!(matches!(
        finished_error.branches[2].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert!(skipped_first.recorded_requests().is_empty());
    assert!(skipped_second.recorded_requests().is_empty());
    assert_eq!(resumed_third.recorded_requests().len(), 1);
    assert_eq!(resumed_third.recorded_requests()[0].messages, vec![input]);
    drop(reopened);

    let terminal = SQLiteParallelStore::open(&path).await.unwrap();
    assert_eq!(
        terminal
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .finished_result(),
        Some(Err(finished_error))
    );
    let replay = restarted
        .resume_checkpointed(&terminal, key)
        .await
        .unwrap_err();
    assert!(matches!(replay.cause, ParallelFailure::UnsafeResume(_)));
    assert_eq!(resumed_third.recorded_requests().len(), 1);
}

#[derive(Default)]
struct Activity {
    invoked: AtomicUsize,
    changed: Notify,
}

struct PendingModel(Arc<Activity>);

impl ChatModel for PendingModel {
    fn name(&self) -> &'static str {
        "pending"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }

    fn generate(&self, _: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async move {
            self.0.invoked.fetch_add(1, Ordering::SeqCst);
            self.0.changed.notify_one();
            std::future::pending().await
        })
    }

    fn stream(&self, _: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async { Err(ModelError::new("unused stream")) })
    }
}

#[tokio::test]
async fn interrupted_and_dropped_runs_require_verified_reconciliation_after_reopen() {
    for interrupt_run in [true, false] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("parallel.db");
        let key = StateKey::new("user", "uncertain").unwrap();
        let activity = Arc::new(Activity::default());
        let pending = Arc::new(PendingModel(activity.clone()));
        let never_called = Arc::new(MockChatModel::new("queued"));
        let original = ParallelPipeline::new(
            vec![
                agent("first", pending.clone()),
                agent("second", pending.clone()),
                agent("third", never_called.clone()),
            ],
            2,
        )
        .unwrap();
        let store = SQLiteParallelStore::open(&path).await.unwrap();
        let mut run = original.run_checkpointed(&store, key.clone(), Msg::user("task"));
        let invoked = async {
            while activity.invoked.load(Ordering::SeqCst) < 2 {
                activity.changed.notified().await;
            }
        };
        tokio::select! {
            result = &mut run => panic!("pending run unexpectedly returned {result:?}"),
            () = invoked => {}
        }
        if interrupt_run {
            original.interrupt_handle().interrupt();
            assert_eq!(run.await.unwrap_err().cause, ParallelFailure::Interrupted);
        } else {
            drop(run);
        }
        assert!(never_called.recorded_requests().is_empty());
        drop(store);
        reconcile_inflight_and_resume(&path, key).await;
    }
}

async fn reconcile_inflight_and_resume(path: &std::path::Path, key: StateKey) {
    let reopened = SQLiteParallelStore::open(path).await.unwrap();
    let uncertain = reopened.load(&key).await.unwrap().unwrap();
    assert_eq!(
        uncertain.checkpoint.branches,
        vec![
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::Ready,
        ]
    );
    let skipped_first = Arc::new(MockChatModel::new("must-not-repeat-first"));
    let skipped_second = Arc::new(MockChatModel::new("must-not-repeat-second"));
    let next = model("next completed");
    let restarted = ParallelPipeline::new(
        vec![
            agent("first", skipped_first.clone()),
            agent("second", skipped_second.clone()),
            agent("third", next.clone()),
        ],
        2,
    )
    .unwrap();
    assert!(matches!(
        restarted
            .resume_checkpointed(&reopened, key.clone())
            .await
            .unwrap_err()
            .cause,
        ParallelFailure::UnsafeResume(_)
    ));
    assert!(matches!(
        restarted
            .reconcile_checkpointed(
                &reopened,
                key.clone(),
                uncertain.revision,
                1,
                verified("wrong-name", "rejected"),
            )
            .await
            .unwrap_err()
            .cause,
        ParallelFailure::UnsafeResume(_)
    ));
    let first = verified("first", "externally verified first");
    let second = verified("second", "externally verified second");
    let once = restarted
        .reconcile_checkpointed(&reopened, key.clone(), uncertain.revision, 1, first.clone())
        .await
        .unwrap();
    assert_eq!(once.revision, uncertain.revision + 1);
    assert!(matches!(
        restarted
            .reconcile_checkpointed(
                &reopened,
                key.clone(),
                uncertain.revision,
                2,
                second.clone(),
            )
            .await
            .unwrap_err()
            .cause,
        ParallelFailure::UnsafeResume(_)
    ));
    assert_eq!(reopened.load(&key).await.unwrap().unwrap(), once);
    let resolved = restarted
        .reconcile_checkpointed(&reopened, key.clone(), once.revision, 2, second.clone())
        .await
        .unwrap();
    assert_eq!(resolved.revision, once.revision + 1);
    drop(reopened);

    let reopened_again = SQLiteParallelStore::open(path).await.unwrap();
    let output = restarted
        .resume_checkpointed(&reopened_again, key.clone())
        .await
        .unwrap();
    assert_eq!(
        output.branches[0].outcome,
        ParallelBranchOutcome::Completed(first)
    );
    assert_eq!(
        output.branches[1].outcome,
        ParallelBranchOutcome::Completed(second)
    );
    assert!(skipped_first.recorded_requests().is_empty());
    assert!(skipped_second.recorded_requests().is_empty());
    assert_eq!(next.recorded_requests().len(), 1);
    assert_eq!(
        reopened_again
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
async fn independent_connections_compare_and_swap_without_overwriting_the_winner() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("parallel.db");
    let key = StateKey::new("user", "race").unwrap();
    let one = SQLiteParallelStore::open(&path).await.unwrap();
    let two = SQLiteParallelStore::open(&path).await.unwrap();
    assert!(one.load(&key).await.unwrap().is_none());
    let (left, right) = tokio::join!(
        one.save(key.clone(), None, checkpoint("left creation")),
        two.save(key.clone(), None, checkpoint("right creation")),
    );
    assert_ne!(left.is_ok(), right.is_ok());
    let winning = left.as_ref().ok().or_else(|| right.as_ref().ok()).unwrap();
    assert_eq!(winning.revision, 1);
    assert_eq!(one.load(&key).await.unwrap().unwrap(), *winning);
    let conflict = left.err().or_else(|| right.err()).unwrap();
    assert!(conflict.message.contains("revision conflict"));

    let (left, right) = tokio::join!(
        one.save(key.clone(), Some(1), checkpoint("left update")),
        two.save(key.clone(), Some(1), checkpoint("right update")),
    );
    assert_ne!(left.is_ok(), right.is_ok());
    let winning = left.as_ref().ok().or_else(|| right.as_ref().ok()).unwrap();
    assert_eq!(winning.revision, 2);
    assert_eq!(two.load(&key).await.unwrap().unwrap(), *winning);
    assert!(
        one.save(key.clone(), Some(1), checkpoint("stale"))
            .await
            .is_err()
    );
    assert_eq!(one.load(&key).await.unwrap().unwrap(), *winning);

    for isolated in [
        StateKey::new("other", "race").unwrap(),
        StateKey::new("user", "other").unwrap(),
    ] {
        assert!(one.load(&isolated).await.unwrap().is_none());
        let saved = one
            .save(isolated.clone(), None, checkpoint("isolated"))
            .await
            .unwrap();
        assert_eq!(saved.revision, 1);
        assert_eq!(two.load(&isolated).await.unwrap().unwrap(), saved);
    }
    assert_eq!(one.load(&key).await.unwrap().unwrap(), *winning);
}

#[tokio::test]
async fn same_database_and_key_keep_agent_sequential_and_parallel_records_independent() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("all-stores.db");
    let key = StateKey::new("user", "shared-key").unwrap();
    let agents = SQLiteStateStore::open(&path).await.unwrap();
    let sequential = SQLitePipelineStore::open(&path).await.unwrap();
    let parallel = SQLiteParallelStore::open(&path).await.unwrap();
    let agent_state = AgentState::new("worker", vec![Msg::user("agent history")]);
    let sequential_checkpoint = PipelineCheckpoint {
        version: PIPELINE_CHECKPOINT_VERSION,
        agent_names: vec!["worker".into()],
        completed: Vec::new(),
        next_input: Msg::user("sequential task"),
        status: PipelineCheckpointStatus::Ready,
    };
    let parallel_checkpoint = checkpoint("parallel task");
    let (agent, sequential_record, parallel_record) = tokio::join!(
        agents.save(key.clone(), None, agent_state.clone()),
        sequential.save(key.clone(), None, sequential_checkpoint.clone()),
        parallel.save(key.clone(), None, parallel_checkpoint.clone()),
    );
    assert_eq!(agent.unwrap().revision(), 1);
    assert_eq!(sequential_record.unwrap().revision, 1);
    assert_eq!(parallel_record.unwrap().revision, 1);
    let updated_parallel = checkpoint("updated parallel task");
    parallel
        .save(key.clone(), Some(1), updated_parallel.clone())
        .await
        .unwrap();
    drop(agents);
    drop(sequential);
    drop(parallel);

    assert_eq!(
        SQLiteStateStore::open(&path)
            .await
            .unwrap()
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .state(),
        &agent_state
    );
    let sequential_record = SQLitePipelineStore::open(&path)
        .await
        .unwrap()
        .load(&key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(sequential_record.revision, 1);
    assert_eq!(sequential_record.checkpoint, sequential_checkpoint);
    let parallel_record = SQLiteParallelStore::open(&path)
        .await
        .unwrap()
        .load(&key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(parallel_record.revision, 2);
    assert_eq!(parallel_record.checkpoint, updated_parallel);
}

#[tokio::test]
async fn rejects_corrupt_json_invalid_revisions_overflow_and_unknown_schema() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("parallel.db");
    let key = StateKey::new("user", "corrupt").unwrap();
    let store = SQLiteParallelStore::open(&path).await.unwrap();
    let initial = store
        .save(key.clone(), None, checkpoint("original"))
        .await
        .unwrap();
    let raw = tokio_rusqlite::Connection::open(&path).await.unwrap();
    raw.call(|db| {
        db.execute(
            "UPDATE agentscope_parallel_checkpoints SET checkpoint_json = 'invalid'",
            [],
        )
    })
    .await
    .unwrap();
    assert!(store.load(&key).await.is_err());
    store
        .save(
            key.clone(),
            Some(initial.revision),
            initial.checkpoint.clone(),
        )
        .await
        .unwrap();

    for revision in [0, -1] {
        raw.call(move |db| {
            db.execute_batch("PRAGMA ignore_check_constraints = ON;")?;
            db.execute(
                "UPDATE agentscope_parallel_checkpoints SET revision = ?1",
                [revision],
            )
        })
        .await
        .unwrap();
        assert!(
            store
                .load(&key)
                .await
                .unwrap_err()
                .message
                .contains("invalid parallel revision")
        );
        assert!(
            store
                .save(key.clone(), Some(0), checkpoint("must not overwrite"))
                .await
                .unwrap_err()
                .message
                .contains("invalid parallel revision")
        );
    }
    raw.call(|db| {
        db.execute(
            "UPDATE agentscope_parallel_checkpoints SET revision = ?1",
            [i64::MAX],
        )
    })
    .await
    .unwrap();
    let error = store
        .save(
            key.clone(),
            Some(i64::MAX.unsigned_abs()),
            checkpoint("overflow"),
        )
        .await
        .unwrap_err();
    assert!(error.message.contains("revision overflow"));
    let retained = store.load(&key).await.unwrap().unwrap();
    assert_eq!(retained.revision, i64::MAX.unsigned_abs());
    assert_eq!(retained.checkpoint, initial.checkpoint);

    raw.call(|db| db.execute("UPDATE agentscope_parallel_schema SET version = 999", []))
        .await
        .unwrap();
    let error = SQLiteParallelStore::open(&path).await.err().unwrap();
    assert!(
        error
            .message
            .contains("unsupported parallel schema version 999")
    );
}

#[tokio::test]
async fn failed_branch_can_be_explicitly_reconciled_without_replaying_it() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("parallel.db");
    let key = StateKey::new("user", "failed-reconcile").unwrap();
    let store = SQLiteParallelStore::open(&path).await.unwrap();
    let saved = store
        .save(
            key.clone(),
            None,
            ParallelCheckpoint {
                version: PARALLEL_CHECKPOINT_VERSION,
                agent_names: vec!["failed".into(), "ready".into()],
                input: Msg::user("task"),
                branches: vec![
                    ParallelBranchCheckpoint::Failed(Box::new(AgentError::Model(ModelError::new(
                        "known failure",
                    )))),
                    ParallelBranchCheckpoint::Ready,
                ],
            },
        )
        .await
        .unwrap();
    drop(store);

    let reopened = SQLiteParallelStore::open(&path).await.unwrap();
    let skipped = Arc::new(MockChatModel::new("must-not-replay"));
    let next = model("done");
    let pipeline = ParallelPipeline::new(
        vec![
            agent("failed", skipped.clone()),
            agent("ready", next.clone()),
        ],
        1,
    )
    .unwrap();
    let replacement = verified("failed", "verified recovery");
    let reconciled = pipeline
        .reconcile_checkpointed(
            &reopened,
            key.clone(),
            saved.revision,
            1,
            replacement.clone(),
        )
        .await
        .unwrap();
    assert_eq!(reconciled.revision, saved.revision + 1);
    let output = pipeline.resume_checkpointed(&reopened, key).await.unwrap();
    assert_eq!(
        output.branches[0].outcome,
        ParallelBranchOutcome::Completed(replacement)
    );
    assert!(skipped.recorded_requests().is_empty());
    assert_eq!(next.recorded_requests().len(), 1);
}
