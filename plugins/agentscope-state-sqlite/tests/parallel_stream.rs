use agentscope::{
    AgentEvent, ChatEvent, ChatEventStream, ChatModel, ChatRequest, ChatResponse, ContentBlock,
    FinishReason, InMemoryMemory, MockChatModel, ModelCapabilities, ModelError, ModelFuture, Msg,
    ParallelBranchCheckpoint, ParallelBranchOutcome, ParallelEvent, ParallelFailure,
    ParallelPipeline, ParallelRecord, ParallelStore, ReActAgent, Role, StateKey, StateStore,
    ToolExecutor, ToolRegistry,
};
use agentscope_state_sqlite::{SQLiteParallelStore, SQLiteStateStore};
use futures_util::StreamExt;
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::Poll,
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
    Arc::new(MockChatModel::new("offline").with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: text.into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]))
}

fn verified(name: &str, text: &str) -> Msg {
    Msg::new(name, Role::Assistant, [ContentBlock::from(text)])
}

#[tokio::test]
async fn reopened_boundary_resumes_only_ready_without_replaying_committed_success_or_failure() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("parallel-stream.db");
    let key = StateKey::new("user", "boundary").unwrap();
    let input = Msg::user("same task after restart");
    let first = model("first completed");
    let failed =
        Arc::new(MockChatModel::new("failed").with_stream_error(ModelError::new("second failed")));
    let third = Arc::new(MockChatModel::new("queued"));
    let original = ParallelPipeline::new(
        vec![
            agent("first", first.clone()),
            agent("second", failed.clone()),
            agent("third", third.clone()),
        ],
        1,
    )
    .unwrap();
    let store = SQLiteParallelStore::open(&path).await.unwrap();
    let mut events = original
        .stream_checkpointed(&store, key.clone(), input.clone())
        .await
        .unwrap();
    loop {
        let event = events.next().await.expect("second branch must finish");
        if matches!(event, ParallelEvent::BranchFinished { result } if result.branch == 2) {
            break;
        }
    }
    let boundary = store.load(&key).await.unwrap().unwrap();
    assert!(matches!(
        boundary.checkpoint.branches[0],
        ParallelBranchCheckpoint::Completed(_)
    ));
    assert!(matches!(
        boundary.checkpoint.branches[1],
        ParallelBranchCheckpoint::Failed(_)
    ));
    assert_eq!(
        boundary.checkpoint.branches[2],
        ParallelBranchCheckpoint::Ready
    );
    assert!(third.recorded_requests().is_empty());
    drop(events);
    drop(store);
    resume_committed_boundary(&path, key, boundary, input).await;
}

async fn resume_committed_boundary(
    path: &Path,
    key: StateKey,
    boundary: ParallelRecord,
    input: Msg,
) {
    let reopened = SQLiteParallelStore::open(path).await.unwrap();
    assert_eq!(reopened.load(&key).await.unwrap().unwrap(), boundary);
    let skipped_first = Arc::new(MockChatModel::new("must-not-repeat-first"));
    let skipped_failed = Arc::new(MockChatModel::new("must-not-repeat-failure"));
    let resumed = model("third completed");
    let restarted = ParallelPipeline::new(
        vec![
            agent("first", skipped_first.clone()),
            agent("second", skipped_failed.clone()),
            agent("third", resumed.clone()),
        ],
        2,
    )
    .unwrap();
    let observed = restarted
        .resume_checkpointed_stream(&reopened, key.clone())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let started: Vec<_> = observed
        .iter()
        .filter_map(|event| match event {
            ParallelEvent::BranchStarted { branch, .. } => Some(*branch),
            _ => None,
        })
        .collect();
    assert_eq!(started, vec![3]);
    let Some(ParallelEvent::Error { error }) = observed.last() else {
        panic!("retained failed branch must produce aggregate Error")
    };
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    assert!(matches!(
        error.branches[0].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert!(matches!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Failed(_)
    ));
    assert!(matches!(
        error.branches[2].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert_eq!(
        reopened
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .finished_result(),
        Some(Err(error.clone()))
    );
    assert!(skipped_first.recorded_requests().is_empty());
    assert!(skipped_failed.recorded_requests().is_empty());
    assert_eq!(resumed.recorded_requests().len(), 1);
    assert_eq!(resumed.recorded_requests()[0].messages, vec![input]);
    assert!(matches!(
        restarted
            .resume_checkpointed_stream(&reopened, key)
            .await
            .err()
            .unwrap()
            .cause,
        ParallelFailure::UnsafeResume(_)
    ));
    assert_eq!(resumed.recorded_requests().len(), 1);
}

#[tokio::test]
async fn dropping_selected_or_agent_terminal_before_branch_finished_keeps_durable_inflight() {
    for after_agent_terminal in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("parallel-stream.db");
        let key = StateKey::new("user", "dropped").unwrap();
        let state_key = StateKey::new("user", "first-agent").unwrap();
        let state_store = SQLiteStateStore::open(&path).await.unwrap();
        let first = model("first completed");
        let second = Arc::new(MockChatModel::new("queued"));
        let first_agent = ReActAgent::from_shared(
            "first",
            first.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_state_store(state_key.clone(), state_store.clone());
        let original = ParallelPipeline::new(
            vec![Arc::new(first_agent), agent("second", second.clone())],
            1,
        )
        .unwrap();
        let store = SQLiteParallelStore::open(&path).await.unwrap();
        let mut events = original
            .stream_checkpointed(&store, key.clone(), Msg::user("task"))
            .await
            .unwrap();
        assert!(matches!(
            events.next().await,
            Some(ParallelEvent::BranchStarted { branch: 1, .. })
        ));
        let evidence = if after_agent_terminal {
            loop {
                match events.next().await.expect("first agent must complete") {
                    ParallelEvent::Agent {
                        branch: 1,
                        event: AgentEvent::Finished { message, .. },
                        ..
                    } => break message,
                    ParallelEvent::BranchFinished { .. } => {
                        panic!("must stop before BranchFinished")
                    }
                    _ => {}
                }
            }
        } else {
            verified("first", "externally verified first")
        };
        let record = store.load(&key).await.unwrap().unwrap();
        assert_eq!(
            record.checkpoint.branches,
            vec![
                ParallelBranchCheckpoint::InFlight,
                ParallelBranchCheckpoint::Ready
            ]
        );
        assert!(second.recorded_requests().is_empty());
        if after_agent_terminal {
            assert_eq!(first.recorded_requests().len(), 1);
            let agent_record = state_store.load(&state_key).await.unwrap().unwrap();
            assert_eq!(agent_record.state().messages().last(), Some(&evidence));
        } else {
            assert!(first.recorded_requests().is_empty());
            assert!(state_store.load(&state_key).await.unwrap().is_none());
        }
        drop(events);
        drop(store);
        drop(original);
        drop(state_store);
        reconcile_and_resume(&path, key, record.revision, evidence).await;
    }
}

async fn reconcile_and_resume(path: &Path, key: StateKey, revision: u64, evidence: Msg) {
    let reopened = SQLiteParallelStore::open(path).await.unwrap();
    let skipped = Arc::new(MockChatModel::new("must-not-replay"));
    let next = model("second completed");
    let restarted = ParallelPipeline::new(
        vec![
            agent("first", skipped.clone()),
            agent("second", next.clone()),
        ],
        2,
    )
    .unwrap();
    assert!(matches!(
        restarted
            .resume_checkpointed_stream(&reopened, key.clone())
            .await
            .err()
            .unwrap()
            .cause,
        ParallelFailure::UnsafeResume(_)
    ));
    assert!(skipped.recorded_requests().is_empty());
    assert!(next.recorded_requests().is_empty());
    let reconciled = restarted
        .reconcile_checkpointed(&reopened, key.clone(), revision, 1, evidence.clone())
        .await
        .unwrap();
    assert_eq!(reconciled.revision, revision + 1);
    drop(reopened);
    let reopened_again = SQLiteParallelStore::open(path).await.unwrap();
    let observed = restarted
        .resume_checkpointed_stream(&reopened_again, key.clone())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let Some(ParallelEvent::Finished { output }) = observed.last() else {
        panic!("reconciled run must finish")
    };
    assert_eq!(
        output.branches[0].outcome,
        ParallelBranchOutcome::Completed(evidence)
    );
    assert!(skipped.recorded_requests().is_empty());
    assert_eq!(next.recorded_requests().len(), 1);
    assert_eq!(
        reopened_again
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
async fn awaited_but_unpolled_stream_remains_ready_across_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("parallel-stream.db");
    let key = StateKey::new("user", "unpolled").unwrap();
    let input = Msg::user("original task");
    let unused = Arc::new(MockChatModel::new("unused"));
    let pipeline = ParallelPipeline::new(vec![agent("worker", unused.clone())], 1).unwrap();
    let store = SQLiteParallelStore::open(&path).await.unwrap();
    let events = pipeline
        .stream_checkpointed(&store, key.clone(), input.clone())
        .await
        .unwrap();
    let prepared = store.load(&key).await.unwrap().unwrap();
    assert_eq!(prepared.revision, 1);
    assert_eq!(
        prepared.checkpoint.branches,
        vec![ParallelBranchCheckpoint::Ready]
    );
    assert!(unused.recorded_requests().is_empty());
    drop(events);
    drop(store);

    let reopened = SQLiteParallelStore::open(&path).await.unwrap();
    assert_eq!(reopened.load(&key).await.unwrap().unwrap(), prepared);
    let resumed = model("done");
    let restarted = ParallelPipeline::new(vec![agent("worker", resumed.clone())], 1).unwrap();
    let observed = restarted
        .resume_checkpointed_stream(&reopened, key)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        observed.last(),
        Some(ParallelEvent::Finished { .. })
    ));
    assert_eq!(resumed.recorded_requests()[0].messages, vec![input]);
}

#[derive(Default)]
struct ActiveStreams {
    opened: AtomicUsize,
    active: AtomicUsize,
    changed: Notify,
}

struct PendingModel(Arc<ActiveStreams>);
struct ActiveStream(Arc<ActiveStreams>);

impl Drop for ActiveStream {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ChatModel for PendingModel {
    fn name(&self) -> &'static str {
        "pending"
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }
    fn generate(&self, _: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async { Err(ModelError::new("unused non-streaming call")) })
    }
    fn stream(&self, _: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async move {
            self.0.opened.fetch_add(1, Ordering::SeqCst);
            self.0.active.fetch_add(1, Ordering::SeqCst);
            self.0.changed.notify_one();
            let guard = ActiveStream(self.0.clone());
            let events = futures_util::stream::poll_fn(move |_| {
                let _alive = &guard;
                Poll::Pending
            });
            Ok(Box::pin(events) as ChatEventStream<'_>)
        })
    }
}

#[tokio::test]
async fn interruption_drops_active_siblings_before_error_and_reopen_keeps_inflight() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("parallel-stream.db");
    let key = StateKey::new("user", "interrupted").unwrap();
    let activity = Arc::new(ActiveStreams::default());
    let pending = Arc::new(PendingModel(activity.clone()));
    let queued = Arc::new(MockChatModel::new("queued"));
    let pipeline = ParallelPipeline::new(
        vec![
            agent("first", pending.clone()),
            agent("second", pending.clone()),
            agent("third", queued.clone()),
        ],
        2,
    )
    .unwrap();
    let store = SQLiteParallelStore::open(&path).await.unwrap();
    let mut events = pipeline
        .stream_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap();
    while activity.opened.load(Ordering::SeqCst) < 2 {
        tokio::select! {
            item = events.next() => assert!(item.is_some()),
            () = activity.changed.notified() => {}
        }
    }
    assert_eq!(activity.active.load(Ordering::SeqCst), 2);
    pipeline.interrupt_handle().interrupt();
    let Some(ParallelEvent::Error { error }) = events.next().await else {
        panic!("interruption must emit terminal Error")
    };
    assert_eq!(error.cause, ParallelFailure::Interrupted);
    // Assert while retaining the outer stream: Error itself must release siblings.
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert!(queued.recorded_requests().is_empty());
    drop(events);
    drop(store);

    let reopened = SQLiteParallelStore::open(&path).await.unwrap();
    assert_eq!(
        reopened
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .branches,
        vec![
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::InFlight,
            ParallelBranchCheckpoint::Ready,
        ]
    );
    assert!(matches!(
        pipeline
            .resume_checkpointed_stream(&reopened, key)
            .await
            .err()
            .unwrap()
            .cause,
        ParallelFailure::UnsafeResume(_)
    ));
}
