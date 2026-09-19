use agentscope::{
    AgentInterruptHandle, ChatResponse, ContentBlock, InMemoryMemory, MockChatModel, Msg,
    PipelineCheckpoint, PipelineCheckpointStatus, PipelineFailure, PipelineRecord, PipelineStore,
    PipelineStoreFuture, ReActAgent, SequentialPipeline, StateKey, ToolExecutor, ToolRegistry,
};
use agentscope_state_sqlite::SQLitePipelineStore;
use std::sync::Arc;

fn pipeline(model: MockChatModel) -> SequentialPipeline {
    let agent = ReActAgent::new("worker", model, ToolExecutor::new(ToolRegistry::new()))
        .unwrap()
        .with_memory(InMemoryMemory::new());
    SequentialPipeline::new(vec![Arc::new(agent)]).unwrap()
}

struct StopAtReady {
    inner: SQLitePipelineStore,
    interrupt: AgentInterruptHandle,
}

impl PipelineStore for StopAtReady {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<PipelineRecord>> {
        self.inner.load(key)
    }

    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        checkpoint: PipelineCheckpoint,
    ) -> PipelineStoreFuture<'_, PipelineRecord> {
        Box::pin(async move {
            let at_boundary = checkpoint.status == PipelineCheckpointStatus::Ready
                && checkpoint.completed.len() == 1;
            let record = self.inner.save(key, expected_revision, checkpoint).await?;
            if at_boundary {
                self.interrupt.interrupt();
            }
            Ok(record)
        })
    }
}

#[tokio::test]
async fn sqlite_ready_boundary_resumes_after_reopen_without_replaying_first_stage() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pipeline.db");
    let key = StateKey::new("user", "ready-boundary").unwrap();
    let writer_model = Arc::new(
        MockChatModel::new("writer")
            .with_response(ChatResponse::completed([ContentBlock::from("draft")])),
    );
    let reviewer_model = Arc::new(
        MockChatModel::new("reviewer")
            .with_response(ChatResponse::completed([ContentBlock::from("reviewed")])),
    );
    let writer = Arc::new(
        ReActAgent::from_shared(
            "writer",
            writer_model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new()),
    );
    let reviewer = Arc::new(
        ReActAgent::from_shared(
            "reviewer",
            reviewer_model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new()),
    );
    let pipeline = SequentialPipeline::new(vec![writer, reviewer]).unwrap();
    let store = StopAtReady {
        inner: SQLitePipelineStore::open(&path).await.unwrap(),
        interrupt: pipeline.interrupt_handle(),
    };
    let error = pipeline
        .run_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap_err();
    assert_eq!(error.cause, PipelineFailure::Interrupted);
    assert_eq!(writer_model.recorded_requests().len(), 1);
    assert!(reviewer_model.recorded_requests().is_empty());
    drop(store);

    let reopened = SQLitePipelineStore::open(&path).await.unwrap();
    let record = reopened.load(&key).await.unwrap().unwrap();
    assert_eq!(record.checkpoint.status, PipelineCheckpointStatus::Ready);
    let restarted_writer_model = Arc::new(MockChatModel::new("must-not-run"));
    let restarted = SequentialPipeline::new(vec![
        Arc::new(
            ReActAgent::from_shared(
                "writer",
                restarted_writer_model.clone(),
                ToolExecutor::new(ToolRegistry::new()),
            )
            .unwrap()
            .with_memory(InMemoryMemory::new()),
        ),
        Arc::new(
            ReActAgent::from_shared(
                "reviewer",
                reviewer_model.clone(),
                ToolExecutor::new(ToolRegistry::new()),
            )
            .unwrap()
            .with_memory(InMemoryMemory::new()),
        ),
    ])
    .unwrap();
    let output = restarted.resume_checkpointed(&reopened, key).await.unwrap();
    assert_eq!(output.message.text_content("").as_deref(), Some("reviewed"));
    assert_eq!(writer_model.recorded_requests().len(), 1);
    assert!(restarted_writer_model.recorded_requests().is_empty());
    assert_eq!(reviewer_model.recorded_requests().len(), 1);
}

#[tokio::test]
async fn sqlite_pipeline_checkpoint_survives_reopen_and_rejects_replay() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pipeline.db");
    let key = StateKey::new("user", "finished").unwrap();
    let store = SQLitePipelineStore::open(&path).await.unwrap();
    let completed = pipeline(
        MockChatModel::new("offline")
            .with_response(ChatResponse::completed([ContentBlock::from("done")])),
    );
    let output = completed
        .run_checkpointed(&store, key.clone(), Msg::user("task"))
        .await
        .unwrap();
    assert_eq!(output.steps.len(), 1);
    drop(store);

    let reopened = SQLitePipelineStore::open(&path).await.unwrap();
    let record = reopened.load(&key).await.unwrap().unwrap();
    assert_eq!(record.revision, 3);
    assert_eq!(record.checkpoint.status, PipelineCheckpointStatus::Finished);
    assert_eq!(record.checkpoint.completed, output.steps);
    let retry = pipeline(MockChatModel::new("unused"));
    assert!(matches!(
        retry.resume_checkpointed(&reopened, key).await,
        Err(agentscope::PipelineError {
            cause: PipelineFailure::UnsafeResume(_),
            ..
        })
    ));
}

#[tokio::test]
async fn sqlite_pipeline_checkpoint_keeps_inflight_on_failure() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pipeline.db");
    let key = StateKey::new("user", "in-flight").unwrap();
    let store = SQLitePipelineStore::open(&path).await.unwrap();
    let failed = pipeline(
        MockChatModel::new("offline").with_error(agentscope::ModelError::new("maybe executed")),
    );
    assert!(
        failed
            .run_checkpointed(&store, key.clone(), Msg::user("task"))
            .await
            .is_err()
    );
    drop(store);

    let reopened = SQLitePipelineStore::open(&path).await.unwrap();
    let record = reopened.load(&key).await.unwrap().unwrap();
    assert_eq!(record.revision, 2);
    assert_eq!(record.checkpoint.status, PipelineCheckpointStatus::InFlight);
    let retry = pipeline(MockChatModel::new("unused"));
    assert!(matches!(
        retry.resume_checkpointed(&reopened, key).await,
        Err(agentscope::PipelineError {
            cause: PipelineFailure::UnsafeResume(_),
            ..
        })
    ));
}
