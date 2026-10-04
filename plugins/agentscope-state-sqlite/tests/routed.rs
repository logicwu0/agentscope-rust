use agentscope::{
    AgentError, AgentState, ChatResponse, ContentBlock, DataBlock, InMemoryMemory, MockChatModel,
    MockTool, ModelError, Msg, PARALLEL_CHECKPOINT_VERSION, PIPELINE_CHECKPOINT_VERSION,
    ParallelBranchCheckpoint, ParallelCheckpoint, ParallelStore, PendingToolCalls,
    PendingToolExecution, PipelineCheckpoint, PipelineCheckpointStatus, PipelineStore,
    ROUTED_CHECKPOINT_VERSION, ReActAgent, Role, RoutedCheckpoint, RoutedCheckpointStatus,
    RoutedFailure, RoutedPipeline, RoutedStore, StateKey, StateStore, ThinkingBlock, ToolCallBlock,
    ToolConfirmation, ToolDefinition, ToolExecutor, ToolRegistry, Usage,
};
use agentscope_state_sqlite::{
    SQLiteParallelStore, SQLitePipelineStore, SQLiteRoutedStore, SQLiteStateStore,
};
use std::sync::Arc;

fn checkpoint(input: &str) -> RoutedCheckpoint {
    RoutedCheckpoint {
        version: ROUTED_CHECKPOINT_VERSION,
        route: "code".into(),
        agent_name: "coder".into(),
        input: Msg::user(input),
        status: RoutedCheckpointStatus::Ready,
    }
}

fn rich_message(name: &str, role: Role) -> Msg {
    Msg::new(
        name,
        role,
        [
            ContentBlock::Thinking(ThinkingBlock::new("private reasoning")),
            ContentBlock::from("visible text"),
            ContentBlock::Data(
                DataBlock::base64("aGVsbG8=", "application/octet-stream")
                    .unwrap()
                    .with_name("result.bin"),
            ),
        ],
    )
    .with_metadata(
        [(
            "private".into(),
            serde_json::json!({"tenant": "one", "tags": [1, 2]}),
        )]
        .into_iter()
        .collect(),
    )
    .with_usage(Usage::new(13, 7).with_reasoning_tokens(2))
}

#[tokio::test]
async fn reopens_rich_messages_and_all_progress_states_with_isolated_keys() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("routed.db");
    let store = SQLiteRoutedStore::open(&path).await.unwrap();
    let keys = [("one", "session"), ("two", "session"), ("one", "other")]
        .map(|(user, session)| StateKey::new(user, session).unwrap());
    let states = [
        RoutedCheckpointStatus::Ready,
        RoutedCheckpointStatus::InFlight,
        RoutedCheckpointStatus::Completed(rich_message("coder", Role::Assistant)),
    ];
    let mut saved = Vec::new();
    for (key, status) in keys.iter().zip(states) {
        assert!(store.load(key).await.unwrap().is_none());
        let mut progress = checkpoint("task");
        progress.input = rich_message("user", Role::User);
        progress.status = status;
        let record = store
            .save(key.clone(), None, progress.clone())
            .await
            .unwrap();
        assert_eq!(record.revision, 1);
        assert_eq!(record.checkpoint, progress);
        saved.push(record);
    }
    drop(store);
    let reopened = SQLiteRoutedStore::open(&path).await.unwrap();
    for (key, record) in keys.iter().zip(saved) {
        assert_eq!(reopened.load(key).await.unwrap().unwrap(), record);
    }
    // The storage layer preserves metadata rather than judging workflow validity.
    let invalid_key = StateKey::new("one", "invalid-metadata").unwrap();
    let mut invalid = checkpoint("kept as supplied");
    invalid.version = 999;
    invalid.route = " ".into();
    invalid.agent_name.clear();
    let stored = reopened
        .save(invalid_key.clone(), None, invalid.clone())
        .await
        .unwrap();
    assert_eq!(stored.checkpoint, invalid);
    assert_eq!(reopened.load(&invalid_key).await.unwrap().unwrap(), stored);
}

#[tokio::test]
async fn reopens_structured_agent_errors_without_losing_recovery_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("errors.db");
    let call = ToolCallBlock::complete("call-1", "write", r#"{"value":42}"#).unwrap();
    let confirmation: PendingToolCalls = serde_json::from_value(serde_json::json!({
        "reply_id": "reply-1", "step": 2, "calls": [call]
    }))
    .unwrap();
    let execution: PendingToolExecution = serde_json::from_value(serde_json::json!({
        "execution_id": "execution-1", "confirmation": confirmation,
        "decisions": [ToolConfirmation::approve("call-1")]
    }))
    .unwrap();
    let errors = [
        AgentError::Model(
            ModelError::new("provider failed")
                .with_code("offline")
                .with_retryable(true),
        ),
        AgentError::ToolConfirmationRequired {
            checkpoint: confirmation,
        },
        AgentError::ToolExecutionInDoubt {
            checkpoint: execution,
        },
    ];
    let store = SQLiteRoutedStore::open(&path).await.unwrap();
    let mut records = Vec::new();
    for (index, error) in errors.into_iter().enumerate() {
        let key = StateKey::new("user", format!("error-{index}")).unwrap();
        let mut progress = checkpoint("original task");
        progress.status = RoutedCheckpointStatus::Failed(Box::new(error));
        records.push((key.clone(), store.save(key, None, progress).await.unwrap()));
    }
    drop(store);
    let reopened = SQLiteRoutedStore::open(&path).await.unwrap();
    for (key, record) in records {
        assert_eq!(reopened.load(&key).await.unwrap().unwrap(), record);
    }
}

#[tokio::test]
async fn independent_connections_have_one_cas_winner_and_reject_stale_or_duplicate_writes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("race.db");
    let key = StateKey::new("user", "race").unwrap();
    let one = SQLiteRoutedStore::open(&path).await.unwrap();
    let two = SQLiteRoutedStore::open(&path).await.unwrap();
    let (left, right) = tokio::join!(
        one.save(key.clone(), None, checkpoint("left create")),
        two.save(key.clone(), None, checkpoint("right create")),
    );
    assert_ne!(left.is_ok(), right.is_ok());
    let created = left.as_ref().ok().or_else(|| right.as_ref().ok()).unwrap();
    assert_eq!(created.revision, 1);
    assert_eq!(one.load(&key).await.unwrap().unwrap(), *created);
    assert!(
        left.err()
            .or_else(|| right.err())
            .unwrap()
            .message
            .contains("revision conflict")
    );
    assert!(
        one.save(key.clone(), None, checkpoint("duplicate"))
            .await
            .is_err()
    );
    let (left, right) = tokio::join!(
        one.save(key.clone(), Some(1), checkpoint("left update")),
        two.save(key.clone(), Some(1), checkpoint("right update")),
    );
    assert_ne!(left.is_ok(), right.is_ok());
    let updated = left.as_ref().ok().or_else(|| right.as_ref().ok()).unwrap();
    assert_eq!(updated.revision, 2);
    assert!(
        one.save(key.clone(), Some(1), checkpoint("stale"))
            .await
            .is_err()
    );
    assert_eq!(two.load(&key).await.unwrap().unwrap(), *updated);
}

#[tokio::test]
async fn four_store_types_share_one_database_and_key_without_crossing_records() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("all-stores.db");
    let key = StateKey::new("user", "same-key").unwrap();
    let agents = SQLiteStateStore::open(&path).await.unwrap();
    let sequential = SQLitePipelineStore::open(&path).await.unwrap();
    let parallel = SQLiteParallelStore::open(&path).await.unwrap();
    let routed = SQLiteRoutedStore::open(&path).await.unwrap();
    let agent_state = AgentState::new("coder", vec![Msg::user("agent history")]);
    let sequential_checkpoint = PipelineCheckpoint {
        version: PIPELINE_CHECKPOINT_VERSION,
        agent_names: vec!["coder".into()],
        completed: Vec::new(),
        next_input: Msg::user("sequential task"),
        status: PipelineCheckpointStatus::Ready,
    };
    let parallel_checkpoint = ParallelCheckpoint {
        version: PARALLEL_CHECKPOINT_VERSION,
        agent_names: vec!["coder".into()],
        input: Msg::user("parallel task"),
        branches: vec![ParallelBranchCheckpoint::Ready],
    };
    let (agent, sequential_record, parallel_record, routed_record) = tokio::join!(
        agents.save(key.clone(), None, agent_state),
        sequential.save(key.clone(), None, sequential_checkpoint),
        parallel.save(key.clone(), None, parallel_checkpoint),
        routed.save(key.clone(), None, checkpoint("routed task")),
    );
    let agent = agent.unwrap();
    let sequential_record = sequential_record.unwrap();
    let parallel_record = parallel_record.unwrap();
    assert_eq!(routed_record.unwrap().revision, 1);
    let updated = routed
        .save(key.clone(), Some(1), checkpoint("updated route"))
        .await
        .unwrap();
    drop(agents);
    drop(sequential);
    drop(parallel);
    drop(routed);
    assert_eq!(
        SQLiteStateStore::open(&path)
            .await
            .unwrap()
            .load(&key)
            .await
            .unwrap()
            .unwrap(),
        agent
    );
    assert_eq!(
        SQLitePipelineStore::open(&path)
            .await
            .unwrap()
            .load(&key)
            .await
            .unwrap()
            .unwrap(),
        sequential_record
    );
    assert_eq!(
        SQLiteParallelStore::open(&path)
            .await
            .unwrap()
            .load(&key)
            .await
            .unwrap()
            .unwrap(),
        parallel_record
    );
    assert_eq!(
        SQLiteRoutedStore::open(&path)
            .await
            .unwrap()
            .load(&key)
            .await
            .unwrap()
            .unwrap(),
        updated
    );
}

#[tokio::test]
async fn rejects_corrupt_json_nonpositive_revisions_and_overflow_without_overwriting() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("corrupt.db");
    let key = StateKey::new("user", "corrupt").unwrap();
    let store = SQLiteRoutedStore::open(&path).await.unwrap();
    let initial = store
        .save(key.clone(), None, checkpoint("original"))
        .await
        .unwrap();
    let raw = tokio_rusqlite::Connection::open(&path).await.unwrap();
    raw.call(|db| {
        db.execute(
            "UPDATE agentscope_routed_checkpoints SET checkpoint_json = 'invalid'",
            [],
        )
    })
    .await
    .unwrap();
    assert!(store.load(&key).await.is_err());
    store
        .save(key.clone(), Some(1), initial.checkpoint.clone())
        .await
        .unwrap();
    for revision in [0_i64, -1] {
        raw.call(move |db| {
            db.execute_batch("PRAGMA ignore_check_constraints = ON;")?;
            db.execute(
                "UPDATE agentscope_routed_checkpoints SET revision = ?1",
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
                .contains("invalid routed revision")
        );
        assert!(
            store
                .save(key.clone(), Some(0), checkpoint("must not write"))
                .await
                .unwrap_err()
                .message
                .contains("invalid routed revision")
        );
    }
    raw.call(|db| {
        db.execute(
            "UPDATE agentscope_routed_checkpoints SET revision = ?1",
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
}

#[tokio::test]
async fn unsupported_schema_is_rejected_without_creating_checkpoint_tables() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("unsupported.db");
    let raw = tokio_rusqlite::Connection::open(&path).await.unwrap();
    raw.call(|db| db.execute_batch(
        "CREATE TABLE agentscope_routed_schema (singleton INTEGER PRIMARY KEY, version INTEGER NOT NULL);
         INSERT INTO agentscope_routed_schema VALUES (1, 999);"
    )).await.unwrap();
    let error = SQLiteRoutedStore::open(&path).await.err().unwrap();
    assert!(
        error
            .message
            .contains("unsupported routed schema version 999")
    );
    let tables: i64 = raw.call(|db| db.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'agentscope_routed_checkpoints'",
        [], |row| row.get(0),
    )).await.unwrap();
    assert_eq!(tables, 0);
    let version: i64 = raw
        .call(|db| {
            db.query_row(
                "SELECT version FROM agentscope_routed_schema WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
        })
        .await
        .unwrap();
    assert_eq!(version, 999);
}

fn resume_agent(name: &str, model: Arc<MockChatModel>, tool: Arc<MockTool>) -> Arc<ReActAgent> {
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool).unwrap();
    Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(registry))
            .unwrap()
            .with_memory(InMemoryMemory::new()),
    )
}

#[tokio::test]
async fn reopened_ready_route_executes_once_while_terminal_and_inflight_never_replay() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("route-recovery.db");
    let ready_key = StateKey::new("user", "ready").unwrap();
    let inflight_key = StateKey::new("user", "inflight").unwrap();
    let input = rich_message("user", Role::User);
    let mut ready = checkpoint("unused");
    ready.input = input.clone();
    let mut inflight = ready.clone();
    inflight.status = RoutedCheckpointStatus::InFlight;
    let store = SQLiteRoutedStore::open(&path).await.unwrap();
    store.save(ready_key.clone(), None, ready).await.unwrap();
    let uncertain = store
        .save(inflight_key.clone(), None, inflight)
        .await
        .unwrap();
    drop(store);

    let reply = rich_message("coder", Role::Assistant);
    let response = ChatResponse::completed(reply.content)
        .with_metadata(reply.metadata)
        .with_usage(reply.usage.unwrap());
    let code_model = Arc::new(MockChatModel::new("code").with_response(response));
    let writing_model = Arc::new(MockChatModel::new("must-not-run"));
    let tool = Arc::new(MockTool::new(
        ToolDefinition::new(
            "unused",
            "must not execute",
            serde_json::json!({"type":"object"}),
        )
        .unwrap(),
    ));
    let coder = resume_agent("coder", code_model.clone(), tool.clone());
    let writer = resume_agent("writer", writing_model.clone(), tool.clone());
    let pipeline = RoutedPipeline::new(vec![
        ("code".into(), coder.clone()),
        ("writing".into(), writer.clone()),
    ])
    .unwrap();
    let reopened = SQLiteRoutedStore::open(&path).await.unwrap();
    let output = pipeline
        .resume_checkpointed(&reopened, ready_key.clone())
        .await
        .unwrap();
    assert_eq!(output.route, "code");
    assert_eq!(output.agent_name, "coder");
    assert_eq!(code_model.recorded_requests().len(), 1);
    assert_eq!(
        code_model.recorded_requests()[0].messages.last(),
        Some(&input)
    );
    assert!(writing_model.recorded_requests().is_empty());
    assert!(tool.recorded_invocations().is_empty());
    assert_eq!(
        coder.snapshot().await.unwrap().messages(),
        &[input, output.message.clone()]
    );
    assert!(writer.snapshot().await.unwrap().messages().is_empty());
    let committed = reopened.load(&ready_key).await.unwrap().unwrap();
    assert_eq!(committed.revision, 3);
    assert_eq!(
        committed.checkpoint.status,
        RoutedCheckpointStatus::Completed(output.message.clone())
    );
    assert_eq!(committed.checkpoint.finished_result(), Some(Ok(output)));
    drop(reopened);

    let terminal_store = SQLiteRoutedStore::open(&path).await.unwrap();
    for key in [&ready_key, &inflight_key] {
        let before = terminal_store.load(key).await.unwrap().unwrap();
        let error = pipeline
            .resume_checkpointed(&terminal_store, key.clone())
            .await
            .unwrap_err();
        assert!(matches!(error.cause, RoutedFailure::UnsafeResume(_)));
        assert_eq!(terminal_store.load(key).await.unwrap().unwrap(), before);
    }
    assert_eq!(
        terminal_store.load(&ready_key).await.unwrap().unwrap(),
        committed
    );
    assert_eq!(
        terminal_store.load(&inflight_key).await.unwrap().unwrap(),
        uncertain
    );
    assert_eq!(code_model.recorded_requests().len(), 1);
    assert!(writing_model.recorded_requests().is_empty());
    assert!(tool.recorded_invocations().is_empty());
}
