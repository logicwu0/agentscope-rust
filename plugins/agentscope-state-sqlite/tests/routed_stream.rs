use agentscope::{
    AgentError, AgentEvent, ChatEvent, ContentBlock, DataBlock, FinishReason, InMemoryMemory,
    MockChatModel, MockTool, Msg, ReActAgent, Role, RoutedCheckpointStatus, RoutedEvent,
    RoutedFailure, RoutedPipeline, RoutedStore, StateKey, StateStore, ToolDefinition, ToolExecutor,
    ToolRegistry, Usage,
};
use agentscope_state_sqlite::{SQLiteRoutedStore, SQLiteStateStore};
use futures_util::StreamExt;
use std::{path::Path, sync::Arc};

fn agent(name: &str, model: Arc<MockChatModel>) -> Arc<ReActAgent> {
    Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))
            .unwrap()
            .with_memory(InMemoryMemory::new()),
    )
}

fn model(text: &str) -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new("offline").with_stream([
        Ok(ChatEvent::ThinkingDelta {
            block_id: "thinking".into(),
            delta: "private fixture reasoning".into(),
        }),
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: text.into(),
        }),
        Ok(ChatEvent::Usage {
            usage: Usage::new(13, 7).with_reasoning_tokens(2),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]))
}

fn input() -> Msg {
    // Data is a local mock fixture, not a claim about provider wire support.
    Msg::new(
        "user",
        Role::User,
        [
            ContentBlock::from("original task"),
            ContentBlock::Data(DataBlock::base64("aGVsbG8=", "application/octet-stream").unwrap()),
        ],
    )
    .with_metadata(
        [("private".into(), serde_json::json!({"tenant": "one"}))]
            .into_iter()
            .collect(),
    )
}

async fn assert_resume_rejected(pipeline: &RoutedPipeline, store: &dyn RoutedStore, key: StateKey) {
    let error = pipeline
        .resume_checkpointed_stream(store, key)
        .await
        .err()
        .expect("in-flight or terminal progress must never replay");
    assert!(matches!(error.cause, RoutedFailure::UnsafeResume(_)));
}

#[tokio::test]
async fn prepared_ready_reopens_and_streams_saved_route_with_complete_committed_reply() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("routed-stream.db");
    let key = StateKey::new("user", "ready").unwrap();
    let original_input = input();
    let unused = Arc::new(MockChatModel::new("must-not-run-before-poll"));
    let pipeline =
        RoutedPipeline::new(vec![("code".into(), agent("coder", unused.clone()))]).unwrap();
    let store = SQLiteRoutedStore::open(&path).await.unwrap();
    let stream = pipeline
        .stream_checkpointed(&store, key.clone(), "code", original_input.clone())
        .await
        .unwrap();
    let ready = store.load(&key).await.unwrap().unwrap();
    assert_eq!(ready.revision, 1);
    assert_eq!(ready.checkpoint.status, RoutedCheckpointStatus::Ready);
    assert_eq!(ready.checkpoint.route, "code");
    assert_eq!(ready.checkpoint.agent_name, "coder");
    assert_eq!(ready.checkpoint.input, original_input);
    assert!(unused.recorded_requests().is_empty());
    drop(stream);
    drop(store);

    let reopened = SQLiteRoutedStore::open(&path).await.unwrap();
    assert_eq!(reopened.load(&key).await.unwrap().unwrap(), ready);
    let selected = model("complete visible reply");
    let other = Arc::new(MockChatModel::new("other-route-must-not-run"));
    let restarted = RoutedPipeline::new(vec![
        ("code".into(), agent("coder", selected.clone())),
        ("new-unrelated-route".into(), agent("other", other.clone())),
    ])
    .unwrap();
    let mut stream = restarted
        .resume_checkpointed_stream(&reopened, key.clone())
        .await
        .unwrap();
    assert!(selected.recorded_requests().is_empty());
    assert!(
        matches!(stream.next().await, Some(RoutedEvent::RouteStarted { route, agent_name }) if route == "code" && agent_name == "coder")
    );
    assert_eq!(
        reopened
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .status,
        RoutedCheckpointStatus::InFlight
    );
    assert!(selected.recorded_requests().is_empty());
    let events = stream.collect::<Vec<_>>().await;
    let Some(RoutedEvent::Finished { output }) = events.last() else {
        panic!("resumed route must finish")
    };
    let observed_reply = events
        .iter()
        .find_map(|event| match event {
            RoutedEvent::Agent {
                event: AgentEvent::Finished { message, .. },
                ..
            } => Some(message),
            _ => None,
        })
        .expect("original Agent Finished must be forwarded");
    assert_eq!(output.message, *observed_reply);
    assert!(
        output
            .message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Thinking(_)))
    );
    assert_eq!(
        output.message.usage,
        Some(Usage::new(13, 7).with_reasoning_tokens(2))
    );
    assert_eq!(
        selected.recorded_requests()[0].messages,
        vec![original_input]
    );
    assert_eq!(selected.recorded_requests().len(), 1);
    assert!(other.recorded_requests().is_empty());
    let committed = reopened.load(&key).await.unwrap().unwrap();
    assert_eq!(committed.revision, 3);
    assert_eq!(
        committed.checkpoint.status,
        RoutedCheckpointStatus::Completed(output.message.clone())
    );
    assert_eq!(
        committed.checkpoint.finished_result(),
        Some(Ok(output.clone()))
    );
    drop(reopened);

    let terminal = SQLiteRoutedStore::open(&path).await.unwrap();
    assert_eq!(terminal.load(&key).await.unwrap().unwrap(), committed);
    assert_resume_rejected(&restarted, &terminal, key).await;
    assert_eq!(selected.recorded_requests().len(), 1);
}

#[tokio::test]
async fn drop_after_selection_or_agent_terminal_requires_reconciliation_without_replay() {
    for after_agent_terminal in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("routed-stream.db");
        let key = StateKey::new("user", "dropped").unwrap();
        let state_key = StateKey::new("user", "child-state").unwrap();
        let state_store = SQLiteStateStore::open(&path).await.unwrap();
        let selected = model("verified local fixture reply");
        let child = Arc::new(
            ReActAgent::from_shared(
                "coder",
                selected.clone(),
                ToolExecutor::new(ToolRegistry::new()),
            )
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_state_store(state_key.clone(), state_store.clone()),
        );
        let pipeline = RoutedPipeline::new(vec![("code".into(), child)]).unwrap();
        let store = SQLiteRoutedStore::open(&path).await.unwrap();
        let mut stream = pipeline
            .stream_checkpointed(&store, key.clone(), "code", input())
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await,
            Some(RoutedEvent::RouteStarted { .. })
        ));
        let verified = if after_agent_terminal {
            loop {
                match stream.next().await.expect("local fixture must finish") {
                    RoutedEvent::Agent {
                        event: AgentEvent::Finished { message, .. },
                        ..
                    } => break message,
                    RoutedEvent::Finished { .. } => {
                        panic!("stop before routed terminal acknowledgement")
                    }
                    _ => {}
                }
            }
        } else {
            // No operation occurred; this is explicit local reconciliation evidence.
            assert!(selected.recorded_requests().is_empty());
            Msg::new(
                "coder",
                Role::Assistant,
                [ContentBlock::from("verified local no-effect fixture")],
            )
        };
        let inflight = store.load(&key).await.unwrap().unwrap();
        assert_eq!(inflight.revision, 2);
        assert_eq!(inflight.checkpoint.status, RoutedCheckpointStatus::InFlight);
        if after_agent_terminal {
            let child_record = state_store.load(&state_key).await.unwrap().unwrap();
            assert_eq!(child_record.state().messages().last(), Some(&verified));
            assert_eq!(selected.recorded_requests().len(), 1);
        } else {
            assert!(state_store.load(&state_key).await.unwrap().is_none());
        }
        drop(stream);
        drop(store);
        drop(pipeline);
        drop(state_store);
        reconcile_after_reopen(&path, key, inflight.revision, verified).await;
        assert_eq!(
            selected.recorded_requests().len(),
            usize::from(after_agent_terminal)
        );
    }
}

async fn reconcile_after_reopen(path: &Path, key: StateKey, revision: u64, verified: Msg) {
    let reopened = SQLiteRoutedStore::open(path).await.unwrap();
    let unused = Arc::new(MockChatModel::new("must-not-replay"));
    let restarted =
        RoutedPipeline::new(vec![("code".into(), agent("coder", unused.clone()))]).unwrap();
    assert_eq!(
        reopened
            .load(&key)
            .await
            .unwrap()
            .unwrap()
            .checkpoint
            .status,
        RoutedCheckpointStatus::InFlight
    );
    assert_resume_rejected(&restarted, &reopened, key.clone()).await;
    let reconciled = restarted
        .reconcile_checkpointed(&reopened, key.clone(), revision, verified.clone())
        .await
        .unwrap();
    assert_eq!(reconciled.revision, revision + 1);
    assert_eq!(
        reconciled.checkpoint.status,
        RoutedCheckpointStatus::Completed(verified)
    );
    assert_eq!(reopened.load(&key).await.unwrap().unwrap(), reconciled);
    let result = reconciled
        .checkpoint
        .finished_result()
        .expect("reconciled checkpoint is terminal")
        .unwrap();
    assert_eq!(result.route, "code");
    assert_resume_rejected(&restarted, &reopened, key).await;
    assert!(unused.recorded_requests().is_empty());
}

#[tokio::test]
async fn confirmation_failure_commits_original_pending_payload_without_approving_tools() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("routed-stream.db");
    let key = StateKey::new("user", "confirmation").unwrap();
    let state_key = StateKey::new("user", "confirmation-child").unwrap();
    let selected = Arc::new(MockChatModel::new("offline-confirmation").with_stream([
        Ok(ChatEvent::ToolCallDelta {
            tool_call_id: "call-1".into(),
            tool_name: "write".into(),
            delta: r#"{"payload":"local"}"#.into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::ToolCalls,
        }),
    ]));
    let tool = Arc::new(
        MockTool::new(
            ToolDefinition::new(
                "write",
                "local fixture only",
                serde_json::json!({"type": "object"}),
            )
            .unwrap(),
        )
        .with_output("must not run without confirmation"),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let state_store = SQLiteStateStore::open(&path).await.unwrap();
    let child = Arc::new(
        ReActAgent::from_shared("coder", selected.clone(), ToolExecutor::new(registry))
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_tool_confirmation_required("write")
            .with_state_store(state_key.clone(), state_store.clone()),
    );
    let pipeline = RoutedPipeline::new(vec![("code".into(), child)]).unwrap();
    let store = SQLiteRoutedStore::open(&path).await.unwrap();
    let events = pipeline
        .stream_checkpointed(&store, key.clone(), "code", input())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let pending = events
        .iter()
        .find_map(|event| match event {
            RoutedEvent::Agent {
                event: AgentEvent::ToolConfirmationRequired { checkpoint },
                ..
            } => Some(checkpoint),
            _ => None,
        })
        .expect("agent confirmation must be forwarded")
        .clone();
    let Some(RoutedEvent::Error { error }) = events.last() else {
        panic!("confirmation must terminate routed execution with its original error")
    };
    let original_error = AgentError::ToolConfirmationRequired {
        checkpoint: pending.clone(),
    };
    assert_eq!(
        error.cause,
        RoutedFailure::Agent(Box::new(original_error.clone()))
    );
    assert_eq!(
        state_store
            .load(&state_key)
            .await
            .unwrap()
            .unwrap()
            .state()
            .pending_tool_calls(),
        Some(&pending)
    );
    assert!(tool.recorded_invocations().is_empty());
    drop(store);
    drop(pipeline);
    drop(state_store);

    let reopened = SQLiteRoutedStore::open(&path).await.unwrap();
    let committed = reopened.load(&key).await.unwrap().unwrap();
    assert_eq!(committed.revision, 3);
    assert_eq!(
        committed.checkpoint.status,
        RoutedCheckpointStatus::Failed(Box::new(original_error))
    );
    assert_eq!(
        committed.checkpoint.finished_result(),
        Some(Err(error.clone()))
    );
    let unused = Arc::new(MockChatModel::new("must-not-approve-or-replay"));
    let restarted =
        RoutedPipeline::new(vec![("code".into(), agent("coder", unused.clone()))]).unwrap();
    assert_resume_rejected(&restarted, &reopened, key).await;
    assert!(unused.recorded_requests().is_empty());
    assert_eq!(selected.recorded_requests().len(), 1);
    assert!(tool.recorded_invocations().is_empty());
}
