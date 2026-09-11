use super::*;
use crate::{AgentEventStream, ContextPolicy, FullContext, RecentTurns, ToolResultBlock};

fn old_history() -> Vec<Msg> {
    vec![
        Msg::user("old question"),
        Msg::assistant("Friday", "old answer"),
    ]
}

fn tool_response() -> ChatResponse {
    ChatResponse::finished(
        [ContentBlock::from(
            ToolCallBlock::complete("call-stream-1", "calculator", r#"{"expression":"6*7"}"#)
                .unwrap(),
        )],
        FinishReason::ToolCalls,
    )
}

fn answer_response() -> ChatResponse {
    ChatResponse::finished([ContentBlock::from("42")], FinishReason::Completed)
}

fn executor(tool: Arc<MockTool>) -> ToolExecutor {
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool).unwrap();
    ToolExecutor::new(registry)
}

async fn consume(stream: AgentEventStream<'_>) {
    let events = stream.collect::<Vec<_>>().await;
    assert!(matches!(
        events.last(),
        Some(Ok(AgentEvent::Finished { .. }))
    ));
    assert!(
        events
            .iter()
            .all(|event| !matches!(event, Err(_) | Ok(AgentEvent::Error { .. })))
    );
}

#[test]
fn default_keeps_full_history_and_custom_policy_is_shared_by_clones() {
    struct CountingPolicy(Arc<AtomicUsize>);
    impl ContextPolicy for CountingPolicy {
        fn select_messages(&self, history: &[Msg]) -> Vec<Msg> {
            self.0.fetch_add(1, Ordering::SeqCst);
            FullContext.select_messages(history)
        }
    }
    block_on(async {
        let model = Arc::new(
            MockChatModel::new("mock")
                .with_response(answer_response())
                .with_response(answer_response()),
        );
        let agent = ReActAgent::from_shared(
            "Friday",
            model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::from_messages(old_history()));
        agent.reply(Msg::user("new")).await.unwrap();
        assert_eq!(model.recorded_requests()[0].messages.len(), 3);
        let calls = Arc::new(AtomicUsize::new(0));
        let policy: Arc<dyn ContextPolicy> = Arc::new(CountingPolicy(calls.clone()));
        let agent = agent.with_shared_context_policy(policy).clone();
        agent.reply(Msg::user("next")).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(model.recorded_requests()[1].messages.len(), 5);
    });
}

#[test]
fn normal_and_streaming_loops_trim_every_request_but_keep_memory() {
    block_on(async {
        for streaming in [false, true] {
            let model = Arc::new(
                MockChatModel::new("mock")
                    .with_response(tool_response())
                    .with_response(answer_response())
                    .with_stream(calculator_call_stream())
                    .with_stream(final_answer_stream()),
            );
            let memory = Arc::new(InMemoryMemory::from_messages(old_history()));
            let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
            let agent = ReActAgent::from_shared("Friday", model.clone(), executor(tool.clone()))
                .unwrap()
                .with_system_prompt("Be precise")
                .with_shared_memory(memory.clone())
                .with_context_policy(RecentTurns::new(1).unwrap());
            if streaming {
                consume(agent.stream(Msg::user("current")).await.unwrap()).await;
            } else {
                agent.reply(Msg::user("current")).await.unwrap();
            }
            assert_model_context(&model);
            let history = memory.messages().await.unwrap();
            assert_eq!(history.len(), 6);
            assert_eq!(history[0].text_content(""), Some("old question".into()));
            assert_eq!(tool.recorded_invocations().len(), 1);
        }
    });
}

fn assert_model_context(model: &MockChatModel) {
    let requests = model.recorded_requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].messages.len(), 2);
    assert_eq!(requests[1].messages.len(), 4);
    for request in &requests {
        assert_eq!(request.messages[0].role, Role::System);
        assert_eq!(
            request.messages[0].text_content(""),
            Some("Be precise".into())
        );
        assert_eq!(request.messages[1].text_content(""), Some("current".into()));
        assert_eq!(request.tools.len(), 1);
    }
    assert!(
        requests[1].messages[2]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolCall(_)))
    );
    assert!(
        requests[1].messages[3]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult(_)))
    );
}

#[test]
fn all_recovery_paths_apply_policy_after_reloading_full_state() {
    block_on(async {
        for streaming in [false, true] {
            for mode in 0..3 {
                let model = Arc::new(
                    MockChatModel::new("mock")
                        .with_response(tool_response())
                        .with_response(answer_response())
                        .with_stream(final_answer_stream()),
                );
                let store = Arc::new(InMemoryStateStore::new());
                let key = StateKey::new("user", "context").unwrap();
                let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
                let build = |memory: InMemoryMemory| {
                    ReActAgent::from_shared("Friday", model.clone(), executor(tool.clone()))
                        .unwrap()
                        .with_system_prompt("Be precise")
                        .with_memory(memory)
                        .with_shared_state_store(key.clone(), store.clone())
                        .with_tool_confirmation_required("calculator")
                        .with_context_policy(RecentTurns::new(1).unwrap())
                };
                let agent = build(InMemoryMemory::from_messages(old_history()));
                let AgentError::ToolConfirmationRequired { checkpoint } =
                    agent.reply(Msg::user("current")).await.unwrap_err()
                else {
                    panic!("expected pause")
                };
                let reply_id = checkpoint.reply_id().to_owned();
                let decisions = vec![ToolConfirmation::approve("call-stream-1")];
                if mode != 0 {
                    let mut stream = agent
                        .stream_resume_tool_calls(&reply_id, decisions.clone())
                        .await
                        .unwrap();
                    assert!(matches!(
                        stream.next().await.unwrap().unwrap(),
                        AgentEvent::ToolStarted { .. }
                    ));
                    drop(stream);
                }
                drop(agent);
                // Rebuild with empty memory: full history/checkpoints must load
                // from the store, while the policy is runtime configuration.
                let agent = build(InMemoryMemory::new());
                let results =
                    vec![ToolResultBlock::success("call-stream-1", "calculator", "42").unwrap()];
                if streaming {
                    let stream = match mode {
                        0 => agent.stream_resume_tool_calls(&reply_id, decisions).await,
                        1 => agent.stream_retry_tool_execution(&reply_id).await,
                        _ => {
                            agent
                                .stream_resolve_tool_execution(&reply_id, results)
                                .await
                        }
                    }
                    .unwrap();
                    consume(stream).await;
                } else {
                    match mode {
                        0 => agent.resume_tool_calls(&reply_id, decisions).await,
                        1 => agent.retry_tool_execution(&reply_id).await,
                        _ => agent.resolve_tool_execution(&reply_id, results).await,
                    }
                    .unwrap();
                }
                assert_model_context(&model);
                let stored = store.load(&key).await.unwrap().unwrap();
                assert_eq!(stored.state().messages().len(), 6);
                assert_eq!(
                    stored.state().messages()[0].text_content(""),
                    Some("old question".into())
                );
                assert!(stored.state().pending_tool_calls().is_none());
                assert!(stored.state().pending_tool_execution().is_none());
                assert_eq!(tool.recorded_invocations().len(), usize::from(mode != 2));
            }
        }
    });
}
