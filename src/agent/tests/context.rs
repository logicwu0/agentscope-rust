use super::*;
use crate::{AgentEventStream, ContextPolicy, FullContext, RecentTurns, ToolResultBlock};
use crate::{TokenBudget, TokenBudgetError, TokenCount, TokenCountAccuracy, TokenCounter};

struct BudgetCounter;
impl TokenCounter for BudgetCounter {
    fn count(&self, request: &ChatRequest) -> Result<TokenCount, TokenBudgetError> {
        let tokens = request
            .messages
            .iter()
            .map(|message| {
                if message
                    .text_content("")
                    .is_some_and(|text| text.starts_with("old"))
                {
                    100
                } else {
                    10
                }
            })
            .sum::<u64>()
            + request.tools.len() as u64 * 20;
        Ok(TokenCount {
            tokens,
            accuracy: TokenCountAccuracy::Estimated,
        })
    }
}

fn configure(agent: ReActAgent, budgeted: bool) -> ReActAgent {
    if budgeted {
        agent.with_token_budget(
            TokenBudget::new(80, 20)
                .unwrap()
                .with_counter(BudgetCounter),
        )
    } else {
        agent.with_context_policy(RecentTurns::new(1).unwrap())
    }
}

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
        for (streaming, budgeted) in [(false, false), (true, false), (false, true), (true, true)] {
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
                .with_shared_memory(memory.clone());
            let agent = configure(agent, budgeted);
            if streaming {
                consume(agent.stream(Msg::user("current")).await.unwrap()).await;
            } else {
                agent.reply(Msg::user("current")).await.unwrap();
            }
            assert_model_context(&model);
            assert_eq!(
                model.recorded_requests()[0].options.max_tokens,
                budgeted.then_some(20)
            );
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
        for (streaming, budgeted) in [(false, false), (true, false), (false, true), (true, true)] {
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
                    let agent =
                        ReActAgent::from_shared("Friday", model.clone(), executor(tool.clone()))
                            .unwrap()
                            .with_system_prompt("Be precise")
                            .with_memory(memory)
                            .with_shared_state_store(key.clone(), store.clone())
                            .with_tool_confirmation_required("calculator");
                    configure(agent, budgeted)
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
                assert_eq!(
                    model.recorded_requests()[1].options.max_tokens,
                    budgeted.then_some(20)
                );
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

#[test]
fn budget_errors_block_model_calls_and_preserve_saved_history() {
    block_on(async {
        for streaming in [false, true] {
            for after_tool in [false, true] {
                let model = Arc::new(
                    MockChatModel::new("mock")
                        .with_response(tool_response())
                        .with_stream(calculator_call_stream()),
                );
                let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
                let store = Arc::new(InMemoryStateStore::new());
                let key = StateKey::new("user", "budget-failure").unwrap();
                // Initial protected request costs 40; after tools it costs 60.
                let window = if after_tool { 60 } else { 30 };
                let agent =
                    ReActAgent::from_shared("Friday", model.clone(), executor(tool.clone()))
                        .unwrap()
                        .with_system_prompt("Be precise")
                        .with_memory(InMemoryMemory::from_messages(old_history()))
                        .with_shared_state_store(key.clone(), store.clone())
                        .with_token_budget(
                            TokenBudget::new(window, 20)
                                .unwrap()
                                .with_counter(BudgetCounter),
                        );
                if streaming {
                    let events = agent
                        .stream(Msg::user("current"))
                        .await
                        .unwrap()
                        .collect::<Vec<_>>()
                        .await;
                    assert!(matches!(
                        events.last(),
                        Some(Ok(AgentEvent::Error {
                            error: AgentError::TokenBudget(TokenBudgetError::Exceeded { .. }),
                            ..
                        }))
                    ));
                    assert!(
                        !events
                            .iter()
                            .any(|event| matches!(event, Ok(AgentEvent::Finished { .. })))
                    );
                } else {
                    assert!(matches!(
                        agent.reply(Msg::user("current")).await,
                        Err(AgentError::TokenBudget(TokenBudgetError::Exceeded { .. }))
                    ));
                }
                assert_eq!(model.recorded_requests().len(), usize::from(after_tool));
                assert_eq!(tool.recorded_invocations().len(), usize::from(after_tool));
                let state = store.load(&key).await.unwrap().unwrap();
                assert_eq!(
                    state.state().messages().len(),
                    if after_tool { 5 } else { 3 }
                );
                assert_eq!(
                    state.state().messages()[0].text_content(""),
                    Some("old question".into())
                );
            }
        }
    });
}

#[test]
fn recovery_budget_errors_keep_committed_results_and_clear_checkpoints() {
    block_on(async {
        for streaming in [false, true] {
            for mode in 0..3 {
                let model = Arc::new(MockChatModel::new("mock").with_response(tool_response()));
                let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
                let store = Arc::new(InMemoryStateStore::new());
                let key = StateKey::new("user", "budget-recovery-failure").unwrap();
                let agent =
                    ReActAgent::from_shared("Friday", model.clone(), executor(tool.clone()))
                        .unwrap()
                        .with_system_prompt("Be precise")
                        .with_memory(InMemoryMemory::from_messages(old_history()))
                        .with_shared_state_store(key.clone(), store.clone())
                        .with_tool_confirmation_required("calculator");
                let AgentError::ToolConfirmationRequired { checkpoint } =
                    agent.reply(Msg::user("current")).await.unwrap_err()
                else {
                    panic!("expected confirmation")
                };
                let id = checkpoint.reply_id();
                let confirmations = vec![ToolConfirmation::approve("call-stream-1")];
                if mode != 0 {
                    let mut events = agent
                        .stream_resume_tool_calls(id, confirmations.clone())
                        .await
                        .unwrap();
                    assert!(matches!(
                        events.next().await.unwrap().unwrap(),
                        AgentEvent::ToolStarted { .. }
                    ));
                    drop(events);
                }
                let agent = agent.with_token_budget(
                    TokenBudget::new(60, 20)
                        .unwrap()
                        .with_counter(BudgetCounter),
                );
                let results =
                    vec![ToolResultBlock::success("call-stream-1", "calculator", "42").unwrap()];
                let error = if streaming {
                    let stream = match mode {
                        0 => agent.stream_resume_tool_calls(id, confirmations).await,
                        1 => agent.stream_retry_tool_execution(id).await,
                        _ => agent.stream_resolve_tool_execution(id, results).await,
                    }
                    .unwrap();
                    let events = stream.collect::<Vec<_>>().await;
                    assert!(
                        !events
                            .iter()
                            .any(|event| matches!(event, Ok(AgentEvent::Finished { .. })))
                    );
                    let Ok(AgentEvent::Error { error, .. }) = events.last().unwrap() else {
                        panic!("expected error")
                    };
                    error.clone()
                } else {
                    match mode {
                        0 => agent.resume_tool_calls(id, confirmations).await,
                        1 => agent.retry_tool_execution(id).await,
                        _ => agent.resolve_tool_execution(id, results).await,
                    }
                    .unwrap_err()
                };
                assert!(matches!(
                    error,
                    AgentError::TokenBudget(TokenBudgetError::Exceeded { .. })
                ));
                assert_eq!(model.recorded_requests().len(), 1);
                assert_eq!(tool.recorded_invocations().len(), usize::from(mode != 2));
                let saved = store.load(&key).await.unwrap().unwrap();
                assert_eq!(saved.state().messages().len(), 5);
                assert!(saved.state().pending_tool_calls().is_none());
                assert!(saved.state().pending_tool_execution().is_none());
                assert!(
                    saved
                        .state()
                        .messages()
                        .last()
                        .unwrap()
                        .content
                        .iter()
                        .any(|block| matches!(block, ContentBlock::ToolResult(_)))
                );
            }
        }
    });
}

#[test]
fn budget_follows_context_policy_and_hooks_see_final_request() {
    struct RecordingCounter(Arc<Mutex<Vec<usize>>>);
    impl TokenCounter for RecordingCounter {
        fn count(&self, request: &ChatRequest) -> Result<TokenCount, TokenBudgetError> {
            self.0.lock().unwrap().push(request.messages.len());
            assert_eq!(request.options.max_tokens, Some(20));
            Ok(TokenCount {
                tokens: request.messages.len() as u64 * 10,
                accuracy: TokenCountAccuracy::Estimated,
            })
        }
    }
    struct RequestHook(Arc<AtomicUsize>);
    impl AgentHook for RequestHook {
        fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a> {
            Box::pin(async move {
                if let AgentHookEvent::BeforeModelCall { request, .. } = event {
                    assert_eq!(request.messages.len(), 2);
                    assert_eq!(request.options.max_tokens, Some(20));
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
                Ok(())
            })
        }
    }
    block_on(async {
        for streaming in [false, true] {
            let counts = Arc::new(Mutex::new(Vec::new()));
            let hooks = Arc::new(AtomicUsize::new(0));
            let model = MockChatModel::new("mock")
                .with_response(answer_response())
                .with_stream(final_answer_stream());
            let agent = ReActAgent::new("Friday", model, ToolExecutor::new(ToolRegistry::new()))
                .unwrap()
                .with_memory(InMemoryMemory::from_messages([
                    Msg::user("first"),
                    Msg::assistant("a", "one"),
                    Msg::user("second"),
                    Msg::assistant("a", "two"),
                    Msg::user("third"),
                    Msg::assistant("a", "three"),
                ]))
                .with_system_prompt("instructions")
                .with_context_policy(RecentTurns::new(2).unwrap())
                .with_token_budget(
                    TokenBudget::new(50, 20)
                        .unwrap()
                        .with_counter(RecordingCounter(counts.clone())),
                )
                .with_hook(RequestHook(hooks.clone()));
            if streaming {
                consume(agent.stream(Msg::user("current")).await.unwrap()).await;
            } else {
                agent.reply(Msg::user("current")).await.unwrap();
            }
            assert_eq!(*counts.lock().unwrap(), vec![4, 2]);
            assert_eq!(hooks.load(Ordering::SeqCst), 1);
            assert_eq!(agent.snapshot().await.unwrap().messages().len(), 8);
        }
    });
}
