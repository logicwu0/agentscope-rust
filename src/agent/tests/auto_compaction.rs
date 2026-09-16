use super::*;
use crate::{
    ContextSummarizer, SummaryError, SummaryFuture, TokenBudget, TokenBudgetError, TokenCount,
    TokenCountAccuracy, TokenCounter,
};

fn history() -> Vec<Msg> {
    vec![
        Msg::system("instructions"),
        Msg::user("old question".repeat(100)),
        Msg::assistant("Friday", "old answer".repeat(100)),
        Msg::user("recent"),
        Msg::assistant("Friday", "answer"),
    ]
}

struct Counter;
impl TokenCounter for Counter {
    fn count(&self, request: &ChatRequest) -> Result<TokenCount, TokenBudgetError> {
        let tokens = request
            .messages
            .iter()
            .map(|m| {
                if m.name == "context_summary" {
                    200
                } else if m.text_content("").is_some_and(|s| s.starts_with("old")) {
                    1000
                } else if m
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult(_)))
                {
                    800
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

struct Summarizer {
    calls: Arc<AtomicUsize>,
    fail: bool,
}
impl ContextSummarizer for Summarizer {
    fn summarize<'a>(&'a self, _: &'a [Msg]) -> SummaryFuture<'a> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if self.fail {
                Err(SummaryError::InvalidResponse)
            } else {
                Ok("Old task completed.".into())
            }
        })
    }
}

fn model() -> Arc<MockChatModel> {
    Arc::new(
        MockChatModel::new("main")
            .with_response(ChatResponse::completed([ContentBlock::from("done")]))
            .with_stream(final_answer_stream()),
    )
}

fn fixture(
    model: Arc<MockChatModel>,
    fail: bool,
) -> (
    ReActAgent,
    Arc<AtomicUsize>,
    Arc<InMemoryStateStore>,
    StateKey,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let store = Arc::new(InMemoryStateStore::new());
    let key = StateKey::new("user", "auto").unwrap();
    let agent = ReActAgent::from_shared("Friday", model, ToolExecutor::new(ToolRegistry::new()))
        .unwrap()
        .with_memory(InMemoryMemory::from_messages(history()))
        .with_shared_state_store(key.clone(), store.clone())
        .with_summarizer(Summarizer {
            calls: calls.clone(),
            fail,
        })
        .with_token_budget(TokenBudget::new(400, 100).unwrap().with_counter(Counter));
    (agent, calls, store, key)
}

#[test]
fn disabled_under_budget_and_context_policy_paths_do_not_summarize() {
    block_on(async {
        for mode in 0..3 {
            let model = model();
            let (mut agent, calls, _, _) = fixture(model.clone(), false);
            if mode != 0 {
                agent = agent.with_auto_compaction(1).unwrap();
            }
            if mode == 1 {
                agent = agent.with_context_policy(crate::RecentTurns::new(1).unwrap());
            }
            if mode == 2 {
                agent
                    .restore(AgentState::new("Friday", history()[3..].to_vec()))
                    .await
                    .unwrap();
            }
            agent.reply(Msg::user("new")).await.unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(model.recorded_requests().len(), 1);
        }
    });
}

#[test]
fn ordinary_reply_compacts_once_including_incoming_message_in_budget() {
    block_on(async {
        let model = model();
        let (agent, calls, store, key) = fixture(model.clone(), false);
        let agent = agent.with_auto_compaction(1).unwrap();
        let original = agent.snapshot().await.unwrap();
        agent.reply(Msg::user("new")).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let request = &model.recorded_requests()[0];
        assert!(request.messages.iter().any(|m| m.name == "context_summary"));
        assert!(
            !request
                .messages
                .iter()
                .any(|m| m.text_content("").is_some_and(|s| s.starts_with("old")))
        );
        assert_eq!(request.options.max_tokens, Some(100));
        let saved = store.load(&key).await.unwrap().unwrap();
        assert_eq!(saved.revision(), 2); // summary commit, then completed reply
        assert_eq!(&saved.state().messages()[..5], original.messages());
        assert_eq!(saved.state().messages().len(), 7);
    });
}

#[test]
fn stream_events_are_lazy_serializable_and_completed_only_after_save() {
    block_on(async {
        let model = model();
        let (agent, calls, store, key) = fixture(model.clone(), false);
        let agent = agent.with_auto_compaction(1).unwrap();
        let mut events = agent.stream(Msg::user("new")).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(matches!(
            events.next().await.unwrap().unwrap(),
            AgentEvent::ContextCompactionStarted { .. }
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let event = events.next().await.unwrap().unwrap();
        assert!(matches!(
            event,
            AgentEvent::ContextCompactionCompleted {
                covered_messages: 3
            }
        ));
        assert_eq!(
            serde_json::from_str::<AgentEvent>(&serde_json::to_string(&event).unwrap()).unwrap(),
            event
        );
        let saved = store.load(&key).await.unwrap().unwrap();
        assert!(saved.state().context_summary().is_some());
        assert_eq!(saved.state().messages().len(), 5); // incoming message not appended yet
        let rest = events.collect::<Vec<_>>().await;
        assert!(matches!(rest.last(), Some(Ok(AgentEvent::Finished { .. }))));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(model.recorded_requests().len(), 1);
        assert_eq!(
            store
                .load(&key)
                .await
                .unwrap()
                .unwrap()
                .state()
                .messages()
                .len(),
            7
        );
    });
}

#[test]
fn failures_preserve_original_state_and_emit_failure_then_terminal_error() {
    block_on(async {
        for streaming in [false, true] {
            for failure in 0..3 {
                let model = model();
                let (mut agent, calls, store, key) = fixture(model.clone(), failure == 0);
                agent = agent.with_auto_compaction(1).unwrap();
                if failure == 1 {
                    agent = agent.with_token_budget(
                        TokenBudget::new(100, 50).unwrap().with_counter(Counter),
                    );
                }
                if failure == 2 {
                    agent
                        .restore(AgentState::new("Friday", history()[..3].to_vec()))
                        .await
                        .unwrap();
                }
                let before = agent.snapshot().await.unwrap();
                if streaming {
                    let events = agent
                        .stream(Msg::user("new"))
                        .await
                        .unwrap()
                        .collect::<Vec<_>>()
                        .await;
                    assert_eq!(events.len(), 3);
                    assert!(matches!(
                        events[0],
                        Ok(AgentEvent::ContextCompactionStarted { .. })
                    ));
                    assert!(matches!(
                        events[1],
                        Ok(AgentEvent::ContextCompactionFailed { .. })
                    ));
                    assert!(matches!(
                        events[2],
                        Ok(AgentEvent::Error { step: None, .. })
                    ));
                } else {
                    assert!(agent.reply(Msg::user("new")).await.is_err());
                }
                assert_eq!(agent.snapshot().await.unwrap(), before);
                assert!(model.recorded_requests().is_empty());
                assert_eq!(calls.load(Ordering::SeqCst), usize::from(failure != 2));
                if failure != 2 {
                    assert!(store.load(&key).await.unwrap().is_none());
                }
            }
        }
    });
}

#[test]
fn incoming_oversize_is_not_saved_and_previous_summary_survives() {
    block_on(async {
        let model = model();
        let (agent, calls, _, _) = fixture(model.clone(), false);
        agent.compact_context(1).await.unwrap();
        agent
            .observe(Msg::user("another completed turn"))
            .await
            .unwrap();
        agent
            .observe(Msg::assistant("Friday", "complete"))
            .await
            .unwrap();
        let before = agent.snapshot().await.unwrap();
        let agent = agent.with_auto_compaction(1).unwrap();
        assert!(matches!(
            agent.reply(Msg::user("old HUGE NEW INPUT")).await,
            Err(AgentError::TokenBudget(_))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 2); // explicit + one automatic attempt
        assert_eq!(agent.snapshot().await.unwrap(), before);
        assert!(model.recorded_requests().is_empty());
    });
}

#[test]
fn stream_drop_and_interrupt_release_locks_without_committing_input() {
    block_on(async {
        let model = model();
        let (agent, calls, _, _) = fixture(model.clone(), false);
        let agent = agent.with_auto_compaction(1).unwrap();
        let before = agent.snapshot().await.unwrap();
        let mut events = agent.stream(Msg::user("new")).await.unwrap();
        assert!(matches!(
            events.next().await.unwrap().unwrap(),
            AgentEvent::ContextCompactionStarted { .. }
        ));
        drop(events);
        assert_eq!(agent.snapshot().await.unwrap(), before);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let mut events = agent.stream(Msg::user("new")).await.unwrap();
        events.next().await.unwrap().unwrap();
        agent.interrupt_handle().interrupt();
        let rest = events.collect::<Vec<_>>().await;
        assert!(matches!(
            rest.last(),
            Some(Ok(AgentEvent::Error {
                error: AgentError::Interrupted,
                ..
            }))
        ));
        assert_eq!(agent.snapshot().await.unwrap(), before);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(model.recorded_requests().is_empty());
    });
}

#[test]
fn oversized_tool_result_does_not_trigger_a_second_attempt() {
    block_on(async {
        for streaming in [false, true] {
            let model = Arc::new(
                MockChatModel::new("main")
                    .with_response(ChatResponse::finished(
                        [ContentBlock::from(
                            ToolCallBlock::complete(
                                "call-stream-1",
                                "calculator",
                                r#"{"expression":"6*7"}"#,
                            )
                            .unwrap(),
                        )],
                        FinishReason::ToolCalls,
                    ))
                    .with_stream(calculator_call_stream()),
            );
            let calls = Arc::new(AtomicUsize::new(0));
            let mut registry = ToolRegistry::new();
            registry
                .register(MockTool::new(calculator_definition()).with_output("large result"))
                .unwrap();
            let agent =
                ReActAgent::from_shared("Friday", model.clone(), ToolExecutor::new(registry))
                    .unwrap()
                    .with_memory(InMemoryMemory::from_messages(history()))
                    .with_summarizer(Summarizer {
                        calls: calls.clone(),
                        fail: false,
                    })
                    .with_token_budget(TokenBudget::new(400, 100).unwrap().with_counter(Counter))
                    .with_auto_compaction(1)
                    .unwrap();
            if streaming {
                let events = agent
                    .stream(Msg::user("calculate"))
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| matches!(e, Ok(AgentEvent::ContextCompactionStarted { .. })))
                        .count(),
                    1
                );
                assert!(matches!(
                    events.last(),
                    Some(Ok(AgentEvent::Error {
                        error: AgentError::TokenBudget(_),
                        ..
                    }))
                ));
            } else {
                assert!(matches!(
                    agent.reply(Msg::user("calculate")).await,
                    Err(AgentError::TokenBudget(_))
                ));
            }
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(model.recorded_requests().len(), 1);
            let state = agent.snapshot().await.unwrap();
            assert!(
                state
                    .messages()
                    .last()
                    .unwrap()
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolResult(_)))
            );
        }
    });
}

#[test]
fn configuration_is_explicit_and_validated_without_main_model_calls() {
    block_on(async {
        let model = model();
        let base = ReActAgent::from_shared(
            "Friday",
            model.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::from_messages(history()));
        assert!(matches!(
            base.clone().with_auto_compaction(0),
            Err(AgentError::Summary(SummaryError::ZeroRecentTurns))
        ));
        let missing = base.clone().with_auto_compaction(1).unwrap();
        assert!(matches!(
            missing.reply(Msg::user("new")).await,
            Err(AgentError::Summary(SummaryError::NotConfigured))
        ));
        let missing = base
            .with_summarizer(Summarizer {
                calls: Arc::new(AtomicUsize::new(0)),
                fail: false,
            })
            .with_auto_compaction(1)
            .unwrap();
        assert!(matches!(
            missing.reply(Msg::user("new")).await,
            Err(AgentError::Summary(SummaryError::AutoRequiresBudget))
        ));
        assert!(model.recorded_requests().is_empty());
        let (agent, calls, _, _) = fixture(model.clone(), false);
        agent
            .with_auto_compaction(1)
            .unwrap()
            .without_auto_compaction()
            .reply(Msg::user("new"))
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn rejected_summary_write_leaves_both_runtime_and_store_unchanged() {
    use crate::{StateRecord, StateStoreError, StateStoreFuture};
    struct Reject(Arc<InMemoryStateStore>);
    impl StateStore for Reject {
        fn load<'a>(&'a self, key: &'a StateKey) -> StateStoreFuture<'a, Option<StateRecord>> {
            self.0.load(key)
        }
        fn save(
            &self,
            _: StateKey,
            _: Option<u64>,
            _: AgentState,
        ) -> StateStoreFuture<'_, StateRecord> {
            Box::pin(async { Err(StateStoreError::new("rejected")) })
        }
    }
    block_on(async {
        let model = model();
        let (agent, calls, store, key) = fixture(model.clone(), false);
        let original = agent.snapshot().await.unwrap();
        store
            .save(key.clone(), None, original.clone())
            .await
            .unwrap();
        let agent = agent
            .with_state_store(key.clone(), Reject(store.clone()))
            .with_auto_compaction(1)
            .unwrap();
        let events = agent
            .stream(Msg::user("new"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(
            events[1],
            Ok(AgentEvent::ContextCompactionFailed {
                error: AgentError::StateStore(_)
            })
        ));
        assert_eq!(store.load(&key).await.unwrap().unwrap().state(), &original);
        let runtime = agent.with_state_store(
            StateKey::new("new", "empty").unwrap(),
            InMemoryStateStore::new(),
        );
        assert_eq!(runtime.snapshot().await.unwrap(), original);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(model.recorded_requests().is_empty());
    });
}

#[test]
fn cancellation_during_summary_preserves_state_and_drop_after_commit_preserves_only_summary() {
    struct Pending;
    impl ContextSummarizer for Pending {
        fn summarize<'a>(&'a self, _: &'a [Msg]) -> SummaryFuture<'a> {
            Box::pin(std::future::pending())
        }
    }
    block_on(async {
        let (agent, _, _, _) = fixture(model(), false);
        let before = agent.snapshot().await.unwrap();
        let agent = agent
            .with_summarizer(Pending)
            .with_auto_compaction(1)
            .unwrap();
        let mut reply = agent.reply(Msg::user("new"));
        assert!(futures_util::poll!(&mut reply).is_pending());
        agent.interrupt_handle().interrupt();
        assert!(matches!(reply.await, Err(AgentError::Interrupted)));
        assert_eq!(agent.snapshot().await.unwrap(), before);
        let (agent, _, store, key) = fixture(model(), false);
        let agent = agent.with_auto_compaction(1).unwrap();
        let mut events = agent.stream(Msg::user("new")).await.unwrap();
        events.next().await.unwrap().unwrap();
        assert!(matches!(
            events.next().await.unwrap().unwrap(),
            AgentEvent::ContextCompactionCompleted { .. }
        ));
        drop(events);
        let state = agent.snapshot().await.unwrap();
        assert!(state.context_summary().is_some());
        assert_eq!(state.messages().len(), 5);
        assert_eq!(store.load(&key).await.unwrap().unwrap().revision(), 1);
    });
}

#[test]
fn paused_and_recovery_paths_never_automatically_compact() {
    block_on(async {
        for streaming in [false, true] {
            for mode in 0..3 {
                let model = Arc::new(
                    MockChatModel::new("main").with_response(ChatResponse::finished(
                        [ContentBlock::from(
                            ToolCallBlock::complete("c", "calculator", r#"{"expression":"6*7"}"#)
                                .unwrap(),
                        )],
                        FinishReason::ToolCalls,
                    )),
                );
                let (agent, calls, _, _) = fixture(model.clone(), false);
                let agent = agent.with_tool_confirmation_required("calculator");
                let AgentError::ToolConfirmationRequired { checkpoint } =
                    agent.reply(Msg::user("calculate")).await.unwrap_err()
                else {
                    panic!("expected pause")
                };
                let agent = agent.with_auto_compaction(1).unwrap();
                assert!(matches!(
                    agent.reply(Msg::user("new")).await,
                    Err(AgentError::ToolConfirmationRequired { .. })
                ));
                let id = checkpoint.reply_id();
                let decisions = vec![ToolConfirmation::approve("c")];
                if mode != 0 {
                    let mut events = agent
                        .stream_resume_tool_calls(id, decisions.clone())
                        .await
                        .unwrap();
                    events.next().await.unwrap().unwrap();
                    drop(events);
                    assert!(matches!(
                        agent.reply(Msg::user("new")).await,
                        Err(AgentError::ToolExecutionInDoubt { .. })
                    ));
                }
                let results =
                    vec![crate::ToolResultBlock::success("c", "calculator", "result").unwrap()];
                if streaming {
                    let events = match mode {
                        0 => agent.stream_resume_tool_calls(id, decisions).await,
                        1 => agent.stream_retry_tool_execution(id).await,
                        _ => agent.stream_resolve_tool_execution(id, results).await,
                    }
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                    assert!(!events.iter().any(|event| matches!(
                        event,
                        Ok(AgentEvent::ContextCompactionStarted { .. })
                    )));
                } else {
                    let _ = match mode {
                        0 => agent.resume_tool_calls(id, decisions).await,
                        1 => agent.retry_tool_execution(id).await,
                        _ => agent.resolve_tool_execution(id, results).await,
                    };
                }
                assert_eq!(calls.load(Ordering::SeqCst), 0);
            }
        }
    });
}
