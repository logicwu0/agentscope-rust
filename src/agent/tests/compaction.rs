use super::*;
use crate::{ChatModelSummarizer, ContextSummarizer, SummaryError, SummaryFuture, TokenBudget};

fn history() -> Vec<Msg> {
    vec![
        Msg::system("Never disclose private data."),
        Msg::user("old question".repeat(100)),
        Msg::assistant("Friday", "old answer".repeat(100)),
        Msg::user("recent question"),
        Msg::assistant("Friday", "recent answer"),
    ]
}

struct StubSummary {
    calls: Arc<AtomicUsize>,
    text: String,
}
impl ContextSummarizer for StubSummary {
    fn summarize<'a>(&'a self, messages: &'a [Msg]) -> SummaryFuture<'a> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(!messages.is_empty());
        Box::pin(async { Ok(self.text.clone()) })
    }
}

fn agent(model: Arc<MockChatModel>, text: &str) -> (ReActAgent, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let agent = ReActAgent::from_shared("Friday", model, ToolExecutor::new(ToolRegistry::new()))
        .unwrap()
        .with_memory(InMemoryMemory::from_messages(history()))
        .with_summarizer(StubSummary {
            calls: calls.clone(),
            text: text.into(),
        });
    (agent, calls)
}

#[test]
fn compaction_is_explicit_persisted_and_used_in_both_reply_paths() {
    block_on(async {
        for streaming in [false, true] {
            let model = Arc::new(
                MockChatModel::new("mock")
                    .with_response(ChatResponse::completed([ContentBlock::from("done")]))
                    .with_stream(final_answer_stream()),
            );
            let (agent, calls) = agent(model.clone(), "Prior goal and result.");
            let store = Arc::new(InMemoryStateStore::new());
            let key = StateKey::new("u", "summary").unwrap();
            let agent = agent.with_shared_state_store(key.clone(), store.clone());
            let original = agent.snapshot().await.unwrap();
            let dynamic: &dyn Agent = &agent;
            let summary = dynamic.compact_context(1).await.unwrap().unwrap();
            assert_eq!(summary.covered_messages(), 3);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert!(model.recorded_requests().is_empty());
            assert!(dynamic.compact_context(1).await.unwrap().is_none());
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            let snapshot = dynamic.snapshot().await.unwrap();
            assert_eq!(snapshot.messages(), original.messages());
            assert_eq!(snapshot.context_summary(), Some(&summary));
            assert_eq!(snapshot.format_version(), 4);
            let json = serde_json::to_string(&snapshot).unwrap();
            assert_eq!(serde_json::from_str::<AgentState>(&json).unwrap(), snapshot);
            // Rebuild without a summarizer: applying a persisted summary needs no model call.
            let restored = ReActAgent::from_shared(
                "Friday",
                model.clone(),
                ToolExecutor::new(ToolRegistry::new()),
            )
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_shared_state_store(key, store)
            .with_context_policy(crate::RecentTurns::new(1).unwrap());
            if streaming {
                let events = restored
                    .stream(Msg::user("next"))
                    .await
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                assert!(matches!(
                    events.last(),
                    Some(Ok(AgentEvent::Finished { .. }))
                ));
            } else {
                restored.reply(Msg::user("next")).await.unwrap();
            }
            let request = &model.recorded_requests()[0];
            assert!(
                request
                    .messages
                    .iter()
                    .any(|m| m.name == "context_summary" && m.role == Role::Assistant)
            );
            assert!(request.messages.iter().any(|m| m.role == Role::System));
            assert!(
                !request
                    .messages
                    .iter()
                    .any(|m| m.text_content("").is_some_and(|s| s.contains("old answer")))
            );
            assert_eq!(restored.snapshot().await.unwrap().messages().len(), 7);
            restored.clear_context_summary().await.unwrap();
            assert!(
                restored
                    .snapshot()
                    .await
                    .unwrap()
                    .context_summary()
                    .is_none()
            );
            assert_eq!(restored.snapshot().await.unwrap().messages().len(), 7);
        }
    });
}

#[test]
fn invalid_summary_and_budget_failures_leave_original_state_unchanged() {
    block_on(async {
        for text in ["", &"long summary".repeat(1000), "small summary"] {
            let (agent, _) = agent(Arc::new(MockChatModel::new("mock")), text);
            let agent = if text == "small summary" {
                agent.with_token_budget(TokenBudget::new(2, 1).unwrap())
            } else {
                agent
            };
            let before = agent.snapshot().await.unwrap();
            assert!(agent.compact_context(1).await.is_err());
            assert_eq!(agent.snapshot().await.unwrap(), before);
        }
    });
}

#[test]
fn validates_sources_before_calling_summary_model() {
    block_on(async {
        let (agent, calls) = agent(Arc::new(MockChatModel::new("mock")), "summary");
        assert!(matches!(
            agent.compact_context(0).await,
            Err(AgentError::Summary(SummaryError::ZeroRecentTurns))
        ));
        assert!(agent.compact_context(2).await.unwrap().is_none());
        let malformed = vec![Msg::user("unfinished"), Msg::user("current")];
        agent
            .restore(AgentState::new("Friday", malformed))
            .await
            .unwrap();
        assert!(matches!(
            agent.compact_context(1).await,
            Err(AgentError::Summary(SummaryError::IncompleteHistory))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn changed_source_is_rejected_even_if_message_ids_are_unchanged() {
    block_on(async {
        let (agent, _) = agent(Arc::new(MockChatModel::new("mock")), "summary");
        agent.compact_context(1).await.unwrap();
        let before = agent.snapshot().await.unwrap();
        let mut json = serde_json::to_value(&before).unwrap();
        json["messages"][1]["content"][0]["text"] = json!("changed");
        let altered = serde_json::from_value(json).unwrap();
        assert!(matches!(
            agent.restore(altered).await,
            Err(AgentError::Summary(SummaryError::StaleSource))
        ));
        assert_eq!(agent.snapshot().await.unwrap(), before);
    });
}

#[test]
fn chat_model_summarizer_uses_one_bounded_tool_free_request() {
    block_on(async {
        let model = Arc::new(
            MockChatModel::new("summary").with_response(ChatResponse::completed([
                ContentBlock::from("Summary facts"),
            ])),
        );
        let summary =
            ChatModelSummarizer::from_shared(model.clone(), TokenBudget::new(10000, 500).unwrap());
        assert_eq!(
            summary.summarize(&history()[..3]).await.unwrap(),
            "Summary facts"
        );
        let requests = model.recorded_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.len(), 2);
        assert_eq!(requests[0].options.max_tokens, Some(500));
        assert!(requests[0].tools.is_empty());
        assert!(
            !requests[0].messages[1]
                .text_content("")
                .unwrap()
                .contains("Never disclose")
        );
        for reason in [FinishReason::Length, FinishReason::ToolCalls] {
            let model = MockChatModel::new("summary").with_response(ChatResponse::finished(
                [ContentBlock::from("partial")],
                reason,
            ));
            let summary = ChatModelSummarizer::new(model, TokenBudget::new(10000, 500).unwrap());
            assert!(matches!(
                summary.summarize(&history()[..3]).await,
                Err(SummaryError::InvalidResponse)
            ));
        }
        let model = Arc::new(MockChatModel::new("summary"));
        let summary =
            ChatModelSummarizer::from_shared(model.clone(), TokenBudget::new(2, 1).unwrap());
        assert!(matches!(
            summary.summarize(&history()[..3]).await,
            Err(SummaryError::Budget(_))
        ));
        assert!(model.recorded_requests().is_empty());
    });
}

#[test]
fn dropping_or_interrupting_compaction_never_installs_a_summary() {
    struct Pending;
    impl ContextSummarizer for Pending {
        fn summarize<'a>(&'a self, _: &'a [Msg]) -> SummaryFuture<'a> {
            Box::pin(std::future::pending())
        }
    }
    block_on(async {
        let (agent, _) = agent(Arc::new(MockChatModel::new("mock")), "unused");
        let agent = agent.with_summarizer(Pending);
        let before = agent.snapshot().await.unwrap();
        let mut future = agent.compact_context(1);
        assert!(futures_util::poll!(&mut future).is_pending());
        assert!(matches!(
            agent.clone().compact_context(1).await,
            Err(AgentError::Summary(SummaryError::Busy))
        ));
        drop(future);
        assert_eq!(agent.snapshot().await.unwrap(), before);
        let mut future = agent.compact_context(1);
        assert!(futures_util::poll!(&mut future).is_pending());
        agent.interrupt_handle().interrupt();
        assert!(matches!(future.await, Err(AgentError::Interrupted)));
        assert_eq!(agent.snapshot().await.unwrap(), before);
    });
}

#[test]
fn failed_persistence_does_not_install_runtime_or_durable_summary() {
    use crate::{StateRecord, StateStoreError, StateStoreFuture};
    struct FailingStore(Arc<InMemoryStateStore>);
    impl StateStore for FailingStore {
        fn load<'a>(&'a self, key: &'a StateKey) -> StateStoreFuture<'a, Option<StateRecord>> {
            self.0.load(key)
        }
        fn save(
            &self,
            _: StateKey,
            _: Option<u64>,
            _: AgentState,
        ) -> StateStoreFuture<'_, StateRecord> {
            Box::pin(async { Err(StateStoreError::new("simulated failure")) })
        }
    }
    block_on(async {
        let inner = Arc::new(InMemoryStateStore::new());
        let key = StateKey::new("u", "failure").unwrap();
        let initial = AgentState::new("Friday", history());
        inner
            .save(key.clone(), None, initial.clone())
            .await
            .unwrap();
        let (agent, _) = agent(Arc::new(MockChatModel::new("main")), "Summary");
        let agent = agent.with_state_store(key.clone(), FailingStore(inner.clone()));
        assert!(matches!(
            agent.compact_context(1).await,
            Err(AgentError::StateStore(_))
        ));
        assert_eq!(inner.load(&key).await.unwrap().unwrap().state(), &initial);
        let runtime = agent.with_state_store(
            StateKey::new("other", "empty").unwrap(),
            InMemoryStateStore::new(),
        );
        assert_eq!(runtime.snapshot().await.unwrap(), initial);
    });
}

#[test]
fn pending_confirmations_and_uncertain_executions_block_compaction() {
    block_on(async {
        let model = Arc::new(
            MockChatModel::new("main").with_response(ChatResponse::finished(
                [ContentBlock::from(
                    ToolCallBlock::complete("c", "calculator", r#"{"expression":"6*7"}"#).unwrap(),
                )],
                FinishReason::ToolCalls,
            )),
        );
        let (agent, calls) = agent(model, "Summary");
        let agent = agent.with_tool_confirmation_required("calculator");
        let AgentError::ToolConfirmationRequired { checkpoint } =
            agent.reply(Msg::user("tool please")).await.unwrap_err()
        else {
            panic!("expected confirmation")
        };
        assert!(matches!(
            agent.compact_context(1).await,
            Err(AgentError::Summary(SummaryError::PendingTools))
        ));
        let mut stream = agent
            .stream_resume_tool_calls(checkpoint.reply_id(), vec![ToolConfirmation::approve("c")])
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            AgentEvent::ToolStarted { .. }
        ));
        drop(stream);
        assert!(matches!(
            agent.compact_context(1).await,
            Err(AgentError::Summary(SummaryError::PendingTools))
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn repeated_compaction_expands_from_original_history_and_supports_legacy_state() {
    block_on(async {
        let (agent, calls) = agent(Arc::new(MockChatModel::new("main")), "Summary");
        agent.compact_context(1).await.unwrap();
        agent.observe(Msg::user("new")).await.unwrap();
        let summary = agent.compact_context(1).await.unwrap().unwrap();
        assert_eq!(summary.covered_messages(), 5);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(agent.snapshot().await.unwrap().messages().len(), 6);
        let mut legacy = serde_json::to_value(AgentState::new("Friday", history())).unwrap();
        legacy["format_version"] = json!(3);
        agent
            .restore(serde_json::from_value(legacy).unwrap())
            .await
            .unwrap();
        assert!(agent.snapshot().await.unwrap().context_summary().is_none());
    });
}

#[test]
fn summary_remains_protected_when_token_budget_drops_old_turns() {
    use crate::{TokenBudgetError, TokenCount, TokenCountAccuracy, TokenCounter};
    struct Counter;
    impl TokenCounter for Counter {
        fn count(&self, request: &ChatRequest) -> Result<TokenCount, TokenBudgetError> {
            Ok(TokenCount {
                tokens: request
                    .messages
                    .iter()
                    .map(|m| if m.name == "context_summary" { 100 } else { 10 })
                    .sum(),
                accuracy: TokenCountAccuracy::Estimated,
            })
        }
    }
    block_on(async {
        let model = Arc::new(MockChatModel::new("main"));
        let (agent, _) = agent(model.clone(), "Summary");
        agent.compact_context(1).await.unwrap();
        let agent =
            agent.with_token_budget(TokenBudget::new(100, 20).unwrap().with_counter(Counter));
        assert!(matches!(
            agent.reply(Msg::user("new")).await,
            Err(AgentError::TokenBudget(_))
        ));
        assert!(model.recorded_requests().is_empty());
        assert!(agent.snapshot().await.unwrap().context_summary().is_some());
    });
}

#[test]
fn completed_tool_exchanges_are_summarized_whole_and_orphans_rejected() {
    block_on(async {
        for balanced in [false, true] {
            let (agent, calls) = agent(Arc::new(MockChatModel::new("main")), "A tool completed.");
            let source = vec![
                Msg::user("old task".repeat(100)),
                Msg::new(
                    "Friday",
                    Role::Assistant,
                    [ContentBlock::from(
                        ToolCallBlock::complete("c", "tool", "{}").unwrap(),
                    )],
                ),
                Msg::new(
                    "tool",
                    Role::Assistant,
                    [ContentBlock::from(
                        crate::ToolResultBlock::success(
                            if balanced { "c" } else { "wrong" },
                            "tool",
                            "result",
                        )
                        .unwrap(),
                    )],
                ),
                Msg::assistant("Friday", "completed"),
                Msg::user("current"),
            ];
            agent
                .restore(AgentState::new("Friday", source.clone()))
                .await
                .unwrap();
            let result = agent.compact_context(1).await;
            if balanced {
                assert_eq!(result.unwrap().unwrap().covered_messages(), 4);
            } else {
                assert!(matches!(
                    result,
                    Err(AgentError::Summary(SummaryError::IncompleteHistory))
                ));
            }
            assert_eq!(calls.load(Ordering::SeqCst), usize::from(balanced));
            assert_eq!(agent.snapshot().await.unwrap().messages(), source);
        }
    });
}

#[test]
fn recovery_continuations_keep_summary_and_original_prefix() {
    block_on(async {
        for streaming in [false, true] {
            for mode in 0..3 {
                let model = Arc::new(
                    MockChatModel::new("main")
                        .with_response(ChatResponse::finished(
                            [ContentBlock::from(
                                ToolCallBlock::complete(
                                    "c",
                                    "calculator",
                                    r#"{"expression":"6*7"}"#,
                                )
                                .unwrap(),
                            )],
                            FinishReason::ToolCalls,
                        ))
                        .with_response(ChatResponse::completed([ContentBlock::from("42")]))
                        .with_stream(final_answer_stream()),
                );
                let mut registry = ToolRegistry::new();
                registry
                    .register(MockTool::new(calculator_definition()).with_output("42"))
                    .unwrap();
                let calls = Arc::new(AtomicUsize::new(0));
                let agent =
                    ReActAgent::from_shared("Friday", model.clone(), ToolExecutor::new(registry))
                        .unwrap()
                        .with_memory(InMemoryMemory::from_messages(history()))
                        .with_state_store(
                            StateKey::new("u", "recover").unwrap(),
                            InMemoryStateStore::new(),
                        )
                        .with_tool_confirmation_required("calculator")
                        .with_summarizer(StubSummary {
                            calls: calls.clone(),
                            text: "Old facts".into(),
                        });
                let summary = agent.compact_context(1).await.unwrap().unwrap();
                let original = agent.snapshot().await.unwrap();
                let AgentError::ToolConfirmationRequired { checkpoint } =
                    agent.reply(Msg::user("calculate")).await.unwrap_err()
                else {
                    panic!("expected pause")
                };
                let id = checkpoint.reply_id();
                let decisions = vec![ToolConfirmation::approve("c")];
                if mode != 0 {
                    let mut events = agent
                        .stream_resume_tool_calls(id, decisions.clone())
                        .await
                        .unwrap();
                    assert!(matches!(
                        events.next().await.unwrap().unwrap(),
                        AgentEvent::ToolStarted { .. }
                    ));
                    drop(events);
                }
                let results =
                    vec![crate::ToolResultBlock::success("c", "calculator", "42").unwrap()];
                if streaming {
                    let events = match mode {
                        0 => agent.stream_resume_tool_calls(id, decisions).await,
                        1 => agent.stream_retry_tool_execution(id).await,
                        _ => agent.stream_resolve_tool_execution(id, results).await,
                    }
                    .unwrap()
                    .collect::<Vec<_>>()
                    .await;
                    assert!(matches!(
                        events.last(),
                        Some(Ok(AgentEvent::Finished { .. }))
                    ));
                } else {
                    match mode {
                        0 => agent.resume_tool_calls(id, decisions).await,
                        1 => agent.retry_tool_execution(id).await,
                        _ => agent.resolve_tool_execution(id, results).await,
                    }
                    .unwrap();
                }
                assert_eq!(model.recorded_requests().len(), 2);
                for request in model.recorded_requests() {
                    assert_summary_request(&request);
                }
                let state = agent.snapshot().await.unwrap();
                assert_eq!(state.context_summary(), Some(&summary));
                assert_eq!(&state.messages()[..5], original.messages());
                assert_eq!(calls.load(Ordering::SeqCst), 1);
            }
        }
    });
}

fn assert_summary_request(request: &ChatRequest) {
    assert_eq!(
        request
            .messages
            .iter()
            .filter(|m| m.name == "context_summary")
            .count(),
        1
    );
    assert!(!request.messages.iter().any(|m| {
        m.text_content("")
            .is_some_and(|s| s.contains("old question"))
    }));
}
