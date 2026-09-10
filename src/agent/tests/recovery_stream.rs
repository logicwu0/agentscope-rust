use super::*;
use crate::{PendingToolCalls, StateRecord, StateStoreError, StateStoreFuture, ToolResultBlock};

struct Fixture {
    agent: ReActAgent,
    pending: PendingToolCalls,
    store: Arc<InMemoryStateStore>,
    key: StateKey,
}

async fn fixture(tool: Arc<dyn Tool>) -> Fixture {
    let model = MockChatModel::new("mock")
        .with_response(ChatResponse::finished(
            [ContentBlock::from(
                ToolCallBlock::complete("call", "calculator", r#"{"expression":"6*7"}"#).unwrap(),
            )],
            FinishReason::ToolCalls,
        ))
        .with_stream([
            Ok(ChatEvent::TextDelta {
                block_id: "text".into(),
                delta: "4".into(),
            }),
            Ok(ChatEvent::TextDelta {
                block_id: "text".into(),
                delta: "2".into(),
            }),
            Ok(ChatEvent::Finished {
                reason: FinishReason::Completed,
            }),
        ]);
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool).unwrap();
    let key = StateKey::new("user", "stream-recovery").unwrap();
    let store = Arc::new(InMemoryStateStore::new());
    let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_shared_state_store(key.clone(), store.clone())
        .with_tool_confirmation_required("calculator");
    let AgentError::ToolConfirmationRequired {
        checkpoint: pending,
    } = agent.reply(Msg::user("calculate")).await.unwrap_err()
    else {
        panic!("expected confirmation")
    };
    Fixture {
        agent,
        pending,
        store,
        key,
    }
}

fn tool() -> Arc<MockTool> {
    Arc::new(MockTool::new(calculator_definition()).with_output("42"))
}

#[test]
fn approval_stream_is_lazy_and_saves_tools_and_final_reply_before_events() {
    block_on(async {
        let tool = tool();
        let fixture = fixture(tool.clone()).await;
        let agent: &dyn Agent = &fixture.agent;
        let unpolled = agent
            .stream_resume_tool_calls(
                fixture.pending.reply_id().into(),
                vec![ToolConfirmation::approve("call")],
            )
            .await
            .unwrap();
        drop(unpolled);
        assert!(
            fixture
                .store
                .load(&fixture.key)
                .await
                .unwrap()
                .unwrap()
                .state()
                .pending_tool_calls()
                .is_some()
        );
        let mut events = agent
            .stream_resume_tool_calls(
                fixture.pending.reply_id().into(),
                vec![ToolConfirmation::approve("call")],
            )
            .await
            .unwrap();
        assert!(tool.recorded_invocations().is_empty());
        assert!(matches!(
            events.next().await.unwrap().unwrap(),
            AgentEvent::ToolStarted { step: 1, .. }
        ));
        let stored = fixture.store.load(&fixture.key).await.unwrap().unwrap();
        assert!(stored.state().pending_tool_calls().is_none());
        assert!(stored.state().pending_tool_execution().is_some());
        assert!(matches!(
            events.next().await.unwrap().unwrap(),
            AgentEvent::ToolFinished { step: 1, .. }
        ));
        let stored = fixture.store.load(&fixture.key).await.unwrap().unwrap();
        assert_eq!(stored.state().messages().len(), 3);
        assert!(stored.state().pending_tool_execution().is_none());
        assert!(
            matches!(events.next().await.unwrap().unwrap(), AgentEvent::TextDelta { step: 2, delta, .. } if delta == "4")
        );
        assert!(
            matches!(events.next().await.unwrap().unwrap(), AgentEvent::TextDelta { step: 2, delta, .. } if delta == "2")
        );
        assert!(matches!(
            events.next().await.unwrap().unwrap(),
            AgentEvent::StepFinished { step: 2, .. }
        ));
        let AgentEvent::Finished { steps, message } = events.next().await.unwrap().unwrap() else {
            panic!("expected final reply")
        };
        assert_eq!(steps, 2);
        assert_eq!(
            fixture
                .store
                .load(&fixture.key)
                .await
                .unwrap()
                .unwrap()
                .state()
                .messages()
                .last(),
            Some(&message)
        );
        assert!(events.next().await.is_none());
        assert_eq!(tool.recorded_invocations().len(), 1);
    });
}

#[test]
fn denial_and_invalid_confirmation_do_not_execute_tools() {
    block_on(async {
        let tool = tool();
        let fixture = fixture(tool.clone()).await;
        assert!(matches!(
            fixture
                .agent
                .stream_resume_tool_calls("wrong", vec![])
                .await,
            Err(AgentError::InvalidToolConfirmation(_))
        ));
        assert!(matches!(
            fixture
                .agent
                .stream_resume_tool_calls(fixture.pending.reply_id(), vec![])
                .await,
            Err(AgentError::InvalidToolConfirmation(_))
        ));
        let events = fixture
            .agent
            .stream_resume_tool_calls(
                fixture.pending.reply_id(),
                vec![ToolConfirmation::deny("call", "denied")],
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(&events[0], Ok(AgentEvent::ToolFinished { result, .. }) if result.state() == ToolResultState::Denied)
        );
        assert!(matches!(
            events.last(),
            Some(Ok(AgentEvent::Finished { steps: 2, .. }))
        ));
        assert!(tool.recorded_invocations().is_empty());
    });
}

async fn leave_execution_pending(fixture: &Fixture) -> crate::PendingToolExecution {
    let mut stream = fixture
        .agent
        .stream_resume_tool_calls(
            fixture.pending.reply_id(),
            vec![ToolConfirmation::approve("call")],
        )
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await.unwrap().unwrap(),
        AgentEvent::ToolStarted { .. }
    ));
    drop(stream);
    fixture
        .store
        .load(&fixture.key)
        .await
        .unwrap()
        .unwrap()
        .state()
        .pending_tool_execution()
        .unwrap()
        .clone()
}

#[test]
fn dropped_execution_stream_retries_with_original_key() {
    block_on(async {
        let tool = tool();
        let fixture = fixture(tool.clone()).await;
        let pending = leave_execution_pending(&fixture).await;
        let events = (&fixture.agent as &dyn Agent)
            .stream_retry_tool_execution(fixture.pending.reply_id().into())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(
            events.last(),
            Some(Ok(AgentEvent::Finished { steps: 2, .. }))
        ));
        let invocations = tool.recorded_invocations();
        assert_eq!(invocations.len(), 1);
        assert_eq!(
            invocations[0].context.idempotency_key(),
            pending.idempotency_key("call").as_deref()
        );
    });
}

#[test]
fn verified_results_stream_without_executing_tools() {
    block_on(async {
        let tool = tool();
        let fixture = fixture(tool.clone()).await;
        leave_execution_pending(&fixture).await;
        let agent: &dyn Agent = &fixture.agent;
        assert!(matches!(
            agent
                .stream_resolve_tool_execution(fixture.pending.reply_id().into(), vec![])
                .await,
            Err(AgentError::InvalidToolExecutionResolution(_))
        ));
        let result = ToolResultBlock::success("call", "calculator", "42").unwrap();
        let events = agent
            .stream_resolve_tool_execution(fixture.pending.reply_id().into(), vec![result])
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(&events[0], Ok(AgentEvent::ToolFinished { .. })));
        assert!(matches!(
            events.last(),
            Some(Ok(AgentEvent::Finished { .. }))
        ));
        assert!(tool.recorded_invocations().is_empty());
    });
}

#[test]
fn dropping_model_continuation_preserves_results_but_not_partial_text() {
    block_on(async {
        let fixture = fixture(tool()).await;
        let mut events = fixture
            .agent
            .stream_resume_tool_calls(
                fixture.pending.reply_id(),
                vec![ToolConfirmation::approve("call")],
            )
            .await
            .unwrap();
        while !matches!(
            events.next().await.unwrap().unwrap(),
            AgentEvent::TextDelta { .. }
        ) {}
        drop(events);
        let state = fixture.agent.snapshot().await.unwrap();
        assert_eq!(state.messages().len(), 3);
        assert!(state.pending_tool_execution().is_none());
        assert!(matches!(
            &state.messages()[2].content[0],
            ContentBlock::ToolResult(_)
        ));
    });
}

#[test]
fn interrupted_stream_keeps_a_restorable_execution_checkpoint() {
    block_on(async {
        let handle = Arc::new(Mutex::new(None));
        let tool = Arc::new(InterruptingTool {
            definition: calculator_definition(),
            handle: handle.clone(),
            invocations: AtomicUsize::new(0),
        });
        let fixture = fixture(tool.clone()).await;
        *handle.lock().unwrap() = Some(fixture.agent.interrupt_handle());
        let events = fixture
            .agent
            .stream_resume_tool_calls(
                fixture.pending.reply_id(),
                vec![ToolConfirmation::approve("call")],
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(
            events.last(),
            Some(Ok(AgentEvent::Error {
                error: AgentError::ToolExecutionInDoubt { .. },
                ..
            }))
        ));
        assert!(!events.iter().any(|event| matches!(
            event,
            Ok(AgentEvent::ToolFinished { .. } | AgentEvent::Finished { .. })
        )));
        assert!(
            fixture
                .agent
                .snapshot()
                .await
                .unwrap()
                .pending_tool_execution()
                .is_some()
        );
        assert_eq!(tool.invocations.load(Ordering::SeqCst), 1);
    });
}

struct InterruptAfterTool(AgentInterruptHandle);
impl AgentHook for InterruptAfterTool {
    fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a> {
        Box::pin(async move {
            if matches!(event, AgentHookEvent::AfterToolCall { .. }) {
                self.0.interrupt();
            }
            Ok(())
        })
    }
}

#[test]
fn interruption_between_tools_and_model_preserves_saved_results() {
    block_on(async {
        let fixture = fixture(tool()).await;
        let handle = fixture.agent.interrupt_handle();
        let agent = fixture.agent.with_hook(InterruptAfterTool(handle));
        let events = agent
            .stream_resume_tool_calls(
                fixture.pending.reply_id(),
                vec![ToolConfirmation::approve("call")],
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(
            events.last(),
            Some(Ok(AgentEvent::Error {
                error: AgentError::Interrupted,
                step: Some(2)
            }))
        ));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Ok(AgentEvent::TextDelta { .. })))
        );
        assert_eq!(agent.snapshot().await.unwrap().messages().len(), 3);
    });
}

#[test]
fn subsequent_confirmation_preserves_original_step_budget() {
    block_on(async {
        let model = MockChatModel::new("mock")
            .with_response(ChatResponse::finished(
                [ContentBlock::from(
                    ToolCallBlock::complete("first", "calculator", r#"{"expression":"1+1"}"#)
                        .unwrap(),
                )],
                FinishReason::ToolCalls,
            ))
            .with_stream([
                Ok(ChatEvent::ToolCallDelta {
                    tool_call_id: "second".into(),
                    tool_name: "calculator".into(),
                    delta: r#"{"expression":"2+2"}"#.into(),
                }),
                Ok(ChatEvent::Finished {
                    reason: FinishReason::ToolCalls,
                }),
            ]);
        let mut registry = ToolRegistry::new();
        registry.register_shared(tool()).unwrap();
        let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_max_steps(3)
            .unwrap()
            .with_tool_confirmation_required("calculator");
        let AgentError::ToolConfirmationRequired { checkpoint } =
            agent.reply(Msg::user("compute")).await.unwrap_err()
        else {
            panic!("expected pause")
        };
        let events = agent
            .stream_resume_tool_calls(
                checkpoint.reply_id(),
                vec![ToolConfirmation::approve("first")],
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(
            matches!(events.last(), Some(Ok(AgentEvent::ToolConfirmationRequired { checkpoint })) if checkpoint.step() == 2)
        );
        assert_eq!(
            agent
                .snapshot()
                .await
                .unwrap()
                .pending_tool_calls()
                .unwrap()
                .step(),
            2
        );
    });
}

struct FailFinalSave {
    inner: Arc<InMemoryStateStore>,
    writes: AtomicUsize,
}
impl StateStore for FailFinalSave {
    fn load<'a>(&'a self, key: &'a StateKey) -> StateStoreFuture<'a, Option<StateRecord>> {
        self.inner.load(key)
    }
    fn save(
        &self,
        key: StateKey,
        revision: Option<u64>,
        state: AgentState,
    ) -> StateStoreFuture<'_, StateRecord> {
        Box::pin(async move {
            if self.writes.fetch_add(1, Ordering::SeqCst) == 2 {
                return Err(StateStoreError::new("final save failed"));
            }
            self.inner.save(key, revision, state).await
        })
    }
}

#[test]
fn final_save_failure_replaces_success_event() {
    block_on(async {
        let fixture = fixture(tool()).await;
        let agent = fixture.agent.with_state_store(
            fixture.key,
            FailFinalSave {
                inner: fixture.store,
                writes: AtomicUsize::new(0),
            },
        );
        let events = agent
            .stream_resume_tool_calls(
                fixture.pending.reply_id(),
                vec![ToolConfirmation::approve("call")],
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(
            events.last(),
            Some(Ok(AgentEvent::Error {
                error: AgentError::StateStore(_),
                ..
            }))
        ));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Ok(AgentEvent::Finished { .. })))
        );
    });
}
