use super::*;
use crate::{
    PendingToolCalls, PendingToolExecution, StateRecord, StateStoreError, StateStoreFuture,
};

fn uncertain_state() -> AgentState {
    let calls = ["approved", "denied"].map(|id| {
        let mut call =
            ToolCallBlock::complete(id, "calculator", r#"{"expression":"6*7"}"#).unwrap();
        call.state = ToolCallState::Asking;
        call
    });
    let message = Msg::new(
        "Friday",
        Role::Assistant,
        calls.clone().map(ContentBlock::from),
    );
    let checkpoint = PendingToolCalls::new(message.id.clone(), 1, calls.to_vec());
    AgentState::new("Friday", vec![Msg::user("calculate"), message]).with_pending_tool_execution(
        Some(PendingToolExecution::new(
            "original-execution".to_owned(),
            checkpoint,
            vec![
                ToolConfirmation::approve("approved"),
                ToolConfirmation::deny("denied", "no"),
            ],
        )),
    )
}

#[test]
fn retry_after_restore_reuses_keys_and_preserves_denials_in_both_modes() {
    for mode in [
        crate::ToolExecutionMode::Sequential,
        crate::ToolExecutionMode::Concurrent,
    ] {
        let state = uncertain_state();
        let execution = state.pending_tool_execution().unwrap().clone();
        let store = Arc::new(InMemoryStateStore::new());
        let key = StateKey::new("user", "retry").unwrap();
        block_on(store.save(key.clone(), None, state)).unwrap();
        let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
        let mut registry = ToolRegistry::new();
        registry.register_shared(tool.clone()).unwrap();
        let agent: Arc<dyn Agent> = Arc::new(
            ReActAgent::new(
                "Friday",
                MockChatModel::new("mock")
                    .with_response(ChatResponse::completed([ContentBlock::from("42")])),
                ToolExecutor::new(registry).with_mode(mode),
            )
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_shared_state_store(key.clone(), store.clone()),
        );
        assert!(matches!(
            block_on(agent.retry_tool_execution("wrong".to_owned())),
            Err(AgentError::InvalidToolExecutionResolution(_))
        ));
        assert!(tool.recorded_invocations().is_empty());
        let reply =
            block_on(agent.retry_tool_execution(execution.confirmation().reply_id().to_owned()))
                .unwrap();
        assert_eq!(reply.text_content(""), Some("42".to_owned()));
        let invocations = tool.recorded_invocations();
        assert_eq!(invocations.len(), 1);
        assert_eq!(
            invocations[0].context.idempotency_key(),
            Some("original-execution:approved")
        );
        let saved = block_on(store.load(&key)).unwrap().unwrap();
        assert!(saved.state().pending_tool_execution().is_none());
        let ContentBlock::ToolResult(denial) = &saved.state().messages()[2].content[1] else {
            panic!("missing denied result")
        };
        assert_eq!(denial.state(), ToolResultState::Denied);
        assert_eq!(
            block_on(agent.retry_tool_execution(execution.confirmation().reply_id().to_owned()))
                .unwrap_err(),
            AgentError::NoPendingToolExecution
        );
    }
}

#[test]
fn interrupted_retry_preserves_the_original_checkpoint() {
    let state = uncertain_state();
    let execution = state.pending_tool_execution().unwrap().clone();
    let handle = Arc::new(Mutex::new(None));
    let tool = Arc::new(InterruptingTool {
        definition: calculator_definition(),
        handle: handle.clone(),
        invocations: AtomicUsize::new(0),
    });
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock"),
        ToolExecutor::new(registry),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new());
    *handle.lock().unwrap() = Some(agent.interrupt_handle());
    block_on(agent.restore(state)).unwrap();
    let error =
        block_on(agent.retry_tool_execution(execution.confirmation().reply_id())).unwrap_err();
    assert_eq!(
        error,
        AgentError::ToolExecutionInDoubt {
            checkpoint: execution.clone()
        }
    );
    assert_eq!(
        block_on(agent.snapshot()).unwrap().pending_tool_execution(),
        Some(&execution)
    );
    assert_eq!(tool.invocations.load(Ordering::SeqCst), 1);
}

struct FailingSaveStore {
    inner: InMemoryStateStore,
}

impl StateStore for FailingSaveStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> StateStoreFuture<'a, Option<StateRecord>> {
        self.inner.load(key)
    }
    fn save(
        &self,
        _key: StateKey,
        _revision: Option<u64>,
        _state: AgentState,
    ) -> StateStoreFuture<'_, StateRecord> {
        Box::pin(async { Err(StateStoreError::new("storage unavailable")) })
    }
}

#[test]
fn retry_does_not_execute_when_checkpoint_save_fails() {
    let state = uncertain_state();
    let execution = state.pending_tool_execution().unwrap().clone();
    let key = StateKey::new("user", "retry").unwrap();
    let inner = InMemoryStateStore::new();
    block_on(inner.save(key.clone(), None, state)).unwrap();
    let tool = Arc::new(MockTool::new(calculator_definition()).with_output("42"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let agent = ReActAgent::new(
        "Friday",
        MockChatModel::new("mock"),
        ToolExecutor::new(registry),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new())
    .with_state_store(key, FailingSaveStore { inner });
    assert!(matches!(
        block_on(agent.retry_tool_execution(execution.confirmation().reply_id())),
        Err(AgentError::StateStore(_))
    ));
    assert!(tool.recorded_invocations().is_empty());
}
