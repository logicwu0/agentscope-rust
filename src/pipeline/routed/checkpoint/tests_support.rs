use crate::*;
use futures_util::future::poll_fn;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
};

pub(super) fn reply(name: &str, text: &str) -> Msg {
    Msg::new(name, Role::Assistant, [ContentBlock::from(text)])
}

pub(super) fn checkpoint(status: RoutedCheckpointStatus) -> RoutedCheckpoint {
    RoutedCheckpoint {
        version: ROUTED_CHECKPOINT_VERSION,
        route: "selected".into(),
        agent_name: "worker".into(),
        input: Msg::user("original input"),
        status,
    }
}

pub(super) fn key(session: &str) -> StateKey {
    StateKey::new("routed-test", session).unwrap()
}

#[derive(Default)]
pub(super) struct Gate {
    pub(super) polls: AtomicUsize,
    pub(super) effects: AtomicUsize,
    pub(super) dropped: AtomicUsize,
    released: AtomicBool,
    waker: Mutex<Option<Waker>>,
}
impl Gate {
    pub(super) fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
    fn poll(&self, cx: &mut Context<'_>) -> Poll<()> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        if self.released.load(Ordering::SeqCst) {
            self.effects.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(())
        } else {
            *self.waker.lock().unwrap() = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
struct DropMarker<'a>(&'a Gate);
impl Drop for DropMarker<'_> {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

pub(super) struct ScriptAgent {
    name: &'static str,
    result: AgentResult<Msg>,
    gate: Option<Arc<Gate>>,
    witness: Option<Arc<Mutex<Vec<RoutedRecord>>>>,
    pub(super) inputs: Mutex<Vec<Msg>>,
    pub(super) calls: AtomicUsize,
    pub(super) child_handle_reads: AtomicUsize,
}
impl ScriptAgent {
    pub(super) fn new(name: &'static str, result: AgentResult<Msg>) -> Arc<Self> {
        Self::make(name, result, None, None)
    }
    pub(super) fn gated(name: &'static str, gate: &Arc<Gate>) -> Arc<Self> {
        Self::make(name, Ok(reply(name, "completed")), Some(gate.clone()), None)
    }
    pub(super) fn witnessed(
        name: &'static str,
        result: AgentResult<Msg>,
        history: &Arc<Mutex<Vec<RoutedRecord>>>,
    ) -> Arc<Self> {
        Self::make(name, result, None, Some(history.clone()))
    }
    fn make(
        name: &'static str,
        result: AgentResult<Msg>,
        gate: Option<Arc<Gate>>,
        witness: Option<Arc<Mutex<Vec<RoutedRecord>>>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            name,
            result,
            gate,
            witness,
            inputs: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
            child_handle_reads: AtomicUsize::new(0),
        })
    }
}
impl Agent for ScriptAgent {
    fn name(&self) -> &str {
        self.name
    }
    fn reply(&self, input: Msg) -> AgentFuture<'_, Msg> {
        // Even creating the child future is observable and requires a durable marker.
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(history) = &self.witness {
            let history = history.lock().unwrap();
            let record = history
                .last()
                .expect("marker must exist before reply creation");
            assert_eq!(record.checkpoint.status, RoutedCheckpointStatus::InFlight);
            assert_eq!(record.checkpoint.agent_name, self.name);
            assert_eq!(record.checkpoint.input, input);
        }
        self.inputs.lock().unwrap().push(input);
        Box::pin(async move {
            if let Some(gate) = &self.gate {
                let _marker = DropMarker(gate);
                poll_fn(|cx| gate.poll(cx)).await;
            }
            self.result.clone()
        })
    }
    fn stream(&self, _: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
        panic!("unexpected child stream invocation")
    }
    fn observe(&self, _: Msg) -> AgentFuture<'_, ()> {
        panic!("unexpected operation")
    }
    fn snapshot(&self) -> AgentFuture<'_, AgentState> {
        panic!("unexpected operation")
    }
    fn compact_context(&self, _: usize) -> AgentFuture<'_, Option<ContextSummary>> {
        panic!("unexpected operation")
    }
    fn clear_context_summary(&self) -> AgentFuture<'_, ()> {
        panic!("unexpected operation")
    }
    fn restore(&self, _: AgentState) -> AgentFuture<'_, ()> {
        panic!("unexpected child restoration")
    }
    fn resume_tool_calls(&self, _: String, _: Vec<ToolConfirmation>) -> AgentFuture<'_, Msg> {
        panic!("unexpected automatic confirmation")
    }
    fn resolve_tool_execution(&self, _: String, _: Vec<ToolResultBlock>) -> AgentFuture<'_, Msg> {
        panic!("unexpected operation")
    }
    fn retry_tool_execution(&self, _: String) -> AgentFuture<'_, Msg> {
        panic!("unexpected automatic retry")
    }
    fn stream_resume_tool_calls(
        &self,
        _: String,
        _: Vec<ToolConfirmation>,
    ) -> AgentFuture<'_, AgentEventStream<'_>> {
        panic!("unexpected operation")
    }
    fn stream_retry_tool_execution(&self, _: String) -> AgentFuture<'_, AgentEventStream<'_>> {
        panic!("unexpected operation")
    }
    fn stream_resolve_tool_execution(
        &self,
        _: String,
        _: Vec<ToolResultBlock>,
    ) -> AgentFuture<'_, AgentEventStream<'_>> {
        panic!("unexpected operation")
    }
    fn interrupt_handle(&self) -> AgentInterruptHandle {
        self.child_handle_reads.fetch_add(1, Ordering::SeqCst);
        AgentInterruptHandle::new()
    }
}

#[derive(Clone, Copy)]
pub(super) enum SavePoint {
    Initial,
    Marker,
    Terminal,
}
impl SavePoint {
    fn matches(self, checkpoint: &RoutedCheckpoint) -> bool {
        matches!(
            (self, &checkpoint.status),
            (Self::Initial, RoutedCheckpointStatus::Ready)
                | (Self::Marker, RoutedCheckpointStatus::InFlight)
                | (
                    Self::Terminal,
                    RoutedCheckpointStatus::Completed(_) | RoutedCheckpointStatus::Failed(_)
                )
        )
    }
}
#[derive(Clone, Copy)]
enum SaveAction {
    None,
    Reject,
    Ambiguous,
    Interrupt,
    Conflict,
    BadRevision,
    BadCheckpoint,
}

pub(super) struct ControlledStore {
    pub(super) inner: InMemoryRoutedStore,
    pub(super) history: Arc<Mutex<Vec<RoutedRecord>>>,
    pub(super) loads: AtomicUsize,
    pub(super) saves: AtomicUsize,
    pub(super) interrupt: Mutex<Option<AgentInterruptHandle>>,
    point: SavePoint,
    action: SaveAction,
    gate: Option<Arc<Gate>>,
}
impl ControlledStore {
    pub(super) fn plain() -> Self {
        Self::make(SavePoint::Initial, SaveAction::None, None)
    }
    pub(super) fn gated(point: SavePoint, gate: &Arc<Gate>) -> Self {
        Self::make(point, SaveAction::None, Some(gate.clone()))
    }
    pub(super) fn reject(point: SavePoint) -> Self {
        Self::make(point, SaveAction::Reject, None)
    }
    pub(super) fn ambiguous(point: SavePoint) -> Self {
        Self::make(point, SaveAction::Ambiguous, None)
    }
    pub(super) fn signal(point: SavePoint) -> Self {
        Self::make(point, SaveAction::Interrupt, None)
    }
    pub(super) fn conflict(point: SavePoint) -> Self {
        Self::make(point, SaveAction::Conflict, None)
    }
    pub(super) fn bad_ack(point: SavePoint, alter_checkpoint: bool) -> Self {
        Self::make(
            point,
            if alter_checkpoint {
                SaveAction::BadCheckpoint
            } else {
                SaveAction::BadRevision
            },
            None,
        )
    }
    fn make(point: SavePoint, action: SaveAction, gate: Option<Arc<Gate>>) -> Self {
        Self {
            inner: InMemoryRoutedStore::new(),
            history: Arc::new(Mutex::new(Vec::new())),
            loads: AtomicUsize::new(0),
            saves: AtomicUsize::new(0),
            interrupt: Mutex::new(None),
            point,
            action,
            gate,
        }
    }
}
impl RoutedStore for ControlledStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<RoutedRecord>> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(key)
    }
    fn save(
        &self,
        key: StateKey,
        expected: Option<u64>,
        checkpoint: RoutedCheckpoint,
    ) -> PipelineStoreFuture<'_, RoutedRecord> {
        Box::pin(async move {
            self.saves.fetch_add(1, Ordering::SeqCst);
            let selected = self.point.matches(&checkpoint);
            if selected && matches!(self.action, SaveAction::Reject) {
                return Err(PipelineStoreError::new("write rejected before commit"));
            }
            if selected && matches!(self.action, SaveAction::Conflict) {
                let mut competing = checkpoint.clone();
                competing.status = RoutedCheckpointStatus::Failed(Box::new(AgentError::Model(
                    ModelError::new("concurrent writer"),
                )));
                self.inner.save(key.clone(), expected, competing).await?;
            }
            let mut record = self.inner.save(key, expected, checkpoint).await?;
            self.history.lock().unwrap().push(record.clone());
            if selected {
                if let Some(gate) = &self.gate {
                    let _marker = DropMarker(gate);
                    poll_fn(|cx| gate.poll(cx)).await;
                }
                match self.action {
                    SaveAction::Ambiguous => {
                        return Err(PipelineStoreError::new("commit acknowledgement lost"));
                    }
                    SaveAction::Interrupt => {
                        self.interrupt.lock().unwrap().as_ref().unwrap().interrupt();
                    }
                    SaveAction::BadRevision => record.revision = 0,
                    SaveAction::BadCheckpoint => {
                        record
                            .checkpoint
                            .input
                            .metadata
                            .insert("corrupted".into(), serde_json::json!(true));
                    }
                    SaveAction::None | SaveAction::Reject | SaveAction::Conflict => {}
                }
            }
            Ok(record)
        })
    }
}

pub(super) struct StaticStore {
    pub(super) record: Option<RoutedRecord>,
    pub(super) error: Option<PipelineStoreError>,
    pub(super) saves: AtomicUsize,
}
impl StaticStore {
    pub(super) fn with_record(record: RoutedRecord) -> Self {
        Self {
            record: Some(record),
            error: None,
            saves: AtomicUsize::new(0),
        }
    }
}
impl RoutedStore for StaticStore {
    fn load<'a>(&'a self, _: &'a StateKey) -> PipelineStoreFuture<'a, Option<RoutedRecord>> {
        Box::pin(async move {
            self.error
                .clone()
                .map_or_else(|| Ok(self.record.clone()), Err)
        })
    }
    fn save(
        &self,
        _: StateKey,
        _: Option<u64>,
        _: RoutedCheckpoint,
    ) -> PipelineStoreFuture<'_, RoutedRecord> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(PipelineStoreError::new(
                "unexpected save of invalid checkpoint",
            ))
        })
    }
}

pub(super) fn confirming_agent(
    store: Arc<InMemoryStateStore>,
    key: StateKey,
) -> (Arc<ReActAgent>, Arc<MockChatModel>, Arc<MockTool>) {
    let tool = Arc::new(
        MockTool::new(
            ToolDefinition::new("write", "write", serde_json::json!({"type":"object"})).unwrap(),
        )
        .with_error(ToolError::in_doubt("remote outcome unknown")),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let model = Arc::new(
        MockChatModel::new("approval").with_response(ChatResponse::finished(
            [ToolCallBlock::complete("call", "write", "{}")
                .unwrap()
                .into()],
            FinishReason::ToolCalls,
        )),
    );
    let agent = Arc::new(
        ReActAgent::from_shared("worker", model.clone(), ToolExecutor::new(registry))
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_tool_confirmation_required("write")
            .with_shared_state_store(key, store),
    );
    (agent, model, tool)
}

pub(super) async fn assert_resume_rejected(
    pipeline: &RoutedPipeline,
    store: &dyn RoutedStore,
    key: StateKey,
) {
    assert!(matches!(
        pipeline
            .resume_checkpointed(store, key)
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::UnsafeResume(_)
    ));
}

pub(super) async fn assert_reconciliation_fences_ready_and_concurrent_write(
    pipeline: &RoutedPipeline,
) {
    let ready_store = InMemoryRoutedStore::new();
    ready_store
        .save(
            key("ready"),
            None,
            checkpoint(RoutedCheckpointStatus::Ready),
        )
        .await
        .unwrap();
    assert!(matches!(
        pipeline
            .reconcile_checkpointed(
                &ready_store,
                key("ready"),
                1,
                reply("worker", "not authorized")
            )
            .await
            .unwrap_err()
            .cause,
        RoutedFailure::UnsafeResume(_)
    ));
    let racing_store = ControlledStore::conflict(SavePoint::Terminal);
    racing_store
        .inner
        .save(
            key("racing"),
            None,
            checkpoint(RoutedCheckpointStatus::InFlight),
        )
        .await
        .unwrap();
    let raced = pipeline
        .reconcile_checkpointed(&racing_store, key("racing"), 1, reply("worker", "verified"))
        .await
        .unwrap_err();
    assert!(matches!(raced.cause, RoutedFailure::Store(reason) if reason.contains("conflict")));
    let concurrent = racing_store
        .inner
        .load(&key("racing"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(concurrent.revision, 2);
    assert!(
        matches!(concurrent.checkpoint.status, RoutedCheckpointStatus::Failed(error) if *error == AgentError::Model(ModelError::new("concurrent writer")))
    );
}
