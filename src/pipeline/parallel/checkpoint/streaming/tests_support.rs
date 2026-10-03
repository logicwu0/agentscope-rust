use crate::*;
use async_stream::stream;
use futures_util::{FutureExt, StreamExt};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::Notify;

pub(super) fn model(text: &str) -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new("offline-stream").with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "shared-text".into(),
            delta: text.into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]))
}

pub(super) fn agent(name: &str, model: Arc<dyn ChatModel>) -> Arc<ReActAgent> {
    Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))
            .unwrap()
            .with_memory(InMemoryMemory::new()),
    )
}

pub(super) fn reply(name: &str, text: &str) -> Msg {
    Msg::new(name, Role::Assistant, [ContentBlock::from(text)])
}

pub(super) fn checkpoint(
    names: &[&str],
    input: Msg,
    branches: Vec<ParallelBranchCheckpoint>,
) -> ParallelCheckpoint {
    ParallelCheckpoint {
        version: PARALLEL_CHECKPOINT_VERSION,
        agent_names: names.iter().map(|name| (*name).into()).collect(),
        input,
        branches,
    }
}

pub(super) fn terminal_count(events: &[ParallelEvent]) -> usize {
    events
        .iter()
        .filter(|event| {
            matches!(
                event,
                ParallelEvent::Finished { .. } | ParallelEvent::Error { .. }
            )
        })
        .count()
}

pub(super) fn error(events: &[ParallelEvent]) -> &ParallelError {
    assert_eq!(terminal_count(events), 1);
    let Some(ParallelEvent::Error { error }) = events.last() else {
        panic!("expected one terminal error")
    };
    error
}

pub(super) fn assert_roundtrips(events: &[ParallelEvent]) {
    for event in events {
        let encoded = serde_json::to_string(event).unwrap();
        assert_eq!(
            serde_json::from_str::<ParallelEvent>(&encoded).unwrap(),
            *event
        );
    }
}

pub(super) fn assert_scripted_failures(error: &ParallelError, failures: [AgentError; 3]) {
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    assert_eq!(error.branches.len(), 6);
    for (index, failure) in [0, 1, 3].into_iter().zip(failures) {
        assert_eq!(
            error.branches[index].outcome,
            ParallelBranchOutcome::Failed(Box::new(failure))
        );
    }
    assert!(
        matches!(&error.branches[2].outcome, ParallelBranchOutcome::Failed(cause) if matches!(cause.as_ref(), AgentError::InvalidModelResponse(reason) if reason.contains("terminal")))
    );
    assert!(matches!(
        error.branches[5].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
}

pub(super) fn drain_ready(
    events: &mut ParallelEventStream<'_>,
    observed: &mut Vec<ParallelEvent>,
) -> bool {
    loop {
        match events.next().now_or_never() {
            Some(Some(event)) => observed.push(event),
            Some(None) => return true,
            None => return false,
        }
    }
}

#[derive(Default)]
pub(super) struct Activity {
    pub(super) active: AtomicUsize,
    pub(super) peak: AtomicUsize,
    pub(super) completion_order: Mutex<Vec<&'static str>>,
}

pub(super) struct GateModel {
    label: &'static str,
    activity: Arc<Activity>,
    witness: Option<(Arc<InMemoryParallelStore>, StateKey, usize)>,
    requests: Mutex<Vec<ChatRequest>>,
    released: AtomicBool,
    release: Notify,
    pub(super) dropped: AtomicUsize,
}

struct ActiveCall<'a>(&'a GateModel);
impl Drop for ActiveCall<'_> {
    fn drop(&mut self) {
        self.0.activity.active.fetch_sub(1, Ordering::SeqCst);
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl GateModel {
    pub(super) fn new(
        label: &'static str,
        activity: &Arc<Activity>,
        witness: Option<(Arc<InMemoryParallelStore>, StateKey, usize)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            label,
            activity: activity.clone(),
            witness,
            requests: Mutex::new(Vec::new()),
            released: AtomicBool::new(false),
            release: Notify::new(),
            dropped: AtomicUsize::new(0),
        })
    }
    pub(super) fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.release.notify_one();
    }
    pub(super) fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl ChatModel for GateModel {
    fn name(&self) -> &'static str {
        "durable-stream-gate"
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }
    fn generate(&self, _: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async { Err(ModelError::new("unused non-streaming operation")) })
    }
    fn stream(&self, request: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async move {
            if let Some((store, key, index)) = &self.witness {
                let record = store
                    .load(key)
                    .await
                    .unwrap()
                    .expect("marker must exist before invocation");
                assert_eq!(
                    record.checkpoint.branches[*index],
                    ParallelBranchCheckpoint::InFlight
                );
            }
            self.requests.lock().unwrap().push(request);
            let active = self.activity.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.activity.peak.fetch_max(active, Ordering::SeqCst);
            let call = ActiveCall(self);
            Ok(Box::pin(stream! {
                let _call = call;
                yield Ok(ChatEvent::TextDelta { block_id: "shared-text".into(), delta: self.label.into() });
                while !self.released.load(Ordering::SeqCst) { self.release.notified().await; }
                self.activity.completion_order.lock().unwrap().push(self.label);
                yield Ok(ChatEvent::Finished { reason: FinishReason::Completed });
            }) as ChatEventStream<'_>)
        })
    }
}

#[derive(Clone, Copy)]
pub(super) enum SavePoint {
    Initial,
    Marker(usize),
    Terminal(usize),
}
impl SavePoint {
    fn matches(self, expected: Option<u64>, checkpoint: &ParallelCheckpoint) -> bool {
        match self {
            Self::Initial => expected.is_none(),
            Self::Marker(index) => matches!(
                checkpoint.branches[index],
                ParallelBranchCheckpoint::InFlight
            ),
            Self::Terminal(index) => matches!(
                checkpoint.branches[index],
                ParallelBranchCheckpoint::Completed(_) | ParallelBranchCheckpoint::Failed(_)
            ),
        }
    }
}

pub(super) struct ControlledStore {
    pub(super) inner: Arc<InMemoryParallelStore>,
    reject: Option<SavePoint>,
    signal_on: Option<SavePoint>,
    ambiguous_on: Option<SavePoint>,
    pub(super) interrupt: Mutex<Option<AgentInterruptHandle>>,
}

impl ControlledStore {
    pub(super) fn reject(point: SavePoint) -> Self {
        Self {
            inner: Arc::new(InMemoryParallelStore::new()),
            reject: Some(point),
            signal_on: None,
            ambiguous_on: None,
            interrupt: Mutex::new(None),
        }
    }
    pub(super) fn interrupt_after(point: SavePoint) -> Self {
        Self {
            inner: Arc::new(InMemoryParallelStore::new()),
            reject: None,
            signal_on: Some(point),
            ambiguous_on: None,
            interrupt: Mutex::new(None),
        }
    }
    pub(super) fn lose_acknowledgement(point: SavePoint) -> Self {
        Self {
            inner: Arc::new(InMemoryParallelStore::new()),
            reject: None,
            signal_on: None,
            ambiguous_on: Some(point),
            interrupt: Mutex::new(None),
        }
    }
}

impl ParallelStore for ControlledStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<ParallelRecord>> {
        self.inner.load(key)
    }
    fn save(
        &self,
        key: StateKey,
        expected: Option<u64>,
        checkpoint: ParallelCheckpoint,
    ) -> PipelineStoreFuture<'_, ParallelRecord> {
        Box::pin(async move {
            if self
                .reject
                .is_some_and(|point| point.matches(expected, &checkpoint))
            {
                return Err(PipelineStoreError::new("deliberate write failure"));
            }
            let signal = self
                .signal_on
                .is_some_and(|point| point.matches(expected, &checkpoint));
            let ambiguous = self
                .ambiguous_on
                .is_some_and(|point| point.matches(expected, &checkpoint));
            let record = self.inner.save(key, expected, checkpoint).await?;
            if ambiguous {
                return Err(PipelineStoreError::new(
                    "write committed, acknowledgement lost",
                ));
            }
            if signal {
                if let Some(handle) = self.interrupt.lock().unwrap().take() {
                    handle.interrupt();
                }
            }
            Ok(record)
        })
    }
}

pub(super) struct ScriptAgent {
    name: &'static str,
    startup_error: Option<AgentError>,
    events: Vec<AgentResult<AgentEvent>>,
    pub(super) inputs: Mutex<Vec<Msg>>,
}

impl ScriptAgent {
    pub(super) fn new(name: &'static str, events: Vec<AgentResult<AgentEvent>>) -> Arc<Self> {
        Arc::new(Self {
            name,
            startup_error: None,
            events,
            inputs: Mutex::new(Vec::new()),
        })
    }
    pub(super) fn startup(name: &'static str, error: AgentError) -> Arc<Self> {
        Arc::new(Self {
            name,
            startup_error: Some(error),
            events: Vec::new(),
            inputs: Mutex::new(Vec::new()),
        })
    }
}

impl Agent for ScriptAgent {
    fn name(&self) -> &str {
        self.name
    }
    fn stream(&self, input: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
        self.inputs.lock().unwrap().push(input);
        Box::pin(async move {
            if let Some(error) = &self.startup_error {
                return Err(error.clone());
            }
            Ok(Box::pin(futures_util::stream::iter(self.events.clone())) as AgentEventStream<'_>)
        })
    }
    fn reply(&self, _: Msg) -> AgentFuture<'_, Msg> {
        panic!("unexpected operation")
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
        panic!("unexpected operation")
    }
    fn resume_tool_calls(&self, _: String, _: Vec<ToolConfirmation>) -> AgentFuture<'_, Msg> {
        panic!("unexpected operation")
    }
    fn resolve_tool_execution(&self, _: String, _: Vec<ToolResultBlock>) -> AgentFuture<'_, Msg> {
        panic!("unexpected operation")
    }
    fn retry_tool_execution(&self, _: String) -> AgentFuture<'_, Msg> {
        panic!("unexpected operation")
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
        panic!("unexpected operation")
    }
}

pub(super) fn confirming_agent(
    store: Arc<InMemoryStateStore>,
    key: StateKey,
) -> (Arc<ReActAgent>, Arc<MockTool>) {
    let tool = Arc::new(
        MockTool::new(
            ToolDefinition::new("write", "write", serde_json::json!({"type":"object"})).unwrap(),
        )
        .with_output("saved"),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let agent = Arc::new(
        ReActAgent::new(
            "blocked",
            MockChatModel::new("approval").with_stream([
                Ok(ChatEvent::ToolCallDelta {
                    tool_call_id: "call".into(),
                    tool_name: "write".into(),
                    delta: "{}".into(),
                }),
                Ok(ChatEvent::Finished {
                    reason: FinishReason::ToolCalls,
                }),
            ]),
            ToolExecutor::new(registry),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("write")
        .with_shared_state_store(key, store),
    );
    (agent, tool)
}
