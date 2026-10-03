use crate::*;
use async_stream::stream;
use futures_core::Stream;
use futures_util::{FutureExt, StreamExt, future::poll_fn};
use std::{
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
};
use tokio::sync::Notify;

pub(super) fn stream_model(text: &str) -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new("offline-stream").with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "shared-block".into(),
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

pub(super) fn confirming_agent(
    store: Arc<dyn StateStore>,
    key: StateKey,
) -> (Arc<ReActAgent>, Arc<MockTool>) {
    let tool = Arc::new(
        MockTool::new(
            ToolDefinition::new("write", "write", serde_json::json!({"type": "object"})).unwrap(),
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

#[derive(Default)]
pub(super) struct Activity {
    pub(super) active: AtomicUsize,
    pub(super) peak: AtomicUsize,
    pub(super) completed: Mutex<Vec<&'static str>>,
}

pub(super) struct GateModel {
    pub(super) label: &'static str,
    activity: Arc<Activity>,
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
    pub(super) fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.release.notify_one();
    }
    pub(super) fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
    fn start(&self, request: ChatRequest) -> ActiveCall<'_> {
        self.requests.lock().unwrap().push(request);
        let active = self.activity.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.activity.peak.fetch_max(active, Ordering::SeqCst);
        ActiveCall(self)
    }
    async fn wait(&self) {
        while !self.released.load(Ordering::SeqCst) {
            self.release.notified().await;
        }
    }
}

impl ChatModel for GateModel {
    fn name(&self) -> &'static str {
        "stream-gate"
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }
    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async move {
            let _call = self.start(request);
            self.wait().await;
            Ok(ChatResponse::completed([ContentBlock::from(self.label)]))
        })
    }
    fn stream(&self, request: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async move {
            let call = self.start(request);
            Ok(Box::pin(stream! {
                let _call = call;
                yield Ok(ChatEvent::TextDelta { block_id: "shared-block".into(), delta: self.label.into() });
                self.wait().await;
                self.activity.completed.lock().unwrap().push(self.label);
                yield Ok(ChatEvent::Finished { reason: FinishReason::Completed });
            }) as ChatEventStream<'_>)
        })
    }
}

pub(super) fn gate(label: &'static str, activity: &Arc<Activity>) -> Arc<GateModel> {
    Arc::new(GateModel {
        label,
        activity: activity.clone(),
        requests: Mutex::new(Vec::new()),
        released: AtomicBool::new(false),
        release: Notify::new(),
        dropped: AtomicUsize::new(0),
    })
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

pub(super) fn assert_roundtrips(events: &[ParallelEvent]) {
    for event in events {
        let encoded = serde_json::to_string(event).unwrap();
        assert_eq!(
            serde_json::from_str::<ParallelEvent>(&encoded).unwrap(),
            *event
        );
    }
}

pub(super) fn wrapped_agent_events(events: &[ParallelEvent]) -> Vec<(usize, &str, AgentEvent)> {
    events
        .iter()
        .filter_map(|event| match event {
            ParallelEvent::Agent {
                branch,
                agent_name,
                event,
            } => Some((*branch, agent_name.as_str(), event.clone())),
            _ => None,
        })
        .collect()
}

pub(super) struct ScriptAgent {
    pub(super) name: &'static str,
    pub(super) startup_error: Option<AgentError>,
    pub(super) events: Vec<AgentResult<AgentEvent>>,
    pub(super) inputs: Mutex<Vec<Msg>>,
    race: Option<(Arc<RaceControl>, bool, bool)>,
}

impl ScriptAgent {
    pub(super) fn new(name: &'static str, events: Vec<AgentResult<AgentEvent>>) -> Arc<Self> {
        Arc::new(Self {
            name,
            startup_error: None,
            events,
            inputs: Mutex::new(Vec::new()),
            race: None,
        })
    }
    pub(super) fn startup(name: &'static str, error: AgentError) -> Arc<Self> {
        Arc::new(Self {
            name,
            startup_error: Some(error),
            events: Vec::new(),
            inputs: Mutex::new(Vec::new()),
            race: None,
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
            if let Some((control, interrupting, startup)) = &self.race {
                if *startup {
                    poll_fn(|cx| control.poll(*interrupting, cx)).await;
                    unreachable!("the startup gate never completes");
                }
                return Ok(Box::pin(RaceStream {
                    control: control.clone(),
                    interrupting: *interrupting,
                }) as AgentEventStream<'_>);
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

#[derive(Default)]
pub(super) struct RaceControl {
    pub(super) handle: Mutex<Option<AgentInterruptHandle>>,
    pending_waker: Mutex<Option<Waker>>,
    pub(super) target_polls: AtomicUsize,
    pub(super) interrupting_polls: AtomicUsize,
}

impl RaceControl {
    fn poll(&self, interrupting: bool, cx: &mut Context<'_>) -> Poll<()> {
        if interrupting {
            if self.interrupting_polls.fetch_add(1, Ordering::SeqCst) == 0 {
                self.handle.lock().unwrap().as_ref().unwrap().interrupt();
                self.pending_waker
                    .lock()
                    .unwrap()
                    .take()
                    .expect("first branch must already be pending")
                    .wake();
            }
        } else {
            self.target_polls.fetch_add(1, Ordering::SeqCst);
            *self.pending_waker.lock().unwrap() = Some(cx.waker().clone());
        }
        Poll::Pending
    }
}

struct RaceStream {
    control: Arc<RaceControl>,
    interrupting: bool,
}

impl Stream for RaceStream {
    type Item = AgentResult<AgentEvent>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let _ = self.control.poll(self.interrupting, cx);
        Poll::Pending
    }
}

pub(super) fn race_agents(startup: bool) -> (Vec<Arc<dyn Agent>>, Arc<RaceControl>) {
    let control = Arc::new(RaceControl::default());
    let agents = [("pending", false), ("interrupting", true)]
        .into_iter()
        .map(|(name, interrupting)| {
            Arc::new(ScriptAgent {
                name,
                startup_error: None,
                events: Vec::new(),
                inputs: Mutex::new(Vec::new()),
                race: Some((control.clone(), interrupting, startup)),
            }) as Arc<dyn Agent>
        })
        .collect();
    (agents, control)
}
