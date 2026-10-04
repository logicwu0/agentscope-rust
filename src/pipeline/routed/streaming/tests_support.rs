use crate::*;
use futures_core::Stream;
use futures_util::future::poll_fn;
use std::{
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
};

pub(super) fn reply(name: &str, text: &str) -> Msg {
    Msg::new(name, Role::Assistant, [ContentBlock::from(text)])
}
pub(super) fn model(text: &str) -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new("offline").with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: text.into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]))
}
pub(super) fn assert_roundtrips(events: &[RoutedEvent]) {
    for event in events {
        let value = serde_json::to_value(event).unwrap();
        assert!(matches!(
            value["type"].as_str(),
            Some("route_started" | "agent" | "finished" | "error")
        ));
        assert_eq!(
            serde_json::from_value::<RoutedEvent>(value).unwrap(),
            *event
        );
    }
}
pub(super) fn terminal_count(events: &[RoutedEvent]) -> usize {
    events
        .iter()
        .filter(|event| {
            matches!(
                event,
                RoutedEvent::Finished { .. } | RoutedEvent::Error { .. }
            )
        })
        .count()
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
struct StartupMarker<'a>(&'a Gate);
impl Drop for StartupMarker<'_> {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

pub(super) struct ScriptAgent {
    name: &'static str,
    events: Vec<AgentResult<AgentEvent>>,
    startup_error: Option<AgentError>,
    startup_gate: Option<Arc<Gate>>,
    event_gate: Option<Arc<Gate>>,
    pub(super) inputs: Mutex<Vec<Msg>>,
    pub(super) stream_calls: AtomicUsize,
    pub(super) reply_calls: AtomicUsize,
    pub(super) item_polls: AtomicUsize,
    pub(super) events_dropped: AtomicUsize,
    pub(super) child_handle_reads: AtomicUsize,
}
impl ScriptAgent {
    fn make(
        name: &'static str,
        events: Vec<AgentResult<AgentEvent>>,
        startup_error: Option<AgentError>,
        startup_gate: Option<Arc<Gate>>,
        event_gate: Option<Arc<Gate>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            name,
            events,
            startup_error,
            startup_gate,
            event_gate,
            inputs: Mutex::new(Vec::new()),
            stream_calls: AtomicUsize::new(0),
            reply_calls: AtomicUsize::new(0),
            item_polls: AtomicUsize::new(0),
            events_dropped: AtomicUsize::new(0),
            child_handle_reads: AtomicUsize::new(0),
        })
    }
    pub(super) fn new(name: &'static str, events: Vec<AgentResult<AgentEvent>>) -> Arc<Self> {
        Self::make(name, events, None, None, None)
    }
    pub(super) fn finished(name: &'static str) -> Arc<Self> {
        Self::new(
            name,
            vec![Ok(AgentEvent::Finished {
                steps: 1,
                message: reply(name, "done"),
            })],
        )
    }
    pub(super) fn startup_error(name: &'static str, error: AgentError) -> Arc<Self> {
        Self::make(name, Vec::new(), Some(error), None, None)
    }
    pub(super) fn startup_gated(name: &'static str, gate: &Arc<Gate>) -> Arc<Self> {
        Self::make(name, Vec::new(), None, Some(gate.clone()), None)
    }
    pub(super) fn event_gated(name: &'static str, gate: &Arc<Gate>) -> Arc<Self> {
        Self::make(
            name,
            vec![Ok(AgentEvent::Finished {
                steps: 1,
                message: reply(name, "done"),
            })],
            None,
            None,
            Some(gate.clone()),
        )
    }
}
struct ScriptEvents<'a> {
    owner: &'a ScriptAgent,
    index: usize,
}
impl Drop for ScriptEvents<'_> {
    fn drop(&mut self) {
        self.owner.events_dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl Stream for ScriptEvents<'_> {
    type Item = AgentResult<AgentEvent>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.owner.item_polls.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.owner.event_gate {
            if gate.poll(cx).is_pending() {
                return Poll::Pending;
            }
        }
        let item = self.owner.events.get(self.index).cloned();
        self.index += usize::from(item.is_some());
        Poll::Ready(item)
    }
}
impl Agent for ScriptAgent {
    fn name(&self) -> &str {
        self.name
    }
    fn stream(&self, input: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        self.inputs.lock().unwrap().push(input);
        Box::pin(async move {
            if let Some(gate) = &self.startup_gate {
                let _marker = StartupMarker(gate);
                poll_fn(|cx| gate.poll(cx)).await;
            }
            if let Some(error) = &self.startup_error {
                return Err(error.clone());
            }
            Ok(Box::pin(ScriptEvents {
                owner: self,
                index: 0,
            }) as AgentEventStream<'_>)
        })
    }
    fn reply(&self, _: Msg) -> AgentFuture<'_, Msg> {
        self.reply_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if let Some(gate) = &self.startup_gate {
                let _marker = StartupMarker(gate);
                poll_fn(|cx| gate.poll(cx)).await;
            }
            Ok(reply(self.name, "plain reply"))
        })
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
        self.child_handle_reads.fetch_add(1, Ordering::SeqCst);
        AgentInterruptHandle::new()
    }
}

pub(super) fn confirming_agent(
    store: Arc<InMemoryStateStore>,
    key: StateKey,
) -> (Arc<ReActAgent>, Arc<MockTool>, Arc<MockChatModel>) {
    let tool = Arc::new(
        MockTool::new(
            ToolDefinition::new("write", "write", serde_json::json!({"type":"object"})).unwrap(),
        )
        .with_error(ToolError::in_doubt("remote outcome unknown")),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let model = Arc::new(MockChatModel::new("confirmation").with_stream([
        Ok(ChatEvent::ToolCallDelta {
            tool_call_id: "call".into(),
            tool_name: "write".into(),
            delta: "{}".into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::ToolCalls,
        }),
    ]));
    let agent = Arc::new(
        ReActAgent::from_shared("worker", model.clone(), ToolExecutor::new(registry))
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_tool_confirmation_required("write")
            .with_shared_state_store(key, store),
    );
    (agent, tool, model)
}
