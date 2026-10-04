use crate::*;
use futures_util::future::poll_fn;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::{Poll, Waker},
};

pub(super) fn model(text: &str) -> Arc<MockChatModel> {
    Arc::new(
        MockChatModel::new("offline")
            .with_response(ChatResponse::completed([ContentBlock::from(text)])),
    )
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

#[derive(Default)]
pub(super) struct ReplyGate {
    pub(super) polls: AtomicUsize,
    pub(super) effects: AtomicUsize,
    pub(super) dropped: AtomicUsize,
    released: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl ReplyGate {
    pub(super) fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
}

struct DropMarker<'a>(&'a ReplyGate);
impl Drop for DropMarker<'_> {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

pub(super) struct ScriptAgent {
    name: &'static str,
    result: AgentResult<Msg>,
    gate: Option<Arc<ReplyGate>>,
    pub(super) inputs: Mutex<Vec<Msg>>,
    pub(super) renamed: AtomicBool,
    pub(super) interrupt_handle_reads: AtomicUsize,
    interrupt: AgentInterruptHandle,
}

impl ScriptAgent {
    pub(super) fn returning(name: &'static str, result: AgentResult<Msg>) -> Arc<Self> {
        Arc::new(Self {
            name,
            result,
            gate: None,
            inputs: Mutex::new(Vec::new()),
            renamed: AtomicBool::new(false),
            interrupt_handle_reads: AtomicUsize::new(0),
            interrupt: AgentInterruptHandle::new(),
        })
    }
    pub(super) fn gated(name: &'static str, gate: &Arc<ReplyGate>) -> Arc<Self> {
        Arc::new(Self {
            name,
            result: Ok(reply(name, "completed")),
            gate: Some(gate.clone()),
            inputs: Mutex::new(Vec::new()),
            renamed: AtomicBool::new(false),
            interrupt_handle_reads: AtomicUsize::new(0),
            interrupt: AgentInterruptHandle::new(),
        })
    }
}

impl Agent for ScriptAgent {
    fn name(&self) -> &str {
        if self.renamed.load(Ordering::SeqCst) {
            "changed-name"
        } else {
            self.name
        }
    }
    fn reply(&self, input: Msg) -> AgentFuture<'_, Msg> {
        // Invocation itself is observable: lazy/unknown/Busy routes must never call it.
        self.inputs.lock().unwrap().push(input);
        Box::pin(async move {
            if let Some(gate) = &self.gate {
                let _drop = DropMarker(gate);
                poll_fn(|cx| {
                    gate.polls.fetch_add(1, Ordering::SeqCst);
                    if gate.released.load(Ordering::SeqCst) {
                        gate.effects.fetch_add(1, Ordering::SeqCst);
                        Poll::Ready(())
                    } else {
                        *gate.waker.lock().unwrap() = Some(cx.waker().clone());
                        Poll::Pending
                    }
                })
                .await;
            }
            self.result.clone()
        })
    }
    fn stream(&self, _: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
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
        self.interrupt_handle_reads.fetch_add(1, Ordering::SeqCst);
        self.interrupt.clone()
    }
}

pub(super) fn uncertain_agent() -> (Arc<ReActAgent>, Arc<MockChatModel>, Arc<MockTool>) {
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
            .with_tool_confirmation_required("write"),
    );
    (agent, model, tool)
}
