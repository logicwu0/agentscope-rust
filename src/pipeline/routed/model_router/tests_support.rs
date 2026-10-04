use crate::*;
use futures_util::future::poll_fn;
use serde_json::{Value, json};
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
pub(super) fn structured(value: Value) -> ContentBlock {
    StructuredOutputBlock::complete(json!({}), value)
        .unwrap()
        .into()
}
pub(super) fn response(route: &str) -> ChatResponse {
    ChatResponse::completed([structured(json!({"route": route}))])
}
pub(super) fn raw_response(raw: &str) -> ChatResponse {
    let mut block = StructuredOutputBlock::streaming(json!({})).unwrap();
    block.append_output_delta(raw).unwrap();
    block.finish().unwrap();
    ChatResponse::completed([block.into()])
}
pub(super) fn pipeline(selected: &Arc<ScriptAgent>, other: &Arc<ScriptAgent>) -> RoutedPipeline {
    RoutedPipeline::new(vec![
        ("selected".into(), selected.clone()),
        ("alias".into(), selected.clone()),
        ("other".into(), other.clone()),
    ])
    .unwrap()
}
pub(super) fn router(pipeline: RoutedPipeline, model: Arc<dyn ChatModel>) -> ModelRouter {
    ModelRouter::from_shared(
        pipeline,
        model,
        vec![
            ("selected".into(), "Primary route".into()),
            ("alias".into(), "Exact alias".into()),
        ],
    )
    .unwrap()
}
pub(super) fn assert_selection_error(error: &RoutedError, expected: RouteSelectionError) {
    assert_eq!(error.route, "");
    assert_eq!(error.agent_name, None);
    assert_eq!(error.cause, RoutedFailure::Selection(expected));
    assert_eq!(
        serde_json::from_value::<RoutedError>(serde_json::to_value(error).unwrap()).unwrap(),
        *error
    );
}

pub(super) fn assert_exact_schema(request: &ChatRequest) {
    let schema = request.structured_output_schema.as_ref().unwrap();
    assert_eq!(schema["required"], json!(["route"]));
    assert_eq!(schema["additionalProperties"], json!(false));
    assert_eq!(schema["properties"].as_object().unwrap().len(), 1);
    let choices = schema["properties"]["route"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .map(Value::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        choices,
        [json!("selected"), json!("alias"), json!(null)]
            .iter()
            .map(Value::to_string)
            .collect()
    );
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

pub(super) struct GateModel {
    pub(super) requests: Mutex<Vec<ChatRequest>>,
    gate: Option<Arc<Gate>>,
    interrupt_on_reply: Option<AgentInterruptHandle>,
    response: ChatResponse,
}
impl GateModel {
    pub(super) fn gated(gate: &Arc<Gate>, response: ChatResponse) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::new(Vec::new()),
            gate: Some(gate.clone()),
            interrupt_on_reply: None,
            response,
        })
    }
    pub(super) fn interrupting(
        interrupt: AgentInterruptHandle,
        response: ChatResponse,
    ) -> Arc<Self> {
        Arc::new(Self {
            requests: Mutex::new(Vec::new()),
            gate: None,
            interrupt_on_reply: Some(interrupt),
            response,
        })
    }
}
impl ChatModel for GateModel {
    fn name(&self) -> &'static str {
        "router-gate"
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }
    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        self.requests.lock().unwrap().push(request);
        Box::pin(async move {
            if let Some(gate) = &self.gate {
                let _marker = DropMarker(gate);
                poll_fn(|cx| gate.poll(cx)).await;
            }
            if let Some(interrupt) = &self.interrupt_on_reply {
                interrupt.interrupt();
            }
            Ok(self.response.clone())
        })
    }
    fn stream(&self, _: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        panic!("selector must use one non-streaming call")
    }
}

pub(super) struct ScriptAgent {
    name: &'static str,
    result: AgentResult<Msg>,
    gate: Option<Arc<Gate>>,
    pub(super) inputs: Mutex<Vec<Msg>>,
    pub(super) child_handle_reads: AtomicUsize,
}
impl ScriptAgent {
    pub(super) fn new(name: &'static str, result: AgentResult<Msg>) -> Arc<Self> {
        Self::make(name, result, None)
    }
    pub(super) fn gated(name: &'static str, original: Msg, gate: &Arc<Gate>) -> Arc<Self> {
        Self::make(name, Ok(original), Some(gate.clone()))
    }
    fn make(name: &'static str, result: AgentResult<Msg>, gate: Option<Arc<Gate>>) -> Arc<Self> {
        Arc::new(Self {
            name,
            result,
            gate,
            inputs: Mutex::new(Vec::new()),
            child_handle_reads: AtomicUsize::new(0),
        })
    }
}
impl Agent for ScriptAgent {
    fn name(&self) -> &str {
        self.name
    }
    fn reply(&self, input: Msg) -> AgentFuture<'_, Msg> {
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
        panic!("unexpected child stream")
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
        panic!("unexpected restore")
    }
    fn resume_tool_calls(&self, _: String, _: Vec<ToolConfirmation>) -> AgentFuture<'_, Msg> {
        panic!("unexpected automatic approval")
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

pub(super) fn uncertain_agent() -> (Arc<ReActAgent>, Arc<MockChatModel>, Arc<MockTool>) {
    let tool = Arc::new(
        MockTool::new(ToolDefinition::new("write", "write", json!({"type":"object"})).unwrap())
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
