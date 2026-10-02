use super::*;
use crate::*;
use futures_util::FutureExt;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::Notify;

fn model(text: &str) -> Arc<MockChatModel> {
    Arc::new(
        MockChatModel::new("offline")
            .with_response(ChatResponse::completed([ContentBlock::from(text)])),
    )
}

fn agent(name: &str, model: Arc<dyn ChatModel>) -> Arc<ReActAgent> {
    Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))
            .unwrap()
            .with_memory(InMemoryMemory::new()),
    )
}

#[derive(Default)]
struct Activity {
    active: AtomicUsize,
    peak: AtomicUsize,
    completion_order: Mutex<Vec<&'static str>>,
}

struct GateModel {
    label: &'static str,
    activity: Arc<Activity>,
    requests: Mutex<Vec<ChatRequest>>,
    released: AtomicBool,
    release: Notify,
    dropped: AtomicUsize,
    response: Result<ChatResponse, ModelError>,
}

struct ActiveCall<'a>(&'a GateModel);

impl Drop for ActiveCall<'_> {
    fn drop(&mut self) {
        self.0.activity.active.fetch_sub(1, Ordering::SeqCst);
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl GateModel {
    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.release.notify_one();
    }

    fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl ChatModel for GateModel {
    fn name(&self) -> &'static str {
        "parallel-gate"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }

    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async move {
            self.requests.lock().unwrap().push(request);
            let active = self.activity.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.activity.peak.fetch_max(active, Ordering::SeqCst);
            let _call = ActiveCall(self);
            while !self.released.load(Ordering::SeqCst) {
                self.release.notified().await;
            }
            self.activity
                .completion_order
                .lock()
                .unwrap()
                .push(self.label);
            self.response.clone()
        })
    }

    fn stream(&self, _: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async { Err(ModelError::new("unused streaming operation")) })
    }
}

fn gate(label: &'static str, activity: &Arc<Activity>) -> Arc<GateModel> {
    gate_response(
        label,
        activity,
        Ok(ChatResponse::completed([ContentBlock::from(label)])),
    )
}

fn gate_response(
    label: &'static str,
    activity: &Arc<Activity>,
    response: Result<ChatResponse, ModelError>,
) -> Arc<GateModel> {
    Arc::new(GateModel {
        label,
        activity: activity.clone(),
        requests: Mutex::new(Vec::new()),
        released: AtomicBool::new(false),
        release: Notify::new(),
        dropped: AtomicUsize::new(0),
        response,
    })
}

#[tokio::test]
async fn bounded_overlap_returns_ordered_results_and_preserves_independent_memory() {
    let activity = Arc::new(Activity::default());
    let models = ["a", "b", "c", "d"].map(|name| gate(name, &activity));
    let agents = ["a", "b", "c", "d"]
        .into_iter()
        .zip(&models)
        .map(|(name, model)| agent(name, model.clone()))
        .collect::<Vec<_>>();
    for a in &agents {
        a.observe(Msg::user(format!("{} private history", a.name())))
            .await
            .unwrap();
    }
    let pipeline = ParallelPipeline::new(
        agents
            .iter()
            .cloned()
            .map(|a| a as Arc<dyn Agent>)
            .collect(),
        2,
    )
    .unwrap();
    let mut input = Msg::user("analyze the same task");
    input
        .metadata
        .insert("shared".into(), serde_json::json!({"value": 7}));
    let mut run = pipeline.run(input.clone());
    assert!(run.as_mut().now_or_never().is_none());
    assert_eq!(activity.active.load(Ordering::SeqCst), 2);
    assert_eq!(models[0].requests().len(), 1);
    assert_eq!(models[1].requests().len(), 1);
    assert!(models[2].requests().is_empty());
    assert!(models[3].requests().is_empty());

    models[1].release();
    assert!(run.as_mut().now_or_never().is_none());
    assert_eq!(models[2].requests().len(), 1);
    assert!(models[3].requests().is_empty());
    models[2].release();
    assert!(run.as_mut().now_or_never().is_none());
    assert_eq!(models[3].requests().len(), 1);
    models[3].release();
    assert!(run.as_mut().now_or_never().is_none());
    models[0].release();
    let output = run.await.unwrap();

    assert_eq!(activity.peak.load(Ordering::SeqCst), 2);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert_eq!(
        *activity.completion_order.lock().unwrap(),
        vec!["b", "c", "d", "a"]
    );
    assert_eq!(output.branches.len(), 4);
    for (index, branch) in output.branches.iter().enumerate() {
        assert_eq!(branch.branch, index + 1);
        assert_eq!(branch.agent_name, agents[index].name());
        let ParallelBranchOutcome::Completed(message) = &branch.outcome else {
            panic!("successful branches must retain their replies")
        };
        assert_eq!(message.name, agents[index].name());
        assert_eq!(
            message.text_content("").as_deref(),
            Some(agents[index].name())
        );
        let requests = models[index].requests();
        assert_eq!(requests[0].messages.len(), 2);
        assert_eq!(requests[0].messages.last(), Some(&input));
        assert_eq!(
            requests[0].messages[0].text_content(""),
            Some(format!("{} private history", agents[index].name()))
        );
        let snapshot = agents[index].snapshot().await.unwrap();
        assert_eq!(snapshot.messages().len(), 3);
        assert_eq!(snapshot.messages()[1], input);
        assert_eq!(snapshot.messages()[2], *message);
    }
    let encoded = serde_json::to_string(&output).unwrap();
    assert_eq!(
        serde_json::from_str::<ParallelOutput>(&encoded).unwrap(),
        output
    );
}

#[tokio::test]
async fn failures_do_not_skip_queued_branches_and_preserve_all_outcomes() {
    let first =
        Arc::new(MockChatModel::new("broken-first").with_error(ModelError::new("first failure")));
    let second = model("second completed");
    let third =
        Arc::new(MockChatModel::new("broken-third").with_error(ModelError::new("third failure")));
    let pipeline = ParallelPipeline::new(
        vec![
            agent("first", first.clone()),
            agent("second", second.clone()),
            agent("third", third.clone()),
        ],
        1,
    )
    .unwrap();
    let error = pipeline.run(Msg::user("task")).await.unwrap_err();
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    assert_eq!(error.branches.len(), 3);
    assert_eq!(first.recorded_requests().len(), 1);
    assert_eq!(second.recorded_requests().len(), 1);
    assert_eq!(third.recorded_requests().len(), 1);
    assert!(matches!(
        &error.branches[0].outcome,
        ParallelBranchOutcome::Failed(cause) if matches!(cause.as_ref(), AgentError::Model(error) if error.message == "first failure")
    ));
    assert!(matches!(
        &error.branches[1].outcome,
        ParallelBranchOutcome::Completed(message) if message.text_content("").as_deref() == Some("second completed")
    ));
    assert!(matches!(
        &error.branches[2].outcome,
        ParallelBranchOutcome::Failed(cause) if matches!(cause.as_ref(), AgentError::Model(error) if error.message == "third failure")
    ));
    let encoded = serde_json::to_string(&error).unwrap();
    assert_eq!(
        serde_json::from_str::<ParallelError>(&encoded).unwrap(),
        error
    );
}

#[tokio::test]
async fn pending_tool_confirmation_is_preserved_and_does_not_block_siblings() {
    let tool = Arc::new(
        MockTool::new(
            ToolDefinition::new("write", "write", serde_json::json!({"type": "object"})).unwrap(),
        )
        .with_output("saved"),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let blocked = Arc::new(
        ReActAgent::new(
            "blocked",
            MockChatModel::new("approval").with_response(ChatResponse::finished(
                [ToolCallBlock::complete("call", "write", "{}")
                    .unwrap()
                    .into()],
                FinishReason::ToolCalls,
            )),
            ToolExecutor::new(registry),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("write"),
    );
    let sibling = model("sibling completed");
    let pipeline =
        ParallelPipeline::new(vec![blocked.clone(), agent("sibling", sibling.clone())], 1).unwrap();
    let error = pipeline.run(Msg::user("task")).await.unwrap_err();
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    let ParallelBranchOutcome::Failed(cause) = &error.branches[0].outcome else {
        panic!("the confirmation checkpoint must be returned as an agent failure")
    };
    let AgentError::ToolConfirmationRequired { checkpoint } = cause.as_ref() else {
        panic!("the original confirmation error must be retained")
    };
    assert_eq!(
        checkpoint,
        blocked
            .snapshot()
            .await
            .unwrap()
            .pending_tool_calls()
            .unwrap()
    );
    assert!(tool.recorded_invocations().is_empty());
    assert_eq!(sibling.recorded_requests().len(), 1);
    assert!(matches!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    let encoded = serde_json::to_string(&error).unwrap();
    assert_eq!(
        serde_json::from_str::<ParallelError>(&encoded).unwrap(),
        error
    );
}

#[tokio::test]
async fn interruption_retains_observed_results_and_distinguishes_queued_branches() {
    let activity = Arc::new(Activity::default());
    let completed = gate("a", &activity);
    let failed = gate_response("b", &activity, Err(ModelError::new("b failure")));
    let active_first = gate("c", &activity);
    let active_second = gate("d", &activity);
    let queued = gate("e", &activity);
    let pipeline = ParallelPipeline::new(
        vec![
            agent("a", completed.clone()),
            agent("b", failed.clone()),
            agent("c", active_first.clone()),
            agent("d", active_second.clone()),
            agent("e", queued.clone()),
        ],
        2,
    )
    .unwrap();
    let mut run = pipeline.run(Msg::user("task"));
    assert!(run.as_mut().now_or_never().is_none());
    completed.release();
    assert!(run.as_mut().now_or_never().is_none());
    assert_eq!(active_first.requests().len(), 1);
    failed.release();
    assert!(run.as_mut().now_or_never().is_none());
    assert_eq!(active_second.requests().len(), 1);
    assert!(queued.requests().is_empty());
    pipeline.interrupt_handle().interrupt();
    let error = run.await.unwrap_err();

    assert_eq!(error.cause, ParallelFailure::Interrupted);
    assert_eq!(error.branches.len(), 5);
    assert!(matches!(
        error.branches[0].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert!(matches!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Failed(_)
    ));
    assert_eq!(
        error.branches[2].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert_eq!(
        error.branches[3].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert_eq!(error.branches[4].outcome, ParallelBranchOutcome::NotStarted);
    assert!(queued.requests().is_empty());
    assert_eq!(active_first.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(active_second.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    let encoded = serde_json::to_string(&error).unwrap();
    assert_eq!(
        serde_json::from_str::<ParallelError>(&encoded).unwrap(),
        error
    );

    active_first.release();
    active_second.release();
    queued.release();
    assert!(pipeline.run(Msg::user("new run")).await.is_err());
    assert_eq!(queued.requests().len(), 1);
}

#[tokio::test]
async fn lazy_runs_clones_and_drop_share_and_release_the_run_lock() {
    let activity = Arc::new(Activity::default());
    let gated = gate("first", &activity);
    let last = model("last completed");
    let first_agent = agent("first", gated.clone());
    let pipeline =
        ParallelPipeline::new(vec![first_agent.clone(), agent("last", last.clone())], 1).unwrap();
    let cloned = pipeline.clone();
    drop(pipeline.run(Msg::user("unpolled")));
    assert!(gated.requests().is_empty());
    assert!(first_agent.snapshot().await.unwrap().messages().is_empty());
    let mut active = pipeline.run(Msg::user("active"));
    assert!(active.as_mut().now_or_never().is_none());
    let busy = cloned.run(Msg::user("overlap")).await.unwrap_err();
    assert_eq!(busy.cause, ParallelFailure::Busy);
    assert!(busy.branches.is_empty());
    assert!(last.recorded_requests().is_empty());
    drop(active);
    assert_eq!(gated.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert!(last.recorded_requests().is_empty());
    gated.release();
    let output = cloned.run(Msg::user("explicit new run")).await.unwrap();
    assert_eq!(output.branches.len(), 2);
    assert_eq!(gated.requests().len(), 2);
    assert_eq!(last.recorded_requests().len(), 1);
}

#[tokio::test]
async fn child_interruption_is_an_agent_failure_while_siblings_finish() {
    let activity = Arc::new(Activity::default());
    let gated = gate("first", &activity);
    let first = agent("first", gated.clone());
    let last = model("last completed");
    let pipeline =
        ParallelPipeline::new(vec![first.clone(), agent("last", last.clone())], 2).unwrap();
    let mut run = pipeline.run(Msg::user("task"));
    assert!(run.as_mut().now_or_never().is_none());
    assert_eq!(last.recorded_requests().len(), 1);
    first.interrupt_handle().interrupt();
    let error = run.await.unwrap_err();
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    assert!(matches!(
        &error.branches[0].outcome,
        ParallelBranchOutcome::Failed(cause) if **cause == AgentError::Interrupted
    ));
    assert!(matches!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert_eq!(gated.dropped.load(Ordering::SeqCst), 1);
}

struct InterruptBeforeReply(Arc<Mutex<Option<AgentInterruptHandle>>>);

impl AgentHook for InterruptBeforeReply {
    fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a> {
        Box::pin(async move {
            if matches!(event, AgentHookEvent::BeforeReply { .. }) {
                self.0.lock().unwrap().as_ref().unwrap().interrupt();
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn synchronous_interrupt_does_not_invoke_already_scheduled_siblings() {
    let activity = Arc::new(Activity::default());
    let gated = gate("first", &activity);
    let control = Arc::new(Mutex::new(None));
    let first = Arc::new(
        ReActAgent::from_shared(
            "first",
            gated.clone(),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_hook(InterruptBeforeReply(control.clone())),
    );
    let sibling_model = model("must not run");
    let sibling = agent("sibling", sibling_model.clone());
    let pipeline = ParallelPipeline::new(vec![first, sibling.clone()], 2).unwrap();
    *control.lock().unwrap() = Some(pipeline.interrupt_handle());
    let error = pipeline.run(Msg::user("task")).await.unwrap_err();
    assert_eq!(error.cause, ParallelFailure::Interrupted);
    assert_eq!(
        error.branches[0].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert_eq!(error.branches[1].outcome, ParallelBranchOutcome::NotStarted);
    assert!(sibling_model.recorded_requests().is_empty());
    assert!(sibling.snapshot().await.unwrap().messages().is_empty());
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
}

struct NameOnlyAgent(&'static str);

// Constructor validation must only inspect names, never invoke agent operations.
impl Agent for NameOnlyAgent {
    fn name(&self) -> &str {
        self.0
    }
    fn reply(&self, _: Msg) -> AgentFuture<'_, Msg> {
        panic!("unexpected operation")
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
        panic!("unexpected operation")
    }
}

#[test]
fn configuration_rejects_empty_zero_blank_and_duplicate_names() {
    assert!(matches!(
        ParallelPipeline::new(vec![], 1),
        Err(PipelineConfigError::Empty)
    ));
    assert!(matches!(
        ParallelPipeline::new(vec![Arc::new(NameOnlyAgent("one"))], 0),
        Err(PipelineConfigError::ZeroConcurrency)
    ));
    assert!(matches!(
        ParallelPipeline::new(
            vec![
                Arc::new(NameOnlyAgent("one")),
                Arc::new(NameOnlyAgent(" \t"))
            ],
            2
        ),
        Err(PipelineConfigError::EmptyName { step: 2 })
    ));
    let same: Arc<dyn Agent> = Arc::new(NameOnlyAgent("same"));
    assert!(matches!(
        ParallelPipeline::new(vec![same.clone(), same], 2),
        Err(PipelineConfigError::DuplicateName(name)) if name == "same"
    ));
    assert!(matches!(
        ParallelPipeline::new(vec![Arc::new(NameOnlyAgent("same")), Arc::new(NameOnlyAgent("same"))], 2),
        Err(PipelineConfigError::DuplicateName(name)) if name == "same"
    ));
    assert!(ParallelPipeline::new(vec![Arc::new(NameOnlyAgent("one"))], 8).is_ok());
}
