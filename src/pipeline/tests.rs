use super::*;
use crate::*;
use futures_util::FutureExt;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
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

#[tokio::test]
async fn writer_reviewer_handoff_and_independent_memory() {
    let writer_model = model("Draft answer");
    let reviewer_model = model("Reviewed answer");
    let writer = agent("writer", writer_model.clone());
    let reviewer = agent("reviewer", reviewer_model.clone());
    writer
        .observe(Msg::user("writer private history"))
        .await
        .unwrap();
    reviewer
        .observe(Msg::user("reviewer private history"))
        .await
        .unwrap();
    let pipeline = SequentialPipeline::new(vec![writer.clone(), reviewer.clone()]).unwrap();
    let input = Msg::user("Write a draft");
    let output = pipeline.run(input.clone()).await.unwrap();
    assert_eq!(output.steps.len(), 2);
    assert_eq!(output.steps[0].agent_name, "writer");
    assert_eq!(output.steps[1].step, 2);
    assert_eq!(output.message, output.steps[1].message);
    assert_eq!(
        output.message.text_content("").as_deref(),
        Some("Reviewed answer")
    );
    let writer_requests = writer_model.recorded_requests();
    assert_eq!(writer_requests[0].messages.last(), Some(&input));
    let requests = reviewer_model.recorded_requests();
    assert_eq!(requests[0].messages.len(), 2);
    let handoff = &requests[0].messages[1];
    assert_eq!(handoff.role, Role::User);
    assert_eq!(handoff.name, "writer");
    assert_eq!(handoff.text_content("").as_deref(), Some("Draft answer"));
    assert_ne!(handoff.id, output.steps[0].message.id);
    assert_eq!(writer.snapshot().await.unwrap().messages().len(), 3);
    assert_eq!(reviewer.snapshot().await.unwrap().messages().len(), 3);
    let encoded = serde_json::to_string(&output).unwrap();
    assert_eq!(
        serde_json::from_str::<PipelineOutput>(&encoded).unwrap(),
        output
    );
}

#[test]
fn handoff_does_not_copy_reasoning_metadata_identity_or_role() {
    let mut original = Msg::new(
        "untrusted-name",
        Role::System,
        [
            ThinkingBlock::new("private reasoning").into(),
            ContentBlock::from("one"),
            ContentBlock::from("two"),
        ],
    );
    original
        .metadata
        .insert("private".into(), json!("do not share"));
    let next = handoff("writer", &original).unwrap();
    assert_eq!(next.role, Role::User);
    assert_eq!(next.name, "writer");
    assert!(next.metadata.is_empty());
    assert!(next.usage.is_none());
    assert_eq!(next.content.len(), 1);
    assert_eq!(next.text_content("").as_deref(), Some("one\ntwo"));
    assert_ne!(next.id, original.id);
    assert_ne!(next.content, original.content);
}

#[test]
fn configuration_rejects_empty_and_duplicate_agents() {
    assert!(matches!(
        SequentialPipeline::new(vec![]),
        Err(PipelineConfigError::Empty)
    ));
    let a = agent("same", model("a"));
    assert!(matches!(
        SequentialPipeline::new(vec![a.clone(), a]),
        Err(PipelineConfigError::DuplicateName(_))
    ));
    assert!(matches!(
        SequentialPipeline::new(vec![agent("same", model("a")), agent("same", model("b"))]),
        Err(PipelineConfigError::DuplicateName(_))
    ));
}

#[tokio::test]
async fn single_stage_and_unpolled_run() {
    let model = model("final");
    let a = agent("one", model.clone());
    let pipeline = SequentialPipeline::new(vec![a.clone()]).unwrap();
    drop(pipeline.run(Msg::user("never polled")));
    assert!(model.recorded_requests().is_empty());
    assert!(a.snapshot().await.unwrap().messages().is_empty());
    let output = pipeline.run(Msg::user("run")).await.unwrap();
    assert_eq!(output.steps.len(), 1);
    assert_eq!(output.message.name, "one");
}

#[tokio::test]
async fn model_failure_keeps_completed_outputs_and_never_calls_later_stages() {
    let first = model("first complete");
    let failed =
        Arc::new(MockChatModel::new("broken").with_error(ModelError::new("offline failure")));
    let last = model("must not run");
    let pipeline = SequentialPipeline::new(vec![
        agent("a", first.clone()),
        agent("b", failed.clone()),
        agent("c", last.clone()),
    ])
    .unwrap();
    let error = pipeline.run(Msg::user("start")).await.unwrap_err();
    assert_eq!(error.step, Some(2));
    assert_eq!(error.agent_name.as_deref(), Some("b"));
    assert_eq!(error.completed.len(), 1);
    assert!(
        matches!(&error.cause, PipelineFailure::Agent(cause) if matches!(cause.as_ref(), AgentError::Model(_)))
    );
    assert_eq!(first.recorded_requests().len(), 1);
    assert_eq!(failed.recorded_requests().len(), 1);
    assert!(last.recorded_requests().is_empty());
    let encoded = serde_json::to_string(&error).unwrap();
    assert_eq!(
        serde_json::from_str::<PipelineError>(&encoded).unwrap(),
        error
    );
}

#[tokio::test]
async fn unusable_intermediate_output_stops_before_next_agent() {
    let replies = [
        ChatResponse::completed([ContentBlock::from("   ")]),
        ChatResponse::completed([ThinkingBlock::new("private only").into()]),
        ChatResponse::completed([
            ContentBlock::from("visible"),
            ToolResultBlock::success("call", "tool", "result")
                .unwrap()
                .into(),
        ]),
    ];
    for reply in replies {
        let first = Arc::new(MockChatModel::new("first").with_response(reply));
        let last = model("must not run");
        let pipeline =
            SequentialPipeline::new(vec![agent("a", first), agent("b", last.clone())]).unwrap();
        let error = pipeline.run(Msg::user("start")).await.unwrap_err();
        assert_eq!(error.step, Some(1));
        assert_eq!(error.completed.len(), 1);
        assert!(matches!(error.cause, PipelineFailure::InvalidHandoff(_)));
        assert!(last.recorded_requests().is_empty());
    }
}

#[tokio::test]
async fn approval_is_not_bypassed_and_manual_stage_resume_does_not_resume_pipeline() {
    let first = model("draft");
    let reviewer_model = Arc::new(
        MockChatModel::new("reviewer")
            .with_response(ChatResponse::finished(
                [ToolCallBlock::complete("call", "write", "{}")
                    .unwrap()
                    .into()],
                FinishReason::ToolCalls,
            ))
            .with_response(ChatResponse::completed([ContentBlock::from("reviewed")])),
    );
    let tool = Arc::new(
        MockTool::new(ToolDefinition::new("write", "write", json!({"type":"object"})).unwrap())
            .with_output("saved"),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let reviewer = Arc::new(
        ReActAgent::from_shared("reviewer", reviewer_model, ToolExecutor::new(registry))
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_tool_confirmation_required("write"),
    );
    let last = model("not automatic");
    let pipeline = SequentialPipeline::new(vec![
        agent("writer", first.clone()),
        reviewer.clone(),
        agent("last", last.clone()),
    ])
    .unwrap();
    let error = pipeline.run(Msg::user("start")).await.unwrap_err();
    let PipelineFailure::Agent(cause) = error.cause else {
        panic!()
    };
    let AgentError::ToolConfirmationRequired { checkpoint } = *cause else {
        panic!()
    };
    assert_eq!(error.completed.len(), 1);
    assert!(tool.recorded_invocations().is_empty());
    let reply = reviewer
        .resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("call")],
        )
        .await
        .unwrap();
    assert_eq!(reply.text_content("").as_deref(), Some("reviewed"));
    assert_eq!(first.recorded_requests().len(), 1);
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert!(last.recorded_requests().is_empty());
}

#[tokio::test]
async fn existing_agent_state_stores_stay_separate() {
    let store = Arc::new(InMemoryStateStore::new());
    let a_key = StateKey::new("user", "writer").unwrap();
    let b_key = StateKey::new("user", "reviewer").unwrap();
    let a = Arc::new(
        ReActAgent::from_shared(
            "writer",
            model("draft"),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_shared_state_store(a_key.clone(), store.clone()),
    );
    let b = Arc::new(
        ReActAgent::from_shared(
            "reviewer",
            model("reviewed"),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_shared_state_store(b_key.clone(), store.clone()),
    );
    SequentialPipeline::new(vec![a, b])
        .unwrap()
        .run(Msg::user("original task"))
        .await
        .unwrap();
    let a = store.load(&a_key).await.unwrap().unwrap();
    let b = store.load(&b_key).await.unwrap().unwrap();
    assert_eq!(
        a.state().messages()[0].text_content("").as_deref(),
        Some("original task")
    );
    assert_eq!(
        b.state().messages()[0].text_content("").as_deref(),
        Some("draft")
    );
    assert_eq!(a.state().messages().len(), 2);
    assert_eq!(b.state().messages().len(), 2);
}

struct GateModel {
    started: Notify,
    release: Notify,
    dropped: AtomicUsize,
}
struct DropMarker<'a>(&'a AtomicUsize);
impl Drop for DropMarker<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl ChatModel for GateModel {
    fn name(&self) -> &'static str {
        "gate"
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::all()
    }
    fn generate(&self, _: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async move {
            let _marker = DropMarker(&self.dropped);
            self.started.notify_one();
            self.release.notified().await;
            Ok(ChatResponse::completed([ContentBlock::from(
                "gate finished",
            )]))
        })
    }
    fn stream(&self, _: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async { Err(ModelError::new("unused")) })
    }
}
fn gate() -> Arc<GateModel> {
    Arc::new(GateModel {
        started: Notify::new(),
        release: Notify::new(),
        dropped: AtomicUsize::new(0),
    })
}

#[tokio::test]
async fn active_run_and_clones_are_busy_interrupt_releases_lock() {
    let gate = gate();
    let last = model("done");
    let pipeline =
        SequentialPipeline::new(vec![agent("a", gate.clone()), agent("b", last.clone())]).unwrap();
    let cloned = pipeline.clone();
    let handle = pipeline.interrupt_handle();
    let (result, ()) = tokio::join!(pipeline.run(Msg::user("first")), async {
        gate.started.notified().await;
        assert!(last.recorded_requests().is_empty());
        let error = cloned.run(Msg::user("overlap")).await.unwrap_err();
        assert_eq!(error.cause, PipelineFailure::Busy);
        assert!(error.completed.is_empty());
        handle.interrupt();
    });
    let error = result.unwrap_err();
    assert_eq!(error.cause, PipelineFailure::Interrupted);
    assert_eq!(error.step, Some(1));
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
    assert!(last.recorded_requests().is_empty());
    gate.release.notify_one();
    let result = pipeline.run(Msg::user("explicit new run")).await.unwrap();
    assert_eq!(result.steps.len(), 2);
}

#[tokio::test]
async fn dropping_run_stops_dispatch_without_rolling_back_completed_memory() {
    let first = model("draft");
    let a = agent("a", first.clone());
    let gate = gate();
    let last = model("unused");
    let pipeline = SequentialPipeline::new(vec![
        a.clone(),
        agent("b", gate.clone()),
        agent("c", last.clone()),
    ])
    .unwrap();
    let mut run = pipeline.run(Msg::user("task"));
    assert!(run.as_mut().now_or_never().is_none());
    drop(run);
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(a.snapshot().await.unwrap().messages().len(), 2);
    assert_eq!(first.recorded_requests().len(), 1);
    assert!(last.recorded_requests().is_empty());
}

#[tokio::test]
async fn child_agent_interruption_is_preserved_with_completed_steps() {
    let gate = gate();
    let second = agent("b", gate.clone());
    let handle = second.interrupt_handle();
    let pipeline = SequentialPipeline::new(vec![agent("a", model("draft")), second]).unwrap();
    let (result, ()) = tokio::join!(pipeline.run(Msg::user("start")), async {
        gate.started.notified().await;
        handle.interrupt();
    });
    let error = result.unwrap_err();
    assert_eq!(error.completed.len(), 1);
    assert_eq!(error.step, Some(2));
    assert!(matches!(error.cause, PipelineFailure::Agent(e) if *e == AgentError::Interrupted));
}

struct InterruptAfterReply(Arc<std::sync::Mutex<Option<AgentInterruptHandle>>>);
impl AgentHook for InterruptAfterReply {
    fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a> {
        Box::pin(async move {
            if matches!(event, AgentHookEvent::AfterReply { .. }) {
                self.0.lock().unwrap().as_ref().unwrap().interrupt();
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn interruption_between_stages_does_not_dispatch_next_agent() {
    let control = Arc::new(std::sync::Mutex::new(None));
    let first = Arc::new(
        ReActAgent::from_shared("a", model("done"), ToolExecutor::new(ToolRegistry::new()))
            .unwrap()
            .with_memory(InMemoryMemory::new())
            .with_hook(InterruptAfterReply(control.clone())),
    );
    let next = model("must not run");
    let pipeline = SequentialPipeline::new(vec![first, agent("b", next.clone())]).unwrap();
    *control.lock().unwrap() = Some(pipeline.interrupt_handle());
    let error = pipeline.run(Msg::user("start")).await.unwrap_err();
    assert_eq!(error.step, Some(2));
    assert_eq!(error.completed.len(), 1);
    assert_eq!(error.cause, PipelineFailure::Interrupted);
    assert!(next.recorded_requests().is_empty());
}

#[tokio::test]
async fn uncertain_agent_execution_is_preserved_without_retry() {
    let remote = Arc::new(
        MockTool::new(ToolDefinition::new("write", "write", json!({"type":"object"})).unwrap())
            .with_error(ToolError::in_doubt("remote outcome unknown")),
    );
    let mut registry = ToolRegistry::new();
    registry.register_shared(remote.clone()).unwrap();
    let blocked = Arc::new(
        ReActAgent::new(
            "blocked",
            MockChatModel::new("model").with_response(ChatResponse::finished(
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
    let Err(AgentError::ToolConfirmationRequired { checkpoint }) =
        blocked.reply(Msg::user("prior task")).await
    else {
        panic!()
    };
    assert!(matches!(
        blocked
            .resume_tool_calls(
                checkpoint.reply_id(),
                vec![ToolConfirmation::approve("call")]
            )
            .await,
        Err(AgentError::ToolExecutionInDoubt { .. })
    ));
    let before = blocked.snapshot().await.unwrap();
    let last = model("unused");
    let pipeline = SequentialPipeline::new(vec![
        agent("first", model("draft")),
        blocked.clone(),
        agent("last", last.clone()),
    ])
    .unwrap();
    let error = pipeline.run(Msg::user("new task")).await.unwrap_err();
    assert_eq!(error.completed.len(), 1);
    assert!(
        matches!(error.cause, PipelineFailure::Agent(e) if matches!(*e, AgentError::ToolExecutionInDoubt {..}))
    );
    assert_eq!(before, blocked.snapshot().await.unwrap());
    assert_eq!(remote.recorded_invocations().len(), 1);
    assert!(last.recorded_requests().is_empty());
}
