use super::*;
use crate::*;
use futures_util::{FutureExt, StreamExt};
use serde_json::json;
use std::sync::{Arc, atomic::Ordering};
#[path = "tests_support.rs"]
mod support;
use support::*;

#[tokio::test]
async fn bounded_streams_interleave_with_namespaced_events_and_ordered_outputs() {
    let activity = Arc::new(Activity::default());
    let models = ["a", "b", "c", "d"].map(|name| gate(name, &activity));
    let pipeline = ParallelPipeline::new(
        ["a", "b", "c", "d"]
            .into_iter()
            .zip(&models)
            .map(|(name, model)| agent(name, model.clone()) as Arc<dyn Agent>)
            .collect(),
        2,
    )
    .unwrap();
    let mut input = Msg::user("same input");
    input.metadata.insert("shared".into(), json!({"number": 7}));
    let mut events = pipeline.stream(input.clone()).await.unwrap();
    assert!(models.iter().all(|model| model.requests().is_empty()));
    let mut observed = Vec::new();
    assert!(!drain_ready(&mut events, &mut observed));
    assert_eq!(activity.active.load(Ordering::SeqCst), 2);
    assert_eq!(models[0].requests().len(), 1);
    assert_eq!(models[1].requests().len(), 1);
    assert!(models[2].requests().is_empty());
    assert!(models[3].requests().is_empty());

    models[1].release();
    assert!(!drain_ready(&mut events, &mut observed));
    assert_eq!(models[2].requests().len(), 1);
    assert!(models[3].requests().is_empty());
    models[2].release();
    assert!(!drain_ready(&mut events, &mut observed));
    assert_eq!(models[3].requests().len(), 1);
    models[3].release();
    assert!(!drain_ready(&mut events, &mut observed));
    models[0].release();
    observed.extend(events.by_ref().collect::<Vec<_>>().await);
    assert!(events.next().await.is_none());
    assert_eq!(activity.peak.load(Ordering::SeqCst), 2);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert_eq!(
        *activity.completed.lock().unwrap(),
        vec!["b", "c", "d", "a"]
    );
    assert_eq!(terminal_count(&observed), 1);
    let ParallelEvent::Finished { output } = observed.last().unwrap() else {
        panic!("expected successful final output")
    };
    assert_eq!(output.branches.len(), 4);
    for (index, model) in models.iter().enumerate() {
        let branch = index + 1;
        assert_eq!(model.requests()[0].messages.last(), Some(&input));
        assert_eq!(output.branches[index].branch, branch);
        assert_eq!(output.branches[index].agent_name, model.label);
        let branch_events: Vec<_> = observed
            .iter()
            .filter(|event| match event {
                ParallelEvent::BranchStarted { branch: id, .. }
                | ParallelEvent::Agent { branch: id, .. } => *id == branch,
                ParallelEvent::BranchFinished { result } => result.branch == branch,
                _ => false,
            })
            .collect();
        assert_eq!(branch_events.len(), 5);
        assert!(
            matches!(branch_events[0], ParallelEvent::BranchStarted { agent_name, .. } if agent_name == model.label)
        );
        assert!(matches!(branch_events[1], ParallelEvent::Agent {
            agent_name, event: AgentEvent::TextDelta { step: 1, block_id, delta }, ..
        } if agent_name == model.label && block_id == "shared-block" && delta == model.label));
        assert!(matches!(
            branch_events[2],
            ParallelEvent::Agent {
                event: AgentEvent::StepFinished {
                    step: 1,
                    reason: FinishReason::Completed
                },
                ..
            }
        ));
        assert!(matches!(
            branch_events[3],
            ParallelEvent::Agent {
                event: AgentEvent::Finished { steps: 1, .. },
                ..
            }
        ));
        assert!(matches!(
            branch_events[4],
            ParallelEvent::BranchFinished {
                result: ParallelBranchResult {
                    outcome: ParallelBranchOutcome::Completed(_),
                    ..
                }
            }
        ));
    }
    assert_roundtrips(&observed);
}

#[tokio::test]
async fn stream_reservation_is_lazy_and_shares_lock_with_runs_and_clones() {
    let activity = Arc::new(Activity::default());
    let gated = gate("one", &activity);
    let a = agent("one", gated.clone());
    let pipeline = ParallelPipeline::new(vec![a.clone()], 1).unwrap();
    let cloned = pipeline.clone();
    drop(pipeline.stream(Msg::user("unpolled preparation")));
    assert!(gated.requests().is_empty());
    let reserved = pipeline.stream(Msg::user("reserved")).await.unwrap();
    assert!(gated.requests().is_empty());
    assert!(a.snapshot().await.unwrap().messages().is_empty());
    let error = cloned.run(Msg::user("busy run")).await.unwrap_err();
    assert_eq!(error.cause, ParallelFailure::Busy);
    assert!(error.branches.is_empty());
    let Err(error) = cloned.stream(Msg::user("busy stream")).await else {
        panic!("a reserved stream must block clone streams")
    };
    assert_eq!(error.cause, ParallelFailure::Busy);
    assert!(error.branches.is_empty());
    drop(reserved);
    let mut run = cloned.run(Msg::user("active run"));
    assert!(run.as_mut().now_or_never().is_none());
    assert_eq!(gated.requests().len(), 1);
    let Err(error) = pipeline.stream(Msg::user("busy stream")).await else {
        panic!("an active run must block stream reservation")
    };
    assert_eq!(error.cause, ParallelFailure::Busy);
    drop(run);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    let fresh = pipeline
        .stream(Msg::user("fresh reservation"))
        .await
        .unwrap();
    drop(fresh);
}

#[tokio::test]
async fn interrupt_before_poll_and_after_start_announcement_leaves_work_unstarted() {
    for announce_first in [false, true] {
        let first = stream_model("first");
        let second = stream_model("second");
        let a = agent("first", first.clone());
        let b = agent("second", second.clone());
        let pipeline = ParallelPipeline::new(vec![a.clone(), b.clone()], 2).unwrap();
        let mut events = pipeline.stream(Msg::user("task")).await.unwrap();
        if announce_first {
            assert!(matches!(
                events.next().await,
                Some(ParallelEvent::BranchStarted { branch: 1, .. })
            ));
        }
        pipeline.interrupt_handle().interrupt();
        let observed = events.by_ref().collect::<Vec<_>>().await;
        assert_eq!(observed.len(), 1);
        let ParallelEvent::Error { error } = &observed[0] else {
            panic!("expected a terminal interruption")
        };
        assert_eq!(error.cause, ParallelFailure::Interrupted);
        assert_eq!(error.branches.len(), 2);
        assert!(
            error
                .branches
                .iter()
                .all(|branch| branch.outcome == ParallelBranchOutcome::NotStarted)
        );
        assert!(first.recorded_requests().is_empty());
        assert!(second.recorded_requests().is_empty());
        assert!(a.snapshot().await.unwrap().messages().is_empty());
        assert!(b.snapshot().await.unwrap().messages().is_empty());
        assert!(events.next().await.is_none());
        let fresh = pipeline
            .stream(Msg::user("new baseline"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert!(matches!(fresh.last(), Some(ParallelEvent::Finished { .. })));
    }
}

#[tokio::test]
async fn interrupt_after_agent_finished_preserves_completed_reply_and_stops_dispatch() {
    let activity = Arc::new(Activity::default());
    let a_model = gate("a", &activity);
    let b_model = gate("b", &activity);
    let queued = stream_model("must not run");
    let pipeline = ParallelPipeline::new(
        vec![
            agent("a", a_model.clone()),
            agent("b", b_model.clone()),
            agent("queued", queued.clone()),
        ],
        2,
    )
    .unwrap();
    let mut events = pipeline.stream(Msg::user("task")).await.unwrap();
    let mut observed = Vec::new();
    assert!(!drain_ready(&mut events, &mut observed));
    assert_eq!(activity.active.load(Ordering::SeqCst), 2);
    a_model.release();
    let completed = loop {
        let event = events.next().await.unwrap();
        let result = match &event {
            ParallelEvent::Agent {
                branch: 1,
                event: AgentEvent::Finished { message, .. },
                ..
            } => Some(message.clone()),
            _ => None,
        };
        observed.push(event);
        if let Some(message) = result {
            break message;
        }
    };
    pipeline.interrupt_handle().interrupt();
    observed.extend(events.by_ref().collect::<Vec<_>>().await);
    let ParallelEvent::Error { error } = observed.last().unwrap() else {
        panic!("expected interruption")
    };
    assert_eq!(error.cause, ParallelFailure::Interrupted);
    assert_eq!(
        error.branches[0].outcome,
        ParallelBranchOutcome::Completed(completed)
    );
    assert_eq!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Interrupted
    );
    assert_eq!(error.branches[2].outcome, ParallelBranchOutcome::NotStarted);
    assert_eq!(terminal_count(&observed), 1);
    assert!(!observed.iter().any(
        |event| matches!(event, ParallelEvent::BranchFinished { result } if result.branch == 2)
    ));
    assert!(queued.recorded_requests().is_empty());
    assert_eq!(b_model.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert!(events.next().await.is_none());
}

#[tokio::test]
async fn dropping_active_stream_cancels_pending_calls_and_releases_lock_without_queued_work() {
    let activity = Arc::new(Activity::default());
    let first = agent("first", stream_model("completed"));
    let gated = gate("gated", &activity);
    let last = stream_model("must not run");
    let pipeline = ParallelPipeline::new(
        vec![
            first.clone(),
            agent("gated", gated.clone()),
            agent("last", last.clone()),
        ],
        1,
    )
    .unwrap();
    let mut events = pipeline.stream(Msg::user("task")).await.unwrap();
    let mut observed = Vec::new();
    assert!(!drain_ready(&mut events, &mut observed));
    assert!(observed.iter().any(
        |event| matches!(event, ParallelEvent::BranchFinished { result } if result.branch == 1)
    ));
    assert_eq!(first.snapshot().await.unwrap().messages().len(), 2);
    assert_eq!(activity.active.load(Ordering::SeqCst), 1);
    assert!(last.recorded_requests().is_empty());
    drop(events);
    assert_eq!(activity.active.load(Ordering::SeqCst), 0);
    assert_eq!(gated.dropped.load(Ordering::SeqCst), 1);
    assert!(last.recorded_requests().is_empty());
    assert_eq!(terminal_count(&observed), 0);
    assert_eq!(first.snapshot().await.unwrap().messages().len(), 2);
    let reserved = pipeline.clone();
    drop(reserved.stream(Msg::user("new reservation")).await.unwrap());
}

#[tokio::test]
async fn confirmation_and_successful_sibling_are_forwarded_and_durably_finalized() {
    let store = Arc::new(InMemoryStateStore::new());
    let blocked_key = StateKey::new("test", "blocked").unwrap();
    let sibling_key = StateKey::new("test", "sibling").unwrap();
    let (blocked, tool) = confirming_agent(store.clone(), blocked_key.clone());
    let sibling = Arc::new(
        ReActAgent::from_shared(
            "sibling",
            stream_model("sibling completed"),
            ToolExecutor::new(ToolRegistry::new()),
        )
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_shared_state_store(sibling_key.clone(), store.clone()),
    );
    let pipeline = ParallelPipeline::new(vec![blocked, sibling], 2).unwrap();
    let observed = pipeline
        .stream(Msg::user("task"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(terminal_count(&observed), 1);
    let confirmation_index = observed
        .iter()
        .position(|event| {
            matches!(
                event,
                ParallelEvent::Agent {
                    branch: 1,
                    event: AgentEvent::ToolConfirmationRequired { .. },
                    ..
                }
            )
        })
        .unwrap();
    let ParallelEvent::Agent {
        event: AgentEvent::ToolConfirmationRequired { checkpoint },
        ..
    } = &observed[confirmation_index]
    else {
        panic!()
    };
    let branch_index = observed
        .iter()
        .position(
            |event| matches!(event, ParallelEvent::BranchFinished { result } if result.branch == 1),
        )
        .unwrap();
    assert!(confirmation_index < branch_index);
    let ParallelEvent::Error { error } = observed.last().unwrap() else {
        panic!("confirmation must fail the pipeline while retaining siblings")
    };
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    assert!(
        matches!(&error.branches[0].outcome, ParallelBranchOutcome::Failed(cause) if matches!(cause.as_ref(), AgentError::ToolConfirmationRequired { checkpoint: stored } if stored == checkpoint))
    );
    assert!(matches!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Completed(_)
    ));
    assert!(tool.recorded_invocations().is_empty());
    let blocked_record = store.load(&blocked_key).await.unwrap().unwrap();
    assert_eq!(
        blocked_record.state().pending_tool_calls(),
        Some(checkpoint)
    );
    assert_eq!(blocked_record.state().messages().len(), 2);
    let sibling_record = store.load(&sibling_key).await.unwrap().unwrap();
    assert_eq!(sibling_record.state().messages().len(), 2);
    assert_eq!(
        sibling_record.state().messages()[1]
            .text_content("")
            .as_deref(),
        Some("sibling completed")
    );
    assert_roundtrips(&observed);
    drop(pipeline.stream(Msg::user("lock released")).await.unwrap());
}

#[tokio::test]
async fn startup_raw_and_missing_terminal_failures_do_not_skip_queued_siblings() {
    let startup_error = AgentError::Model(ModelError::new("startup failure"));
    let startup = ScriptAgent::startup("startup", startup_error.clone());
    let raw_error = AgentError::Model(ModelError::new("raw failure"));
    let raw = ScriptAgent::new("raw", vec![Err(raw_error.clone())]);
    let delta = AgentEvent::TextDelta {
        step: 17,
        block_id: "text".into(),
        delta: "partial".into(),
    };
    let missing = ScriptAgent::new("missing", vec![Ok(delta.clone())]);
    let event_error = AgentError::Model(ModelError::new("terminal event failure"));
    let error_event = AgentEvent::Error {
        step: Some(23),
        error: event_error.clone(),
    };
    let terminal = ScriptAgent::new("terminal", vec![Ok(error_event.clone())]);
    let reply = Msg::new(
        "success",
        Role::Assistant,
        [ContentBlock::from("completed")],
    );
    let final_event = AgentEvent::Finished {
        steps: 42,
        message: reply.clone(),
    };
    let success = ScriptAgent::new("success", vec![Ok(final_event.clone())]);
    let pipeline = ParallelPipeline::new(
        vec![
            startup.clone(),
            raw.clone(),
            missing.clone(),
            terminal.clone(),
            success.clone(),
        ],
        1,
    )
    .unwrap();
    let input = Msg::user("task");
    let observed = pipeline
        .stream(input.clone())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(terminal_count(&observed), 1);
    let ParallelEvent::Error { error } = observed.last().unwrap() else {
        panic!("expected collected failures")
    };
    assert_eq!(error.cause, ParallelFailure::AgentFailures);
    assert_eq!(error.branches.len(), 5);
    assert_eq!(
        error.branches[0].outcome,
        ParallelBranchOutcome::Failed(Box::new(startup_error))
    );
    assert_eq!(
        error.branches[1].outcome,
        ParallelBranchOutcome::Failed(Box::new(raw_error))
    );
    assert!(
        matches!(&error.branches[2].outcome, ParallelBranchOutcome::Failed(cause) if matches!(cause.as_ref(), AgentError::InvalidModelResponse(reason) if reason.contains("terminal")))
    );
    assert_eq!(
        error.branches[3].outcome,
        ParallelBranchOutcome::Failed(Box::new(event_error))
    );
    assert_eq!(
        error.branches[4].outcome,
        ParallelBranchOutcome::Completed(reply)
    );
    assert_eq!(
        observed
            .iter()
            .filter(|event| matches!(event, ParallelEvent::BranchStarted { .. }))
            .count(),
        5
    );
    assert_eq!(
        observed
            .iter()
            .filter(|event| matches!(event, ParallelEvent::BranchFinished { .. }))
            .count(),
        5
    );
    assert_eq!(
        wrapped_agent_events(&observed),
        vec![
            (3, "missing", delta),
            (4, "terminal", error_event),
            (5, "success", final_event)
        ]
    );
    for agent in [startup, raw, missing, terminal, success] {
        assert_eq!(*agent.inputs.lock().unwrap(), vec![input.clone()]);
    }
    assert_roundtrips(&observed);
}

#[tokio::test]
async fn sibling_interruption_prevents_resumed_startup_or_stream_polling() {
    for startup in [false, true] {
        let (agents, control) = race_agents(startup);
        let pipeline = ParallelPipeline::new(agents, 2).unwrap();
        *control.handle.lock().unwrap() = Some(pipeline.interrupt_handle());
        let observed = pipeline
            .stream(Msg::user("task"))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let ParallelEvent::Error { error } = observed.last().unwrap() else {
            panic!("the synchronous signal must interrupt both pending branches")
        };
        assert_eq!(error.cause, ParallelFailure::Interrupted);
        assert!(
            error
                .branches
                .iter()
                .all(|branch| branch.outcome == ParallelBranchOutcome::Interrupted)
        );
        assert_eq!(
            control.target_polls.load(Ordering::SeqCst),
            1,
            "the pending operation must not be polled again after the sibling interrupts"
        );
        assert_eq!(control.interrupting_polls.load(Ordering::SeqCst), 1);
        assert_eq!(terminal_count(&observed), 1);
    }
}
