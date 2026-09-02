use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use serde_json::json;

use crate::{
    AgentError, AgentEvent, AgentEventStream, AgentHookError, ChatEvent, ContentBlock,
    FinishReason, MemoryError, ModelError, Msg, ToolCallBlock, ToolError, ToolResultBlock, Usage,
};

#[test]
fn model_events_lift_into_their_agent_step() {
    let usage = Usage::new(3, 5);
    let cases = [
        (
            ChatEvent::TextDelta {
                block_id: "text-1".to_owned(),
                delta: "hello".to_owned(),
            },
            AgentEvent::TextDelta {
                step: 2,
                block_id: "text-1".to_owned(),
                delta: "hello".to_owned(),
            },
        ),
        (
            ChatEvent::ThinkingDelta {
                block_id: "thinking-1".to_owned(),
                delta: "reason".to_owned(),
            },
            AgentEvent::ThinkingDelta {
                step: 2,
                block_id: "thinking-1".to_owned(),
                delta: "reason".to_owned(),
            },
        ),
        (
            ChatEvent::ToolCallDelta {
                tool_call_id: "call-1".to_owned(),
                tool_name: "calculator".to_owned(),
                delta: "{}".to_owned(),
            },
            AgentEvent::ToolCallDelta {
                step: 2,
                tool_call_id: "call-1".to_owned(),
                tool_name: "calculator".to_owned(),
                delta: "{}".to_owned(),
            },
        ),
        (
            ChatEvent::StructuredOutputDelta {
                block_id: "output-1".to_owned(),
                schema: json!({"type": "object"}),
                delta: "{}".to_owned(),
            },
            AgentEvent::StructuredOutputDelta {
                step: 2,
                block_id: "output-1".to_owned(),
                schema: json!({"type": "object"}),
                delta: "{}".to_owned(),
            },
        ),
        (
            ChatEvent::Usage { usage },
            AgentEvent::Usage { step: 2, usage },
        ),
        (
            ChatEvent::Finished {
                reason: FinishReason::ToolCalls,
            },
            AgentEvent::StepFinished {
                step: 2,
                reason: FinishReason::ToolCalls,
            },
        ),
        (
            ChatEvent::Error {
                error: ModelError::new("provider failed").with_code("provider_error"),
            },
            AgentEvent::Error {
                step: Some(2),
                error: AgentError::Model(
                    ModelError::new("provider failed").with_code("provider_error"),
                ),
            },
        ),
    ];

    for (model_event, expected) in cases {
        assert_eq!(AgentEvent::from_chat_event(2, model_event), expected);
    }
}

#[test]
fn lifecycle_events_round_trip_through_json() {
    let call = ToolCallBlock::complete("call-1", "calculator", r#"{"value":42}"#).unwrap();
    let result = ToolResultBlock::success("call-1", "calculator", "42").unwrap();
    let events = vec![
        AgentEvent::ToolStarted {
            step: 1,
            call: call.clone(),
        },
        AgentEvent::ToolFinished { step: 1, result },
        AgentEvent::Finished {
            steps: 2,
            message: Msg::assistant("Friday", "42"),
        },
        AgentEvent::Error {
            step: None,
            error: AgentError::MaxStepsExceeded { max_steps: 8 },
        },
    ];

    let encoded = serde_json::to_value(&events).unwrap();
    let decoded: Vec<AgentEvent> = serde_json::from_value(encoded.clone()).unwrap();

    assert_eq!(decoded, events);
    assert_eq!(encoded[0]["type"], "tool_started");
    assert_eq!(encoded[0]["call"]["id"], call.id());
    assert_eq!(encoded[3]["error"]["kind"], "max_steps_exceeded");
    assert!(encoded[3].get("step").is_none());
}

#[test]
fn agent_event_stream_is_object_safe_and_sendable() {
    let event = AgentEvent::Finished {
        steps: 1,
        message: Msg::new(
            "Friday",
            crate::Role::Assistant,
            [ContentBlock::from("done")],
        ),
    };
    let mut events: AgentEventStream<'static> = Box::pin(stream::iter([Ok(event.clone())]));

    let received = block_on(events.next()).unwrap().unwrap();

    assert_eq!(received, event);
}

#[test]
fn every_agent_error_variant_round_trips_through_json() {
    let errors = vec![
        AgentError::EmptyName,
        AgentError::ZeroMaxSteps,
        AgentError::Model(ModelError::new("model failed").with_code("model_failure")),
        AgentError::Tool(ToolError::new("tool failed").with_code("tool_failure")),
        AgentError::Memory(MemoryError::new("memory failed").with_code("memory_failure")),
        AgentError::MemoryNotConfigured,
        AgentError::UnsupportedStateVersion {
            found: 2,
            supported: 1,
        },
        AgentError::StateAgentMismatch {
            expected: "Friday".to_owned(),
            found: "Saturday".to_owned(),
        },
        AgentError::Hook(AgentHookError::new("hook failed").with_code("hook_failure")),
        AgentError::Interrupted,
        AgentError::InvalidModelResponse("partial response".to_owned()),
        AgentError::MaxStepsExceeded { max_steps: 8 },
    ];

    for error in errors {
        let encoded = serde_json::to_string(&error).unwrap();
        let decoded: AgentError = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, error);
    }
}
