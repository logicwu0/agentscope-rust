use std::sync::{Arc, Mutex};

use futures_executor::block_on;

use crate::{AgentHook, AgentHookError, AgentHookEvent, AgentHookFuture, ChatRequest, Msg};

#[derive(Default)]
struct RecordingHook {
    events: Mutex<Vec<AgentHookEvent>>,
}

impl AgentHook for RecordingHook {
    fn on_event<'a>(&'a self, event: &'a AgentHookEvent) -> AgentHookFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        })
    }
}

#[test]
fn hook_trait_is_object_safe_and_events_round_trip_through_json() {
    let hook: Arc<dyn AgentHook> = Arc::new(RecordingHook::default());
    let event = AgentHookEvent::BeforeModelCall {
        step: 2,
        request: ChatRequest::new([Msg::user("Hello")]),
    };

    block_on(hook.on_event(&event)).unwrap();
    let encoded = serde_json::to_value(&event).unwrap();
    let decoded: AgentHookEvent = serde_json::from_value(encoded.clone()).unwrap();

    assert_eq!(decoded, event);
    assert_eq!(encoded["type"], "before_model_call");
    assert_eq!(encoded["step"], 2);
}

#[test]
fn observation_hook_events_round_trip_through_json() {
    let events = [
        AgentHookEvent::BeforeObserve {
            message: Msg::assistant("planner", "Use exact arithmetic."),
        },
        AgentHookEvent::AfterObserve {
            message: Msg::assistant("planner", "Use exact arithmetic."),
        },
    ];

    let encoded = serde_json::to_value(&events).unwrap();
    let decoded: Vec<AgentHookEvent> = serde_json::from_value(encoded.clone()).unwrap();

    assert_eq!(decoded, events);
    assert_eq!(encoded[0]["type"], "before_observe");
    assert_eq!(encoded[1]["type"], "after_observe");
}

#[test]
fn hook_errors_are_structured_and_round_trip_through_json() {
    let error = AgentHookError::new("metrics backend unavailable").with_code("metrics_down");

    let encoded = serde_json::to_string(&error).unwrap();
    let decoded: AgentHookError = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, error);
    assert_eq!(
        error.to_string(),
        "agent hook error metrics_down: metrics backend unavailable"
    );
}
