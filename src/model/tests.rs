//! Model response tests.

use serde_json::{Value, json};

use super::{ChatResponse, FinishReason};
use crate::message::{ContentBlock, Metadata, Role, ThinkingBlock, ToolCallBlock, Usage};

#[test]
fn finish_reasons_use_provider_neutral_wire_values() {
    for (reason, expected) in [
        (FinishReason::Completed, "completed"),
        (FinishReason::Length, "length"),
        (FinishReason::ToolCalls, "tool_calls"),
        (FinishReason::ContentFilter, "content_filter"),
        (FinishReason::Interrupted, "interrupted"),
    ] {
        assert_eq!(serde_json::to_value(reason).unwrap(), expected);
        assert_eq!(reason.to_string(), expected);
    }
}

#[test]
fn partial_response_has_no_finish_reason() {
    let response = ChatResponse::partial([ContentBlock::from("Hel")]);
    let value = serde_json::to_value(response).unwrap();

    assert_eq!(value["type"], "chat_response");
    assert_eq!(value["is_last"], false);
    assert_eq!(value["content"][0]["text"], "Hel");
    assert!(value.get("finished_reason").is_none());
    assert!(value.get("usage").is_none());
}

#[test]
fn completed_response_round_trips_with_usage_and_metadata() {
    let mut metadata = Metadata::new();
    metadata.insert("provider_request_id".into(), json!("req-42"));
    let original = ChatResponse::completed([ContentBlock::from("Done")])
        .with_usage(Usage::new(90, 12))
        .with_metadata(metadata);

    let value = serde_json::to_value(&original).unwrap();
    assert_eq!(value["finished_reason"], "completed");
    assert_eq!(value["usage"]["input_tokens"], 90);
    assert_eq!(value["metadata"]["provider_request_id"], "req-42");
    assert_eq!(
        serde_json::from_value::<ChatResponse>(value).unwrap(),
        original
    );
}

#[test]
fn response_text_excludes_non_text_blocks() {
    let response = ChatResponse::completed([
        ContentBlock::from(ThinkingBlock::new("internal")),
        ContentBlock::from("First"),
        ContentBlock::from("Second"),
    ]);

    assert_eq!(
        response.text_content("\n").as_deref(),
        Some("First\nSecond")
    );
}

#[test]
fn response_exposes_tool_calls_in_content_order() {
    let first = ToolCallBlock::complete("call-1", "search", "{}").unwrap();
    let second = ToolCallBlock::complete("call-2", "weather", "{}").unwrap();
    let response = ChatResponse::finished(
        [
            ContentBlock::from(first),
            ContentBlock::from("Calling tools"),
            ContentBlock::from(second),
        ],
        FinishReason::ToolCalls,
    );

    assert_eq!(
        response
            .tool_calls()
            .map(ToolCallBlock::name)
            .collect::<Vec<_>>(),
        ["search", "weather"]
    );
}

#[test]
fn response_converts_to_assistant_message() {
    let response =
        ChatResponse::completed([ContentBlock::from("Done")]).with_usage(Usage::new(40, 5));
    let message = response.into_assistant_msg("Friday");

    assert_eq!(message.name, "Friday");
    assert_eq!(message.role, Role::Assistant);
    assert_eq!(message.text_content("\n").as_deref(), Some("Done"));
    assert_eq!(message.usage, Some(Usage::new(40, 5)));
}

#[test]
fn deserialization_accepts_omitted_response_metadata() {
    let response: ChatResponse = serde_json::from_value(json!({
        "content": [],
        "is_last": false
    }))
    .unwrap();

    assert_eq!(response.content, Vec::<ContentBlock>::new());
    assert_eq!(response.finish_reason, None);
    assert!(response.metadata.is_empty());
    assert_eq!(
        serde_json::to_value(response).unwrap()["type"],
        Value::String("chat_response".into())
    );
}
