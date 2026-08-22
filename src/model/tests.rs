//! Model response tests.

use serde_json::{Value, json};

use super::{
    ChatEvent, ChatResponse, ChatResponseAccumulator, ChatStreamError, FinishReason, ModelError,
};
use crate::message::{
    ContentBlock, Metadata, Role, StructuredOutputState, ThinkingBlock, ToolCallBlock,
    ToolCallState, Usage,
};

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

#[test]
fn chat_events_use_tagged_wire_format() {
    let event = ChatEvent::TextDelta {
        block_id: "text-1".into(),
        delta: "Hello".into(),
    };
    let value = serde_json::to_value(&event).unwrap();

    assert_eq!(value["type"], "text_delta");
    assert_eq!(value["block_id"], "text-1");
    assert_eq!(value["delta"], "Hello");
    assert_eq!(serde_json::from_value::<ChatEvent>(value).unwrap(), event);
}

#[test]
fn accumulator_merges_all_supported_delta_types() {
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"]
    });
    let mut accumulator = ChatResponseAccumulator::new();

    for event in [
        ChatEvent::ThinkingDelta {
            block_id: "thinking-1".into(),
            delta: "Inspect ".into(),
        },
        ChatEvent::ThinkingDelta {
            block_id: "thinking-1".into(),
            delta: "first.".into(),
        },
        ChatEvent::TextDelta {
            block_id: "text-1".into(),
            delta: "The ".into(),
        },
        ChatEvent::TextDelta {
            block_id: "text-1".into(),
            delta: "answer".into(),
        },
        ChatEvent::ToolCallDelta {
            tool_call_id: "call-1".into(),
            tool_name: "calculate".into(),
            delta: "{\"value\":".into(),
        },
        ChatEvent::ToolCallDelta {
            tool_call_id: "call-1".into(),
            tool_name: "calculate".into(),
            delta: "42}".into(),
        },
        ChatEvent::StructuredOutputDelta {
            block_id: "structured-1".into(),
            schema: schema.clone(),
            delta: "{\"answer\":".into(),
        },
        ChatEvent::StructuredOutputDelta {
            block_id: "structured-1".into(),
            schema,
            delta: "42}".into(),
        },
        ChatEvent::Usage {
            usage: Usage::new(80, 10),
        },
        ChatEvent::Usage {
            usage: Usage::new(20, 5),
        },
    ] {
        accumulator.apply(event).unwrap();
    }

    assert_eq!(accumulator.response().usage, None);
    accumulator
        .apply(ChatEvent::Finished {
            reason: FinishReason::ToolCalls,
        })
        .unwrap();
    let response = accumulator.into_response().unwrap();

    assert!(response.is_last);
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(response.usage, Some(Usage::new(100, 15)));
    let ContentBlock::Thinking(thinking) = &response.content[0] else {
        panic!("expected thinking block");
    };
    assert_eq!(thinking.thinking, "Inspect first.");
    assert!(thinking.finished_at.is_some());
    let ContentBlock::Text(text) = &response.content[1] else {
        panic!("expected text block");
    };
    assert_eq!(text.text, "The answer");
    assert!(text.finished_at.is_some());
    let ContentBlock::ToolCall(tool_call) = &response.content[2] else {
        panic!("expected tool call block");
    };
    assert_eq!(tool_call.input(), r#"{"value":42}"#);
    assert_eq!(tool_call.state, ToolCallState::Finished);
    let ContentBlock::StructuredOutput(structured) = &response.content[3] else {
        panic!("expected structured-output block");
    };
    assert_eq!(structured.parsed_output().unwrap(), json!({"answer": 42}));
    assert_eq!(structured.state(), StructuredOutputState::Complete);
}

#[test]
fn failed_finish_does_not_partially_complete_response() {
    let mut accumulator = ChatResponseAccumulator::new();
    accumulator
        .apply(ChatEvent::TextDelta {
            block_id: "text-1".into(),
            delta: "Calling".into(),
        })
        .unwrap();
    accumulator
        .apply(ChatEvent::ToolCallDelta {
            tool_call_id: "call-1".into(),
            tool_name: "search".into(),
            delta: "{".into(),
        })
        .unwrap();

    assert!(matches!(
        accumulator.apply(ChatEvent::Finished {
            reason: FinishReason::ToolCalls
        }),
        Err(ChatStreamError::ToolCall(_))
    ));
    assert!(!accumulator.response().is_last);
    let ContentBlock::Text(text) = &accumulator.response().content[0] else {
        panic!("expected text block");
    };
    assert!(text.finished_at.is_none());

    accumulator
        .apply(ChatEvent::ToolCallDelta {
            tool_call_id: "call-1".into(),
            tool_name: "search".into(),
            delta: "}".into(),
        })
        .unwrap();
    accumulator
        .apply(ChatEvent::Finished {
            reason: FinishReason::ToolCalls,
        })
        .unwrap();
    assert!(accumulator.into_response().unwrap().is_last);
}

#[test]
fn accumulator_rejects_conflicting_block_identity() {
    let mut accumulator = ChatResponseAccumulator::new();
    accumulator
        .apply(ChatEvent::TextDelta {
            block_id: "shared-1".into(),
            delta: "text".into(),
        })
        .unwrap();

    assert!(matches!(
        accumulator.apply(ChatEvent::ThinkingDelta {
            block_id: "shared-1".into(),
            delta: "thinking".into()
        }),
        Err(ChatStreamError::BlockTypeMismatch(block_id)) if block_id == "shared-1"
    ));

    accumulator
        .apply(ChatEvent::ToolCallDelta {
            tool_call_id: "call-1".into(),
            tool_name: "search".into(),
            delta: String::new(),
        })
        .unwrap();
    assert!(matches!(
        accumulator.apply(ChatEvent::ToolCallDelta {
            tool_call_id: "call-1".into(),
            tool_name: "weather".into(),
            delta: String::new()
        }),
        Err(ChatStreamError::ToolNameMismatch { .. })
    ));
}

#[test]
fn terminal_events_close_the_accumulator() {
    let mut completed = ChatResponseAccumulator::new();
    completed
        .apply(ChatEvent::Finished {
            reason: FinishReason::Completed,
        })
        .unwrap();
    assert!(completed.is_finished());
    assert!(matches!(
        completed.apply(ChatEvent::Usage {
            usage: Usage::new(1, 1)
        }),
        Err(ChatStreamError::AlreadyFinished)
    ));

    let model_error = ModelError::new("rate limited")
        .with_code("rate_limit")
        .with_retryable(true);
    let mut failed = ChatResponseAccumulator::new();
    failed
        .apply(ChatEvent::Error {
            error: model_error.clone(),
        })
        .unwrap();
    assert!(failed.is_finished());
    assert!(matches!(
        failed.into_response(),
        Err(ChatStreamError::Model(error)) if error == model_error
    ));
}

#[test]
fn unfinished_accumulator_has_no_final_response() {
    let accumulator = ChatResponseAccumulator::new();

    assert!(matches!(
        accumulator.into_response(),
        Err(ChatStreamError::NotFinished)
    ));
}

#[test]
fn structured_output_finish_is_transactional() {
    let schema = json!({"type": "object"});
    let mut accumulator = ChatResponseAccumulator::new();
    accumulator
        .apply(ChatEvent::StructuredOutputDelta {
            block_id: "structured-1".into(),
            schema: schema.clone(),
            delta: "{".into(),
        })
        .unwrap();

    assert!(matches!(
        accumulator.apply(ChatEvent::Finished {
            reason: FinishReason::Completed
        }),
        Err(ChatStreamError::StructuredOutput(_))
    ));
    assert!(!accumulator.response().is_last);
    let ContentBlock::StructuredOutput(block) = &accumulator.response().content[0] else {
        panic!("expected structured-output block");
    };
    assert_eq!(block.state(), StructuredOutputState::Streaming);

    accumulator
        .apply(ChatEvent::StructuredOutputDelta {
            block_id: "structured-1".into(),
            schema,
            delta: "}".into(),
        })
        .unwrap();
    accumulator
        .apply(ChatEvent::Finished {
            reason: FinishReason::Completed,
        })
        .unwrap();
    assert!(accumulator.into_response().unwrap().is_last);
}

#[test]
fn accumulator_rejects_invalid_block_identity_and_schema_changes() {
    let mut accumulator = ChatResponseAccumulator::new();
    assert!(matches!(
        accumulator.apply(ChatEvent::TextDelta {
            block_id: "  ".into(),
            delta: "ignored".into()
        }),
        Err(ChatStreamError::EmptyBlockId)
    ));

    accumulator
        .apply(ChatEvent::StructuredOutputDelta {
            block_id: "structured-1".into(),
            schema: json!({"type": "object"}),
            delta: String::new(),
        })
        .unwrap();
    assert!(matches!(
        accumulator.apply(ChatEvent::StructuredOutputDelta {
            block_id: "structured-1".into(),
            schema: json!({"type": "array"}),
            delta: String::new()
        }),
        Err(ChatStreamError::SchemaMismatch(block_id)) if block_id == "structured-1"
    ));
}

#[test]
fn model_error_event_round_trips() {
    let error = ModelError::new("temporarily unavailable")
        .with_code("unavailable")
        .with_retryable(true);
    let event = ChatEvent::Error {
        error: error.clone(),
    };
    let value = serde_json::to_value(&event).unwrap();

    assert_eq!(value["type"], "error");
    assert_eq!(value["error"]["code"], "unavailable");
    assert_eq!(value["error"]["retryable"], true);
    assert_eq!(
        error.to_string(),
        "model error unavailable: temporarily unavailable"
    );
    assert_eq!(serde_json::from_value::<ChatEvent>(value).unwrap(), event);
}
