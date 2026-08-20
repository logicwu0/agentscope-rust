//! Message module tests.

use serde_json::{Value, json};

use super::{
    ContentBlock, DataBlock, DataBlockError, DataSource, Metadata, Msg, PermissionBehavior,
    PermissionRule, Role, StructuredOutputBlock, StructuredOutputError, StructuredOutputState,
    ThinkingBlock, ThinkingBlockError, ToolCallBlock, ToolCallError, ToolCallState,
    ToolResultBlock, ToolResultContent, ToolResultError, ToolResultOutput, ToolResultState,
};

#[test]
fn roles_use_agentscope_wire_values() {
    assert_eq!(serde_json::to_value(Role::System).unwrap(), "system");
    assert_eq!(serde_json::to_value(Role::User).unwrap(), "user");
    assert_eq!(serde_json::to_value(Role::Assistant).unwrap(), "assistant");
}

#[test]
fn user_message_has_generated_identity_and_text_block() {
    let message = Msg::user("hello");

    assert_eq!(message.name, "user");
    assert_eq!(message.role, Role::User);
    assert_eq!(message.id.len(), 32);
    assert!(!message.created_at.is_empty());
    assert_eq!(message.text_content("\n").as_deref(), Some("hello"));

    let value = serde_json::to_value(message).unwrap();
    assert_eq!(value["content"][0]["type"], "text");
    assert_eq!(value["content"][0]["text"], "hello");
}

#[test]
fn json_round_trip_preserves_message_and_metadata() {
    let mut metadata = Metadata::new();
    metadata.insert("request_id".into(), json!("req-42"));
    metadata.insert("attempt".into(), json!(2));
    let original = Msg::assistant("Friday", "Done").with_metadata(metadata);

    let json = serde_json::to_string(&original).unwrap();
    let restored: Msg = serde_json::from_str(&json).unwrap();

    assert_eq!(restored, original);
    assert_eq!(restored.metadata["request_id"], "req-42");
    assert_eq!(restored.metadata["attempt"], 2);
}

#[test]
fn deserialization_accepts_missing_metadata() {
    let message: Msg = serde_json::from_value(json!({
        "name": "system",
        "content": [{
            "type": "text",
            "text": "Follow the policy.",
            "id": "block-1",
            "created_at": "2026-08-12T12:00:00.000000",
            "finished_at": "2026-08-12T12:00:00.000000"
        }],
        "role": "system",
        "id": "message-1",
        "created_at": "2026-08-12T12:00:00.000000"
    }))
    .unwrap();

    assert!(message.metadata.is_empty());
    assert!(matches!(message.content[0], ContentBlock::Text(_)));
}

#[test]
fn empty_message_has_no_text_content() {
    let message = Msg::new("Friday", Role::Assistant, []);

    assert_eq!(message.text_content("\n"), None);
    assert_eq!(
        serde_json::to_value(message).unwrap()["content"],
        Value::Array(vec![])
    );
}

#[test]
fn url_data_block_uses_agentscope_wire_format() {
    let block = DataBlock::url("https://example.com/image.png", "image/png")
        .unwrap()
        .with_name("diagram.png");
    let DataSource::Url(source) = &block.source else {
        panic!("expected URL source");
    };
    assert_eq!(source.url().as_str(), "https://example.com/image.png");
    assert_eq!(source.media_type(), "image/png");
    let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

    assert_eq!(value["type"], "data");
    assert_eq!(value["source"]["type"], "url");
    assert_eq!(value["source"]["url"], "https://example.com/image.png");
    assert_eq!(value["source"]["media_type"], "image/png");
    assert_eq!(value["name"], "diagram.png");
}

#[test]
fn base64_data_block_uses_agentscope_wire_format() {
    let block = DataBlock::base64("aGVsbG8=", "text/plain").unwrap();
    let DataSource::Base64(source) = &block.source else {
        panic!("expected Base64 source");
    };
    assert_eq!(source.data(), "aGVsbG8=");
    assert_eq!(source.media_type(), "text/plain");
    let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

    assert_eq!(value["type"], "data");
    assert_eq!(value["source"]["type"], "base64");
    assert_eq!(value["source"]["data"], "aGVsbG8=");
    assert_eq!(value["source"]["media_type"], "text/plain");
}

#[test]
fn mixed_message_round_trip_preserves_text_and_data() {
    let original = Msg::new(
        "user",
        Role::User,
        [
            ContentBlock::from("Describe this image"),
            ContentBlock::from(
                DataBlock::url("https://example.com/cat.jpg", "image/jpeg").unwrap(),
            ),
        ],
    );

    let json = serde_json::to_string(&original).unwrap();
    let restored: Msg = serde_json::from_str(&json).unwrap();

    assert_eq!(restored, original);
    assert_eq!(
        restored.text_content("\n").as_deref(),
        Some("Describe this image")
    );
    assert!(matches!(restored.content[1], ContentBlock::Data(_)));
}

#[test]
fn constructors_reject_invalid_data_sources() {
    assert!(matches!(
        DataBlock::url("relative/image.png", "image/png"),
        Err(DataBlockError::InvalidUrl(_))
    ));
    assert!(matches!(
        DataBlock::base64("not base64!", "image/png"),
        Err(DataBlockError::InvalidBase64(_))
    ));
    assert_eq!(
        DataBlock::base64("", "  ").unwrap_err(),
        DataBlockError::EmptyMediaType
    );
}

#[test]
fn deserialization_rejects_invalid_data_sources() {
    let invalid_url = json!({
        "type": "data",
        "id": "block-1",
        "source": {
            "type": "url",
            "url": "relative/image.png",
            "media_type": "image/png"
        },
        "name": null,
        "created_at": "2026-08-12T12:00:00.000000",
        "finished_at": null
    });
    let invalid_base64 = json!({
        "type": "data",
        "id": "block-2",
        "source": {
            "type": "base64",
            "data": "not base64!",
            "media_type": "image/png"
        },
        "name": null,
        "created_at": "2026-08-12T12:00:00.000000",
        "finished_at": null
    });

    assert!(serde_json::from_value::<ContentBlock>(invalid_url).is_err());
    assert!(serde_json::from_value::<ContentBlock>(invalid_base64).is_err());
}

#[test]
fn thinking_block_uses_agentscope_wire_format() {
    let block = ThinkingBlock::new("I should inspect the request first.");
    let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

    assert_eq!(value["type"], "thinking");
    assert_eq!(value["thinking"], "I should inspect the request first.");
    assert_eq!(value["id"].as_str().unwrap().len(), 32);
    assert!(value["created_at"].is_string());
    assert!(value["finished_at"].is_null());
}

#[test]
fn thinking_provider_fields_survive_json_round_trip() {
    let block = ThinkingBlock::new("")
        .with_extra_field("signature", "sig-123")
        .unwrap()
        .with_extra_field("redacted_thinking_data", "encrypted-payload")
        .unwrap();
    let original = Msg::new("Friday", Role::Assistant, [ContentBlock::from(block)]);

    let json = serde_json::to_string(&original).unwrap();
    let restored: Msg = serde_json::from_str(&json).unwrap();
    let ContentBlock::Thinking(restored_block) = &restored.content[0] else {
        panic!("expected thinking block");
    };

    assert_eq!(restored, original);
    assert_eq!(restored_block.extra_fields()["signature"], "sig-123");
    assert_eq!(
        restored_block.extra_fields()["redacted_thinking_data"],
        "encrypted-payload"
    );
}

#[test]
fn text_content_excludes_thinking_blocks() {
    let message = Msg::new(
        "Friday",
        Role::Assistant,
        [
            ContentBlock::from(ThinkingBlock::new("Internal reasoning")),
            ContentBlock::from("Final answer"),
        ],
    );

    assert_eq!(message.text_content("\n").as_deref(), Some("Final answer"));
}

#[test]
fn thinking_extensions_cannot_replace_standard_fields() {
    for field in ["type", "thinking", "id", "created_at", "finished_at"] {
        assert_eq!(
            ThinkingBlock::new("reasoning")
                .with_extra_field(field, "replacement")
                .unwrap_err(),
            ThinkingBlockError::ReservedField(field.to_owned())
        );
    }
}

#[test]
fn tool_call_states_use_agentscope_wire_values() {
    for (state, expected) in [
        (ToolCallState::Pending, "pending"),
        (ToolCallState::Asking, "asking"),
        (ToolCallState::Allowed, "allowed"),
        (ToolCallState::Submitted, "submitted"),
        (ToolCallState::Finished, "finished"),
    ] {
        assert_eq!(serde_json::to_value(state).unwrap(), expected);
        assert_eq!(state.to_string(), expected);
    }
}

#[test]
fn complete_tool_call_uses_agentscope_wire_format() {
    let rule = PermissionRule {
        tool_name: "get_weather".into(),
        rule_content: Some("Hangzhou".into()),
        behavior: PermissionBehavior::Ask,
        source: "model".into(),
    };
    let block = ToolCallBlock::complete("call-123", "get_weather", r#"{"city":"Hangzhou"}"#)
        .unwrap()
        .with_suggested_rules(vec![rule]);
    assert_eq!(block.id(), "call-123");
    assert_eq!(block.name(), "get_weather");
    let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

    assert_eq!(value["type"], "tool_call");
    assert_eq!(value["id"], "call-123");
    assert_eq!(value["name"], "get_weather");
    assert_eq!(value["input"], r#"{"city":"Hangzhou"}"#);
    assert_eq!(value["state"], "pending");
    assert_eq!(value["suggested_rules"][0]["behavior"], "ask");
}

#[test]
fn streaming_tool_call_accepts_partial_input() {
    let mut block = ToolCallBlock::streaming("call-1", "get_weather").unwrap();
    block.append_input("{\"city\":\"");
    assert!(matches!(
        block.parsed_input(),
        Err(ToolCallError::InvalidInput(_))
    ));

    block.append_input("Hangzhou\"}");
    assert_eq!(block.parsed_input().unwrap(), json!({"city": "Hangzhou"}));
}

#[test]
fn completed_tool_calls_require_valid_identity_and_json() {
    assert!(matches!(
        ToolCallBlock::complete("", "get_weather", "{}"),
        Err(ToolCallError::EmptyId)
    ));
    assert!(matches!(
        ToolCallBlock::complete("call-1", "  ", "{}"),
        Err(ToolCallError::EmptyName)
    ));
    assert!(matches!(
        ToolCallBlock::complete("call-1", "get_weather", "{"),
        Err(ToolCallError::InvalidInput(_))
    ));
}

#[test]
fn tool_call_json_round_trip_preserves_streaming_input_and_rules() {
    let json = json!({
        "type": "tool_call",
        "id": "call-7",
        "name": "search",
        "input": "{\"query\":",
        "state": "asking",
        "suggested_rules": [{
            "tool_name": "search",
            "rule_content": null,
            "behavior": "allow",
            "source": "userSettings"
        }],
        "created_at": "2026-08-19T12:00:00.000000",
        "finished_at": null
    });

    let block: ContentBlock = serde_json::from_value(json.clone()).unwrap();
    let ContentBlock::ToolCall(tool_call) = &block else {
        panic!("expected tool call block");
    };
    assert_eq!(tool_call.name(), "search");
    assert_eq!(tool_call.input(), "{\"query\":");
    assert_eq!(tool_call.state, ToolCallState::Asking);
    assert_eq!(serde_json::to_value(block).unwrap(), json);
}

#[test]
fn deserialization_rejects_blank_tool_identity() {
    for (id, name) in [("", "search"), ("call-1", " ")] {
        let value = json!({
            "type": "tool_call",
            "id": id,
            "name": name,
            "input": "",
            "state": "pending",
            "suggested_rules": [],
            "created_at": "2026-08-19T12:00:00.000000",
            "finished_at": null
        });
        assert!(serde_json::from_value::<ContentBlock>(value).is_err());
    }
}

#[test]
fn text_content_excludes_tool_calls() {
    let message = Msg::new(
        "Friday",
        Role::Assistant,
        [
            ContentBlock::from(ToolCallBlock::complete("call-1", "search", "{}").unwrap()),
            ContentBlock::from("I will search for that."),
        ],
    );

    assert_eq!(
        message.text_content("\n").as_deref(),
        Some("I will search for that.")
    );
}

#[test]
fn tool_result_states_use_agentscope_wire_values() {
    for (state, expected) in [
        (ToolResultState::Running, "running"),
        (ToolResultState::Success, "success"),
        (ToolResultState::Error, "error"),
        (ToolResultState::Interrupted, "interrupted"),
        (ToolResultState::Denied, "denied"),
    ] {
        assert_eq!(serde_json::to_value(state).unwrap(), expected);
        assert_eq!(state.to_string(), expected);
    }
}

#[test]
fn running_tool_result_uses_agentscope_wire_format() {
    let block = ToolResultBlock::running("call-1", "get_weather").unwrap();
    assert_eq!(block.id(), "call-1");
    assert_eq!(block.name(), "get_weather");
    assert_eq!(block.state(), ToolResultState::Running);
    let value = serde_json::to_value(ContentBlock::from(block)).unwrap();

    assert_eq!(value["type"], "tool_result");
    assert_eq!(value["output"], json!([]));
    assert_eq!(value["state"], "running");
    assert_eq!(value["metadata"], json!({}));
    assert!(value["finished_at"].is_null());
}

#[test]
fn successful_raw_tool_result_round_trips() {
    let original = ToolResultBlock::success("call-2", "get_weather", "Sunny").unwrap();
    assert_eq!(original.state(), ToolResultState::Success);
    assert!(original.finished_at.is_some());
    assert_eq!(original.output(), &ToolResultOutput::Text("Sunny".into()));

    let json = serde_json::to_string(&ContentBlock::from(original.clone())).unwrap();
    let restored: ContentBlock = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, ContentBlock::from(original));
}

#[test]
fn streaming_tool_result_merges_text_and_preserves_multimodal_output() {
    let mut block = ToolResultBlock::running("call-3", "inspect_image").unwrap();
    block.append_text_delta("Found ");
    block.append_text_delta("a cat.");
    block.append_data(DataBlock::url("https://example.com/cat.jpg", "image/jpeg").unwrap());
    block.append_text_delta(" High confidence.");
    block.finish(ToolResultState::Success).unwrap();

    let value = serde_json::to_value(ContentBlock::from(block.clone())).unwrap();
    assert_eq!(value["output"][0]["type"], "text");
    assert_eq!(value["output"][0]["text"], "Found a cat.");
    assert_eq!(value["output"][1]["type"], "data");
    assert_eq!(value["output"][2]["text"], " High confidence.");

    let restored: ContentBlock = serde_json::from_value(value).unwrap();
    assert_eq!(restored, ContentBlock::from(block));
}

#[test]
fn appending_to_raw_output_converts_it_to_structured_blocks() {
    let json = json!({
        "type": "tool_result",
        "id": "call-4",
        "name": "search",
        "output": "First",
        "state": "running",
        "metadata": {},
        "created_at": "2026-08-19T12:00:00.000000",
        "finished_at": null
    });
    let ContentBlock::ToolResult(mut block) = serde_json::from_value::<ContentBlock>(json).unwrap()
    else {
        panic!("expected tool result block");
    };

    block.append_text_delta(" second");
    let ToolResultOutput::Blocks(blocks) = block.output() else {
        panic!("expected structured output");
    };
    let ToolResultContent::Text(text) = &blocks[0] else {
        panic!("expected text output");
    };
    assert_eq!(text.text, "First second");
}

#[test]
fn tool_result_finish_requires_terminal_state() {
    let mut block = ToolResultBlock::running("call-5", "search").unwrap();
    assert_eq!(
        block.finish(ToolResultState::Running).unwrap_err(),
        ToolResultError::NonTerminalState
    );
    assert_eq!(block.state(), ToolResultState::Running);
    assert!(block.finished_at.is_none());
}

#[test]
fn tool_result_metadata_and_terminal_state_round_trip() {
    let mut metadata = Metadata::new();
    metadata.insert("status_code".into(), json!(403));
    let original = ToolResultBlock::finished(
        "call-6",
        "write_file",
        "Permission denied",
        ToolResultState::Denied,
    )
    .unwrap()
    .with_metadata(metadata);

    let value = serde_json::to_value(ContentBlock::from(original.clone())).unwrap();
    assert_eq!(value["state"], "denied");
    assert_eq!(value["metadata"]["status_code"], 403);
    assert_eq!(
        serde_json::from_value::<ContentBlock>(value).unwrap(),
        ContentBlock::from(original)
    );
}

#[test]
fn tool_result_rejects_blank_identity() {
    assert_eq!(
        ToolResultBlock::running("", "search").unwrap_err(),
        ToolResultError::EmptyId
    );
    assert_eq!(
        ToolResultBlock::running("call-1", "  ").unwrap_err(),
        ToolResultError::EmptyName
    );

    let invalid = json!({
        "type": "tool_result",
        "id": "call-1",
        "name": "",
        "output": "",
        "state": "running",
        "metadata": {},
        "created_at": "2026-08-19T12:00:00.000000",
        "finished_at": null
    });
    assert!(serde_json::from_value::<ContentBlock>(invalid).is_err());
}

#[test]
fn text_content_excludes_tool_results() {
    let message = Msg::new(
        "Friday",
        Role::Assistant,
        [
            ContentBlock::from(
                ToolResultBlock::success("call-7", "search", "internal result").unwrap(),
            ),
            ContentBlock::from("Here is the answer."),
        ],
    );

    assert_eq!(
        message.text_content("\n").as_deref(),
        Some("Here is the answer.")
    );
}

#[test]
fn structured_output_states_use_wire_values() {
    for (state, expected) in [
        (StructuredOutputState::Streaming, "streaming"),
        (StructuredOutputState::Complete, "complete"),
    ] {
        assert_eq!(serde_json::to_value(state).unwrap(), expected);
        assert_eq!(state.to_string(), expected);
    }
}

#[test]
fn streaming_structured_output_accepts_partial_json() {
    let schema = json!({
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    });
    let mut block = StructuredOutputBlock::streaming(schema).unwrap();
    block.append_output_delta("{\"city\":\"").unwrap();
    assert!(matches!(
        block.parsed_output(),
        Err(StructuredOutputError::InvalidOutput(_))
    ));

    block.append_output_delta("Hangzhou\"}").unwrap();
    block.finish().unwrap();

    assert_eq!(block.state(), StructuredOutputState::Complete);
    assert_eq!(block.parsed_output().unwrap(), json!({"city": "Hangzhou"}));
    assert!(block.finished_at.is_some());
}

#[test]
fn complete_structured_output_uses_wire_format() {
    let schema = json!({"type": "array", "items": {"type": "integer"}});
    let block = StructuredOutputBlock::complete(schema.clone(), json!([1, 2, 3])).unwrap();
    let value = serde_json::to_value(ContentBlock::from(block.clone())).unwrap();

    assert_eq!(value["type"], "structured_output");
    assert_eq!(value["schema"], schema);
    assert_eq!(value["output"], "[1,2,3]");
    assert_eq!(value["state"], "complete");
    assert_eq!(block.schema(), &schema);
    assert_eq!(block.raw_output(), "[1,2,3]");

    let restored: ContentBlock = serde_json::from_value(value).unwrap();
    assert_eq!(restored, ContentBlock::from(block));
}

#[test]
fn structured_output_rejects_invalid_schema_roots() {
    for schema in [json!(null), json!("object"), json!(["object"])] {
        assert!(matches!(
            StructuredOutputBlock::streaming(schema),
            Err(StructuredOutputError::InvalidSchema)
        ));
    }

    assert!(StructuredOutputBlock::streaming(json!(true)).is_ok());
    assert!(StructuredOutputBlock::streaming(json!(false)).is_ok());
}

#[test]
fn completed_structured_output_requires_valid_json() {
    let value = json!({
        "type": "structured_output",
        "schema": {"type": "object"},
        "output": "{",
        "state": "complete",
        "id": "block-1",
        "created_at": "2026-08-20T12:00:00.000000",
        "finished_at": "2026-08-20T12:00:01.000000"
    });

    assert!(serde_json::from_value::<ContentBlock>(value).is_err());
}

#[test]
fn completed_structured_output_is_immutable() {
    let mut block = StructuredOutputBlock::complete(json!({}), json!({"ok": true})).unwrap();

    assert!(matches!(
        block.append_output_delta(" "),
        Err(StructuredOutputError::AlreadyComplete)
    ));
    assert!(matches!(
        block.finish(),
        Err(StructuredOutputError::AlreadyComplete)
    ));
}

#[test]
fn text_content_excludes_structured_output() {
    let message = Msg::new(
        "Friday",
        Role::Assistant,
        [
            ContentBlock::from(
                StructuredOutputBlock::complete(json!({}), json!({"answer": 42})).unwrap(),
            ),
            ContentBlock::from("Done"),
        ],
    );

    assert_eq!(message.text_content("\n").as_deref(), Some("Done"));
}
