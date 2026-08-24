//! Translation between provider-neutral types and `OpenAI` JSON.

use serde_json::{Map, Value, json};

use crate::message::{
    ContentBlock, Metadata, Msg, Role, StructuredOutputBlock, ThinkingBlock, ToolCallBlock,
    ToolResultContent, ToolResultOutput, Usage,
};

use super::super::{ChatRequest, ChatResponse, FinishReason, ModelError};

const RESERVED_REQUEST_FIELDS: [&str; 10] = [
    "model",
    "messages",
    "stream",
    "tools",
    "response_format",
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "stop",
];

pub(super) fn encode_request(model: &str, request: &ChatRequest) -> Result<Value, ModelError> {
    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model.to_owned()));
    body.insert("stream".to_owned(), Value::Bool(false));
    body.insert(
        "messages".to_owned(),
        Value::Array(encode_messages(&request.messages)?),
    );

    insert_option(&mut body, "temperature", request.options.temperature);
    insert_option(&mut body, "max_tokens", request.options.max_tokens);
    insert_option(&mut body, "top_p", request.options.top_p);
    insert_option(&mut body, "seed", request.options.seed);
    if !request.options.stop.is_empty() {
        body.insert("stop".to_owned(), json!(request.options.stop));
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(
                request
                    .tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }
    if let Some(schema) = &request.structured_output_schema {
        body.insert(
            "response_format".to_owned(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": true,
                    "schema": schema,
                }
            }),
        );
    }
    for (key, value) in &request.options.extra {
        if RESERVED_REQUEST_FIELDS.contains(&key.as_str()) {
            return Err(ModelError::new(format!(
                "provider option `{key}` cannot replace a standard request field"
            ))
            .with_code("reserved_option"));
        }
        body.insert(key.clone(), value.clone());
    }
    Ok(Value::Object(body))
}

pub(super) fn decode_response(
    response: &Value,
    structured_output_schema: Option<&Value>,
) -> Result<ChatResponse, ModelError> {
    let object = response
        .as_object()
        .ok_or_else(|| invalid_response("response root must be an object"))?;
    let choice = object
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_response("response did not contain a choice"))?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_response("choice did not contain a message"))?;

    let mut content = Vec::new();
    if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
        if !reasoning.is_empty() {
            content.push(ContentBlock::Thinking(ThinkingBlock::new(reasoning)));
        }
    }
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            if let Some(schema) = structured_output_schema {
                let output = serde_json::from_str::<Value>(text).map_err(|error| {
                    invalid_response(format!("structured output was invalid JSON: {error}"))
                })?;
                let block = StructuredOutputBlock::complete(schema.clone(), output)
                    .map_err(|error| invalid_response(error.to_string()))?;
                content.push(ContentBlock::StructuredOutput(block));
            } else {
                content.push(ContentBlock::from(text.to_owned()));
            }
        }
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            content.push(ContentBlock::ToolCall(decode_tool_call(tool_call)?));
        }
    }

    let reason = decode_finish_reason(choice.get("finish_reason"))?;
    let mut result = ChatResponse::finished(content, reason);
    if let Some(id) = object.get("id").and_then(Value::as_str) {
        id.clone_into(&mut result.id);
    }
    if let Some(usage) = object.get("usage") {
        result.usage = Some(decode_usage(usage)?);
    }
    result.metadata = response_metadata(object);
    Ok(result)
}

fn encode_messages(messages: &[Msg]) -> Result<Vec<Value>, ModelError> {
    let mut encoded = Vec::new();
    for message in messages {
        let mut text = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();
        for block in &message.content {
            match block {
                ContentBlock::Text(block) => text.push(block.text.clone()),
                ContentBlock::StructuredOutput(block) => {
                    text.push(block.raw_output().to_owned());
                }
                ContentBlock::Thinking(_) => {}
                ContentBlock::ToolCall(block) => tool_calls.push(encode_tool_call(block)),
                ContentBlock::ToolResult(block) => {
                    tool_results.push(json!({
                        "role": "tool",
                        "tool_call_id": block.id(),
                        "content": encode_tool_result(block.output())?,
                    }));
                }
                ContentBlock::Data(_) => {
                    return Err(ModelError::new(
                        "OpenAIChatModel does not yet support multimodal input",
                    )
                    .with_code("unsupported_content"));
                }
            }
        }
        if !text.is_empty() || !tool_calls.is_empty() {
            let mut wire = Map::new();
            wire.insert(
                "role".to_owned(),
                Value::String(role_name(message.role).to_owned()),
            );
            if text.is_empty() {
                wire.insert("content".to_owned(), Value::Null);
            } else {
                wire.insert("content".to_owned(), Value::String(text.join("\n")));
            }
            if !tool_calls.is_empty() {
                if message.role != Role::Assistant {
                    return Err(ModelError::new(
                        "tool calls can only be sent in assistant messages",
                    )
                    .with_code("invalid_message"));
                }
                wire.insert("tool_calls".to_owned(), Value::Array(tool_calls));
            }
            encoded.push(Value::Object(wire));
        }
        encoded.extend(tool_results);
    }
    Ok(encoded)
}

fn encode_tool_call(block: &ToolCallBlock) -> Value {
    json!({
        "id": block.id(),
        "type": "function",
        "function": {
            "name": block.name(),
            "arguments": block.input(),
        }
    })
}

fn encode_tool_result(output: &ToolResultOutput) -> Result<String, ModelError> {
    match output {
        ToolResultOutput::Text(text) => Ok(text.clone()),
        ToolResultOutput::Blocks(blocks) => {
            let mut text = Vec::new();
            for block in blocks {
                match block {
                    ToolResultContent::Text(block) => text.push(block.text.clone()),
                    ToolResultContent::Data(_) => {
                        return Err(ModelError::new(
                            "OpenAIChatModel does not yet support multimodal tool results",
                        )
                        .with_code("unsupported_content"));
                    }
                }
            }
            Ok(text.join("\n"))
        }
    }
}

fn decode_tool_call(value: &Value) -> Result<ToolCallBlock, ModelError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_response("tool call must be an object"))?;
    let id = required_string(object, "id", "tool call")?;
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_response("tool call did not contain a function"))?;
    let name = required_string(function, "name", "tool call function")?;
    let arguments = required_string(function, "arguments", "tool call function")?;
    ToolCallBlock::complete(id, name, arguments)
        .map_err(|error| invalid_response(format!("invalid tool call: {error}")))
}

fn decode_finish_reason(value: Option<&Value>) -> Result<FinishReason, ModelError> {
    match value.and_then(Value::as_str) {
        Some("stop") => Ok(FinishReason::Completed),
        Some("length") => Ok(FinishReason::Length),
        Some("tool_calls" | "function_call") => Ok(FinishReason::ToolCalls),
        Some("content_filter") => Ok(FinishReason::ContentFilter),
        Some(other) => Err(invalid_response(format!("unknown finish reason `{other}`"))),
        None => Err(invalid_response("choice did not contain a finish reason")),
    }
}

fn decode_usage(value: &Value) -> Result<Usage, ModelError> {
    let usage = value
        .as_object()
        .ok_or_else(|| invalid_response("usage must be an object"))?;
    let input = optional_u64(usage, "prompt_tokens").unwrap_or_default();
    let output = optional_u64(usage, "completion_tokens").unwrap_or_default();
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| optional_u64(details, "cached_tokens"))
        .or_else(|| optional_u64(usage, "prompt_cache_hit_tokens"));
    let reasoning = usage
        .get("completion_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| optional_u64(details, "reasoning_tokens"));
    let mut result = Usage::new(input, output);
    if let Some(cached) = cached {
        result = result.with_cached_input_tokens(cached);
    }
    if let Some(reasoning) = reasoning {
        result = result.with_reasoning_tokens(reasoning);
    }
    Ok(result)
}

fn response_metadata(object: &Map<String, Value>) -> Metadata {
    let mut metadata = Metadata::new();
    for key in ["model", "created", "system_fingerprint", "service_tier"] {
        if let Some(value) = object.get(key) {
            metadata.insert(key.to_owned(), value.clone());
        }
    }
    metadata
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a str, ModelError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_response(format!("{context} did not contain string field `{key}`")))
}

fn optional_u64(object: &Map<String, Value>, key: &str) -> Option<u64> {
    object.get(key).and_then(Value::as_u64)
}

fn insert_option<T: serde::Serialize>(body: &mut Map<String, Value>, key: &str, value: Option<T>) {
    if let Some(value) = value {
        if let Ok(value) = serde_json::to_value(value) {
            body.insert(key.to_owned(), value);
        }
    }
}

fn invalid_response(message: impl Into<String>) -> ModelError {
    ModelError::new(message).with_code("invalid_response")
}
