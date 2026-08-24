//! Server-sent event decoding for OpenAI-compatible streams.

use std::collections::{BTreeMap, VecDeque};

use futures_util::stream;
use serde_json::Value;

use crate::message::generate_id;

use super::{
    super::{ChatEvent, ChatEventStream, FinishReason, ModelError, ModelResult},
    transport_error, wire,
};

pub(super) fn decode_stream(
    response: reqwest::Response,
    structured_output_schema: Option<Value>,
) -> ChatEventStream<'static> {
    let state = SseState::new(response, structured_output_schema);
    Box::pin(stream::unfold(state, |state| async move {
        next_event(state).await
    }))
}

struct SseState {
    response: reqwest::Response,
    buffer: Vec<u8>,
    pending: VecDeque<ModelResult<ChatEvent>>,
    text_block_id: String,
    thinking_block_id: String,
    structured_output_schema: Option<Value>,
    tool_calls: BTreeMap<u64, PartialToolCall>,
    finish_reason: Option<FinishReason>,
    closed: bool,
}

impl SseState {
    fn new(response: reqwest::Response, structured_output_schema: Option<Value>) -> Self {
        Self {
            response,
            buffer: Vec::new(),
            pending: VecDeque::new(),
            text_block_id: generate_id(),
            thinking_block_id: generate_id(),
            structured_output_schema,
            tool_calls: BTreeMap::new(),
            finish_reason: None,
            closed: false,
        }
    }

    fn fail(&mut self, error: ModelError) {
        self.pending.push_back(Err(error));
        self.closed = true;
    }
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    announced: bool,
}

async fn next_event(mut state: SseState) -> Option<(ModelResult<ChatEvent>, SseState)> {
    loop {
        if let Some(event) = state.pending.pop_front() {
            return Some((event, state));
        }
        if state.closed {
            return None;
        }
        if let Some(frame) = take_frame(&mut state.buffer) {
            if let Err(error) = process_frame(&mut state, &frame) {
                state.fail(error);
            }
            continue;
        }

        match state.response.chunk().await {
            Ok(Some(chunk)) => state.buffer.extend_from_slice(&chunk),
            Ok(None) => {
                if state.buffer.is_empty() {
                    state.fail(invalid_stream("SSE stream ended before `data: [DONE]`"));
                } else {
                    let remaining = std::mem::take(&mut state.buffer);
                    if let Err(error) = process_frame(&mut state, &remaining) {
                        state.fail(error);
                    } else if !state.closed {
                        state.fail(invalid_stream("SSE stream ended before `data: [DONE]`"));
                    }
                }
            }
            Err(error) => state.fail(transport_error(&error)),
        }
    }
}

fn take_frame(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let (position, delimiter_length) = find_delimiter(buffer)?;
    let frame = buffer[..position].to_vec();
    buffer.drain(..position + delimiter_length);
    Some(frame)
}

fn find_delimiter(buffer: &[u8]) -> Option<(usize, usize)> {
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, 4));
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2));
    match (crlf, lf) {
        (Some(left), Some(right)) => Some(if left.0 <= right.0 { left } else { right }),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn process_frame(state: &mut SseState, frame: &[u8]) -> Result<(), ModelError> {
    let frame = std::str::from_utf8(frame)
        .map_err(|error| invalid_stream(format!("SSE event was not UTF-8: {error}")))?;
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() {
        return Ok(());
    }
    if data == "[DONE]" {
        state.closed = true;
        match state.finish_reason.take() {
            Some(reason) => state.pending.push_back(Ok(ChatEvent::Finished { reason })),
            None => state.fail(invalid_stream(
                "SSE stream ended without a provider finish reason",
            )),
        }
        return Ok(());
    }

    let chunk = serde_json::from_str::<Value>(&data)
        .map_err(|error| invalid_stream(format!("SSE data was invalid JSON: {error}")))?;
    if let Some(error) = chunk.get("error") {
        return Err(decode_provider_error(error));
    }
    process_chunk(state, &chunk)
}

fn process_chunk(state: &mut SseState, chunk: &Value) -> Result<(), ModelError> {
    if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
        state
            .pending
            .push_back(wire::decode_usage(usage).map(|usage| ChatEvent::Usage { usage }));
    }

    let Some(choice) = chunk
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
    else {
        return Ok(());
    };
    if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
        process_delta(state, delta)?;
    }
    if let Some(reason) = choice
        .get("finish_reason")
        .filter(|reason| !reason.is_null())
    {
        state.finish_reason = Some(wire::decode_finish_reason(Some(reason))?);
    }
    Ok(())
}

fn process_delta(
    state: &mut SseState,
    delta: &serde_json::Map<String, Value>,
) -> Result<(), ModelError> {
    if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
        if !reasoning.is_empty() {
            state.pending.push_back(Ok(ChatEvent::ThinkingDelta {
                block_id: state.thinking_block_id.clone(),
                delta: reasoning.to_owned(),
            }));
        }
    }
    if let Some(content) = delta.get("content").and_then(Value::as_str) {
        if !content.is_empty() {
            if let Some(schema) = &state.structured_output_schema {
                state
                    .pending
                    .push_back(Ok(ChatEvent::StructuredOutputDelta {
                        block_id: state.text_block_id.clone(),
                        schema: schema.clone(),
                        delta: content.to_owned(),
                    }));
            } else {
                state.pending.push_back(Ok(ChatEvent::TextDelta {
                    block_id: state.text_block_id.clone(),
                    delta: content.to_owned(),
                }));
            }
        }
    }
    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            process_tool_call_delta(state, tool_call)?;
        }
    }
    Ok(())
}

fn process_tool_call_delta(state: &mut SseState, value: &Value) -> Result<(), ModelError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_stream("tool-call delta must be an object"))?;
    let index = object
        .get("index")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid_stream("tool-call delta did not contain an index"))?;
    let partial = state.tool_calls.entry(index).or_default();
    if let Some(id) = object.get("id").and_then(Value::as_str) {
        partial.id = Some(id.to_owned());
    }
    let function = object.get("function").and_then(Value::as_object);
    if let Some(name) = function
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
    {
        partial.name = Some(name.to_owned());
    }
    let arguments = function
        .and_then(|function| function.get("arguments"))
        .and_then(Value::as_str);
    if arguments.is_some() || !partial.announced {
        let id = partial
            .id
            .clone()
            .ok_or_else(|| invalid_stream("first tool-call delta did not contain an identifier"))?;
        let name = partial.name.clone().ok_or_else(|| {
            invalid_stream("first tool-call delta did not contain a function name")
        })?;
        state.pending.push_back(Ok(ChatEvent::ToolCallDelta {
            tool_call_id: id,
            tool_name: name,
            delta: arguments.unwrap_or_default().to_owned(),
        }));
        partial.announced = true;
    }
    Ok(())
}

fn decode_provider_error(error: &Value) -> ModelError {
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("provider reported an SSE error");
    let code = error
        .get("code")
        .or_else(|| error.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("stream_error");
    ModelError::new(message).with_code(code)
}

fn invalid_stream(message: impl Into<String>) -> ModelError {
    ModelError::new(message).with_code("invalid_stream")
}
