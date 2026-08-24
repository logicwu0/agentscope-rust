use std::collections::HashMap;

use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};

use crate::{
    ChatModel, ChatRequest, ContentBlock, FinishReason, GenerateOptions, ModelCapability, Msg,
    OpenAIChatModel, ToolDefinition,
};

use super::wire;

#[test]
fn debug_output_redacts_api_key() {
    let model = OpenAIChatModel::new("test-model", "super-secret").unwrap();
    let debug = format!("{model:?}");

    assert!(debug.contains("[REDACTED]"));
    assert!(!debug.contains("super-secret"));
}

#[test]
fn capabilities_match_non_streaming_mvp() {
    let model = OpenAIChatModel::new("test-model", "secret").unwrap();
    let capabilities = model.capabilities();

    assert!(capabilities.supports(ModelCapability::ToolCalls));
    assert!(capabilities.supports(ModelCapability::StructuredOutput));
    assert!(!capabilities.supports(ModelCapability::Streaming));
    assert!(!capabilities.supports(ModelCapability::MultimodalInput));
}

#[test]
fn request_encodes_options_tools_and_structured_output() {
    let tool = ToolDefinition::new(
        "weather",
        "Get the weather",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    )
    .unwrap();
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    });
    let request = ChatRequest::new([Msg::system("Be concise"), Msg::user("Hello")])
        .with_options(
            GenerateOptions::new()
                .with_temperature(0.2)
                .with_max_tokens(128)
                .with_stop(["END"]),
        )
        .with_tools([tool])
        .with_structured_output_schema(schema)
        .unwrap();

    let body = wire::encode_request("test-model", &request).unwrap();

    assert_eq!(body["model"], "test-model");
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["content"], "Hello");
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["tools"][0]["function"]["name"], "weather");
    assert_eq!(body["response_format"]["type"], "json_schema");
}

#[test]
fn response_decodes_reasoning_tool_calls_and_usage() {
    let response = json!({
        "id": "chatcmpl-123",
        "model": "deepseek-chat",
        "created": 1_700_000_000,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "reasoning_content": "I should use a tool.",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{\"city\":\"杭州\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "prompt_cache_hit_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 3}
        }
    });

    let decoded = wire::decode_response(&response, None).unwrap();

    assert_eq!(decoded.id, "chatcmpl-123");
    assert_eq!(decoded.finish_reason, Some(FinishReason::ToolCalls));
    assert!(matches!(decoded.content[0], ContentBlock::Thinking(_)));
    let tool_call = decoded.tool_calls().next().unwrap();
    assert_eq!(tool_call.name(), "weather");
    assert_eq!(tool_call.parsed_input().unwrap()["city"], "杭州");
    let usage = decoded.usage.unwrap();
    assert_eq!(usage.input_tokens, 12);
    assert_eq!(usage.output_tokens, 7);
    assert_eq!(usage.cached_input_tokens, Some(5));
    assert_eq!(usage.reasoning_tokens, Some(3));
}

#[tokio::test]
async fn generate_sends_bearer_request_and_decodes_text() {
    let provider_response = json!({
        "id": "chatcmpl-local",
        "model": "deepseek-chat",
        "choices": [{
            "message": {"role": "assistant", "content": "OK"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 9, "completion_tokens": 1}
    });
    let (base_url, request_rx) = serve_once(200, provider_response).await;
    let model = OpenAIChatModel::builder()
        .model("deepseek-chat")
        .api_key("local-test-key")
        .base_url(base_url)
        .build()
        .unwrap();

    let response = model
        .generate(ChatRequest::new([Msg::user("Reply OK")]))
        .await
        .unwrap();
    let received = request_rx.await.unwrap();

    assert_eq!(response.text_content(""), Some("OK".to_owned()));
    assert_eq!(response.usage.unwrap().total_tokens(), 10);
    assert!(received.starts_with("POST /chat/completions HTTP/1.1"));
    assert!(
        received
            .to_ascii_lowercase()
            .contains("authorization: bearer local-test-key")
    );
    let body: Value = serde_json::from_str(received.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["model"], "deepseek-chat");
    assert_eq!(body["stream"], false);
}

#[tokio::test]
async fn rate_limit_error_is_retryable_and_preserves_provider_code() {
    let (base_url, _) = serve_once(
        429,
        json!({"error": {"message": "slow down", "type": "rate_limit_error"}}),
    )
    .await;
    let model = OpenAIChatModel::builder()
        .model("test-model")
        .api_key("secret")
        .base_url(base_url)
        .build()
        .unwrap();

    let error = model
        .generate(ChatRequest::new([Msg::user("Hello")]))
        .await
        .unwrap_err();

    assert_eq!(error.message, "slow down");
    assert_eq!(error.code.as_deref(), Some("rate_limit_error"));
    assert!(error.retryable);
}

async fn serve_once(status: u16, response: Value) -> (String, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_http_request(&mut socket).await;
        let body = serde_json::to_string(&response).unwrap();
        let reason = if status == 200 {
            "OK"
        } else {
            "Too Many Requests"
        };
        let reply = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(reply.as_bytes()).await.unwrap();
        request_tx.send(request).ok();
    });
    (format!("http://{address}"), request_rx)
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    loop {
        let read = socket.read(&mut buffer).await.unwrap();
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..read]);
        if let Some(header_end) = find_header_end(&bytes) {
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = parse_content_length(&headers);
            if bytes.len() >= header_end + 4 + content_length {
                break;
            }
        }
    }
    String::from_utf8(bytes).unwrap()
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(headers: &str) -> usize {
    let headers = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim()))
        .collect::<HashMap<_, _>>();
    headers
        .get("content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or_default()
}
