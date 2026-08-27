use std::sync::Arc;

use futures_executor::block_on;
use serde_json::{Value, json};

use crate::{
    Metadata, MockTool, Tool, ToolCallBlock, ToolContext, ToolDefinition, ToolError,
    ToolResultOutput, ToolResultState,
};

fn weather_definition() -> ToolDefinition {
    ToolDefinition::new(
        "weather",
        "Get the weather for a city",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    )
    .unwrap()
}

#[test]
fn tool_trait_object_invokes_and_preserves_call_identity() {
    let tool: Arc<dyn Tool> = Arc::new(MockTool::new(weather_definition()).with_output("sunny"));
    let call = ToolCallBlock::complete("call-1", "weather", r#"{"city":"杭州"}"#).unwrap();

    let result = block_on(tool.invoke(&call, ToolContext::new())).unwrap();

    assert_eq!(result.id(), "call-1");
    assert_eq!(result.name(), "weather");
    assert_eq!(result.state(), ToolResultState::Success);
    assert_eq!(result.output(), &ToolResultOutput::Text("sunny".to_owned()));
}

#[test]
fn mock_records_parsed_input_and_context() {
    let mut metadata = Metadata::new();
    metadata.insert(
        "session_id".to_owned(),
        Value::String("session-1".to_owned()),
    );
    let context = ToolContext::new().with_metadata(metadata.clone());
    let tool = MockTool::new(weather_definition()).with_output("cloudy");
    let call = ToolCallBlock::complete("call-2", "weather", r#"{"city":"上海"}"#).unwrap();

    block_on(tool.invoke(&call, context)).unwrap();
    let invocations = tool.recorded_invocations();

    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].input, json!({"city": "上海"}));
    assert_eq!(invocations[0].context.metadata, metadata);
}

#[test]
fn invoke_rejects_tool_name_mismatch_without_execution() {
    let tool = MockTool::new(weather_definition()).with_output("unused");
    let call = ToolCallBlock::complete("call-3", "calculator", "{}").unwrap();

    let error = block_on(tool.invoke(&call, ToolContext::new())).unwrap_err();

    assert_eq!(error.code.as_deref(), Some("tool_name_mismatch"));
    assert!(tool.recorded_invocations().is_empty());
}

#[test]
fn invoke_rejects_incomplete_json_without_execution() {
    let tool = MockTool::new(weather_definition()).with_output("unused");
    let mut call = ToolCallBlock::streaming("call-4", "weather").unwrap();
    call.append_input(r#"{"city":"#);

    let error = block_on(tool.invoke(&call, ToolContext::new())).unwrap_err();

    assert_eq!(error.code.as_deref(), Some("invalid_tool_input"));
    assert!(tool.recorded_invocations().is_empty());
}

#[test]
fn mock_preserves_scripted_errors_and_reports_exhaustion() {
    let scripted = ToolError::new("weather service unavailable")
        .with_code("upstream_unavailable")
        .with_retryable(true);
    let tool = MockTool::new(weather_definition()).with_error(scripted.clone());
    let first = ToolCallBlock::complete("call-5", "weather", "{}").unwrap();
    let second = ToolCallBlock::complete("call-6", "weather", "{}").unwrap();

    let first_error = block_on(tool.invoke(&first, ToolContext::new())).unwrap_err();
    let second_error = block_on(tool.invoke(&second, ToolContext::new())).unwrap_err();

    assert_eq!(first_error, scripted);
    assert_eq!(second_error.code.as_deref(), Some("mock_exhausted"));
    assert_eq!(tool.recorded_invocations().len(), 2);
}

#[test]
fn tool_error_round_trips_through_json() {
    let error = ToolError::new("try later")
        .with_code("temporary")
        .with_retryable(true);

    let json = serde_json::to_string(&error).unwrap();
    let decoded: ToolError = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded, error);
    assert_eq!(error.to_string(), "tool error temporary: try later");
}
