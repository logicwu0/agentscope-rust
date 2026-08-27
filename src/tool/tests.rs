use std::sync::Arc;

use futures_executor::block_on;
use serde_json::{Value, json};

use crate::{
    Metadata, MockTool, Tool, ToolCallBlock, ToolContext, ToolDefinition, ToolError, ToolRegistry,
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

#[test]
fn registry_exports_definitions_in_stable_name_order() {
    let calculator =
        ToolDefinition::new("calculator", "Calculate", json!({"type": "object"})).unwrap();
    let mut registry = ToolRegistry::new();
    registry
        .register(MockTool::new(weather_definition()).with_output("sunny"))
        .unwrap();
    registry
        .register(MockTool::new(calculator).with_output("42"))
        .unwrap();

    let names = registry
        .definitions()
        .into_iter()
        .map(|definition| definition.name)
        .collect::<Vec<_>>();

    assert_eq!(names, ["calculator", "weather"]);
    assert_eq!(registry.len(), 2);
    assert!(registry.contains("weather"));
    assert!(!registry.is_empty());
}

#[test]
fn registry_rejects_duplicate_names_and_invalid_schemas() {
    let mut registry = ToolRegistry::new();
    registry
        .register(MockTool::new(weather_definition()))
        .unwrap();

    let duplicate = registry
        .register(MockTool::new(weather_definition()))
        .unwrap_err();
    let invalid_definition =
        ToolDefinition::new("broken", "Broken", json!({"type": "not-a-json-type"})).unwrap();
    let invalid = registry
        .register(MockTool::new(invalid_definition))
        .unwrap_err();

    assert_eq!(duplicate.code.as_deref(), Some("duplicate_tool"));
    assert_eq!(invalid.code.as_deref(), Some("invalid_tool_schema"));
    assert_eq!(registry.len(), 1);
}

#[test]
fn registry_does_not_resolve_external_schema_references() {
    let definition = ToolDefinition::new(
        "external",
        "External schema",
        json!({"$ref": "https://example.com/tool-input.json"}),
    )
    .unwrap();
    let mut registry = ToolRegistry::new();

    let error = registry.register(MockTool::new(definition)).unwrap_err();

    assert_eq!(error.code.as_deref(), Some("invalid_tool_schema"));
    assert!(registry.is_empty());
}

#[test]
fn registry_validates_arguments_before_dispatch() {
    let tool = Arc::new(MockTool::new(weather_definition()).with_output("sunny"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let invalid = ToolCallBlock::complete("call-7", "weather", r#"{"city":7}"#).unwrap();
    let valid = ToolCallBlock::complete("call-8", "weather", r#"{"city":"杭州"}"#).unwrap();

    let error = block_on(registry.invoke(&invalid, ToolContext::new())).unwrap_err();
    let result = block_on(registry.invoke(&valid, ToolContext::new())).unwrap();

    assert_eq!(error.code.as_deref(), Some("tool_schema_mismatch"));
    assert!(error.message.contains("/city"));
    assert_eq!(result.id(), "call-8");
    assert_eq!(tool.recorded_invocations().len(), 1);
}

#[test]
fn registry_reports_unknown_tools_without_dispatch() {
    let registry = ToolRegistry::new();
    let call = ToolCallBlock::complete("call-9", "missing", "{}").unwrap();

    let error = block_on(registry.invoke(&call, ToolContext::new())).unwrap_err();

    assert_eq!(error.code.as_deref(), Some("unknown_tool"));
}

#[test]
fn registry_can_return_and_remove_shared_tools() {
    let tool = Arc::new(MockTool::new(weather_definition()));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool).unwrap();

    let fetched = registry.get("weather").unwrap();
    let removed = registry.remove("weather").unwrap();

    assert_eq!(fetched.definition().name, "weather");
    assert_eq!(removed.definition().name, "weather");
    assert!(registry.is_empty());
    assert!(registry.get("weather").is_none());
}
