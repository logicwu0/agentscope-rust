use agentscope::{
    ChatEvent, ChatEventStream, ChatModel, ChatRequest, ChatResponse, ContentBlock, FinishReason,
    ModelCapabilities, ModelFuture, Msg, Role, Tool, ToolCallBlock, ToolContext, ToolDefinition,
    ToolError, ToolFuture, ToolResultOutput,
};
use serde_json::{Value, json};

pub struct Multiply {
    definition: ToolDefinition,
}

impl Multiply {
    pub fn new() -> Self {
        Self { definition: ToolDefinition::new("multiply", "Multiply two integers", json!({"type":"object", "properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false})).expect("valid static schema") }
    }
}

impl Tool for Multiply {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }
    fn execute(&self, input: Value, _context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        Box::pin(async move {
            let a = input["a"]
                .as_i64()
                .ok_or_else(|| ToolError::new("a must be an integer"))?;
            let b = input["b"]
                .as_i64()
                .ok_or_else(|| ToolError::new("b must be an integer"))?;
            let result = a
                .checked_mul(b)
                .ok_or_else(|| ToolError::new("multiplication overflow"))?;
            Ok(result.to_string().into())
        })
    }
}

pub struct OfflineModel;

impl OfflineModel {
    fn response(request: &ChatRequest) -> ChatResponse {
        let last = request.messages.last();
        if let Some(result) = last.and_then(|msg| {
            msg.content.iter().find_map(|block| {
                if let ContentBlock::ToolResult(result) = block {
                    Some(result)
                } else {
                    None
                }
            })
        }) {
            return ChatResponse::completed([ContentBlock::from(format!(
                "Tool result: {:?}",
                result.output()
            ))]);
        }
        let text = last
            .and_then(|msg| msg.text_content(""))
            .unwrap_or_default();
        let words = text.split_whitespace().collect::<Vec<_>>();
        if let ["multiply", a, b] = words.as_slice() {
            if let (Ok(a), Ok(b)) = (a.parse::<i64>(), b.parse::<i64>()) {
                let call = ToolCallBlock::complete(
                    format!("call-{}", Msg::user("").id),
                    "multiply",
                    json!({"a":a,"b":b}).to_string(),
                )
                .expect("valid call");
                return ChatResponse::finished([ContentBlock::from(call)], FinishReason::ToolCalls);
            }
        }
        let turns = request
            .messages
            .iter()
            .filter(|msg| msg.role == Role::User)
            .count();
        ChatResponse::completed([ContentBlock::from(format!("Offline turn {turns}: {text}"))])
    }
}

impl ChatModel for OfflineModel {
    fn name(&self) -> &'static str {
        "offline"
    }
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::new()
            .with(agentscope::ModelCapability::Streaming, true)
            .with(agentscope::ModelCapability::ToolCalls, true)
    }
    fn generate(&self, request: ChatRequest) -> ModelFuture<'_, ChatResponse> {
        Box::pin(async move { Ok(Self::response(&request)) })
    }
    fn stream(&self, request: ChatRequest) -> ModelFuture<'_, ChatEventStream<'_>> {
        Box::pin(async move {
            let response = Self::response(&request);
            let mut events = Vec::new();
            for call in response.tool_calls() {
                events.push(Ok(ChatEvent::ToolCallDelta {
                    tool_call_id: call.id().into(),
                    tool_name: call.name().into(),
                    delta: call.input().into(),
                }));
            }
            if let Some(text) = response.text_content("") {
                for character in text.chars() {
                    events.push(Ok(ChatEvent::TextDelta {
                        block_id: "text".into(),
                        delta: character.to_string(),
                    }));
                }
            }
            events.push(Ok(ChatEvent::Finished {
                reason: response.finish_reason.unwrap_or(FinishReason::Completed),
            }));
            Ok(Box::pin(futures_util::stream::iter(events)) as ChatEventStream<'_>)
        })
    }
}
