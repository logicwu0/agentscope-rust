use super::*;
use crate::{GenerateOptions, Msg, ToolCallBlock, ToolDefinition, ToolResultBlock};

struct MessageCounter;
impl TokenCounter for MessageCounter {
    fn count(&self, request: &ChatRequest) -> Result<TokenCount, TokenBudgetError> {
        Ok(TokenCount {
            tokens: request.messages.len() as u64 * 10 + request.tools.len() as u64 * 20,
            accuracy: TokenCountAccuracy::Exact,
        })
    }
}

fn history() -> Vec<Msg> {
    vec![
        Msg::system("instructions"),
        Msg::user("old"),
        Msg::assistant("a", "old answer"),
        Msg::user("recent"),
        Msg::assistant("a", "recent answer"),
        Msg::user("current"),
    ]
}

#[test]
fn configuration_and_output_reservation_are_validated() {
    for (window, output) in [(0, 0), (10, 0), (10, 10), (9, 10)] {
        assert!(matches!(
            TokenBudget::new(window, output),
            Err(TokenBudgetError::InvalidConfiguration)
        ));
    }
    let budget = TokenBudget::new(100, 20)
        .unwrap()
        .with_counter(MessageCounter);
    assert_eq!(budget.context_window(), 100);
    assert_eq!(budget.reserved_output(), 20);
    assert_eq!(budget.input_limit(), 80);
    for output in [0, 21] {
        let request =
            ChatRequest::new([]).with_options(GenerateOptions::new().with_max_tokens(output));
        assert!(
            matches!(budget.apply(request), Err(TokenBudgetError::InvalidOutputLimit { requested, .. }) if requested == output)
        );
    }
    assert_eq!(
        budget
            .apply(ChatRequest::new([]))
            .unwrap()
            .options
            .max_tokens,
        Some(20)
    );
    let request = ChatRequest::new([]).with_options(GenerateOptions::new().with_max_tokens(5));
    assert_eq!(budget.apply(request).unwrap().options.max_tokens, Some(5));
}

#[test]
fn cuts_minimum_complete_turns_and_keeps_system_and_current() {
    let request = ChatRequest::new(history());
    let budget = TokenBudget::new(60, 20)
        .unwrap()
        .with_counter(MessageCounter);
    let selected = budget.apply(request.clone()).unwrap();
    assert_eq!(
        selected.messages,
        [
            request.messages[..1].to_vec(),
            request.messages[3..].to_vec()
        ]
        .concat()
    );
    assert_eq!(request.messages.len(), 6);
    let selected = TokenBudget::new(40, 20)
        .unwrap()
        .with_counter(MessageCounter)
        .apply(request.clone())
        .unwrap();
    assert_eq!(
        selected.messages,
        vec![request.messages[0].clone(), request.messages[5].clone()]
    );
}

#[test]
fn fixed_tool_schemas_are_counted_and_not_pruned() {
    let tool = ToolDefinition::new("lookup", "Lookup", json!({"type":"object"})).unwrap();
    let request = ChatRequest::new(history()).with_tools([tool.clone()]);
    let selected = TokenBudget::new(60, 20)
        .unwrap()
        .with_counter(MessageCounter)
        .apply(request)
        .unwrap();
    assert_eq!(selected.messages.len(), 2);
    assert_eq!(selected.tools, vec![tool]);
}

#[test]
fn oversized_protected_content_returns_structured_error() {
    let budget = TokenBudget::new(39, 20)
        .unwrap()
        .with_counter(MessageCounter);
    let error = budget.apply(ChatRequest::new(history())).unwrap_err();
    assert_eq!(
        error,
        TokenBudgetError::Exceeded {
            input: TokenCount {
                tokens: 20,
                accuracy: TokenCountAccuracy::Exact
            },
            input_limit: 19,
            reserved_output: 20,
        }
    );
    assert!(error.to_string().contains("Exact"));
    let event = crate::AgentEvent::Error {
        step: Some(1),
        error: crate::AgentError::TokenBudget(error),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert_eq!(
        serde_json::from_str::<crate::AgentEvent>(&json).unwrap(),
        event
    );
}

#[test]
fn no_user_history_is_not_silently_discarded() {
    let request = ChatRequest::new([
        Msg::system("instructions"),
        Msg::assistant("other", "observe"),
    ]);
    let budget = TokenBudget::new(21, 20)
        .unwrap()
        .with_counter(MessageCounter);
    assert!(matches!(
        budget.apply(request),
        Err(TokenBudgetError::Exceeded { .. })
    ));
}

#[test]
fn cross_turn_tool_dependencies_cannot_be_cut_to_satisfy_budget() {
    let call = ToolCallBlock::complete("c", "tool", "{}").unwrap();
    let result = ToolResultBlock::success("c", "tool", "result").unwrap();
    let request = ChatRequest::new([
        Msg::user("old"),
        Msg::new("a", Role::Assistant, [ContentBlock::from(call)]),
        Msg::user("current"),
        Msg::new("tool", Role::Assistant, [ContentBlock::from(result)]),
    ]);
    let budget = TokenBudget::new(50, 20)
        .unwrap()
        .with_counter(MessageCounter);
    assert!(matches!(
        budget.apply(request),
        Err(TokenBudgetError::Exceeded {
            input: TokenCount { tokens: 40, .. },
            ..
        })
    ));
}

#[test]
fn estimator_includes_multilingual_content_system_tools_and_schema() {
    let base = ChatRequest::new([Msg::user("Hello 你好 😀")]);
    let initial = HeuristicTokenCounter.count(&base).unwrap();
    assert_eq!(initial.accuracy, TokenCountAccuracy::Estimated);
    let mut larger = base.clone();
    larger
        .messages
        .insert(0, Msg::system("system instructions".repeat(50)));
    assert!(HeuristicTokenCounter.count(&larger).unwrap().tokens > initial.tokens);
    larger = base.clone().with_tools([ToolDefinition::new(
        "tool",
        "description".repeat(50),
        json!({"type":"object"}),
    )
    .unwrap()]);
    assert!(HeuristicTokenCounter.count(&larger).unwrap().tokens > initial.tokens);
    larger = base
        .clone()
        .with_structured_output_schema(json!({"description":"schema".repeat(50)}))
        .unwrap();
    assert!(HeuristicTokenCounter.count(&larger).unwrap().tokens > initial.tokens);
    let mut metadata_only = base;
    metadata_only.messages[0].id = "x".repeat(1000);
    metadata_only.messages[0]
        .metadata
        .insert("local".into(), json!("not sent".repeat(1000)));
    assert_eq!(
        HeuristicTokenCounter.count(&metadata_only).unwrap(),
        initial
    );
}

#[test]
fn counter_errors_fail_closed() {
    struct Failing;
    impl TokenCounter for Failing {
        fn count(&self, _: &ChatRequest) -> Result<TokenCount, TokenBudgetError> {
            Err(TokenBudgetError::Counter("unavailable tokenizer".into()))
        }
    }
    let counter: Arc<dyn TokenCounter> = Arc::new(Failing);
    let budget = TokenBudget::new(100, 20)
        .unwrap()
        .with_shared_counter(counter);
    assert!(matches!(
        budget.apply(ChatRequest::new(history())),
        Err(TokenBudgetError::Counter(_))
    ));
}

#[test]
fn heuristic_rejects_multimodal_messages_and_tool_results() {
    let data = crate::DataBlock::url("https://example.com/image.png", "image/png").unwrap();
    let direct = ContentBlock::from(data.clone());
    let result = ContentBlock::from(
        ToolResultBlock::success("call", "tool", vec![ToolResultContent::Data(data)]).unwrap(),
    );
    for block in [direct, result] {
        let request = ChatRequest::new([Msg::new("user", Role::User, [block])]);
        assert_eq!(
            HeuristicTokenCounter.count(&request),
            Err(TokenBudgetError::UnsupportedContent)
        );
    }
}

#[test]
fn heuristic_counts_large_tool_results_and_reports_estimated_overflow() {
    let result = ToolResultBlock::success("call", "tool", "output".repeat(1000)).unwrap();
    let request = ChatRequest::new([
        Msg::user("current"),
        Msg::new("tool", Role::Assistant, [ContentBlock::from(result)]),
    ]);
    assert!(matches!(
        TokenBudget::new(100, 20).unwrap().apply(request),
        Err(TokenBudgetError::Exceeded {
            input: TokenCount {
                accuracy: TokenCountAccuracy::Estimated,
                ..
            },
            ..
        })
    ));
}
