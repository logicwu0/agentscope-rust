use crate::*;
use futures_util::StreamExt;
use std::sync::{Arc, Mutex};

struct Recall {
    text: String,
    inputs: Arc<Mutex<Vec<Vec<Msg>>>>,
}
impl ContextAugmenter for Recall {
    fn augment<'a>(&'a self, input: ContextInput<'a>) -> ContextAugmentationFuture<'a> {
        Box::pin(async move {
            assert_eq!(input.agent_name, "test");
            assert!(input.step > 0);
            self.inputs.lock().unwrap().push(input.messages.to_vec());
            Ok(vec![String::new(), self.text.clone()])
        })
    }
}

fn agent(model: Arc<MockChatModel>) -> ReActAgent {
    ReActAgent::from_shared("test", model, ToolExecutor::new(ToolRegistry::new())).unwrap()
}

#[tokio::test]
async fn augmentation_is_ordered_request_local_and_shared_by_streaming() {
    let model = Arc::new(
        MockChatModel::new("mock")
            .with_response(ChatResponse::completed([ContentBlock::from("done")]))
            .with_stream([
                Ok(ChatEvent::TextDelta {
                    block_id: "a".into(),
                    delta: "done".into(),
                }),
                Ok(ChatEvent::Finished {
                    reason: FinishReason::Completed,
                }),
            ]),
    );
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let memory = Arc::new(InMemoryMemory::new());
    let agent = agent(model.clone())
        .with_system_prompt("system")
        .with_shared_memory(memory.clone())
        .with_context_augmenter(Recall {
            text: "first".into(),
            inputs: inputs.clone(),
        })
        .with_context_augmenter(Recall {
            text: "second".into(),
            inputs: inputs.clone(),
        });
    agent.reply(Msg::user("task")).await.unwrap();
    let events = agent
        .stream(Msg::user("next task"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Ok(AgentEvent::Finished { .. })))
    );
    let requests = model.recorded_requests();
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request.messages[0].role, Role::System);
        assert_eq!(request.messages[1].role, Role::User);
        assert!(
            serde_json::to_string(&request.messages[1])
                .unwrap()
                .contains("first")
        );
        assert!(
            serde_json::to_string(&request.messages[2])
                .unwrap()
                .contains("second")
        );
        assert!(request.tools.is_empty());
    }
    {
        let inputs = inputs.lock().unwrap();
        assert_eq!(inputs.len(), 4);
        assert_eq!(inputs[0], inputs[1]);
        assert_eq!(inputs[2], inputs[3]);
    }
    assert!(
        !serde_json::to_string(&memory.messages().await.unwrap())
            .unwrap()
            .contains("Supplementary context")
    );
    assert!(
        !serde_json::to_string(&agent.snapshot().await.unwrap())
            .unwrap()
            .contains("Supplementary context")
    );
}

struct Failing;
impl ContextAugmenter for Failing {
    fn augment<'a>(&'a self, _: ContextInput<'a>) -> ContextAugmentationFuture<'a> {
        Box::pin(async {
            Err(ContextAugmentationError {
                message: "retrieval unavailable".into(),
            })
        })
    }
}

#[tokio::test]
async fn retrieval_failure_stops_before_model_in_both_modes() {
    let model = Arc::new(MockChatModel::new("mock"));
    let agent = agent(model.clone()).with_context_augmenter(Failing);
    let error = agent.reply(Msg::user("task")).await.unwrap_err();
    assert!(matches!(error, AgentError::ContextAugmentation(_)));
    let json = serde_json::to_string(&error).unwrap();
    assert_eq!(serde_json::from_str::<AgentError>(&json).unwrap(), error);
    let events = agent
        .stream(Msg::user("task"))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        &events[..],
        [Ok(AgentEvent::Error {
            error: AgentError::ContextAugmentation(_),
            ..
        })]
    ));
    assert!(model.recorded_requests().is_empty());
}

#[tokio::test]
async fn supplementary_context_cannot_bypass_token_budget() {
    let model = Arc::new(MockChatModel::new("mock"));
    let agent = agent(model.clone())
        .with_token_budget(TokenBudget::new(100, 10).unwrap())
        .with_context_augmenter(Recall {
            text: "experience ".repeat(1000),
            inputs: Arc::default(),
        });
    assert!(matches!(
        agent.reply(Msg::user("task")).await,
        Err(AgentError::TokenBudget(_))
    ));
    assert!(model.recorded_requests().is_empty());
}

struct Waiting(tokio::sync::Notify);

#[tokio::test]
async fn augmentation_preserves_tool_confirmation_and_retrieves_after_recovery() {
    let model = Arc::new(
        MockChatModel::new("mock")
            .with_response(ChatResponse::finished(
                [ContentBlock::from(
                    ToolCallBlock::complete("call", "lookup", "{}").unwrap(),
                )],
                FinishReason::ToolCalls,
            ))
            .with_response(ChatResponse::completed([ContentBlock::from("done")])),
    );
    let definition =
        ToolDefinition::new("lookup", "lookup", serde_json::json!({"type":"object"})).unwrap();
    let tool = Arc::new(MockTool::new(definition).with_output("result"));
    let mut registry = ToolRegistry::new();
    registry.register_shared(tool.clone()).unwrap();
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let agent = ReActAgent::from_shared("test", model.clone(), ToolExecutor::new(registry))
        .unwrap()
        .with_memory(InMemoryMemory::new())
        .with_tool_confirmation_required("lookup")
        .with_context_augmenter(Recall {
            text: "reference".into(),
            inputs: inputs.clone(),
        });
    let Err(AgentError::ToolConfirmationRequired { checkpoint }) =
        agent.reply(Msg::user("task")).await
    else {
        panic!("confirmation must remain required");
    };
    assert!(tool.recorded_invocations().is_empty());
    agent
        .resume_tool_calls(
            checkpoint.reply_id(),
            vec![ToolConfirmation::approve("call")],
        )
        .await
        .unwrap();
    assert_eq!(tool.recorded_invocations().len(), 1);
    assert_eq!(inputs.lock().unwrap().len(), 2);
    let requests = model.recorded_requests();
    assert_eq!(requests[0].tools, requests[1].tools);
    for request in requests {
        assert_eq!(
            serde_json::to_string(&request.messages)
                .unwrap()
                .matches("Supplementary context")
                .count(),
            1
        );
    }
}
impl ContextAugmenter for Waiting {
    fn augment<'a>(&'a self, _: ContextInput<'a>) -> ContextAugmentationFuture<'a> {
        Box::pin(async {
            self.0.notify_one();
            std::future::pending().await
        })
    }
}

#[tokio::test]
async fn pending_retrieval_is_interruptible() {
    let model = Arc::new(MockChatModel::new("mock"));
    let provider = Arc::new(Waiting(tokio::sync::Notify::new()));
    let agent = agent(model.clone()).with_shared_context_augmenter(provider.clone());
    let handle = agent.interrupt_handle();
    let (result, ()) = tokio::join!(agent.reply(Msg::user("task")), async {
        provider.0.notified().await;
        handle.interrupt();
    });
    assert!(matches!(result, Err(AgentError::Interrupted)));
    assert!(model.recorded_requests().is_empty());
}
