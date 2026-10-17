//! External-crate checks: custom algorithms implement only the execution contract.
use std::sync::Arc;

use agentscope::{
    Agent, AgentError, AgentEvent, AgentEventStream, AgentFuture, AgentInterruptHandle,
    ContextManagedAgent, InMemoryMemory, MockChatModel, Msg, ParallelPipeline, ReActAgent,
    RecoverableAgent, RoutedPipeline, SequentialPipeline, StatefulAgent, ToolExecutor,
    ToolRegistry,
};
use futures_util::StreamExt;

#[derive(Default)]
struct EchoAgent(AgentInterruptHandle);

// No fake snapshot, compaction or recovery methods are needed.
impl Agent for EchoAgent {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn reply(&self, message: Msg) -> AgentFuture<'_, Msg> {
        let token = self.0.token();
        Box::pin(async move {
            if token.is_interrupted() {
                return Err(AgentError::Interrupted);
            }
            Ok(Msg::assistant(
                "echo",
                message.text_content("").unwrap_or_default(),
            ))
        })
    }

    fn stream(&self, message: Msg) -> AgentFuture<'_, AgentEventStream<'_>> {
        Box::pin(async move {
            let message = self.reply(message).await?;
            Ok(
                Box::pin(futures_util::stream::iter([Ok(AgentEvent::Finished {
                    steps: 0,
                    message,
                })])) as AgentEventStream<'_>,
            )
        })
    }

    fn interrupt_handle(&self) -> AgentInterruptHandle {
        self.0.clone()
    }
}

#[tokio::test]
async fn minimal_agent_runs_in_all_pipeline_types() {
    let agent: Arc<dyn Agent> = Arc::new(EchoAgent::default());
    let message = Msg::user("hello");
    assert_eq!(
        agent.reply(message.clone()).await.unwrap().text_content(""),
        Some("hello".into())
    );
    let events = agent
        .stream(message.clone())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        events.as_slice(),
        [Ok(AgentEvent::Finished { steps: 0, .. })]
    ));
    assert!(
        SequentialPipeline::new(vec![agent.clone()])
            .unwrap()
            .run(message.clone())
            .await
            .is_ok()
    );
    assert!(
        ParallelPipeline::new(vec![agent.clone()], 1)
            .unwrap()
            .run(message.clone())
            .await
            .is_ok()
    );
    assert!(
        RoutedPipeline::new(vec![("echo".into(), agent)])
            .unwrap()
            .run("echo", message)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn external_agents_can_observe_operation_scoped_cancellation() {
    let agent = EchoAgent::default();
    let old = agent.reply(Msg::user("old"));
    agent.interrupt_handle().interrupt();
    assert!(matches!(old.await, Err(AgentError::Interrupted)));
    assert!(agent.reply(Msg::user("new")).await.is_ok());
    let handle = agent.interrupt_handle();
    let mut token = handle.token();
    handle.interrupt();
    token.cancelled().await;
    assert!(token.is_interrupted());
}

#[tokio::test]
async fn react_exposes_each_optional_capability_as_a_trait_object() {
    let agent = ReActAgent::new(
        "test",
        MockChatModel::new("mock"),
        ToolExecutor::new(ToolRegistry::new()),
    )
    .unwrap()
    .with_memory(InMemoryMemory::new());
    let stateful: &dyn StatefulAgent = &agent;
    stateful.observe(Msg::user("observed")).await.unwrap();
    let snapshot = stateful.snapshot().await.unwrap();
    stateful.restore(snapshot.clone()).await.unwrap();
    assert_eq!(snapshot, stateful.snapshot().await.unwrap());
    let context: &dyn ContextManagedAgent = &agent;
    context.clear_context_summary().await.unwrap();
    let recovery: &dyn RecoverableAgent = &agent;
    assert!(matches!(
        recovery.retry_tool_execution("missing".into()).await,
        Err(AgentError::NoPendingToolExecution)
    ));
}
