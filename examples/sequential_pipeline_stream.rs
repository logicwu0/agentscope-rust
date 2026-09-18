//! Offline streaming writer -> reviewer pipeline with stage-aware events.
use agentscope::{
    AgentEvent, ChatEvent, FinishReason, InMemoryMemory, MockChatModel, Msg, PipelineEvent,
    ReActAgent, SequentialPipeline, ToolExecutor, ToolRegistry,
};
use futures_util::StreamExt;
use std::{error::Error, sync::Arc};

fn model(text: &str) -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new("offline-stream").with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: text.into(),
        }),
        Ok(ChatEvent::Finished {
            reason: FinishReason::Completed,
        }),
    ]))
}

fn agent(name: &str, model: Arc<MockChatModel>) -> Result<Arc<ReActAgent>, Box<dyn Error>> {
    Ok(Arc::new(
        ReActAgent::from_shared(name, model, ToolExecutor::new(ToolRegistry::new()))?
            .with_memory(InMemoryMemory::new()),
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let pipeline = SequentialPipeline::new(vec![
        agent(
            "writer",
            model("Draft: keep credentials outside source control."),
        )?,
        agent(
            "reviewer",
            model("Reviewed: use an ignored local config or environment variables."),
        )?,
    ])?;
    let mut events = pipeline
        .stream(Msg::user("Explain how to keep API credentials private."))
        .await?;

    while let Some(event) = events.next().await {
        match event {
            PipelineEvent::StageStarted {
                pipeline_step,
                agent_name,
            } => println!("stage {pipeline_step} started: {agent_name}"),
            PipelineEvent::Agent {
                pipeline_step,
                agent_name,
                event: AgentEvent::TextDelta { delta, .. },
            } => println!("stage {pipeline_step} {agent_name}: {delta}"),
            PipelineEvent::StageCompleted { stage } => {
                println!("stage {} completed: {}", stage.step, stage.agent_name);
            }
            PipelineEvent::Finished { output } => println!(
                "pipeline complete: {}",
                output.message.text_content("").unwrap_or_default()
            ),
            PipelineEvent::Error { error } => return Err(error.into()),
            PipelineEvent::Agent { .. } => {}
        }
    }
    Ok(())
}
