//! Offline parallel text streams with branch-aware events and independent memories.
use agentscope::{
    Agent, AgentEvent, ChatEvent, FinishReason, InMemoryMemory, MockChatModel, Msg,
    ParallelBranchOutcome, ParallelEvent, ParallelPipeline, ReActAgent, ToolExecutor, ToolRegistry,
};
use futures_util::StreamExt;
use std::{error::Error, sync::Arc};

fn model(name: &str, first: &str, second: &str) -> Arc<MockChatModel> {
    Arc::new(MockChatModel::new(name).with_stream([
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: first.into(),
        }),
        Ok(ChatEvent::TextDelta {
            block_id: "text".into(),
            delta: second.into(),
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
    let scripts = [
        (
            "security",
            "Security: isolate credentials; ",
            "grant minimal permissions.",
        ),
        (
            "reliability",
            "Reliability: bound execution; ",
            "verify uncertain side effects.",
        ),
        (
            "operations",
            "Operations: isolate session state; ",
            "inspect failures.",
        ),
    ];
    let mut models = Vec::new();
    let mut analysts = Vec::new();
    for (name, first, second) in scripts {
        let model = model(name, first, second);
        analysts.push(agent(name, model.clone())?);
        models.push(model);
    }
    let pipeline = ParallelPipeline::new(
        analysts
            .iter()
            .map(|agent| agent.clone() as Arc<dyn Agent>)
            .collect(),
        2,
    )?;
    let question = Msg::user("What should we consider when building an agent service?");
    let mut events = pipeline.stream(question.clone()).await?;
    assert!(
        models
            .iter()
            .all(|model| model.recorded_requests().is_empty())
    );

    let mut text = vec![String::new(); scripts.len()];
    let mut finished = None;
    while let Some(event) = events.next().await {
        match event {
            ParallelEvent::BranchStarted { branch, agent_name } => {
                println!("branch {branch} started: {agent_name}");
            }
            ParallelEvent::Agent {
                branch,
                agent_name,
                event: AgentEvent::TextDelta { delta, .. },
            } => {
                println!("branch {branch} {agent_name}: {delta}");
                text[branch - 1].push_str(&delta);
            }
            ParallelEvent::BranchFinished { result } => {
                println!("branch {} finished: {}", result.branch, result.agent_name);
            }
            ParallelEvent::Finished { output } => {
                println!(
                    "parallel stream complete: {} branches",
                    output.branches.len()
                );
                finished = Some(output);
            }
            ParallelEvent::Error { error } => return Err(error.into()),
            ParallelEvent::Agent { .. } => {}
        }
    }
    let output = finished.ok_or("parallel stream ended without its terminal result")?;
    assert_eq!(output.branches.len(), scripts.len());
    for (index, branch) in output.branches.iter().enumerate() {
        assert_eq!(branch.branch, index + 1);
        assert_eq!(branch.agent_name, scripts[index].0);
        let ParallelBranchOutcome::Completed(message) = &branch.outcome else {
            unreachable!("a successful stream has only completed branches");
        };
        assert_eq!(message.text_content(""), Some(text[index].clone()));
        let requests = models[index].recorded_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.last(), Some(&question));
        assert_eq!(analysts[index].snapshot().await?.messages().len(), 2);
    }
    println!("Each analyst received the same question and retained its own two-message history.");
    Ok(())
}
