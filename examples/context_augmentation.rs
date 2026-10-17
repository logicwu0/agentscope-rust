//! Offline algorithm extension example; no credentials or network required.
use agentscope::{
    ChatResponse, ContentBlock, ContextAugmentationFuture, ContextAugmenter, ContextInput,
    MockChatModel, Msg, ReActAgent, ToolExecutor, ToolRegistry,
};

struct ExperienceRecall;

impl ContextAugmenter for ExperienceRecall {
    fn augment<'a>(&'a self, input: ContextInput<'a>) -> ContextAugmentationFuture<'a> {
        Box::pin(async move {
            // Replace this fixture with authorized retrieval from your own algorithm.
            // This is an integration example, not a ReasoningBank implementation.
            println!("Retrieving experience for model step {}", input.step);
            Ok(vec![
                "For paginated results, inspect every page before computing a total.".into(),
            ])
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model =
        MockChatModel::new("offline").with_response(ChatResponse::completed([ContentBlock::from(
            "Check all result pages.",
        )]));
    let agent = ReActAgent::new("researcher", model, ToolExecutor::new(ToolRegistry::new()))?
        .with_context_augmenter(ExperienceRecall);
    println!(
        "{:?}",
        agent
            .reply(Msg::user("How should I count paginated results?"))
            .await?
    );
    Ok(())
}
