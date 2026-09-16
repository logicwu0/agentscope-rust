use super::offload::*;
use crate::*;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Store(Mutex<Vec<String>>);
impl OffloadStore for Store {
    fn put<'a>(&'a self, text: &'a str) -> ToolFuture<'a, String> {
        self.0.lock().unwrap().push(text.into());
        Box::pin(async { Ok("blob".into()) })
    }
    fn read<'a>(&'a self, _: &'a str, _: usize, _: usize) -> ToolFuture<'a, OffloadedTextChunk> {
        Box::pin(async { Err(ToolError::new("unused")) })
    }
}

#[tokio::test]
async fn projection_preserves_identity_and_bounds_escaped_unicode_preview() {
    let store = Arc::new(Store::default());
    let config = ToolResultOffload::new(store.clone(), 1024, 512, 128).unwrap();
    let result = ToolResultBlock::success("call", "tool", "\0🙂".repeat(1000)).unwrap();
    let original = Msg::new("tool", Role::User, [result.clone().into()]);
    let mut projected = vec![original.clone()];
    config.project(&mut projected).await.unwrap();
    let ContentBlock::ToolResult(changed) = &projected[0].content[0] else {
        panic!()
    };
    let ToolResultOutput::Text(text) = changed.output() else {
        panic!()
    };
    assert!(text.len() <= 1024);
    assert_eq!(changed.clone().with_output(result.output().clone()), result);
    assert_eq!(projected[0].id, original.id);
    assert_eq!(store.0.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn blocks_errors_running_and_read_pages_are_not_offloaded() {
    let store = Arc::new(Store::default());
    let config = ToolResultOffload::new(store.clone(), 1024, 64, 128).unwrap();
    let text = "x".repeat(3000);
    let results = [
        ToolResultBlock::success(
            "a",
            "tool",
            vec![ToolResultContent::Text(TextBlock::new(&text))],
        )
        .unwrap(),
        ToolResultBlock::finished("b", "tool", &*text, ToolResultState::Error).unwrap(),
        ToolResultBlock::running("c", "tool")
            .unwrap()
            .with_output(text.clone()),
        ToolResultBlock::success("d", "read_offloaded_text", &*text).unwrap(),
    ];
    let original = vec![Msg::new(
        "tool",
        Role::User,
        results.into_iter().map(ContentBlock::from),
    )];
    let mut projected = original.clone();
    config.project(&mut projected).await.unwrap();
    assert_eq!(projected, original);
    assert!(store.0.lock().unwrap().is_empty());
}
