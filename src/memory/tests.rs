use std::sync::Arc;

use futures_executor::block_on;

use crate::{InMemoryMemory, Memory, Msg};

#[test]
fn in_memory_memory_preserves_order_and_returns_snapshots() {
    let memory = InMemoryMemory::from_messages([Msg::system("Be concise")]);
    block_on(memory.append(vec![Msg::user("Hello"), Msg::assistant("bot", "Hi")])).unwrap();

    let mut snapshot = block_on(memory.messages()).unwrap();
    snapshot.push(Msg::user("not persisted"));
    let stored = block_on(memory.messages()).unwrap();

    assert_eq!(stored.len(), 3);
    assert_eq!(stored[0].text_content(""), Some("Be concise".to_owned()));
    assert_eq!(stored[1].text_content(""), Some("Hello".to_owned()));
    assert_eq!(stored[2].text_content(""), Some("Hi".to_owned()));
}

#[test]
fn memory_is_object_safe_and_can_be_cleared() {
    let memory: Arc<dyn Memory> = Arc::new(InMemoryMemory::new());

    block_on(memory.append(vec![Msg::user("Hello")])).unwrap();
    assert_eq!(block_on(memory.messages()).unwrap().len(), 1);
    block_on(memory.clear()).unwrap();

    assert!(block_on(memory.messages()).unwrap().is_empty());
}
