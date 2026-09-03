use std::sync::Arc;

use futures_executor::block_on;

use super::{InMemoryStateStore, StateKey, StateStore};
use crate::{AgentState, Msg};

#[test]
fn state_store_is_object_safe_and_versions_updates() {
    let store: Arc<dyn StateStore> = Arc::new(InMemoryStateStore::new());
    let key = StateKey::new("user-1", "session-1").unwrap();
    let first_state = AgentState::new("Friday", vec![Msg::user("first")]);

    assert!(block_on(store.load(&key)).unwrap().is_none());
    let first = block_on(store.save(key.clone(), None, first_state)).unwrap();
    assert_eq!(first.revision(), 1);
    let encoded = serde_json::to_string(&first).unwrap();
    assert_eq!(
        serde_json::from_str::<super::StateRecord>(&encoded).unwrap(),
        first
    );
    assert!(
        serde_json::from_value::<super::StateRecord>(serde_json::json!({
            "revision": 0,
            "state": AgentState::new("Friday", Vec::new()),
        }))
        .is_err()
    );
    let second_state = AgentState::new("Friday", vec![Msg::user("second")]);
    let second = block_on(store.save(key.clone(), Some(1), second_state.clone())).unwrap();

    assert_eq!(second.revision(), 2);
    assert_eq!(
        block_on(store.load(&key)).unwrap().unwrap().state(),
        &second_state
    );
}

#[test]
fn state_store_rejects_stale_writes_without_overwriting() {
    let store = InMemoryStateStore::new();
    let key = StateKey::new("user-1", "session-1").unwrap();
    let original = AgentState::new("Friday", vec![Msg::user("original")]);
    block_on(store.save(key.clone(), None, original.clone())).unwrap();

    let error = block_on(store.save(
        key.clone(),
        None,
        AgentState::new("Friday", vec![Msg::user("stale")]),
    ))
    .unwrap_err();

    assert_eq!(error.code.as_deref(), Some("revision_conflict"));
    assert!(error.retryable);
    assert_eq!(
        block_on(store.load(&key)).unwrap().unwrap().state(),
        &original
    );
}

#[test]
fn state_store_isolates_users_and_sessions() {
    let store = InMemoryStateStore::new();
    let first = StateKey::new("user-1", "session-1").unwrap();
    let second = StateKey::new("user-1", "session-2").unwrap();
    let third = StateKey::new("user-2", "session-1").unwrap();

    for (key, text) in [(&first, "first"), (&second, "second"), (&third, "third")] {
        block_on(store.save(
            key.clone(),
            None,
            AgentState::new("Friday", vec![Msg::user(text)]),
        ))
        .unwrap();
    }

    assert_eq!(
        block_on(store.load(&second))
            .unwrap()
            .unwrap()
            .state()
            .messages()[0]
            .text_content(""),
        Some("second".to_owned())
    );
}

#[test]
fn state_keys_validate_identity_and_round_trip_through_json() {
    let key = StateKey::new("user-1", "session-1").unwrap();
    let encoded = serde_json::to_string(&key).unwrap();

    assert_eq!(serde_json::from_str::<StateKey>(&encoded).unwrap(), key);
    assert_eq!(
        StateKey::new(" ", "session").unwrap_err().code.as_deref(),
        Some("invalid_user_id")
    );
    assert_eq!(
        StateKey::new("user", " ").unwrap_err().code.as_deref(),
        Some("invalid_session_id")
    );
    assert!(serde_json::from_str::<StateKey>(r#"{"user_id":" ","session_id":"session"}"#).is_err());
}
