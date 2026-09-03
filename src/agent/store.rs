//! Versioned persistence for per-session agent state.

use std::{collections::BTreeMap, fmt, future::Future, pin::Pin, sync::Mutex};

use serde::{Deserialize, Deserializer, Serialize};

use super::AgentState;

/// Result returned by a state-store operation.
pub type StateStoreResult<T> = Result<T, StateStoreError>;

/// A boxed asynchronous state-store operation.
pub type StateStoreFuture<'a, T> = Pin<Box<dyn Future<Output = StateStoreResult<T>> + Send + 'a>>;

/// Stable identity of one user's conversation session.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct StateKey {
    user_id: String,
    session_id: String,
}

impl<'de> Deserialize<'de> for StateKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct StateKeyWire {
            user_id: String,
            session_id: String,
        }

        let wire = StateKeyWire::deserialize(deserializer)?;
        Self::new(wire.user_id, wire.session_id).map_err(serde::de::Error::custom)
    }
}

impl StateKey {
    /// Creates a validated user/session key.
    ///
    /// # Errors
    ///
    /// Returns [`StateStoreError`] when either identifier is blank.
    pub fn new(
        user_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> StateStoreResult<Self> {
        let user_id = user_id.into();
        let session_id = session_id.into();
        if user_id.trim().is_empty() {
            return Err(
                StateStoreError::new("state user id cannot be empty").with_code("invalid_user_id")
            );
        }
        if session_id.trim().is_empty() {
            return Err(StateStoreError::new("state session id cannot be empty")
                .with_code("invalid_session_id"));
        }
        Ok(Self {
            user_id,
            session_id,
        })
    }

    /// Returns the user identifier.
    #[must_use]
    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    /// Returns the session identifier.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

/// One persisted agent state and its optimistic-concurrency revision.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StateRecord {
    revision: u64,
    state: AgentState,
}

impl<'de> Deserialize<'de> for StateRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct StateRecordWire {
            revision: u64,
            state: AgentState,
        }

        let wire = StateRecordWire::deserialize(deserializer)?;
        if wire.revision == 0 {
            return Err(serde::de::Error::custom(
                "state record revision must be greater than zero",
            ));
        }
        Ok(Self {
            revision: wire.revision,
            state: wire.state,
        })
    }
}

impl StateRecord {
    /// Returns the monotonically increasing store revision.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the persisted agent state.
    #[must_use]
    pub const fn state(&self) -> &AgentState {
        &self.state
    }

    pub(crate) fn into_state(self) -> AgentState {
        self.state
    }
}

/// Object-safe asynchronous storage for versioned agent state.
pub trait StateStore: Send + Sync {
    /// Loads the current record, or `None` when the session has never been saved.
    fn load<'a>(&'a self, key: &'a StateKey) -> StateStoreFuture<'a, Option<StateRecord>>;

    /// Saves a state only if its current revision equals `expected_revision`.
    ///
    /// Pass `None` to create a previously absent record. Successful writes
    /// return the new revision.
    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        state: AgentState,
    ) -> StateStoreFuture<'_, StateRecord>;
}

/// Thread-safe process-local state storage with compare-and-swap updates.
#[derive(Default)]
pub struct InMemoryStateStore {
    records: Mutex<BTreeMap<StateKey, StateRecord>>,
}

impl InMemoryStateStore {
    /// Creates an empty state store.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            records: Mutex::new(BTreeMap::new()),
        }
    }
}

impl StateStore for InMemoryStateStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> StateStoreFuture<'a, Option<StateRecord>> {
        Box::pin(async move { Ok(lock(&self.records).get(key).cloned()) })
    }

    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        state: AgentState,
    ) -> StateStoreFuture<'_, StateRecord> {
        Box::pin(async move {
            let mut records = lock(&self.records);
            let actual_revision = records.get(&key).map(StateRecord::revision);
            if actual_revision != expected_revision {
                return Err(StateStoreError::conflict(
                    expected_revision,
                    actual_revision,
                ));
            }
            let revision = actual_revision.unwrap_or(0).checked_add(1).ok_or_else(|| {
                StateStoreError::new("state revision cannot be incremented")
                    .with_code("revision_overflow")
            })?;
            let record = StateRecord { revision, state };
            records.insert(key, record.clone());
            Ok(record)
        })
    }
}

impl fmt::Debug for InMemoryStateStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InMemoryStateStore")
            .field("record_count", &lock(&self.records).len())
            .finish()
    }
}

/// A structured state persistence failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StateStoreError {
    /// Human-readable failure description.
    pub message: String,
    /// Stable backend or application error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Whether retrying after reloading state may succeed.
    #[serde(default)]
    pub retryable: bool,
}

impl StateStoreError {
    /// Creates a non-retryable state-store error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
            retryable: false,
        }
    }

    /// Attaches a stable error code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Marks whether retrying the operation may succeed.
    #[must_use]
    pub const fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    fn conflict(expected: Option<u64>, actual: Option<u64>) -> Self {
        Self::new(format!(
            "state revision conflict: expected {expected:?}, found {actual:?}"
        ))
        .with_code("revision_conflict")
        .with_retryable(true)
    }
}

impl fmt::Display for StateStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.code {
            Some(code) => write!(formatter, "state store error {code}: {}", self.message),
            None => write!(formatter, "state store error: {}", self.message),
        }
    }
}

impl std::error::Error for StateStoreError {}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
