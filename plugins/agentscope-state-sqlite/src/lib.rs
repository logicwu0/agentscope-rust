//! `SQLite` persistence for complete `AgentScope` agent snapshots.

use agentscope::{
    AgentState, StateKey, StateRecord, StateStore, StateStoreError, StateStoreFuture,
    StateStoreResult,
};
use std::{path::Path, time::Duration};
use tokio_rusqlite::{
    Connection, Error, params,
    rusqlite::{self, OptionalExtension, TransactionBehavior},
};

/// A cloneable `SQLite` state store. Clones share a connection; independent
/// connections use immediate transactions to serialize revision checks/writes.
/// Agent snapshots are stored as JSON, isolated by user and session.
#[derive(Clone)]
pub struct SQLiteStateStore {
    connection: Connection,
}

impl SQLiteStateStore {
    /// Opens a database, initializing the plugin's versioned tables atomically.
    ///
    /// # Errors
    /// Returns an error for inaccessible databases or unsupported schema versions.
    pub async fn open(path: impl AsRef<Path>) -> StateStoreResult<Self> {
        let connection = Connection::open(path).await.map_err(sql_error)?;
        connection
            .call(|db| {
                db.busy_timeout(Duration::from_secs(5)).map_err(sql_error)?;
                let tx = db
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sql_error)?;
                tx.execute_batch(
                    "CREATE TABLE IF NOT EXISTS agentscope_state_schema (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1), version INTEGER NOT NULL);
                INSERT OR IGNORE INTO agentscope_state_schema VALUES (1, 1);",
                )
                .map_err(sql_error)?;
                let version: i64 = tx
                    .query_row(
                        "SELECT version FROM agentscope_state_schema WHERE singleton = 1",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(sql_error)?;
                if version != 1 {
                    return Err(StateStoreError::new(format!(
                        "unsupported SQLite state schema version {version}"
                    ))
                    .with_code("unsupported_schema_version"));
                }
                tx.execute_batch(
                    "CREATE TABLE IF NOT EXISTS agentscope_states (
                user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision > 0), state_json TEXT NOT NULL,
                PRIMARY KEY(user_id, session_id));",
                )
                .map_err(sql_error)?;
                tx.commit().map_err(sql_error)
            })
            .await
            .map_err(worker_error)?;
        Ok(Self { connection })
    }
}

impl StateStore for SQLiteStateStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> StateStoreFuture<'a, Option<StateRecord>> {
        let key = key.clone();
        Box::pin(async move {
            self.connection.call(move |db| {
                let row: Option<(i64, String)> = db.query_row(
                    "SELECT revision, state_json FROM agentscope_states WHERE user_id = ?1 AND session_id = ?2",
                    params![key.user_id(), key.session_id()], |row| Ok((row.get(0)?, row.get(1)?)),
                ).optional().map_err(sql_error)?;
                row.map(|(revision, json)| {
                    let revision = u64::try_from(revision).map_err(|_| StateStoreError::new("invalid stored revision").with_code("invalid_revision"))?;
                    let state = serde_json::from_str(&json).map_err(|error| StateStoreError::new(error.to_string()).with_code("invalid_state_json"))?;
                    StateRecord::new(revision, state)
                }).transpose()
            }).await.map_err(worker_error)
        })
    }

    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        state: AgentState,
    ) -> StateStoreFuture<'_, StateRecord> {
        Box::pin(async move {
            let json = serde_json::to_string(&state).map_err(|error| {
                StateStoreError::new(error.to_string()).with_code("invalid_state_json")
            })?;
            self.connection.call(move |db| {
                let tx = db.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql_error)?;
                let actual: Option<u64> = tx.query_row(
                    "SELECT revision FROM agentscope_states WHERE user_id = ?1 AND session_id = ?2",
                    params![key.user_id(), key.session_id()], |row| row.get(0),
                ).optional().map_err(sql_error)?;
                if actual != expected_revision { return Err(StateStoreError::conflict(expected_revision, actual)); }
                let revision = actual.unwrap_or(0).checked_add(1).and_then(|n| i64::try_from(n).ok())
                    .ok_or_else(|| StateStoreError::new("SQLite revision overflow").with_code("revision_overflow"))?;
                tx.execute("INSERT INTO agentscope_states (user_id, session_id, revision, state_json) VALUES (?1, ?2, ?3, ?4)
                    ON CONFLICT(user_id, session_id) DO UPDATE SET revision = excluded.revision, state_json = excluded.state_json",
                    params![key.user_id(), key.session_id(), revision, json]).map_err(sql_error)?;
                let record = StateRecord::new(revision.unsigned_abs(), state)?;
                tx.commit().map_err(sql_error)?;
                Ok(record)
            }).await.map_err(worker_error)
        })
    }
}

// Takes ownership to serve directly as a Result::map_err callback.
#[allow(clippy::needless_pass_by_value)]
fn sql_error(error: rusqlite::Error) -> StateStoreError {
    let retryable = matches!(&error, rusqlite::Error::SqliteFailure(code, _) if matches!(
        code.code, rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked));
    StateStoreError::new(error.to_string())
        .with_code("sqlite_error")
        .with_retryable(retryable)
}

fn worker_error(error: Error<StateStoreError>) -> StateStoreError {
    match error {
        Error::Error(error) => error,
        error => StateStoreError::new(error.to_string()).with_code("sqlite_worker_error"),
    }
}
