//! Independent `SQLite` table for revisioned parallel branch checkpoints.

use agentscope::{
    ParallelCheckpoint, ParallelRecord, ParallelStore, PipelineStoreError, PipelineStoreFuture,
    StateKey,
};
use std::{path::Path, time::Duration};
use tokio_rusqlite::{
    Connection, Error, params,
    rusqlite::{self, OptionalExtension, TransactionBehavior},
};

/// Parallel checkpoint storage, isolated from agent and sequential state tables.
/// Clones share a connection; independent connections serialize revision checks
/// and writes using immediate transactions.
#[derive(Clone)]
pub struct SQLiteParallelStore {
    connection: Connection,
}

impl SQLiteParallelStore {
    /// Opens a database and atomically initializes the parallel checkpoint schema.
    /// # Errors
    /// Returns an error for inaccessible databases or unsupported schemas.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, PipelineStoreError> {
        let connection = Connection::open(path).await.map_err(sql_error)?;
        connection
            .call(|db| {
                db.busy_timeout(Duration::from_secs(5)).map_err(sql_error)?;
                let tx = db
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sql_error)?;
                tx.execute_batch(
                    "CREATE TABLE IF NOT EXISTS agentscope_parallel_schema (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        version INTEGER NOT NULL);
                     INSERT OR IGNORE INTO agentscope_parallel_schema VALUES (1, 1);",
                )
                .map_err(sql_error)?;
                let version: i64 = tx
                    .query_row(
                        "SELECT version FROM agentscope_parallel_schema WHERE singleton = 1",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(sql_error)?;
                if version != 1 {
                    return Err(PipelineStoreError::new(format!(
                        "unsupported parallel schema version {version}"
                    )));
                }
                tx.execute_batch(
                    "CREATE TABLE IF NOT EXISTS agentscope_parallel_checkpoints (
                        user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK(revision > 0),
                        checkpoint_json TEXT NOT NULL,
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

impl ParallelStore for SQLiteParallelStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<ParallelRecord>> {
        let key = key.clone();
        Box::pin(async move {
            self.connection
                .call(move |db| {
                    let row: Option<(i64, String)> = db
                        .query_row(
                            "SELECT revision, checkpoint_json FROM agentscope_parallel_checkpoints
                             WHERE user_id = ?1 AND session_id = ?2",
                            params![key.user_id(), key.session_id()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        )
                        .optional()
                        .map_err(sql_error)?;
                    row.map(|(revision, json)| {
                        let revision = positive_revision(revision)?;
                        let checkpoint = serde_json::from_str(&json)
                            .map_err(|error| PipelineStoreError::new(error.to_string()))?;
                        Ok(ParallelRecord {
                            revision,
                            checkpoint,
                        })
                    })
                    .transpose()
                })
                .await
                .map_err(worker_error)
        })
    }

    fn save(
        &self,
        key: StateKey,
        expected_revision: Option<u64>,
        checkpoint: ParallelCheckpoint,
    ) -> PipelineStoreFuture<'_, ParallelRecord> {
        Box::pin(async move {
            let json = serde_json::to_string(&checkpoint)
                .map_err(|error| PipelineStoreError::new(error.to_string()))?;
            self.connection
                .call(move |db| {
                    let tx = db
                        .transaction_with_behavior(TransactionBehavior::Immediate)
                        .map_err(sql_error)?;
                    let actual: Option<i64> = tx
                        .query_row(
                            "SELECT revision FROM agentscope_parallel_checkpoints
                             WHERE user_id = ?1 AND session_id = ?2",
                            params![key.user_id(), key.session_id()],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(sql_error)?;
                    let actual = actual.map(positive_revision).transpose()?;
                    if actual != expected_revision {
                        return Err(PipelineStoreError::new(format!(
                            "parallel revision conflict: expected {expected_revision:?}, actual {actual:?}"
                        )));
                    }
                    let revision = actual
                        .unwrap_or(0)
                        .checked_add(1)
                        .and_then(|revision| i64::try_from(revision).ok())
                        .ok_or_else(|| PipelineStoreError::new("parallel revision overflow"))?;
                    tx.execute(
                        "INSERT INTO agentscope_parallel_checkpoints
                            (user_id, session_id, revision, checkpoint_json)
                         VALUES (?1, ?2, ?3, ?4)
                         ON CONFLICT(user_id, session_id) DO UPDATE SET
                            revision = excluded.revision,
                            checkpoint_json = excluded.checkpoint_json",
                        params![key.user_id(), key.session_id(), revision, json],
                    )
                    .map_err(sql_error)?;
                    tx.commit().map_err(sql_error)?;
                    Ok(ParallelRecord {
                        revision: revision.unsigned_abs(),
                        checkpoint,
                    })
                })
                .await
                .map_err(worker_error)
        })
    }
}

fn positive_revision(revision: i64) -> Result<u64, PipelineStoreError> {
    u64::try_from(revision)
        .ok()
        .filter(|revision| *revision > 0)
        .ok_or_else(|| PipelineStoreError::new("invalid parallel revision"))
}

#[allow(clippy::needless_pass_by_value)]
fn sql_error(error: rusqlite::Error) -> PipelineStoreError {
    PipelineStoreError::new(error.to_string())
}

fn worker_error(error: Error<PipelineStoreError>) -> PipelineStoreError {
    match error {
        Error::Error(error) => error,
        error => PipelineStoreError::new(error.to_string()),
    }
}
