//! Separate `SQLite` table for revisioned pipeline checkpoints.

use agentscope::{
    PipelineCheckpoint, PipelineRecord, PipelineStore, PipelineStoreError, PipelineStoreFuture,
    StateKey,
};
use std::{path::Path, time::Duration};
use tokio_rusqlite::{
    Connection, Error, params,
    rusqlite::{self, OptionalExtension, TransactionBehavior},
};

/// `SQLite` pipeline checkpoint storage, separate from agent state tables.
#[derive(Clone)]
pub struct SQLitePipelineStore {
    connection: Connection,
}

impl SQLitePipelineStore {
    /// Opens a database and initializes the pipeline checkpoint table.
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
                    "CREATE TABLE IF NOT EXISTS agentscope_pipeline_schema (
                        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                        version INTEGER NOT NULL);
                     INSERT OR IGNORE INTO agentscope_pipeline_schema VALUES (1, 1);",
                )
                .map_err(sql_error)?;
                let version: i64 = tx
                    .query_row(
                        "SELECT version FROM agentscope_pipeline_schema WHERE singleton = 1",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(sql_error)?;
                if version != 1 {
                    return Err(PipelineStoreError::new(format!(
                        "unsupported pipeline schema version {version}"
                    )));
                }
                tx.execute_batch(
                    "CREATE TABLE IF NOT EXISTS agentscope_pipeline_checkpoints (
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

impl PipelineStore for SQLitePipelineStore {
    fn load<'a>(&'a self, key: &'a StateKey) -> PipelineStoreFuture<'a, Option<PipelineRecord>> {
        let key = key.clone();
        Box::pin(async move {
            self.connection
                .call(move |db| {
                    let row: Option<(i64, String)> = db
                        .query_row(
                            "SELECT revision, checkpoint_json FROM agentscope_pipeline_checkpoints
                             WHERE user_id = ?1 AND session_id = ?2",
                            params![key.user_id(), key.session_id()],
                            |row| Ok((row.get(0)?, row.get(1)?)),
                        )
                        .optional()
                        .map_err(sql_error)?;
                    row.map(|(revision, json)| {
                        let revision = u64::try_from(revision)
                            .ok()
                            .filter(|revision| *revision > 0)
                            .ok_or_else(|| PipelineStoreError::new("invalid pipeline revision"))?;
                        let checkpoint = serde_json::from_str(&json)
                            .map_err(|error| PipelineStoreError::new(error.to_string()))?;
                        Ok(PipelineRecord {
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
        checkpoint: PipelineCheckpoint,
    ) -> PipelineStoreFuture<'_, PipelineRecord> {
        Box::pin(async move {
            let json = serde_json::to_string(&checkpoint)
                .map_err(|error| PipelineStoreError::new(error.to_string()))?;
            self.connection
                .call(move |db| {
                    let tx = db
                        .transaction_with_behavior(TransactionBehavior::Immediate)
                        .map_err(sql_error)?;
                    let actual: Option<u64> = tx
                        .query_row(
                            "SELECT revision FROM agentscope_pipeline_checkpoints
                             WHERE user_id = ?1 AND session_id = ?2",
                            params![key.user_id(), key.session_id()],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(sql_error)?;
                    if actual != expected_revision {
                        return Err(PipelineStoreError::new(format!(
                            "pipeline revision conflict: expected {expected_revision:?}, actual {actual:?}"
                        )));
                    }
                    let revision = actual
                        .unwrap_or(0)
                        .checked_add(1)
                        .and_then(|revision| i64::try_from(revision).ok())
                        .ok_or_else(|| PipelineStoreError::new("pipeline revision overflow"))?;
                    tx.execute(
                        "INSERT INTO agentscope_pipeline_checkpoints
                            (user_id, session_id, revision, checkpoint_json)
                         VALUES (?1, ?2, ?3, ?4)
                         ON CONFLICT(user_id, session_id) DO UPDATE SET
                            revision = excluded.revision,
                            checkpoint_json = excluded.checkpoint_json",
                        params![key.user_id(), key.session_id(), revision, json],
                    )
                    .map_err(sql_error)?;
                    tx.commit().map_err(sql_error)?;
                    Ok(PipelineRecord {
                        revision: revision.unsigned_abs(),
                        checkpoint,
                    })
                })
                .await
                .map_err(worker_error)
        })
    }
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
