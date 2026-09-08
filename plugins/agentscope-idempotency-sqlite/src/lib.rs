//! Durable `SQLite` idempotency records for `AgentScope` tools.

use std::{path::Path, time::Duration};

use agentscope::{
    IdempotencyClaim, IdempotencyRequest, IdempotencyStore, ToolError, ToolFuture, ToolResult,
    ToolResultOutput,
};
use tokio_rusqlite::{
    Connection, Error, params,
    rusqlite::{self, OptionalExtension, TransactionBehavior},
};
use uuid::Uuid;

/// Durable tool deduplication, separate from agent-state storage.
///
/// Claims commit before tools execute. Unfinished claims never expire or rerun.
/// Records are not automatically deleted; namespace changes create new identities.
/// The database contains full inputs, context metadata, and results.
#[derive(Clone)]
pub struct SQLiteIdempotencyStore {
    connection: Connection,
}

struct Row {
    request: String,
    owner: String,
    result: Option<String>,
}

impl SQLiteIdempotencyStore {
    /// Opens a database and initializes the plugin's versioned tables.
    ///
    /// # Errors
    /// Returns an error for storage failures or unsupported schema versions.
    pub async fn open(path: impl AsRef<Path>) -> ToolResult<Self> {
        let connection = Connection::open(path).await.map_err(sql_error)?;
        connection
            .call(|db| {
                db.busy_timeout(Duration::from_secs(5)).map_err(sql_error)?;
                let tx = db
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sql_error)?;
                tx.execute_batch(
                    "CREATE TABLE IF NOT EXISTS agentscope_idempotency_schema (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1), version INTEGER NOT NULL);
                INSERT OR IGNORE INTO agentscope_idempotency_schema VALUES (1, 1);",
                )
                .map_err(sql_error)?;
                let version: i64 = tx
                    .query_row(
                        "SELECT version FROM agentscope_idempotency_schema WHERE singleton = 1",
                        [],
                        |row| row.get(0),
                    )
                    .map_err(sql_error)?;
                if version != 1 {
                    return Err(ToolError::new(format!(
                        "unsupported SQLite idempotency schema version {version}"
                    ))
                    .with_code("unsupported_schema_version"));
                }
                tx.execute_batch(
                    "CREATE TABLE IF NOT EXISTS agentscope_idempotency (
                namespace TEXT NOT NULL, tool_name TEXT NOT NULL, call_key TEXT NOT NULL,
                request_json TEXT NOT NULL, owner TEXT NOT NULL, result_json TEXT,
                PRIMARY KEY(namespace, tool_name, call_key));",
                )
                .map_err(sql_error)?;
                tx.commit().map_err(sql_error)
            })
            .await
            .map_err(worker_error)?;
        Ok(Self { connection })
    }

    fn finish(
        &self,
        request: IdempotencyRequest,
        token: Option<String>,
        result: ToolResult<ToolResultOutput>,
    ) -> ToolFuture<'_, ()> {
        Box::pin(async move {
            request.validate()?;
            if result
                .as_ref()
                .is_err_and(|error| error.code.as_deref() == Some("idempotency_in_doubt"))
            {
                return Err(ToolError::new(
                    "an uncertain outcome cannot complete an idempotency record",
                )
                .with_code("idempotency_in_doubt"));
            }
            let encoded = serde_json::to_string(&result).map_err(json_error)?;
            self.connection
                .call(move |db| {
                    let tx = db
                        .transaction_with_behavior(TransactionBehavior::Immediate)
                        .map_err(sql_error)?;
                    let row = read_row(&tx, &request)?.ok_or_else(|| {
                        ToolError::new("idempotency record does not exist")
                            .with_code("idempotency_not_found")
                    })?;
                    validate_request(&row, &request)?;
                    if token.as_ref().is_some_and(|token| *token != row.owner) {
                        return Err(ToolError::new("stale idempotency owner token")
                            .with_code("idempotency_stale_owner"));
                    }
                    if let Some(stored) = row.result {
                        let stored: ToolResult<ToolResultOutput> =
                            serde_json::from_str(&stored).map_err(json_error)?;
                        if stored != result {
                            return Err(conflict(
                                "completed idempotency results cannot be changed",
                            ));
                        }
                    } else {
                        tx.execute(
                            "UPDATE agentscope_idempotency SET result_json = ?4
                        WHERE namespace = ?1 AND tool_name = ?2 AND call_key = ?3",
                            params![
                                request.namespace(),
                                request.tool_name(),
                                request.key(),
                                encoded
                            ],
                        )
                        .map_err(sql_error)?;
                    }
                    tx.commit().map_err(sql_error)
                })
                .await
                .map_err(worker_error)
        })
    }
}

impl IdempotencyStore for SQLiteIdempotencyStore {
    fn claim(&self, request: IdempotencyRequest) -> ToolFuture<'_, IdempotencyClaim> {
        Box::pin(async move {
            request.validate()?;
            let encoded = serde_json::to_string(&request).map_err(json_error)?;
            self.connection.call(move |db| {
                let tx = db.transaction_with_behavior(TransactionBehavior::Immediate).map_err(sql_error)?;
                let claim = if let Some(row) = read_row(&tx, &request)? {
                    validate_request(&row, &request)?;
                    match row.result {
                        Some(json) => IdempotencyClaim::Completed { result: serde_json::from_str(&json).map_err(json_error)? },
                        None => IdempotencyClaim::InDoubt,
                    }
                } else {
                    let token = Uuid::new_v4().simple().to_string();
                    tx.execute("INSERT INTO agentscope_idempotency (namespace, tool_name, call_key, request_json, owner)
                        VALUES (?1, ?2, ?3, ?4, ?5)", params![request.namespace(), request.tool_name(), request.key(), encoded, token]).map_err(sql_error)?;
                    IdempotencyClaim::Acquired { token }
                };
                tx.commit().map_err(sql_error)?;
                Ok(claim)
            }).await.map_err(worker_error)
        })
    }

    fn complete(
        &self,
        request: IdempotencyRequest,
        token: String,
        result: ToolResult<ToolResultOutput>,
    ) -> ToolFuture<'_, ()> {
        self.finish(request, Some(token), result)
    }

    fn reconcile(
        &self,
        request: IdempotencyRequest,
        result: ToolResult<ToolResultOutput>,
    ) -> ToolFuture<'_, ()> {
        self.finish(request, None, result)
    }
}

fn read_row(db: &rusqlite::Connection, request: &IdempotencyRequest) -> ToolResult<Option<Row>> {
    db.query_row(
        "SELECT request_json, owner, result_json FROM agentscope_idempotency
        WHERE namespace = ?1 AND tool_name = ?2 AND call_key = ?3",
        params![request.namespace(), request.tool_name(), request.key()],
        |row| {
            Ok(Row {
                request: row.get(0)?,
                owner: row.get(1)?,
                result: row.get(2)?,
            })
        },
    )
    .optional()
    .map_err(sql_error)
}

fn validate_request(row: &Row, request: &IdempotencyRequest) -> ToolResult<()> {
    let stored: IdempotencyRequest = serde_json::from_str(&row.request).map_err(json_error)?;
    stored.validate()?;
    if stored != *request {
        return Err(conflict(
            "idempotency key was reused with different input or context",
        ));
    }
    Ok(())
}

fn conflict(message: &str) -> ToolError {
    ToolError::new(message).with_code("idempotency_conflict")
}

// Ownership matches the Result::map_err callback signature.
#[allow(clippy::needless_pass_by_value)]
fn sql_error(error: rusqlite::Error) -> ToolError {
    ToolError::new(error.to_string()).with_code("idempotency_storage_error")
}

#[allow(clippy::needless_pass_by_value)]
fn json_error(error: serde_json::Error) -> ToolError {
    ToolError::new(error.to_string()).with_code("invalid_idempotency_json")
}

fn worker_error(error: Error<ToolError>) -> ToolError {
    match error {
        Error::Error(error) => error,
        error => ToolError::new(error.to_string()).with_code("idempotency_storage_error"),
    }
}
