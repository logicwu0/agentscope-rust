//! Asynchronous SQLite-backed conversation memory.

use std::{fmt, path::Path, time::Duration};

use tokio_rusqlite::{Connection, Error as AsyncSqliteError, params, rusqlite};

use crate::Msg;

use super::{Memory, MemoryError, MemoryFuture, MemoryResult};

const SCHEMA_VERSION: i64 = 1;
const CREATE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS agentscope_memory_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version   INTEGER NOT NULL
);
INSERT OR IGNORE INTO agentscope_memory_schema (singleton, version) VALUES (1, 1);

CREATE TABLE IF NOT EXISTS agentscope_memory_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    message_id   TEXT NOT NULL,
    message_json TEXT NOT NULL,
    UNIQUE (session_id, message_id)
);
CREATE INDEX IF NOT EXISTS agentscope_memory_session_order
    ON agentscope_memory_messages (session_id, id);
";

/// Persistent conversation history for one session in a `SQLite` database.
#[derive(Clone)]
pub struct SQLiteMemory {
    connection: Connection,
    session_id: String,
}

impl SQLiteMemory {
    /// Opens or creates a `SQLite` database and selects one conversation session.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] when the session identifier is blank, the
    /// database cannot be opened, or its schema cannot be initialized.
    pub async fn open(path: impl AsRef<Path>, session_id: impl Into<String>) -> MemoryResult<Self> {
        let session_id = validate_session_id(session_id.into())?;
        let connection = Connection::open(path)
            .await
            .map_err(|error| map_sqlite_error("sqlite_open", &error))?;
        Self::initialize(connection, session_id).await
    }

    /// Opens a temporary in-memory `SQLite` database for one session.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] when the session identifier is blank or the
    /// database cannot be opened or initialized.
    pub async fn open_in_memory(session_id: impl Into<String>) -> MemoryResult<Self> {
        let session_id = validate_session_id(session_id.into())?;
        let connection = Connection::open_in_memory()
            .await
            .map_err(|error| map_sqlite_error("sqlite_open", &error))?;
        Self::initialize(connection, session_id).await
    }

    /// Returns the conversation session identifier.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn initialize(connection: Connection, session_id: String) -> MemoryResult<Self> {
        let version = connection
            .call(|database| {
                database.busy_timeout(Duration::from_secs(5))?;
                database.execute_batch(CREATE_SCHEMA)?;
                database.query_row(
                    "SELECT version FROM agentscope_memory_schema WHERE singleton = 1",
                    [],
                    |row| row.get::<_, i64>(0),
                )
            })
            .await
            .map_err(|error| map_async_error("sqlite_initialize", &error))?;
        if version != SCHEMA_VERSION {
            return Err(MemoryError::new(format!(
                "unsupported SQLite memory schema version {version}; expected {SCHEMA_VERSION}"
            ))
            .with_code("unsupported_schema_version"));
        }
        Ok(Self {
            connection,
            session_id,
        })
    }
}

impl Memory for SQLiteMemory {
    fn messages(&self) -> MemoryFuture<'_, Vec<Msg>> {
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        Box::pin(async move {
            let serialized = connection
                .call(move |database| {
                    let mut statement = database.prepare(
                        "SELECT message_json
                         FROM agentscope_memory_messages
                         WHERE session_id = ?1
                         ORDER BY id ASC",
                    )?;
                    statement
                        .query_map([session_id], |row| row.get::<_, String>(0))?
                        .collect::<rusqlite::Result<Vec<_>>>()
                })
                .await
                .map_err(|error| map_async_error("sqlite_read", &error))?;

            serialized
                .into_iter()
                .map(|message| {
                    serde_json::from_str(&message).map_err(|error| {
                        MemoryError::new(format!("stored message is invalid JSON: {error}"))
                            .with_code("invalid_message_json")
                    })
                })
                .collect()
        })
    }

    fn append(&self, messages: Vec<Msg>) -> MemoryFuture<'_, ()> {
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        Box::pin(async move {
            if messages.is_empty() {
                return Ok(());
            }
            let serialized = messages
                .into_iter()
                .map(|message| {
                    let message_id = message.id.clone();
                    serde_json::to_string(&message)
                        .map(|json| (message_id, json))
                        .map_err(|error| {
                            MemoryError::new(format!("message could not be serialized: {error}"))
                                .with_code("message_serialization")
                        })
                })
                .collect::<MemoryResult<Vec<_>>>()?;

            connection
                .call(move |database| {
                    let transaction = database
                        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                    {
                        let mut statement = transaction.prepare(
                            "INSERT INTO agentscope_memory_messages
                                (session_id, message_id, message_json)
                             VALUES (?1, ?2, ?3)",
                        )?;
                        for (message_id, message_json) in serialized {
                            statement.execute(params![session_id, message_id, message_json])?;
                        }
                    }
                    transaction.commit()
                })
                .await
                .map_err(|error| map_async_error("sqlite_write", &error))
        })
    }

    fn clear(&self) -> MemoryFuture<'_, ()> {
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        Box::pin(async move {
            connection
                .call(move |database| {
                    database.execute(
                        "DELETE FROM agentscope_memory_messages WHERE session_id = ?1",
                        [session_id],
                    )?;
                    Ok(())
                })
                .await
                .map_err(|error| map_async_error("sqlite_clear", &error))
        })
    }
}

impl fmt::Debug for SQLiteMemory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SQLiteMemory")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

fn validate_session_id(session_id: String) -> MemoryResult<String> {
    if session_id.trim().is_empty() {
        Err(MemoryError::new("memory session id cannot be empty").with_code("invalid_session_id"))
    } else {
        Ok(session_id)
    }
}

fn map_async_error(code: &str, error: &AsyncSqliteError<rusqlite::Error>) -> MemoryError {
    let retryable = match error {
        AsyncSqliteError::Error(error) => is_retryable(error),
        _ => false,
    };
    MemoryError::new(error.to_string())
        .with_code(code)
        .with_retryable(retryable)
}

fn map_sqlite_error(code: &str, error: &rusqlite::Error) -> MemoryError {
    MemoryError::new(error.to_string())
        .with_code(code)
        .with_retryable(is_retryable(error))
}

fn is_retryable(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(failure, _)
            if matches!(
                failure.code,
                rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
            )
    )
}
