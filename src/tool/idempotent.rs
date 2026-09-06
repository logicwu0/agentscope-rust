//! Bounded, process-local deduplication of side-effecting tool invocations.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;

use super::{TOOL_IDEMPOTENCY_KEY, Tool, ToolContext, ToolError, ToolFuture, ToolResult};
use crate::{ToolDefinition, ToolResultOutput};

const DEFAULT_CAPACITY: usize = 1024;

/// An opt-in, process-local idempotency wrapper around one tool.
///
/// Identical keys, input, and metadata reuse one terminal result (including
/// errors). Concurrent duplicates wait for the first invocation. Cancellation
/// or panic during execution leaves an uncertain record that blocks reexecution.
/// Missing keys pass through normally; blank or non-string keys are rejected.
///
/// Clones share records. Independent wrappers and process restarts do not.
/// Records are never evicted: once capacity is reached, new keyed operations
/// fail closed. This is not a durable exactly-once guarantee; external services
/// should also honor the key for recovery across process failures.
#[derive(Clone)]
pub struct IdempotentTool {
    inner: Arc<dyn Tool>,
    records: Arc<Mutex<BTreeMap<String, Arc<Record>>>>,
    capacity: usize,
}

struct Record {
    input: Value,
    context: ToolContext,
    state: AsyncMutex<ExecutionState>,
}

enum ExecutionState {
    Ready,
    Uncertain,
    Completed(ToolResult<ToolResultOutput>),
}

impl IdempotentTool {
    /// Wraps a tool with a maximum of 1024 remembered keys.
    #[must_use]
    pub fn new<T: Tool + 'static>(tool: T) -> Self {
        Self::from_shared(Arc::new(tool))
    }

    /// Wraps a shared tool with a maximum of 1024 remembered keys.
    #[must_use]
    pub fn from_shared(tool: Arc<dyn Tool>) -> Self {
        Self::with_capacity(tool, DEFAULT_CAPACITY)
    }

    /// Creates a wrapper with a fixed record limit. Zero rejects all keyed calls.
    #[must_use]
    pub fn with_capacity(tool: Arc<dyn Tool>, capacity: usize) -> Self {
        Self {
            inner: tool,
            records: Arc::new(Mutex::new(BTreeMap::new())),
            capacity,
        }
    }

    fn record(&self, key: &str, input: &Value, context: &ToolContext) -> ToolResult<Arc<Record>> {
        let mut records = self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(record) = records.get(key) {
            if record.input != *input || record.context != *context {
                return Err(ToolError::new(
                    "idempotency key was reused with different input or context",
                )
                .with_code("idempotency_conflict"));
            }
            return Ok(record.clone());
        }
        if records.len() >= self.capacity {
            return Err(ToolError::new("idempotency record capacity reached")
                .with_code("idempotency_capacity_exceeded"));
        }
        let record = Arc::new(Record {
            input: input.clone(),
            context: context.clone(),
            state: AsyncMutex::new(ExecutionState::Ready),
        });
        records.insert(key.to_owned(), record.clone());
        Ok(record)
    }
}

impl Tool for IdempotentTool {
    fn definition(&self) -> &ToolDefinition {
        self.inner.definition()
    }

    fn execute(&self, input: Value, context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        Box::pin(async move {
            let Some(value) = context.metadata.get(TOOL_IDEMPOTENCY_KEY) else {
                return self.inner.execute(input, context).await;
            };
            let key = value
                .as_str()
                .filter(|key| !key.trim().is_empty())
                .ok_or_else(|| {
                    ToolError::new("idempotency key must be a non-blank string")
                        .with_code("invalid_idempotency_key")
                })?;
            let record = self.record(key, &input, &context)?;
            let mut state = record.state.lock().await;
            match &*state {
                ExecutionState::Completed(result) => return result.clone(),
                ExecutionState::Uncertain => return Err(ToolError::new(
                    "previous invocation was cancelled or panicked; reconcile its external outcome",
                )
                .with_code("idempotency_in_doubt")),
                ExecutionState::Ready => {}
            }
            // Dropping the future releases the mutex but retains this marker.
            *state = ExecutionState::Uncertain;
            let result = self.inner.execute(input, context).await;
            *state = ExecutionState::Completed(result.clone());
            result
        })
    }
}

#[cfg(test)]
mod tests;
