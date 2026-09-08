//! Storage-neutral durable idempotency contracts and tool adapter.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{TOOL_IDEMPOTENCY_KEY, Tool, ToolContext, ToolError, ToolFuture, ToolResult};
use crate::{ToolDefinition, ToolResultOutput};

/// The identity and complete request bound to an idempotency record.
/// Choose a stable namespace per tenant and tool implementation version.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IdempotencyRequest {
    namespace: String,
    tool_name: String,
    input: Value,
    context: ToolContext,
}

impl IdempotencyRequest {
    /// Creates a request carrying a non-blank idempotency key in its context.
    ///
    /// # Errors
    /// Returns an error for blank identity or a missing/invalid key.
    pub fn new(
        namespace: impl Into<String>,
        tool_name: impl Into<String>,
        input: Value,
        context: ToolContext,
    ) -> ToolResult<Self> {
        let request = Self {
            namespace: namespace.into(),
            tool_name: tool_name.into(),
            input,
            context,
        };
        request.validate()?;
        Ok(request)
    }

    /// Validates identities, including values deserialized from storage.
    ///
    /// # Errors
    /// Returns an error for blank identity or a missing/invalid key.
    pub fn validate(&self) -> ToolResult<()> {
        if self.namespace.trim().is_empty() || self.tool_name.trim().is_empty() {
            return Err(
                ToolError::new("idempotency namespace and tool name must not be blank")
                    .with_code("invalid_idempotency_identity"),
            );
        }
        if self
            .context
            .idempotency_key()
            .is_none_or(|key| key.trim().is_empty())
        {
            return Err(ToolError::new("idempotency key must be a non-blank string")
                .with_code("invalid_idempotency_key"));
        }
        Ok(())
    }

    /// Returns the caller-selected tenant/version namespace.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns the tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Returns the key, or an empty string for invalid deserialized requests.
    #[must_use]
    pub fn key(&self) -> &str {
        self.context.idempotency_key().unwrap_or_default()
    }
}

/// Outcome of atomically claiming a durable invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IdempotencyClaim {
    /// This caller owns execution; completion must supply this opaque token.
    Acquired { token: String },
    /// A terminal result is already available and must be reused.
    Completed {
        result: ToolResult<ToolResultOutput>,
    },
    /// Execution is active or its external outcome is unknown. Do not execute.
    InDoubt,
}

/// Durable, object-safe coordination for tool execution.
///
/// Implementations must atomically bind a key to the full request, durably claim
/// it before returning `Acquired`, and reject mismatched requests. Active claims
/// must never expire into automatic execution. Completed results are immutable.
pub trait IdempotencyStore: Send + Sync {
    /// Claims a new invocation or returns its existing status/result.
    fn claim(&self, request: IdempotencyRequest) -> ToolFuture<'_, IdempotencyClaim>;

    /// Durably records an owner's terminal result; stale tokens must be rejected.
    fn complete(
        &self,
        request: IdempotencyRequest,
        token: String,
        result: ToolResult<ToolResultOutput>,
    ) -> ToolFuture<'_, ()>;

    /// Explicitly supplies a verified external outcome for an existing record.
    /// The application must first ensure its original worker is stopped or fenced.
    /// Unknown requests and attempts to change completed results must be rejected.
    fn reconcile(
        &self,
        request: IdempotencyRequest,
        result: ToolResult<ToolResultOutput>,
    ) -> ToolFuture<'_, ()>;
}

/// Deduplicates tool invocations using a caller-selected durable store.
///
/// Missing keys pass through. Errors as well as successes are cached. Concurrent
/// duplicates of unfinished calls return `idempotency_in_doubt` without waiting.
/// Reuse the same namespace/store across restarts. This does not make external
/// effects transactional with the store; reconcile interrupted executions.
#[derive(Clone)]
pub struct PersistentIdempotentTool {
    namespace: String,
    inner: Arc<dyn Tool>,
    store: Arc<dyn IdempotencyStore>,
}

impl PersistentIdempotentTool {
    /// Wraps an owned tool and storage backend.
    ///
    /// # Errors
    /// Returns an error if the namespace is blank.
    pub fn new<T: Tool + 'static, S: IdempotencyStore + 'static>(
        namespace: impl Into<String>,
        tool: T,
        store: S,
    ) -> ToolResult<Self> {
        Self::from_shared(namespace, Arc::new(tool), Arc::new(store))
    }

    /// Wraps shared tool and storage trait objects.
    ///
    /// # Errors
    /// Returns an error if the namespace is blank.
    pub fn from_shared(
        namespace: impl Into<String>,
        inner: Arc<dyn Tool>,
        store: Arc<dyn IdempotencyStore>,
    ) -> ToolResult<Self> {
        let namespace = namespace.into();
        if namespace.trim().is_empty() {
            return Err(ToolError::new("idempotency namespace must not be blank")
                .with_code("invalid_idempotency_identity"));
        }
        Ok(Self {
            namespace,
            inner,
            store,
        })
    }
}

impl Tool for PersistentIdempotentTool {
    fn definition(&self) -> &ToolDefinition {
        self.inner.definition()
    }

    fn execute(&self, input: Value, context: ToolContext) -> ToolFuture<'_, ToolResultOutput> {
        Box::pin(async move {
            if !context.metadata.contains_key(TOOL_IDEMPOTENCY_KEY) {
                return self.inner.execute(input, context).await;
            }
            let request = IdempotencyRequest::new(
                &self.namespace,
                &self.definition().name,
                input.clone(),
                context.clone(),
            )?;
            // Even a failed claim may have committed before the connection failed.
            let claim = self.store.claim(request.clone()).await.map_err(|error| {
                if error.code.as_deref() == Some("idempotency_conflict") {
                    error
                } else {
                    in_doubt()
                }
            })?;
            let token = match claim {
                IdempotencyClaim::Acquired { token } => token,
                IdempotencyClaim::Completed { result } => return result,
                IdempotencyClaim::InDoubt => return Err(in_doubt()),
            };
            let result = self.inner.execute(input, context).await;
            if result
                .as_ref()
                .is_err_and(|error| error.code.as_deref() == Some("idempotency_in_doubt"))
            {
                return result;
            }
            self.store
                .complete(request, token, result.clone())
                .await
                .map_err(|_| in_doubt())?;
            result
        })
    }
}

fn in_doubt() -> ToolError {
    ToolError::new("durable tool execution is active or uncertain; reconcile its external outcome")
        .with_code("idempotency_in_doubt")
}
