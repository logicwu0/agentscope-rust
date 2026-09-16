//! Explicitly launched local MCP servers, adapted to `AgentScope` tools.
//! Only the stdio tools subset is supported; no HTTP, sampling, roots or OAuth.
#![forbid(unsafe_code)]

mod tool;
mod transport;

use agentscope::{ToolError, ToolRegistry, ToolResult};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    path::PathBuf,
    sync::Arc,
    time::Duration,
};
use tokio::sync::Mutex;
pub use tool::McpTool;
use transport::Session;

/// Launch configuration. No shell is used and no environment variables are
/// inherited. Only start a trusted, explicitly selected executable; this is not
/// a sandbox. Arguments and environment may contain secrets, so no Debug/serde.
pub struct StdioConfig {
    /// Trusted executable. Prefer an absolute path to avoid search-path ambiguity.
    pub program: PathBuf,
    /// Literal arguments; never evaluated by a shell.
    pub args: Vec<OsString>,
    /// Explicit environment allowlist, empty by default.
    pub env: BTreeMap<OsString, OsString>,
    /// Optional working directory, not a filesystem access restriction.
    pub current_dir: Option<PathBuf>,
    /// Per-request deadline, also applied separately to waiting for the connection.
    pub timeout: Duration,
    /// Grace period after closing stdin, before killing the direct child.
    pub shutdown_timeout: Duration,
    /// Maximum UTF-8 JSON line size in either direction (excluding newline).
    pub max_frame_bytes: usize,
    /// Total number of tools allowed across discovery pages.
    pub max_tools: usize,
    /// Opt-in terminal logging. Default false; stderr is discarded, never parsed.
    pub inherit_stderr: bool,
}

impl StdioConfig {
    /// Creates conservative defaults without launching a process.
    #[must_use]
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            current_dir: None,
            timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(2),
            max_frame_bytes: 4 * 1024 * 1024,
            max_tools: 1024,
            inherit_stderr: false,
        }
    }
}

/// Shared live connection. Calls are serialized, including concurrent tool
/// batches. Drop all clients/tools to kill the direct child, or call `close` to
/// close stdin and wait first. No automatic retries, reconnect or process restart.
#[derive(Clone)]
pub struct McpClient {
    session: Arc<Mutex<Option<Session>>>,
    timeout: Duration,
    shutdown_timeout: Duration,
    max_tools: usize,
    protocol_version: String,
}

pub(crate) fn error(code: &str, message: &str) -> ToolError {
    ToolError::new(message).with_code(code)
}

impl McpClient {
    /// Spawns a trusted server and completes version/capability negotiation.
    /// Supports protocol 2025-11-25 and 2025-06-18, requiring tools capability.
    /// # Errors
    /// Invalid configuration, spawn failure, negotiation failure or timeout.
    pub async fn connect(config: StdioConfig) -> ToolResult<Self> {
        if config.program.as_os_str().is_empty()
            || config.timeout.is_zero()
            || config.shutdown_timeout.is_zero()
            || config.max_frame_bytes < 1024
            || config.max_tools == 0
        {
            return Err(error("mcp_config", "invalid MCP stdio configuration"));
        }
        let session = Session::spawn(&config)?;
        let mut client = Self {
            session: Arc::new(Mutex::new(Some(session))),
            timeout: config.timeout,
            shutdown_timeout: config.shutdown_timeout,
            max_tools: config.max_tools,
            protocol_version: String::new(),
        };
        let result = client
            .request(
                "initialize",
                json!({"protocolVersion":"2025-11-25", "capabilities":{},
            "clientInfo":{"name":"agentscope-rust","version":env!("CARGO_PKG_VERSION")}}),
            )
            .await?;
        let version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .filter(|v| matches!(*v, "2025-11-25" | "2025-06-18"))
            .ok_or_else(|| error("mcp_version", "unsupported MCP protocol version"))?;
        if !result
            .pointer("/capabilities/tools")
            .is_some_and(Value::is_object)
            || !result
                .pointer("/serverInfo/name")
                .is_some_and(Value::is_string)
            || !result
                .pointer("/serverInfo/version")
                .is_some_and(Value::is_string)
        {
            return Err(error(
                "mcp_capabilities",
                "MCP server must provide tools capability and server info",
            ));
        }
        client.protocol_version = version.into();
        client.notify_initialized().await?;
        Ok(client)
    }

    #[must_use]
    pub fn protocol_version(&self) -> &str {
        &self.protocol_version
    }

    /// Lists all pages and creates tools named `<namespace>__<remote_name>`.
    /// Names must be ASCII alphanumeric/underscore/hyphen, at most 64 bytes after
    /// namespacing. Rejects collisions, invalid schemas, loops and excessive lists.
    /// Server annotations/instructions never grant permissions or modify prompts.
    /// # Errors
    /// Discovery, schema or name validation failure. No partial list is returned.
    pub async fn tools(&self, namespace: &str) -> ToolResult<Vec<McpTool>> {
        if !tool::valid_name(namespace) {
            return Err(error("mcp_name", "invalid MCP namespace"));
        }
        let mut tools = Vec::new();
        let mut cursors = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut params = json!({});
        for _ in 0..128 {
            let page = self.request("tools/list", params).await?;
            let entries = page
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| error("mcp_protocol", "invalid MCP tool list"))?;
            if entries.len() > self.max_tools.saturating_sub(tools.len()) {
                return Err(error(
                    "mcp_limit",
                    "MCP tool count exceeds configured limit",
                ));
            }
            for entry in entries {
                let tool = McpTool::from_wire(self.clone(), namespace, entry)?;
                if !names.insert(tool.remote_name().to_owned()) {
                    return Err(error("mcp_name", "duplicate MCP tool name"));
                }
                tools.push(tool);
            }
            match page.get("nextCursor") {
                None => return Ok(tools),
                Some(Value::String(cursor))
                    if !cursor.is_empty() && cursors.insert(cursor.clone()) =>
                {
                    params = json!({"cursor":cursor});
                }
                _ => return Err(error("mcp_protocol", "invalid or repeated MCP list cursor")),
            }
        }
        Err(error("mcp_limit", "too many MCP tool discovery pages"))
    }

    /// Creates a new registry after fully validated discovery. Existing registries
    /// can instead register tools returned by `tools` individually.
    /// # Errors
    /// Discovery or registration failure.
    pub async fn registry(&self, namespace: &str) -> ToolResult<ToolRegistry> {
        let mut registry = ToolRegistry::new();
        for tool in self.tools(namespace).await? {
            registry.register(tool)?;
        }
        Ok(registry)
    }

    async fn notify_initialized(&self) -> ToolResult<()> {
        let mut guard = self.session.lock().await;
        let mut session = guard
            .take()
            .ok_or_else(|| error("mcp_closed", "MCP connection closed"))?;
        tokio::time::timeout(self.timeout, session.notify("notifications/initialized"))
            .await
            .map_err(|_| error("mcp_timeout", "MCP initialization notification timed out"))??;
        *guard = Some(session);
        Ok(())
    }

    pub(crate) async fn request(&self, method: &str, params: Value) -> ToolResult<Value> {
        let mut guard = tokio::time::timeout(self.timeout, self.session.lock())
            .await
            .map_err(|_| error("mcp_queue_timeout", "MCP request timed out before dispatch"))?;
        let mut session = guard
            .take()
            .ok_or_else(|| error("mcp_closed", "MCP connection closed; reconnect explicitly"))?;
        let result = tokio::time::timeout(self.timeout, session.exchange(method, params)).await;
        match result {
            Ok(Ok(response)) => {
                *guard = Some(session);
                response
            }
            Ok(Err(e)) => Err(if method == "tools/call" {
                error(
                    "tool_execution_in_doubt",
                    "MCP transport failed after dispatch may have begun; execution outcome unknown, do not automatically retry",
                )
            } else {
                e
            }),
            Err(_) => Err(error(
                if method == "tools/call" {
                    "tool_execution_in_doubt"
                } else {
                    "mcp_timeout"
                },
                "MCP request timed out; connection closed, execution may have occurred; no automatic retry",
            )),
        }
        // On timeout, transport error or future cancellation the taken session
        // drops (kills child, aborts reader), and the shared slot remains closed.
    }

    /// Closes this connection for all clones and tool adapters. Pending calls are
    /// allowed up to their deadline; then close stdin, wait, and kill if needed.
    /// Only the direct child is managed, not arbitrary grandchildren.
    /// # Errors
    /// Lock wait or process shutdown failure.
    pub async fn close(&self) -> ToolResult<()> {
        let mut guard = tokio::time::timeout(self.timeout, self.session.lock())
            .await
            .map_err(|_| {
                error(
                    "mcp_queue_timeout",
                    "MCP close waiting for active request timed out",
                )
            })?;
        if let Some(mut session) = guard.take() {
            session.close(self.shutdown_timeout).await?;
        }
        Ok(())
    }
}
