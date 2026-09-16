//! Bounded newline JSON-RPC framing and child ownership.
use super::{StdioConfig, error};
use agentscope::{ToolError, ToolResult};
use serde_json::{Value, json};
use std::{process::Stdio, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Mutex, mpsc},
    task::JoinHandle,
};

type Input = Arc<Mutex<Option<ChildStdin>>>;

pub(crate) struct Session {
    child: Child,
    input: Input,
    responses: mpsc::Receiver<ToolResult<Value>>,
    reader: JoinHandle<()>,
    next_id: u64,
    max_frame: usize,
}

impl Session {
    pub(crate) fn spawn(config: &StdioConfig) -> ToolResult<Self> {
        let mut command = Command::new(&config.program);
        command
            .args(&config.args)
            .env_clear()
            .envs(&config.env)
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(if config.inherit_stderr {
                Stdio::inherit()
            } else {
                Stdio::null()
            });
        if let Some(dir) = &config.current_dir {
            command.current_dir(dir);
        }
        let mut child = command
            .spawn()
            .map_err(|_| error("mcp_spawn", "cannot start MCP server"))?;
        let input = Arc::new(Mutex::new(child.stdin.take()));
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| error("mcp_spawn", "MCP stdout unavailable"))?;
        let (tx, responses) = mpsc::channel(8);
        let reader_input = input.clone();
        let max_frame = config.max_frame_bytes;
        let reader = tokio::spawn(async move {
            if let Err(e) = read_loop(stdout, &reader_input, &tx, max_frame).await {
                let _ = tx.send(Err(e)).await;
            }
        });
        Ok(Self {
            child,
            input,
            responses,
            reader,
            next_id: 1,
            max_frame,
        })
    }

    pub(crate) async fn notify(&mut self, method: &str) -> ToolResult<()> {
        write(
            &self.input,
            &json!({"jsonrpc":"2.0","method":method}),
            self.max_frame,
        )
        .await
    }

    // Outer errors invalidate the transport; inner errors are valid RPC errors.
    pub(crate) async fn exchange(
        &mut self,
        method: &str,
        params: Value,
    ) -> ToolResult<ToolResult<Value>> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| error("mcp_limit", "MCP request IDs exhausted"))?;
        write(
            &self.input,
            &json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}),
            self.max_frame,
        )
        .await?;
        let response = self
            .responses
            .recv()
            .await
            .ok_or_else(|| error("mcp_closed", "MCP server disconnected"))??;
        if response.get("id").and_then(Value::as_u64) != Some(id) {
            return Err(error("mcp_protocol", "unexpected MCP response ID"));
        }
        match (response.get("result"), response.get("error")) {
            (Some(result), None) if result.is_object() => Ok(Ok(result.clone())),
            (None, Some(err))
                if err.get("code").is_some_and(Value::is_i64)
                    && err.get("message").is_some_and(Value::is_string) =>
            {
                // Server messages may contain secrets; do not copy them into local errors.
                Ok(Err(error(
                    "mcp_rpc",
                    "MCP server returned a JSON-RPC error",
                )))
            }
            _ => Err(error("mcp_protocol", "invalid MCP response envelope")),
        }
    }

    pub(crate) async fn close(&mut self, grace: Duration) -> ToolResult<()> {
        self.reader.abort();
        self.input.lock().await.take();
        if let Ok(result) = tokio::time::timeout(grace, self.child.wait()).await {
            result.map_err(|_| error("mcp_shutdown", "cannot wait for MCP server"))?;
        } else {
            self.child
                .start_kill()
                .map_err(|_| error("mcp_shutdown", "cannot stop MCP server"))?;
            tokio::time::timeout(grace, self.child.wait())
                .await
                .map_err(|_| error("mcp_shutdown", "MCP server shutdown timed out"))?
                .map_err(|_| error("mcp_shutdown", "cannot reap MCP server"))?;
        }
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

async fn write(input: &Input, value: &Value, max_frame: usize) -> ToolResult<()> {
    let mut bytes = serde_json::to_vec(value)
        .map_err(|_| error("mcp_protocol", "cannot encode MCP request"))?;
    if bytes.len() > max_frame {
        return Err(error("mcp_limit", "MCP request exceeds frame limit"));
    }
    bytes.push(b'\n');
    let mut guard = input.lock().await;
    let stdin = guard
        .as_mut()
        .ok_or_else(|| error("mcp_closed", "MCP stdin closed"))?;
    stdin
        .write_all(&bytes)
        .await
        .map_err(|_| error("mcp_io", "cannot write MCP request"))?;
    stdin
        .flush()
        .await
        .map_err(|_| error("mcp_io", "cannot flush MCP request"))
}

async fn read_loop(
    stdout: ChildStdout,
    input: &Input,
    tx: &mpsc::Sender<ToolResult<Value>>,
    limit: usize,
) -> ToolResult<()> {
    let mut reader = BufReader::new(stdout);
    loop {
        let line = read_frame(&mut reader, limit).await?;
        let value: Value = serde_json::from_slice(&line)
            .map_err(|_| error("mcp_protocol", "invalid MCP JSON frame"))?;
        if value.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Err(error("mcp_protocol", "invalid MCP JSON-RPC version"));
        }
        if let Some(method) = value.get("method") {
            let method = method
                .as_str()
                .ok_or_else(|| error("mcp_protocol", "invalid MCP method"))?;
            if value.get("result").is_some() || value.get("error").is_some() {
                return Err(error("mcp_protocol", "invalid MCP request envelope"));
            }
            if let Some(id) = value.get("id") {
                if !id.is_string() && !id.is_i64() && !id.is_u64() {
                    return Err(error("mcp_protocol", "invalid MCP request ID"));
                }
                let response = if method == "ping" {
                    json!({"jsonrpc":"2.0","id":id,"result":{}})
                } else {
                    json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":"Unsupported client method"}})
                };
                write(input, &response, limit).await?;
            }
            // Notifications (including progress/listChanged) are ignored. V1
            // uses explicit discovery snapshots, no automatic registry mutation.
        } else {
            tx.try_send(Ok(value))
                .map_err(|_| error("mcp_protocol", "unsolicited MCP response overflow"))?;
        }
    }
}

async fn read_frame(
    reader: &mut BufReader<ChildStdout>,
    limit: usize,
) -> Result<Vec<u8>, ToolError> {
    let mut line = Vec::new();
    loop {
        let bytes = reader
            .fill_buf()
            .await
            .map_err(|_| error("mcp_io", "cannot read MCP stdout"))?;
        if bytes.is_empty() {
            return Err(error("mcp_closed", "MCP server closed stdout"));
        }
        let newline = bytes.iter().position(|b| *b == b'\n');
        let count = newline.unwrap_or(bytes.len());
        if count > limit.saturating_sub(line.len()) {
            return Err(error("mcp_limit", "MCP response exceeds frame limit"));
        }
        line.extend_from_slice(&bytes[..count]);
        reader.consume(count + usize::from(newline.is_some()));
        if newline.is_some() {
            return Ok(line);
        }
    }
}
