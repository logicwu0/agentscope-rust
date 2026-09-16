# agentscope-mcp

Opt-in, Rust 1.85-compatible MCP **stdio tools subset**, not a complete MCP SDK.
Supports protocol `2025-11-25` and `2025-06-18`: newline-delimited JSON-RPC,
initialize/initialized, paginated tools/list, tools/call and server ping handling.

## Usage

```rust,no_run
use agentscope::{InMemoryMemory, ReActAgent, ToolExecutor};
use agentscope_mcp::{McpClient, StdioConfig};

// Explicit trusted executable; no shell or package installation.
let mut config = StdioConfig::new("/absolute/path/to/mcp-server");
config.args.push("--some-server-option".into());
// Environment is EMPTY. Pass only required variables, including PATH if needed.
let client = McpClient::connect(config).await?;
let registry = client.registry("local").await?;
let agent = ReActAgent::new("Friday", model, ToolExecutor::new(registry))?
    .with_memory(InMemoryMemory::new())
    .with_tool_confirmation_required("local__some_tool");
// Execute replies / collect actual user approvals; optionally add StateStore/offload.
client.close().await?;
```

`tools(namespace)` returns adapters for selective registration. `registry` returns
a new registry only after full discovery/schema validation. Model names are
`<namespace>__<remote_name>`: ASCII letters/digits/underscore/hyphen, at most 64
bytes total. Non-portable names are rejected, not renamed. Input and optional
output schemas are validated locally. Text blocks join with newlines; when
structuredContent is present it is retained alongside text in a JSON object.
Existing large-result offload therefore applies. `isError: true` becomes a tool
error. Unsupported content or invalid output reports uncertain execution because
the remote tool may already have succeeded.

## Safety and limits

- NOT a sandbox: starting a server executes code with application privileges,
  before any Agent confirmation. Use trusted executables, reviewed arguments and
  least-privilege workspaces. `current_dir` is not a security boundary.
- No credentials/config files are created, no environment is inherited. Explicitly
  pass an allowlist via `config.env`. ToolContext metadata/idempotency keys are not
  forwarded. Generic MCP servers do not guarantee idempotency.
- Server instructions/annotations never grant permission or alter prompts.
  Descriptions and results are untrusted data. Select tools and configure
  confirmation by namespaced name; consequential tools should use confirmation
  plus StateStore. Without confirmation there is no durable execution checkpoint.
- One connection serializes calls, including concurrent Agent tool batches.
  Default deadline is 30 seconds per request plus a separate queue-wait deadline
  of the same length. Discovery allows 128 pages/1024 tools. Frames default to
  4 MiB in either direction, bounded before parsing. Progress does not reset the
  deadline. Notifications are ignored; tool lists are explicit discovery snapshots.
- Timeout, malformed transport, oversized response or cancellation closes the
  connection and kills the direct child. No automatic retry/reconnect/restart.
  Effects may have happened: `tool_execution_in_doubt` retains confirmed Agent
  checkpoints and uncertain idempotency records. Reconcile externally before any
  retry. A valid JSON-RPC error is distinct from transport failure.
- `close()` closes stdin, waits (default 2 seconds), then directly kills/waits if
  necessary; no intermediate SIGTERM. Dropping all client/tool handles kills the
  direct child best-effort via Tokio. This is NOT process-tree supervision; avoid
  servers that leave detached children. Keep the runtime alive for cleanup.
- Stderr is discarded by default. `inherit_stderr` explicitly enables potentially
  sensitive terminal logs. Local transport/RPC errors omit raw diagnostics,
  command args and environment; tool errors intentionally carry server content.
- No HTTP/SSE, OAuth, MCP Server API, roots, sampling, elicitation, resources,
  prompts, tasks, multimodal mapping or automatic tool refresh. Unsupported server
  requests get Method Not Found. Timeouts tear down the connection rather than
  implementing resumable MCP cancellation. No full conformance claim.

## Offline verification

On macOS/Linux with `/usr/bin/python3` (standard library only):

```shell
cargo test -p agentscope-mcp
cargo run -p agentscope-mcp --example stdio
```

Uses a real local fixture process and deterministic mock model, no API keys.
Tests include discovery, protocol errors, frame bounds, concurrent calls,
timeout/drop cleanup, approval/uncertain recovery and large-result offload.

References: [lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle),
[stdio](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports),
[tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).
