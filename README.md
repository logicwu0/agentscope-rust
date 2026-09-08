# AgentScope Rust

[![CI](https://github.com/logicwu0/agentscope-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/logicwu0/agentscope-rust/actions/workflows/ci.yml)

[English](README.md) | [简体中文](README.zh-CN.md)

A community-driven, Rust-native agent framework inspired by
[AgentScope](https://github.com/agentscope-ai/agentscope).

> [!IMPORTANT]
> This project is in its early design stage. It is not currently affiliated
> with or endorsed by the AgentScope maintainers, and it is not ready for
> production use.

## Vision

AgentScope Rust aims to bring AgentScope's core ideas to the Rust ecosystem
without mechanically translating its Python API.

The project intends to provide:

- Rust-native abstractions for messages, models, tools, memory, and agents
- Async and streaming execution built on the Rust async ecosystem
- Strongly typed, provider-neutral data structures
- Cancellation, timeouts, backpressure, and predictable error handling
- Interoperability through standards such as MCP, A2A, and OpenTelemetry
- Small, reliable binaries suitable for services, CLIs, edge workloads, and
  embedded agent runtimes

Conceptual and protocol compatibility with AgentScope is a goal. Exact API and
behavioral compatibility will be evaluated feature by feature.

## Current Status

**Milestone 4 — minimal agent loop**

The project foundation and continuous integration are in place. The public API
now provides roles, text, thinking, validated multimodal data blocks,
streaming-aware tool calls, multimodal tool results, and streaming structured
JSON output blocks, token usage accounting, provider-neutral chat model
responses, deterministic streaming event accumulation, and an object-safe
asynchronous chat model interface with a deterministic mock. `OpenAIChatModel`
can now call and stream from OpenAI-compatible chat-completions APIs, including
DeepSeek, with tool calls, structured output, token usage, timeouts, and
structured provider errors. The SSE decoder handles arbitrary HTTP chunk
boundaries and provider-side stream errors. The first tool-layer API now adds
an object-safe asynchronous `Tool` trait, invocation contexts, structured tool
errors, a deterministic mock, and a named registry with precompiled local JSON
Schema validation. A batch executor runs calls sequentially by default or
concurrently when requested, preserves input order, and converts individual
dispatch failures into structured tool-result errors. The first non-streaming
`ReActAgent` now connects model generation, tool execution, observations, and
the final response in a bounded loop. An object-safe `Memory` interface and
thread-safe `InMemoryMemory` can preserve complete conversations across agent
replies. The optional `sqlite` feature adds transactional, session-isolated
`SQLiteMemory` persistence for local applications and single-node services. A
serializable `AgentEvent` protocol defines model deltas, tool execution, step
completion, final replies, and terminal errors. `ReActAgent::stream` now emits
those events in real time across the complete model-tool-model loop. Read-only,
object-safe asynchronous `AgentHook`s can observe reply, observation, model,
and tool lifecycle boundaries in deterministic registration order.
`Agent::observe` stores external messages in configured conversation memory
without triggering a model call. A cloneable `AgentInterruptHandle` provides
cooperative checkpoints around model calls, streaming chunks, and tool
execution. Interrupted tool calls are closed with persisted `interrupted`
results so later conversation state remains structurally valid. Cancellation
does not roll back external side effects already performed by a tool. A
versioned, JSON-serializable `AgentState` can now snapshot and atomically
restore complete conversation history through the object-safe `Agent` API.
Runtime interruption controls are intentionally excluded from persisted state.
The object-safe `StateStore` API now keys records by user and session, protects
writes with optimistic revision checks, and includes a thread-safe
`InMemoryStateStore`. A bound `ReActAgent` automatically loads and saves state
around replies, streams polled through their terminal event, and observations.
Named tools can now require explicit human confirmation. The agent persists a
`PendingToolCalls` checkpoint, emits a structured pause outcome, rejects
unrelated messages while paused, and resumes with per-call approve or deny
decisions. Denied calls become model-visible `denied` tool results.

```rust
use std::time::Duration;

use agentscope::{ChatModel, ChatRequest, Msg, OpenAIChatModel, RetryPolicy};

let model = OpenAIChatModel::builder()
    .model("deepseek-chat")
    .api_key_from_env("DEEPSEEK_API_KEY")?
    .base_url("https://api.deepseek.com")
    .retry_policy(
        RetryPolicy::new(2)
            .with_initial_delay(Duration::from_millis(250))
            .with_max_delay(Duration::from_secs(10)),
    )
    .build()?;

let response = model
    .generate(ChatRequest::new([Msg::user("Hello")]))
    .await?;
```

Run the complete example without putting the key in source code or a committed
configuration file:

```shell
cargo run --example hooks
cargo run --example interruption
cargo run --example state
cargo run --example session_state
cargo run --example tool_confirmation
DEEPSEEK_API_KEY='your-key' cargo run --example deepseek
DEEPSEEK_API_KEY='your-key' cargo run --example deepseek_stream
DEEPSEEK_API_KEY='your-key' cargo run --example deepseek_react
DEEPSEEK_API_KEY='your-key' cargo run --example deepseek_react_stream
```

```rust
let memory = Arc::new(SQLiteMemory::open("agentscope.db", "session-1").await?);
let agent = ReActAgent::new("Friday", model, tool_executor)?
    .with_max_steps(8)?
    .with_shared_memory(memory);

agent.observe(Msg::assistant("planner", "Use exact arithmetic.")).await?;
let reply = agent.reply(Msg::user("What is 6 * 7?")).await?;
```

Enable the persistent backend with `features = ["sqlite"]` in your dependency.
Call `snapshot` or `restore` only while no reply is active on that agent.
State-bound streams must be polled through their terminal event to perform their
final save.
Before an approved tool runs, the agent now persists a `PendingToolExecution`
checkpoint and supplies a stable key through `ToolContext::idempotency_key()`.
If execution stops before its result is saved, a restored agent returns
`AgentError::ToolExecutionInDoubt` instead of automatically running the tool
again. After checking the external system, submit a terminal `ToolResultBlock`
with `resolve_tool_execution` to continue the original reply.

If the application explicitly authorizes another attempt, call
`agent.retry_tool_execution(checkpoint.confirmation().reply_id()).await?`.
This retries all originally approved calls using their original idempotency
keys, including calls that may already have succeeded; denied calls stay denied.
An interrupted retry keeps the same uncertain checkpoint. The method is also
available through `dyn Agent`. No automatic retry is performed.

This prevents silent duplicate execution, but cannot guarantee exactly-once
external side effects by itself. Side-effecting tools should honor the supplied
idempotency key, and applications must reconcile uncertain outcomes before
resuming.

Wrap a side-effecting tool before registering it to enable process-local
deduplication: `registry.register(IdempotentTool::new(my_tool))?`.
Clones of the wrapper share cached results. Identical keys, JSON inputs, and all
context metadata reuse the original result, including errors; conflicting input
or context is rejected. Concurrent duplicates wait for one execution. Calls
without a key run normally; blank/non-string keys are rejected. Cancellation or
panic leaves an uncertain record and blocks another execution until the external
outcome is reconciled via the agent's `resolve_tool_execution` API.

The default capacity is 1024 keys, configurable with `IdempotentTool::with_capacity`.
Records are never evicted; new keyed calls fail when full. These records live only
in that wrapper's memory and are lost on restart. Durable deduplication still
requires a storage plugin or idempotency support in the external service.
Run the offline example with `cargo run --example idempotent_tool`.

### SQLite agent state plugin

`plugins/agentscope-state-sqlite` provides `SQLiteStateStore` independently of
the core crate's `sqlite` memory feature. Bind it with
`agent.with_state_store(key, SQLiteStateStore::open(path).await?)` and configure
`InMemoryMemory` as the agent's working memory. Complete snapshots, including
confirmation and uncertain-execution checkpoints, survive process restarts.
Writes check revisions in an immediate SQLite transaction. Schema version 1 uses
plugin-specific tables; unsupported versions fail without migration.

Run these commands separately to demonstrate an actual process restart (use a
fresh database path). The demo's `resume` command explicitly approves the tools:

```shell
cargo run -p agentscope-state-sqlite --example restart -- pause /tmp/agentscope-demo.db
cargo run -p agentscope-state-sqlite --example restart -- resume /tmp/agentscope-demo.db
```

The plugin persists agent state, not `IdempotentTool`'s result cache. For durable
tool deduplication, use the separate plugin below.

### SQLite idempotency plugin

`agentscope-idempotency-sqlite` implements the core `IdempotencyStore` contract.
Register `PersistentIdempotentTool` with a stable tenant/tool-version namespace:

```rust
let store = SQLiteIdempotencyStore::open("agent.db").await?;
registry.register(PersistentIdempotentTool::new("tenant-a:notes:v1", my_tool, store)?)?;
```

An atomic claim is saved before tool execution. Completed successes and errors
are reused across restarts for the same namespace, tool name, key, JSON input,
and complete context. Conflicting input/context is rejected. Unlike the in-memory
wrapper, concurrent duplicates of an unfinished invocation return
`idempotency_in_doubt` immediately; they do not wait or take over execution.
Cancelled executions, panics, and failed completion writes retain uncertain
records. The agent preserves its execution checkpoint for these outcomes.

After stopping/fencing the original worker and verifying the external outcome,
call `store.reconcile(request, verified_result).await?` with the original
`IdempotencyRequest`. Then `agent.retry_tool_execution(reply_id)` retrieves the
stored result and continues the conversation without invoking that tool again.
Completed results cannot be overwritten. `resolve_tool_execution` alone resolves
agent state; it does not update the separate idempotency store.

Records never expire or auto-retry. The database stores full inputs, context
metadata, and outputs; protect it as application data. External effects are not
transactional with SQLite, so an interrupted invocation still requires outcome
reconciliation. Retention policies are a future extension.

Run these commands as separate processes, using a fresh database for `first`:

```shell
cargo run -p agentscope-idempotency-sqlite --example idempotency_restart -- first /tmp/agentscope-idempotency.db
cargo run -p agentscope-idempotency-sqlite --example idempotency_restart -- replay /tmp/agentscope-idempotency.db
```

The first process executes once; the replay process executes zero tools and
returns the same result. Both SQLite plugins can share the same database file.

## Roadmap / TODO

The roadmap is intentionally incremental. Interfaces will be stabilized only
after they have been exercised by working examples.

### Milestone 0 — Foundation

- [x] Create the repository and Cargo package
- [x] Adopt the Apache-2.0 license
- [x] Configure Rust 2024, rustfmt, and Clippy
- [x] Forbid unsafe Rust in the initial codebase
- [x] Add continuous integration for formatting, linting, and tests
- [ ] Add contribution and security guidelines

### Milestone 1 — Core Types

- [x] Define `Msg`, roles, metadata, and text content blocks
- [x] Add multimodal data blocks
- [x] Define thinking blocks with provider-specific extension fields
- [x] Define streaming-aware tool-call blocks and permission suggestions
- [x] Define streaming and multimodal tool-result blocks
- [x] Define structured-output blocks
- [x] Introduce a provider-neutral token usage type
- [x] Define shared chat response and finish reason types
- [x] Define model errors and streaming event types
- [x] Define object state snapshot and restore conventions
- [ ] Add JSON serialization and compatibility fixtures

### Milestone 2 — Model Layer

- [x] Define a provider-neutral asynchronous `ChatModel` trait
- [x] Implement an OpenAI-compatible chat model
- [x] Support SSE streaming responses
- [x] Map provider token usage
- [x] Support tool calling and structured output
- [x] Add request timeouts, exponential retries, and `Retry-After` handling
- [ ] Add explicit cancellation support
- [x] Add mock models for deterministic tests

### Milestone 3 — Tools

- [x] Define an object-safe asynchronous tool interface
- [x] Execute tool-call batches sequentially or concurrently
- [x] Add bounded process-local idempotent tool deduplication
- [x] Add durable idempotency records and external-outcome reconciliation
- [ ] Add idempotency record inspection and retention policies
- [ ] Add streaming tool execution
- [x] Implement a tool registry and JSON Schema input validation
- [ ] Generate JSON Schema from Rust types
- [ ] Support tool groups and dynamic tool selection
- [ ] Add tool execution middleware
- [ ] Add a procedural macro for ergonomic Rust tool definitions

### Milestone 4 — Memory and Agents

- [x] Define an object-safe asynchronous `Memory` trait
- [x] Define an object-safe asynchronous `Agent` trait
- [x] Implement thread-safe in-memory conversation history
- [x] Implement transactional SQLite conversation history
- [x] Implement a minimal non-streaming `ReActAgent`
- [x] Define a serializable streaming agent event protocol
- [x] Implement streaming `ReActAgent` execution
- [x] Add read-only asynchronous lifecycle hooks
- [x] Add direct external-message observation
- [x] Add instance-level cooperative interruption
- [x] Add versioned manual agent state snapshot and atomic restoration
- [x] Add a revisioned per-session `StateStore` and automatic restoration
- [x] Add persisted tool-confirmation checkpoints and resume decisions
- [x] Persist approved-tool execution checkpoints and stable idempotency keys
- [x] Reconcile uncertain tool executions with externally supplied results
- [x] Explicitly retry uncertain executions using the original idempotency keys
- [ ] Add external tool execution and persisted permission rules
- [ ] Add per-session resumable interruption
- [x] Add the independent SQLite `StateStore` plugin
- [ ] Provide single-agent and multi-agent examples

### Storage Plugins

- [ ] Extract SQLite into an `agentscope-memory-sqlite` crate before API stabilization
- [ ] Add an `agentscope-memory-postgres` plugin
- [ ] Add an `agentscope-memory-redis` plugin
- [ ] Add message pagination, retention, and expiration policies
- [ ] Define storage migration and plugin compatibility policies

### Milestone 5 — Interoperability

- [ ] Implement an MCP client
- [ ] Evaluate MCP server support
- [ ] Add A2A interoperability
- [ ] Add OpenTelemetry tracing
- [ ] Evaluate trace compatibility with AgentScope Studio
- [ ] Publish cross-language message compatibility tests

### Milestone 6 — Production Readiness

- [ ] Define a stable public API and compatibility policy
- [ ] Add integration, concurrency, cancellation, and failure-path tests
- [ ] Add benchmarks and memory profiling
- [ ] Audit dependencies and establish an MSRV policy
- [ ] Publish API documentation and runnable tutorials
- [ ] Prepare the first crates.io release

## Non-Goals for the First MVP

The first usable release will focus on a complete model–tool–agent loop. It
will not initially attempt to cover every AgentScope feature, model provider,
voice workflow, RAG integration, training system, or deployment platform.

## Development

The project currently targets Rust 1.85 or newer and the Rust 2024 edition.

```shell
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

## Contributing

Design discussions, compatibility research, examples, and implementation
contributions are welcome. Because the public API is still being designed,
please open an issue before starting a large change.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
