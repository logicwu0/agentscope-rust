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
`SQLiteMemory` persistence for local applications and single-node services.

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
DEEPSEEK_API_KEY='your-key' cargo run --example deepseek
DEEPSEEK_API_KEY='your-key' cargo run --example deepseek_stream
DEEPSEEK_API_KEY='your-key' cargo run --example deepseek_react
```

```rust
let memory = Arc::new(SQLiteMemory::open("agentscope.db", "session-1").await?);
let agent = ReActAgent::new("Friday", model, tool_executor)?
    .with_max_steps(8)?
    .with_shared_memory(memory);

let reply = agent.reply(Msg::user("What is 6 * 7?")).await?;
```

Enable the persistent backend with `features = ["sqlite"]` in your dependency.

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
- [ ] Define object state snapshot and restore conventions
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
- [ ] Add observation, hooks, interruption, and human-in-the-loop support
- [ ] Add state persistence and session restoration
- [ ] Provide single-agent and multi-agent examples

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
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Contributing

Design discussions, compatibility research, examples, and implementation
contributions are welcome. Because the public API is still being designed,
please open an issue before starting a large change.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
