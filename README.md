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

**Milestone 1 — core types**

The project foundation and continuous integration are in place. The first
public API now provides roles, text content blocks, messages, JSON metadata,
and serialization. Multimodal and tool-related blocks are next.

```rust
// Target API direction — illustrative only, not implemented yet.
let agent = ReActAgent::builder()
    .name("Friday")
    .model(OpenAIChatModel::from_env("qwen-plus")?)
    .tool(weather)
    .memory(InMemoryMemory::new())
    .build()?;

let reply = agent
    .reply(Msg::user("What is the weather in Hangzhou today?"))
    .await?;
```

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
- [ ] Add multimodal data blocks
- [ ] Define tool-use, tool-result, thinking, and structured-output blocks
- [ ] Introduce shared error, result, usage, and streaming event types
- [ ] Define object state snapshot and restore conventions
- [ ] Add JSON serialization and compatibility fixtures

### Milestone 2 — Model Layer

- [ ] Define a provider-neutral asynchronous `ChatModel` trait
- [ ] Implement an OpenAI-compatible chat model
- [ ] Support streaming responses and token usage
- [ ] Support tool calling and structured output
- [ ] Add cancellation, timeout, retry, and rate-limit handling
- [ ] Add mock models for deterministic tests

### Milestone 3 — Tools

- [ ] Define async and streaming tool interfaces
- [ ] Implement a tool registry and JSON Schema generation
- [ ] Support tool groups and dynamic tool selection
- [ ] Add tool execution middleware
- [ ] Add a procedural macro for ergonomic Rust tool definitions

### Milestone 4 — Memory and Agents

- [ ] Define `Memory` and `Agent` traits
- [ ] Implement in-memory conversation history
- [ ] Implement a minimal `ReActAgent`
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
