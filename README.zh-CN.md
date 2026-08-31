# AgentScope Rust

[![CI](https://github.com/logicwu0/agentscope-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/logicwu0/agentscope-rust/actions/workflows/ci.yml)

[English](README.md) | [简体中文](README.zh-CN.md)

一个受 [AgentScope](https://github.com/agentscope-ai/agentscope) 启发、由社区驱动的
Rust 原生 Agent 框架。

> [!IMPORTANT]
> 项目尚处于早期设计阶段，目前与 AgentScope 维护团队没有隶属或官方合作关系，
> 也尚未达到生产可用状态。

## 项目愿景

AgentScope Rust 希望把 AgentScope 的核心理念带入 Rust 生态，但不会机械地翻译
Python API。

项目计划提供：

- Rust 原生的消息、模型、工具、记忆和 Agent 抽象
- 基于 Rust 异步生态的异步与流式执行
- 强类型、与模型供应商无关的数据结构
- 取消、超时、背压以及可预测的错误处理
- 通过 MCP、A2A、OpenTelemetry 等标准实现互操作
- 适用于服务端、CLI、边缘负载和嵌入式 Agent Runtime 的小型可靠二进制程序

我们会以兼容 AgentScope 的核心概念和协议为目标；是否追求完全一致的 API 与行为，
将根据每项功能的实际价值分别评估。

## 当前状态

**里程碑 4——最小 Agent 循环**

工程基础和持续集成已经就绪。公开 API 已实现角色、文本、思考、经过校验的多模态
数据块、支持流式参数的工具调用、多模态工具结果、流式结构化 JSON 输出块、
Token 用量统计、与供应商无关的对话模型响应、确定性的流式事件聚合、可作为 trait
对象使用的异步对话模型接口以及确定性 Mock。`OpenAIChatModel` 现已能够调用包括
DeepSeek 在内的 OpenAI 兼容 Chat Completions API，并支持 SSE 流式响应、工具调用、
结构化输出、Token 用量、超时和结构化供应商错误。SSE 解码器可处理任意 HTTP 分片
边界以及供应商返回的流内错误。工具层现已提供可作为 trait 对象使用的异步 `Tool`
接口、调用上下文、结构化工具错误、确定性 Mock，以及使用预编译本地 JSON Schema
校验的具名注册表。批量执行器默认顺序运行工具，也可显式并发执行；它会保持结果顺序，
并将单个调用的分发失败转换为结构化工具错误结果。首个非流式 `ReActAgent` 已将模型
生成、工具执行、观察结果和最终回答连接成带有步数上限的完整循环。可作为 trait 对象
使用的 `Memory` 接口和线程安全的 `InMemoryMemory` 可以在 Agent 的多次回复之间保留
完整对话。可选的 `sqlite` feature 提供 `SQLiteMemory`，为本地应用和单机服务实现具备
事务与会话隔离能力的持久化。

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
    .generate(ChatRequest::new([Msg::user("你好")]))
    .await?;
```

无需将 Key 写入源码或提交的配置文件，即可运行完整示例：

```shell
DEEPSEEK_API_KEY='你的-key' cargo run --example deepseek
DEEPSEEK_API_KEY='你的-key' cargo run --example deepseek_stream
DEEPSEEK_API_KEY='你的-key' cargo run --example deepseek_react
```

```rust
let memory = Arc::new(SQLiteMemory::open("agentscope.db", "session-1").await?);
let agent = ReActAgent::new("Friday", model, tool_executor)?
    .with_max_steps(8)?
    .with_shared_memory(memory);

let reply = agent.reply(Msg::user("6 乘以 7 等于多少？")).await?;
```

在依赖配置中加入 `features = ["sqlite"]` 即可启用持久化后端。

## 路线图 / TODO

项目将采用渐进式开发。只有经过可运行示例验证的接口，才会逐步进入稳定状态。

### 里程碑 0——工程基础

- [x] 创建代码仓库和 Cargo 包
- [x] 采用 Apache-2.0 许可证
- [x] 配置 Rust 2024、rustfmt 和 Clippy
- [x] 初始代码库禁止使用 unsafe Rust
- [x] 添加格式检查、Lint 和测试的持续集成
- [ ] 添加贡献指南和安全策略

### 里程碑 1——核心类型

- [x] 定义 `Msg`、角色、元数据和文本内容块
- [x] 添加多模态数据块
- [x] 定义支持供应商扩展字段的思考内容块
- [x] 定义支持流式参数和权限建议的工具调用内容块
- [x] 定义支持流式和多模态输出的工具结果内容块
- [x] 定义结构化输出内容块
- [x] 引入与供应商无关的 Token 用量类型
- [x] 定义通用对话响应和结束原因类型
- [x] 定义模型错误和流式事件类型
- [ ] 定义对象状态快照与恢复约定
- [ ] 添加 JSON 序列化和跨语言兼容测试数据

### 里程碑 2——模型层

- [x] 定义与供应商无关的异步 `ChatModel` trait
- [x] 实现 OpenAI 兼容对话模型
- [x] 支持 SSE 流式响应
- [x] 映射供应商 Token 用量统计
- [x] 支持工具调用和结构化输出
- [x] 支持请求超时、指数退避重试和 `Retry-After`
- [ ] 支持显式取消
- [x] 添加用于确定性测试的模拟模型

### 里程碑 3——工具系统

- [x] 定义可作为 trait 对象使用的异步工具接口
- [x] 支持顺序或并发执行批量工具调用
- [ ] 支持流式工具执行
- [x] 实现工具注册表和 JSON Schema 输入校验
- [ ] 从 Rust 类型生成 JSON Schema
- [ ] 支持工具分组和动态工具选择
- [ ] 添加工具执行中间件
- [ ] 提供便于定义 Rust 工具的过程宏

### 里程碑 4——记忆与 Agent

- [x] 定义可作为 trait 对象使用的异步 `Memory` trait
- [x] 定义可作为 trait 对象使用的异步 `Agent` trait
- [x] 实现线程安全的内存会话历史
- [x] 实现具备事务能力的 SQLite 会话历史
- [x] 实现最小可用的非流式 `ReActAgent`
- [ ] 支持观察、Hook、中断和 Human-in-the-loop
- [ ] 支持状态持久化和会话恢复
- [ ] 提供单 Agent 与多 Agent 示例

### 里程碑 5——互操作

- [ ] 实现 MCP Client
- [ ] 评估 MCP Server 支持
- [ ] 添加 A2A 互操作能力
- [ ] 添加 OpenTelemetry 链路追踪
- [ ] 评估与 AgentScope Studio 的 Trace 兼容性
- [ ] 发布跨语言消息兼容测试

### 里程碑 6——生产就绪

- [ ] 确定稳定的公开 API 与兼容性策略
- [ ] 添加集成、并发、取消和故障路径测试
- [ ] 添加性能基准和内存分析
- [ ] 审计依赖并确定 MSRV 策略
- [ ] 发布 API 文档和可运行教程
- [ ] 准备第一个 crates.io 版本

## 首个 MVP 暂不覆盖

首个可用版本会专注打通完整的“模型—工具—Agent”循环，不会立即覆盖 AgentScope 的
全部功能、所有模型供应商、语音工作流、RAG 集成、训练系统或各种部署平台。

## 开发

项目当前使用 Rust 1.85 或更高版本以及 Rust 2024 Edition。

```shell
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## 参与贡献

欢迎参与设计讨论、兼容性研究、示例和代码实现。由于公开 API 尚在设计中，开始较大的
改动前，请先创建 Issue 讨论。

## 许可证

项目使用 Apache License 2.0，详见 [LICENSE](LICENSE)。
