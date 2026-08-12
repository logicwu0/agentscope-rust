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

**里程碑 1——核心类型**

工程基础和持续集成已经就绪。第一批公开 API 已实现角色、文本内容块、消息、JSON
元数据和序列化；下一步将实现多模态与工具相关内容块。

```rust
// 目标 API 方向——仅作示意，目前尚未实现。
let agent = ReActAgent::builder()
    .name("Friday")
    .model(OpenAIChatModel::from_env("qwen-plus")?)
    .tool(weather)
    .memory(InMemoryMemory::new())
    .build()?;

let reply = agent
    .reply(Msg::user("杭州今天天气怎么样？"))
    .await?;
```

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
- [ ] 添加多模态数据块
- [ ] 定义工具调用、工具结果、思考和结构化输出内容块
- [ ] 引入通用错误、结果、用量和流式事件类型
- [ ] 定义对象状态快照与恢复约定
- [ ] 添加 JSON 序列化和跨语言兼容测试数据

### 里程碑 2——模型层

- [ ] 定义与供应商无关的异步 `ChatModel` trait
- [ ] 实现 OpenAI 兼容的对话模型
- [ ] 支持流式响应和 Token 用量统计
- [ ] 支持工具调用和结构化输出
- [ ] 支持取消、超时、重试和限流处理
- [ ] 添加用于确定性测试的模拟模型

### 里程碑 3——工具系统

- [ ] 定义异步与流式工具接口
- [ ] 实现工具注册表和 JSON Schema 生成
- [ ] 支持工具分组和动态工具选择
- [ ] 添加工具执行中间件
- [ ] 提供便于定义 Rust 工具的过程宏

### 里程碑 4——记忆与 Agent

- [ ] 定义 `Memory` 和 `Agent` trait
- [ ] 实现内存会话历史
- [ ] 实现最小可用的 `ReActAgent`
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
