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
事务与会话隔离能力的持久化。可序列化的 `AgentEvent` 协议覆盖模型增量、工具执行、
步骤结束、最终回复和终止错误。`ReActAgent::stream` 现已在完整的模型—工具—模型循环中
实时发送这些事件。只读、可作为 trait 对象使用的异步 `AgentHook` 可以按照确定的注册
顺序观察回复、外部消息、模型和工具生命周期边界。`Agent::observe` 可以把外部消息写入
已配置的会话 Memory，而不会触发模型调用。可克隆的 `AgentInterruptHandle` 在模型调用、
流式分片和工具执行周围提供协作式检查点。被中断的工具调用会生成并持久化
`interrupted` 结果，保证后续会话状态在结构上仍然完整；但中断不会回滚工具已经产生的
外部副作用。版本化且可序列化为 JSON 的 `AgentState` 现可通过 `Agent` trait 对象接口
快照完整对话历史，并以原子方式恢复。运行期中断控制不会写入持久化状态。
可作为 trait 对象使用的 `StateStore` 现会按用户和会话定位记录，通过乐观 revision
校验保护写入，并提供线程安全的 `InMemoryStateStore`。绑定状态存储后，`ReActAgent`
会在回复、读取至终止事件的流以及外部消息观察前后自动加载和保存状态。
现在可以把指定工具配置为必须经过人工确认。Agent 会持久化 `PendingToolCalls`
检查点并发出结构化暂停结果；暂停期间拒绝无关消息，收到逐个工具的批准或拒绝决定后
继续执行。被拒绝的调用会转换为模型可见的 `denied` 工具结果。

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
cargo run --example hooks
cargo run --example interruption
cargo run --example state
cargo run --example session_state
cargo run --example tool_confirmation
DEEPSEEK_API_KEY='你的-key' cargo run --example deepseek
DEEPSEEK_API_KEY='你的-key' cargo run --example deepseek_stream
DEEPSEEK_API_KEY='你的-key' cargo run --example deepseek_react
DEEPSEEK_API_KEY='你的-key' cargo run --example deepseek_react_stream
```

```rust
let memory = Arc::new(SQLiteMemory::open("agentscope.db", "session-1").await?);
let agent = ReActAgent::new("Friday", model, tool_executor)?
    .with_max_steps(8)?
    .with_shared_memory(memory);

agent.observe(Msg::assistant("planner", "请使用精确计算。")).await?;
let reply = agent.reply(Msg::user("6 乘以 7 等于多少？")).await?;
```

在依赖配置中加入 `features = ["sqlite"]` 即可启用持久化后端。
请仅在目标 Agent 没有正在执行的回复时调用 `snapshot` 或 `restore`。
绑定状态存储的流必须读取到终止事件，才能执行最终保存。
获批工具执行前，Agent 现在会先持久化 `PendingToolExecution` 检查点，并通过
`ToolContext::idempotency_key()` 提供稳定幂等键。如果执行结果尚未保存进程就停止，
恢复后的 Agent 会返回 `AgentError::ToolExecutionInDoubt`，不会自动再次调用工具。
应用检查外部系统后，可通过 `resolve_tool_execution` 提交终态 `ToolResultBlock`，继续
原来的回复。

如果应用明确授权再次执行，可调用
`agent.retry_tool_execution(checkpoint.confirmation().reply_id()).await?`。
它会使用原来的幂等键重试所有原先获批的调用（包括可能已经成功的调用），被拒绝的
调用仍保持拒绝。重试再次中断时保留同一份不确定检查点。此方法也支持 `dyn Agent`，
不会自动触发重试。

这能避免静默重复执行，但框架本身仍无法保证外部副作用严格 exactly-once。具有副作用
的工具应正确处理幂等键，应用也必须先核对结果不确定的执行，再恢复会话。

注册工具前可用 `registry.register(IdempotentTool::new(my_tool))?` 开启进程内去重。
包装器的克隆共享缓存；同键、同 JSON 参数和全部上下文元数据的请求复用原结果（包括
错误），参数或上下文冲突则拒绝。并发重复请求等待同一次执行。未提供幂等键时正常
执行，空白或非字符串键会被拒绝。执行中取消或 panic 会留下不确定记录并阻止重跑，
需通过 Agent 的 `resolve_tool_execution` 提交外部核对结果来恢复会话。

默认容量为 1024 个键，可通过 `IdempotentTool::with_capacity` 配置。记录不会被淘汰，
容量满后拒绝新键。这些记录只保存在该包装器的内存中，重启后丢失；跨进程去重仍需
持久化插件或外部服务支持。离线示例：`cargo run --example idempotent_tool`。

### SQLite Agent 状态插件

`plugins/agentscope-state-sqlite` 提供独立的 `SQLiteStateStore`，无需启用核心库的
`sqlite` 会话记忆功能。通过
`agent.with_state_store(key, SQLiteStateStore::open(path).await?)` 绑定，并配置
`InMemoryMemory` 作为运行时记忆。完整快照（含人工确认、结果不确定的执行检查点）
可跨进程重启恢复。写入在 SQLite 即时事务中检查 revision。数据库 schema v1 使用
插件专属表；遇到未知版本会拒绝打开，不会自动迁移。

分别运行以下命令验证真实进程退出再恢复（使用新的数据库路径）。示例的 `resume`
命令代表调用方明确批准执行：

```shell
cargo run -p agentscope-state-sqlite --example restart -- pause /tmp/agentscope-demo.db
cargo run -p agentscope-state-sqlite --example restart -- resume /tmp/agentscope-demo.db
```

本插件保存 Agent 状态，不保存 `IdempotentTool` 的结果缓存。工具持久化去重使用下面的独立插件。

### SQLite 幂等记录插件

`agentscope-idempotency-sqlite` 实现核心 `IdempotencyStore` 接口。使用稳定的租户/工具版本
命名空间注册 `PersistentIdempotentTool`：

```rust
let store = SQLiteIdempotencyStore::open("agent.db").await?;
registry.register(PersistentIdempotentTool::new("tenant-a:notes:v1", my_tool, store)?)?;
```

工具执行前先原子保存执行权记录。命名空间、工具名、幂等键、JSON 参数及完整上下文
相同的请求，重启后也会复用已完成结果（包括成功和错误）。同键参数或上下文冲突会
被拒绝。与内存包装器不同，未完成调用的并发重复请求立即返回 `idempotency_in_doubt`，
不会等待或接管执行。取消、panic 和完成结果写入失败会留下不确定记录，Agent 也会
保留对应执行检查点。

先确保原工作进程已停止或无法再产生副作用，并核对外部执行结果，再以原
`IdempotencyRequest` 调用 `store.reconcile(request, verified_result).await?`。
随后调用 `agent.retry_tool_execution(reply_id)`，即可读取已核对结果并继续会话，无需
再次执行该工具。已完成结果不可覆盖。单独调用 `resolve_tool_execution` 只更新 Agent
状态，不会同步修改独立的幂等记录。

记录不会自动过期或触发重试。数据库保存完整输入、上下文元数据和输出，应作为应用
数据保护。外部副作用与 SQLite 不在同一事务内，执行中断仍需核对结果；记录保留策略
留作后续扩展。

用两个独立进程运行示例，`first` 使用新的数据库路径：

```shell
cargo run -p agentscope-idempotency-sqlite --example idempotency_restart -- first /tmp/agentscope-idempotency.db
cargo run -p agentscope-idempotency-sqlite --example idempotency_restart -- replay /tmp/agentscope-idempotency.db
```

第一个进程执行一次工具，第二个进程执行零次并返回相同结果。两个 SQLite 插件可共用
一个数据库文件。

### 交互式命令行聊天

workspace 示例 `agentscope-chat` 串联了流式聊天、`multiply` 工具人工确认、SQLite
会话状态与持久化幂等去重：

```shell
# 离线模式，不需要 API Key 或网络：
cargo run -p agentscope-chat -- --offline --session demo
# 本地环境已导出 DEEPSEEK_API_KEY 后：
cargo run -p agentscope-chat -- --user alice --session deepseek
```

输入 `multiply 6 7`，在待确认时 `/quit` 退出；用相同数据库、用户和会话重新启动后，
输入 `/approve` 继续执行。通过 `/history` 查看恢复的对话。离线模式只回显普通消息并
解析 `multiply A B`，用于确定性验证，不是真实语言模型。

命令包括 `/help`、`/status`、`/history`、`/approve`（批准全部待确认调用）、
`/deny [原因]`（拒绝全部）、`/retry`、`/resolve CALL_ID 已核对的结果文本`、`/quit`。
`/resolve` 接受已核对的成功文本结果并写入幂等存储，随后 `/retry` 复用结果继续会话。
处理不确定调用前，应先核对外部结果并停止原执行进程。

普通聊天和批准/恢复后的回答均使用流式输出，CLI 同时显示工具开始、结束事件。同一会话请只运行
一个进程。轮次之间 `/quit` 或输入结束会保留完成状态；生成中强制退出可能丢失尚未
提交的文本。默认数据库为 `agentscope-chat.db`（已加入 gitignore），可用 `--db PATH`
指定其他路径。API Key 仅从环境变量读取，不通过聊天输入或命令行参数传递，也不会
自动读取本地 `.env` 文件。离线测试与真实聊天建议使用不同会话。

可作为 trait 对象使用的 Agent 接口新增 `stream_resume_tool_calls`、
`stream_retry_tool_execution`、`stream_resolve_tool_execution`，均返回
`AgentEventStream`；原有非流式方法继续保留。参数会在返回流之前校验，工具只在轮询
流后执行。恢复时保留原步骤编号和最大步数预算。提交外部结果时会发出 `ToolFinished`，
不产生 `ToolStarted`。

恢复流会在工具执行前保存检查点，在继续调用模型前保存完整工具结果。工具开始事件
先于批量执行，结束事件在批量完成后按调用顺序发出。应读取至终止事件以完成最终保存，
保存失败会返回终止错误。执行工具时丢弃流会留下待核对/重试检查点；后续模型输出期间
丢弃流会保留工具结果，但不保存部分文本，可发送新消息继续已保存的对话。另一个进程
重试或核对同一会话前，必须先停止原执行进程。

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
- [x] 定义对象状态快照与恢复约定
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
- [x] 支持有容量上限的进程内工具幂等去重
- [x] 支持持久化幂等记录与外部结果核对
- [ ] 支持幂等记录查询管理与保留策略
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
- [x] 定义可序列化的流式 Agent 事件协议
- [x] 实现流式 `ReActAgent` 执行
- [x] 支持只读异步生命周期 Hook
- [x] 支持直接观察外部消息
- [x] 支持实例级协作式中断
- [x] 支持版本化的手动 Agent 状态快照与原子恢复
- [x] 支持带 revision 的会话级 `StateStore` 与自动恢复
- [x] 支持持久化的工具确认检查点与恢复决定
- [x] 支持工具确认、重试与外部结果恢复的流式续接
- [x] 持久化获批工具的执行检查点并提供稳定幂等键
- [x] 使用外部确认的结果恢复执行结果不确定的工具调用
- [x] 使用原幂等键显式重试结果不确定的工具执行
- [ ] 支持外部工具执行与持久化权限规则
- [ ] 支持会话级可恢复中断
- [x] 添加独立的 SQLite `StateStore` 插件
- [x] 提供带持久化会话恢复的交互式单 Agent 命令行示例
- [ ] 提供多 Agent 示例

### 存储插件

- [ ] 在 API 稳定前将 SQLite 提取为 `agentscope-memory-sqlite` crate
- [ ] 添加 `agentscope-memory-postgres` 插件
- [ ] 添加 `agentscope-memory-redis` 插件
- [ ] 支持消息分页、保留期限和过期策略
- [ ] 定义存储迁移与插件兼容策略

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
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

## 参与贡献

欢迎参与设计讨论、兼容性研究、示例和代码实现。由于公开 API 尚在设计中，开始较大的
改动前，请先创建 Issue 讨论。

## 许可证

项目使用 Apache License 2.0，详见 [LICENSE](LICENSE)。
