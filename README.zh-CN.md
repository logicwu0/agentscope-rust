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

### 模型输入上下文策略

`ReActAgent` 默认使用 `FullContext`，发送完整历史。可显式选择最近 N 个用户轮次：

```rust
use agentscope::RecentTurns;

let agent = agent.with_context_policy(RecentTurns::new(3)?);
```

一个轮次从 `Role::User` 消息开始，包含之后的模型和工具消息，直到下一条用户消息；
N 包含当前轮，必须大于零。系统消息以及 Agent 配置的系统提示词始终保留。历史不足
N 轮时不裁剪。遇到跨轮工具结果会向前扩展到对应调用所在轮，保留完整工具往返，
因此这不是严格的消息数或 Token 上限，也不会修复原本就不完整的历史。

策略只影响每次模型请求，不改变 `Memory`、快照或 `StateStore` 中的完整历史。
普通、流式、确认、重试及外部结果恢复路径共用同一套组装逻辑，模型调用前的 Hook
看到的是裁剪后的请求。策略是运行时配置，重建 Agent 时需要重新设置，不随状态保存。
可实现同步的 `ContextPolicy` trait 自定义选择逻辑；自定义策略必须保证消息顺序、
当前轮和工具调用/结果配对，不应执行 I/O。共享策略使用 `with_shared_context_policy`。

检索尚未实现；摘要和大工具文本卸载需单独显式配置，见下文。离线验证：
`cargo run --example context`，模型最后一次看到 2 条消息，而快照保留全部 6 条历史。

### 每次模型请求的 Token 预算

可在上下文策略裁剪**之后**额外启用窗口预算：

```rust
use agentscope::TokenBudget;

// 示例数值，请根据实际模型配置。
let agent = agent.with_token_budget(TokenBudget::new(8192, 1024)?);
```

以上配置预留 1024 个输出 Token，允许 7168 个输入 Token，输入计入系统消息、工具
定义及历史。输出预留必须大于零且小于窗口。未设置 `GenerateOptions::max_tokens`
时会使用预留值；显式设置更小的正数会保留，零或超过预留的值则报错。不要通过供应商
扩展选项覆盖该上限。即使显式输出上限更小，仍扣除完整预留空间。

默认 `HeuristicTokenCounter` 使用“输入序列化后的 UTF-8 字节数除以三向上取整，再加
16”的启发式估算，计入角色、名称、内容、工具、结构化输出 Schema 和供应商扩展选项，
明确标注为 `Estimated`。它不是模型分词器，也不是保证不超限的上界。应留出余量，或
通过 `with_counter` / `with_shared_counter` 接入模型专用 `TokenCounter`；计数结果
`TokenCount` 区分 `Estimated` 和 `Exact`，精确性由自定义实现负责，包括实际请求编码。
默认估算器不猜测图片、音频等消息或多模态工具结果的成本，会要求专用计数器。
计数失败时不发送模型请求。

超预算时按完整旧用户轮次裁剪并重新计数，保留系统消息及工具依赖。如果当前轮等必须
保留的内容仍放不下，返回 `AgentError::TokenBudget(TokenBudgetError::Exceeded { .. })`，
携带输入计数、精度、输入额度和输出预留；流式路径发送终止错误而不是成功事件。
自定义上下文策略仍须保证当前轮与工具调用/结果完整。

完整历史仍会保存，包括预算失败时刚提交的用户消息。若工具执行后才因结果过大而超限，
工具观察结果不会丢失；恢复路径也会保存结果并清除已完成的检查点。调整预算或策略后可
继续已保存的会话，不应盲目重跑已完成的工具。预算和计数器与上下文策略一样属于运行时
配置，重建 Agent 时需要重新设置。这不是累计费用/回复总 Token 限制，也不包含自动摘要。
离线示例：`cargo run --example token_budget`。

### 显式历史摘要压缩

用选定的 `ChatModel` 和独立 `TokenBudget` 创建 `ChatModelSummarizer`，通过
`with_summarizer` 配置；自定义异步 `ContextSummarizer` 可通过
`with_shared_summarizer` 共享。可以使用同一个模型，也可以单独配置摘要模型。
未显式开启自动压缩时，只有手动调用才会请求摘要：

```rust
let summary = agent.compact_context(3).await?; // 至少保留最近 3 个用户轮次原文
// 以后只清除摘要，不删除任何原始消息：
agent.clear_context_summary().await?;
```

两个方法都支持 `dyn Agent`。提交成功返回 `Some(ContextSummary)`；没有新增可压缩
前缀时返回 `None`，不调用模型。仅处理已完成、工具调用与终态结果完整配对的旧轮次。
同一个 Agent 或其克隆存在活动操作时返回 `Busy`；待确认或执行结果不确定的检查点须
先处理。不要直接并发修改共享 Memory，也不要从摘要器重新调用同一 Agent；独立进程
仍依赖存储 revision 检查来防止覆盖。

内置摘要器只发起一次非流式、无工具的模型请求，把原文作为数据放在一个用户消息中，
通过提示词要求保留目标、约束、事实、决策、工具结果和未完成事项，但不保证不漏信息。
系统消息始终保留原文，不交给摘要器压缩；思考块不发送给摘要模型。此适配器暂不支持
多模态原文。摘要输入超过独立预算会报错，不会静默裁掉待总结原文；空白、截断或请求
工具调用的输出会被拒绝。此适配器不添加分块摘要或额外重试层；模型本身的重试
配置仍可能生效。使用真实模型会产生额外 Token 费用，摘要也可能遗漏或歪曲事实。

`AgentState` 升级为 **v4**，独立保存摘要、覆盖前缀长度及原文 SHA-256 指纹，完整历史
不被改写。仍可读取没有摘要的 v1–v3 状态；SQLite 状态插件无需修改数据库 schema。
旧版本 Agent 无法读取 v4 状态。重建 Agent 后无需摘要器即可使用已保存摘要，但再次
压缩仍需重新配置摘要器。单独的 `Memory` / `SQLiteMemory` 只保存原文，摘要跨重启恢复
需要快照或 `StateStore`。

模型输入由配置的系统提示词、assistant 角色的参考摘要（不是系统指令）、历史系统
消息和未总结的后续原文组成。上下文策略继续处理后续原文，Token 预算会单独保护摘要，
不会为满足预算而静默丢掉它。普通、流式和恢复调用共用同一套组装逻辑；原文前缀被修改
时会拒绝使用过期摘要。指纹只用于一致性检查，不证明摘要事实正确，也不是存储真实性认证。

再次压缩时总结更长的原始前缀，而不是对旧摘要再总结。候选上下文必须在序列化字节数上
变小（不保证实际 Token 一定减少），并通过 Agent 已配置的 Token 预算才会提交。
模型或校验失败、中断及被存储拒绝的写入不会覆盖旧摘要。先保存状态，再更新运行时摘要；
如果存储写入确认不确定，应重新加载确认实际提交结果。清除摘要后可重新使用原文，但仍
受上下文策略和预算限制。

离线示例（确定性 Mock 模型）：`cargo run --example compaction`。

### 可选的新回复前自动压缩

配置 Memory、摘要器和 Agent 的 `TokenBudget` 后，用
`agent.with_auto_compaction(1)?` 显式开启，`without_auto_compaction()` 可关闭但不删除摘要。
默认关闭，属于运行时配置，不随状态保存；开启意味着允许额外摘要模型调用和费用。
参数 N 表示至少保留 N 个**已有**最近用户轮次，此外还会保护本次新输入；不会总结本次输入。
第一版采用这个保守边界，即使压缩更多近期原文可能让请求放得下，也不会自动这样做。

每次新 `reply` / `stream` 开始前，检查“已投影历史 + 当前输入”，位置在 `ContextPolicy`
之后、预算自动丢弃旧轮次之前。预算足够就不调用摘要器，仅输入超预算触发一次尝试。
缺少配置、不支持的内容或其他计数错误直接失败。候选摘要连同保留上下文和当前输入必须
在不进一步裁剪的情况下满足预算，否则报错，不重复摘要，也不静默退回裁剪历史。
没有可压缩前缀时也会报错，但不调用摘要模型。模型自身的重试配置仍可能生效。

流式接口发出 `ContextCompactionStarted`，提交后发出 `ContextCompactionCompleted`；
尝试失败则先发出 `ContextCompactionFailed`，再发终止 `Error`。这些是模型调用前事件，
不占 ReAct 步数。绑定存储时，只有摘要成功保存后才会发完成事件；尝试前的配置等错误
只发终止错误。非流式接口只返回结果或错误。预检失败或取消不会覆盖旧摘要，也不会追加
本次输入；在完成事件后丢弃流会保留已保存摘要，但继续轮询之前不会追加新输入。
后续主模型失败不会撤销已经提交的摘要；存储确认不确定时仍需重新加载确认实际提交结果。

每次新回复最多尝试一次。中途工具步骤，以及确认、重试、外部结果恢复路径继续使用原有
预算行为，不触发自动压缩。待处理检查点仍会拒绝无关新消息。开启自动压缩的新回复会与
同一 Agent 克隆上的操作串行化，不要从 Hook、模型或摘要器重新调用同一 Agent。
独立实例/进程之间仍依赖存储 revision 保护写入。

离线 SDK 示例：`cargo run --example auto_compaction`。CLI 显式开启方式（数值是示例，
不会自动探测模型窗口）：

```shell
cargo run -p agentscope-chat -- --auto-compact 1 --context-window 8192 --output-reserve 1024
```

CLI 会显示压缩事件。真实模式复用 DeepSeek 模型作摘要，摘要窗口设为配置窗口的两倍，
输出预留相同。没有 `--auto-compact` 时不会应用这套自动预算/摘要配置；使用 `--offline`
时采用明确标注的占位摘要器，只验证流程，**不保留语义事实**。估算计数和有损摘要的限制
仍然适用。本次无需升级状态格式。

### 大工具文本卸载与分段读取（可选）

核心提供 `OffloadStore` 接口，`agentscope-offload-file` 插件提供本地文件实现。
默认关闭；通过 `with_tool_result_offload` 启用并自动注册 `read_offloaded_text` 工具。
已有同名工具时拒绝启用。示例配置（应用需依赖本地文件插件）：

```rust
let store = std::sync::Arc::new(
    agentscope_offload_file::FileOffloadStore::new(".agentscope-offload/session-123")?
);
let agent = agent.with_tool_result_offload(
    agentscope::ToolResultOffload::new(store, 8192, 512, 2048)?
)?;
```

三个数分别为卸载阈值、预览上限和单次读回上限，单位均为 UTF-8 字节，不是 Token。
成功的 `ToolResultOutput::Text` 超阈值后，先完整写入存储，再在模型输入中替换为 JSON
引用（ID、原始长度、预览、读取工具名和上限）。JSON 转义后仍会限制引用长度不超过阈值。
模型用 `id`、`offset`、`max_bytes` 读取，按返回的 `next_offset` 继续，直到 `eof`。
偏移必须是 UTF-8 字符边界，读取上限至少为 4 字节；JSON 包装和转义额外占用上下文，
因此仍建议设置 `TokenBudget`，本功能不保证任意请求都能放进模型窗口。

投影位于摘要/历史选择之后、预算检查之前，普通、流式和恢复路径共用；读回页不递归卸载。
原始会话、工具事件、确认检查点和幂等结果不变，工具调用 ID、结果状态及时间戳也保留。
存储失败会停止当前模型请求，不把已成功执行的工具改成失败，也不自动重试工具。
摘要器仍总结原始历史，其独立预算限制不变。

文件插件使用内容哈希去重、临时文件与不可覆盖发布、文件及目录同步；分段读取不会载入
整个文件。文件 I/O 在阻塞线程池执行，中断可能留下已成功写入但尚未使用的文件。
按会话提供私有目录，重启时重新配置相同目录和开关；共享目录会让会话共享读取权限。
目录及祖先目录必须由可信应用控制，本插件不是文件系统沙箱；内容标识不是用户鉴权。
只接受合法内容 ID，不接受任意路径，拒绝符号链接文件。文件含原始敏感内容，不加密，
不要提交或公开目录。仓库已忽略 `/.agentscope-offload/`，自定义目录需自行忽略。

第一版不处理 Blocks/多模态、错误或未完成结果；不提供自动清理、存储配额或分块摘要。
也不减少原始会话存储/内存用量。SDK 离线验收：

```shell
cargo run -p agentscope-offload-file --example offload
```

### 本地 MCP 工具插件（可选）

独立的 `agentscope-mcp` 插件默认不启动程序。通过 `StdioConfig` 明确配置可信程序、
参数和必要环境变量，`McpClient::connect(config).await?` 完成握手，
`client.registry("local").await?` 生成可传给 `ToolExecutor` 的注册表。
工具名为 `local__原名`，沿用确认、流式恢复、幂等包装及大文本结果卸载。

第一版支持协议 `2025-11-25` / `2025-06-18` 的初始化、分页发现、工具调用和 ping，
保留文本/结构化 JSON，校验输入/输出 Schema；不支持的多模态结果显式报错。
同一连接串行调度，有请求时限和帧大小限制。默认不继承环境变量，不转发
`ToolContext` 元数据或幂等键，Server 注解不会自动授予权限。

超时、断连或取消后关闭连接，不自动重连或重试；副作用可能已发生，经过确认的
调用保留“不确定执行”检查点，幂等包装也不会将其缓存为已完成失败。
先核对外部结果再决定恢复，MCP 不保证相同幂等键重试安全。
`close()` 关闭 stdin、等待退出，必要时强制终止直属子进程，不管理整个进程树。
启动 Server 本身就会执行代码，必须信任并授权；本插件不是沙箱。

HTTP/OAuth、动态刷新、资源/提示词和多模态映射留在 TODO；不宣称完整协议兼容。
配置及边界见[插件说明](plugins/agentscope-mcp/README.md)。离线示例（macOS/Linux，
需 `/usr/bin/python3`，无需第三方 Python 包）：
`cargo run -p agentscope-mcp --example stdio`。

### 最小顺序多 Agent Pipeline

`SequentialPipeline` 按固定顺序调用 `Arc<dyn Agent>`，例如“起草 → 审核”：

```rust
let pipeline = agentscope::SequentialPipeline::new(vec![
    std::sync::Arc::new(writer),
    std::sync::Arc::new(reviewer),
])?;
let result = pipeline.run(agentscope::Msg::user("请起草并审核这份说明")).await?;
println!("{}", result.message.text_content("").unwrap_or_default());
```

传入列表必须非空，Agent 名称不可为空或重复。第一位收到原始输入；后续只收到上一位
输出的公开文本（多个文本块以换行连接），作为新的 `User` 消息，来源名为上一阶段名。
不转发原始任务、完整历史、思考块、元数据或 Token 用量，不提升为系统指令；文本本身
仍是不可信输入。仅含思考/空白的中间输出，以及带工具、多模态或结构化内容的混合输出
会停止交接，不静默丢弃这些内容。最终阶段直接保留 Agent 返回的原始消息。

每位 Agent 需配置独立 Memory；使用同一存储后端时，选择不同 `StateKey`。
Pipeline 不读写或合并它们的记忆，但无法检测调用方主动共享的底层 Memory。
同一 Pipeline 及其克隆不允许重叠运行（返回 `Busy`）；不要同时在其他地方运行这些实例。
不同次运行仍使用各自已有的 Agent 历史，不自动清空。

结果包含最终 `message` 和按顺序排列的 `steps`。`PipelineError` 包含一基阶段编号、
Agent 名称、失败原因及已完成输出；确认和“不确定执行”错误原样保留，后续阶段不启动。
中间交接失败时，编号指产生该输出的阶段，该阶段也在 `completed` 中。
这些原始输出可能含思考或私有元数据，不应直接作为公开日志或全部转发给下一个模型。

`pipeline.interrupt_handle().interrupt()` 会停止当前 Pipeline 并丢弃活动 reply future，
不广播中断其他 Agent 任务；直接丢弃运行 future 也停止后续调度。已发生的工具副作用、
内存修改或存储提交不回滚；活动步骤可能需要外部核对，持久化仍取决于 Agent 本身。
第一版只有非流式顺序执行，不是 Agent 的替代实现，也没有 Pipeline 事务、持久化快照、
确认后自动续接或自动重试。每次 `run` 都从第一步开始，**不是恢复**；失败后不要盲目重跑。
可通过保留的 Agent 引用处理确认/核对，然后由调用方显式安排后续工作。

确定性离线示例：`cargo run --example sequential_pipeline`。

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

- [x] 模型输入 `ContextPolicy` 与最近用户轮次裁剪，保持完整持久化历史
- [x] 每次模型请求的输入 Token 预算、输出预留与可替换计数器
- [x] 显式异步历史摘要、独立持久化、校验与失败回滚
- [x] 显式开启、每次新回复最多一次的自动压缩与流式生命周期事件
- [x] 可选的大工具纯文本结果卸载、本地文件插件与有界读回
- [ ] 分块摘要、多模态结果卸载与卸载文件配额/清理策略
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
- [x] 最小非流式顺序 Pipeline 与起草/审核双 Agent 示例
- [ ] Pipeline 流式事件、持久化恢复、并行/路由与任务委派

### 存储插件

- [ ] 在 API 稳定前将 SQLite 提取为 `agentscope-memory-sqlite` crate
- [ ] 添加 `agentscope-memory-postgres` 插件
- [ ] 添加 `agentscope-memory-redis` 插件
- [ ] 支持消息分页、保留期限和过期策略
- [ ] 定义存储迁移与插件兼容策略

### 里程碑 5——互操作

- [x] 独立 MCP Client 插件：本地 stdio、工具发现与调用
- [ ] MCP HTTP/OAuth、动态刷新、资源/提示词与多模态结果映射
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
