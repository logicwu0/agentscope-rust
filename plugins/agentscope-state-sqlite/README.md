# SQLite state and pipeline stores

This optional workspace plugin provides `SQLiteStateStore` for agent snapshots,
`SQLitePipelineStore` for sequential progress, `SQLiteParallelStore` for parallel
branch progress, and `SQLiteRoutedStore` for explicit routed progress. Each uses
independent tables and schema metadata, so they can share a database and even a
`StateKey` without overwriting each other. Separate agents still need distinct
state keys to preserve independent conversations.

```rust
use agentscope::{RoutedStore, StateKey};
use agentscope_state_sqlite::SQLiteRoutedStore;

let store = SQLiteRoutedStore::open("agentscope.db").await?;
let key = StateKey::new("alice", "route-run-1")?;
let output = pipeline.run_checkpointed(&store, key.clone(), "code", input).await?;
let committed = store.load(&key).await?;
```

The routed store persists the complete original input and status payload,
including full replies, thinking/data blocks, metadata, usage, and structured
agent failures. It uses `agentscope_routed_schema` (schema version 1) and
`agentscope_routed_checkpoints`, keyed by user and session. `save` uses an
immediate transaction for compare-and-swap: `None` creates only when absent;
`Some(revision)` updates only that revision. A successful write returns the exact
supplied checkpoint with its incremented positive revision. Independent
connections use a five-second busy timeout. Loading corrupt JSON or nonpositive
stored revisions fails. Unsupported schemas and revision overflow are rejected
without automatic migration or overwriting the existing record.

The storage layer does not validate workflow metadata. The core pipeline checks
checkpoint version, route/captured-name binding and safe recovery state. Names
are symbolic compatibility checks, not proof of identical models, credentials,
policies or agent state. Resume executes
only `Ready`; `InFlight` requires external verification and explicit reconciliation,
and terminal progress is read rather than replayed. The store does not approve
tools, retry an uncertain execution or snapshot the selected agent. Bind that
agent to its own durable state when needed; agent and routed writes are separate
transactions. Stop the original worker before recovery and re-read the store
after an ambiguous write error. Protect this database as private application data.
Routed checkpointed streaming remains a future extension.

## 简体中文

本插件提供 Agent 快照、顺序 Pipeline、并行 Pipeline 和显式路由的四种 SQLite
存储。各自使用独立表和 schema 元数据，同一数据库、同一个 `StateKey` 也不会互相
覆盖；不同 Agent 的会话仍应选择不同状态 key。

`SQLiteRoutedStore` 使用 schema v1 的 `agentscope_routed_schema` 和
`agentscope_routed_checkpoints`，完整保存原始输入、回复、思考/数据块、元数据、
Token 用量与结构化 Agent 错误。即时事务内比较 revision：`None` 仅新建不存在的
记录，`Some(revision)` 仅更新匹配版本；成功后原样返回检查点和递增的正 revision。
独立连接使用五秒 busy timeout。读取损坏 JSON 或非正 revision 时返回错误。
未知 schema 和 revision 溢出会被拒绝，不自动迁移或覆盖已有记录。

存储层不判断流程元数据是否有效；核心 Pipeline 校验版本、route/捕获名称绑定与恢复
状态。名称仅提供符号兼容性，不证明模型、凭证、策略或 Agent 状态相同。
仅 `Ready` 可恢复执行，`InFlight` 需外部核对并显式提交结果；终态读取而不重跑。
插件不会批准工具、重试不确定执行或保存所选 Agent 的记忆。需要时，Agent 仍需单独
绑定持久化状态，Agent 与路由写入不构成原子事务。恢复前停止原工作进程，写入结果
不确定时重新读取记录。数据库可能含私有数据，应作为应用数据保护。路由带检查点
的流式执行留作后续扩展。
