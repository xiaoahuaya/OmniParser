# OpenClaw 计划执行机制学习笔记（完整阅读版）

## 1. 阅读目标与范围

本文聚焦 OpenClaw 在“任务如何被计划、执行、重试、落盘”的完整链路，覆盖：

- 入口与调度：如何避免并发冲突，如何按会话串行
- 执行尝试（attempt）：一次运行内部如何组装上下文、跑工具、处理超时
- 会话与存储：会话 ID/Key、JSONL 转录、`sessions.json` 维护
- 失败恢复：重试、压缩（compaction）、工具结果截断、模型/认证降级
- 可迁移设计：如何集成到 OmniParser/OmniTool 现有架构

---

## 2. 核心结论（先看版）

- OpenClaw 的“计划执行”本质是**队列化的多轮 Agent Loop**，不是单独 planner 模块。
- 核心策略是“双层串行 + 全局并发上限”：
  - 同一会话：严格串行（`session:<key>` lane）
  - 不同会话：允许并发（`main` lane 并发受限）
- 失败处理不是一次性重试，而是**分支化恢复决策树**：
  - 认证刷新/轮换
  - thinking level 自动回退
  - 上下文溢出触发 compaction
  - 工具结果过大触发截断并重跑
- 会话状态与历史持久化拆成两层：
  - `sessions.json`（索引元数据）
  - 每个 `sessionId` 对应 JSONL transcript（完整历史）

---

## 3. 端到端执行链路

```mermaid
flowchart LR
  A["Inbound Message / CLI"] --> B["Resolve Session\n(sessionKey/sessionId/reset/freshness)"]
  B --> C["enqueueSession(session:key)"]
  C --> D["enqueueGlobal(main lane)"]
  D --> E["runEmbeddedPiAgent loop"]
  E --> F["attemptRunEmbeddedPiAgent"]
  F --> G["Session lock + SessionManager init"]
  G --> H["Subscribe lifecycle/tool/assistant events"]
  H --> I["LLM + tool loop"]
  I --> J{"Success?"}
  J -->|Yes| K["Update session store + usage + transcript"]
  J -->|No| L["Retry/compaction/truncate/failover branches"]
  L --> E
  K --> M["Return payloads + status"]
```

---

## 4. 关键模块拆解

## 4.1 入口与队列（防并发冲突）

关键实现：

- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\run.ts`
  - `runEmbeddedPiAgent(...)`
  - 先 `enqueueSession(...)` 再 `enqueueGlobal(...)`
- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\lanes.ts`
  - 规范化 lane key（`session:<key>`）
- `F:\ideacode\openclaw\src\process\command-queue.ts`
  - lane-aware FIFO、并发上限、等待通知、lane 清理

设计要点：

- 保证同一个 `sessionKey` 同时只有一个 run 在写会话文件。
- 全局并发可控，避免所有会话同时打满 LLM/工具资源。
- 用户感知层面可先发“typing/已排队”信号，避免等待无反馈。

## 4.2 Attempt 执行内核（单次尝试）

关键实现：

- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\run\attempt.ts`
  - `SessionManager.open(...)`
  - `prepareSessionManagerForRun(...)`
  - `flushPendingToolResultsAfterIdle(...)`

关键行为：

- 进入 attempt 前会修复/预热 session 文件，减少损坏与冷启动成本。
- 获取会话写锁，防止并发写 transcript。
- 订阅生命周期事件，将流式状态映射到统一事件总线。
- 结束时 flush 工具结果，避免“工具实际完成但结果没落盘”的时间差。

## 4.3 生命周期事件与对外可观测

关键实现：

- `F:\ideacode\openclaw\src\agents\pi-embedded-subscribe.ts`
- `F:\ideacode\openclaw\src\agents\pi-embedded-subscribe.handlers.lifecycle.ts`

特点：

- 生命周期分相位（开始、工具执行、结束、错误）。
- 失败时会格式化为用户可读错误（而不是直接抛底层 provider 异常）。
- 这层是 UI 友好日志、监控事件、统计埋点的最佳插点。

---

## 5. 会话模型与存储机制

## 5.1 Session 解析与重置策略

关键实现：

- `F:\ideacode\openclaw\src\auto-reply\reply\session.ts`
- `F:\ideacode\openclaw\src\config\sessions\session-key.ts`
- `F:\ideacode\openclaw\src\config\sessions\reset.ts`

关键点：

- 根据 channel/chat/thread 计算 `sessionKey`。
- 通过 freshness policy 判断是否沿用旧 `sessionId`。
- `/new`、`/reset`、定时 stale 都可能触发 session 轮换。
- reset 时会归档旧 transcript，避免历史泄漏或文件堆积。

## 5.2 双层持久化

关键实现：

- `F:\ideacode\openclaw\src\config\sessions\store.ts`
- `F:\ideacode\openclaw\src\commands\agent\session-store.ts`

结构：

- `sessions.json`：`sessionKey -> SessionEntry`（索引与运行状态）
- `sessionId.jsonl`：消息/工具/压缩条目（事实历史）

维护能力：

- prune（按时间清理 stale 条目）
- cap（限制最大条目数）
- disk budget（磁盘预算清扫）
- archived transcript 清理（含 reset 归档生命周期）

---

## 6. 重试与恢复决策树（OpenClaw 的强项）

关键实现：

- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\run.ts`
  - 运行循环重试上限：`resolveMaxRunRetryIterations(...)`
  - 上下文溢出：`MAX_OVERFLOW_COMPACTION_ATTEMPTS`
  - compaction 分支决策 + tool result truncate 分支

恢复策略（高层）：

1. 常规失败 -> 继续 run-loop（受总重试次数保护）
2. 认证相关失败 -> 刷新 token/切换 profile 后重试
3. thinking level 不兼容 -> 降级 thinking 后重试
4. context overflow ->
   - attempt 内已压缩：先重跑一次
   - 未压缩：触发显式 compaction
   - 若工具结果超大：截断工具结果再重试
   - 超过上限：给出可解释失败

设计价值：

- 避免“同一种错误无限重试”。
- 将恢复动作显式化并可观测（diag id、branch、attempt/maxAttempt）。
- 失败结果对用户友好，不暴露过多底层噪声。

---

## 7. 与“计划执行”关系的正确理解

在 OpenClaw 中，计划执行不等于单次“生成计划然后照做”，而是：

- 会话分片（sessionKey）
- 队列编排（lane）
- 尝试循环（attempt loop）
- 工具调用与事件流
- 异常分支恢复
- 状态持久化和历史压缩

即：**计划是运行期不断再规划（re-plan）并落盘可恢复的过程**。

---

## 8. 可迁移到 OmniParser 的设计清单

以下是最值得先借鉴的 8 点：

1. 会话 lane 串行 + 全局 lane 并发上限
2. 动作确认门失败后，不重复原动作，进入恢复分支
3. attempt 级统一执行上下文（锁、订阅、超时、flush）
4. 明确错误类型到恢复动作的映射表
5. compaction 与工具输出截断的双保险
6. 双层存储：`session-index.json` + `runs/<sessionId>.jsonl`
7. 生命周期事件标准化（start/step/tool/error/end）
8. 维护任务（prune/cap/disk budget）定期执行

---

## 9. 给 OmniParser 的最小落地蓝图（不改业务功能先改执行骨架）

阶段 1：执行骨架

- 增加 `session_queue.py`（按 `session_key` 串行）
- 增加 `global_queue.py`（全局并发阈值）
- 所有任务统一走 `run_task(session_key, task)` 入口

阶段 2：Attempt 机制

- 增加 `attempt_runner.py`
- 标准化：`prepare -> execute -> classify_error -> recover -> retry`
- 每轮 attempt 产生日志事件和结构化状态

阶段 3：状态与存储

- 增加 `state/sessions.json`
- 增加 `state/transcripts/<session_id>.jsonl`
- 加入 rotate/prune/归档清理

阶段 4：恢复策略

- 先落地 4 类恢复：
  - 焦点不确定：主动探测插入字符 + 回滚校验
  - 点击无变化：执行恢复动作链（Esc/Ctrl+L/URL 或重新定位输入）
  - 上下文过载：摘要压缩 + 继续
  - 工具输出过大：截断并继续

---

## 10. 关键源码/文档索引（本次已阅读）

概念文档：

- `F:\ideacode\openclaw\docs\concepts\agent-loop.md`
- `F:\ideacode\openclaw\docs\concepts\session.md`
- `F:\ideacode\openclaw\docs\concepts\queue.md`
- `F:\ideacode\openclaw\docs\concepts\retry.md`
- `F:\ideacode\openclaw\docs\concepts\compaction.md`
- `F:\ideacode\openclaw\docs\reference\session-management-compaction.md`
- `F:\ideacode\openclaw\docs\tools\llm-task.md`

核心执行代码：

- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\run.ts`
- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\run\attempt.ts`
- `F:\ideacode\openclaw\src\agents\pi-embedded-subscribe.ts`
- `F:\ideacode\openclaw\src\agents\pi-embedded-subscribe.handlers.lifecycle.ts`
- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\lanes.ts`
- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\runs.ts`
- `F:\ideacode\openclaw\src\agents\pi-embedded-runner\session-manager-init.ts`

会话与存储：

- `F:\ideacode\openclaw\src\auto-reply\reply\session.ts`
- `F:\ideacode\openclaw\src\auto-reply\reply\session-run-accounting.ts`
- `F:\ideacode\openclaw\src\auto-reply\reply\session-updates.ts`
- `F:\ideacode\openclaw\src\auto-reply\reply\session-usage.ts`
- `F:\ideacode\openclaw\src\config\sessions\store.ts`
- `F:\ideacode\openclaw\src\config\sessions\reset.ts`
- `F:\ideacode\openclaw\src\config\sessions\session-key.ts`
- `F:\ideacode\openclaw\src\config\sessions\types.ts`
- `F:\ideacode\openclaw\src\commands\agent.ts`
- `F:\ideacode\openclaw\src\commands\agent\session.ts`
- `F:\ideacode\openclaw\src\commands\agent\session-store.ts`
- `F:\ideacode\openclaw\src\process\command-queue.ts`
- `F:\ideacode\openclaw\src\acp\session.ts`

---

## 11. 结论

如果你要把“任务自然语言 -> 稳定自动执行”做成默认能力，最优先不是继续堆 prompt，而是先把 OpenClaw 这套执行控制骨架迁进来：

- 先调度，再执行；
- 先可恢复，再提速；
- 先落盘可追溯，再做更复杂规划。

这会直接降低你当前遇到的三类问题：重复点击死循环、流程偏航、失败不可恢复。

