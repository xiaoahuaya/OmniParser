# OmniTool 7x24 最小可用改造 TODO（通用版）

目标：在不重构主架构的前提下，将现有多节点代理提升到“可持续运行、可恢复、可维护”的 7x24 级别。

## 设计原则（借鉴 OpenClaw）

1. 状态机优先：每次任务必须有清晰阶段（预处理/规划/执行/恢复/完成/失败）。
2. 重试有边界：只对可恢复错误重试，且有次数上限、退避和抖动。
3. 运行前预检：关键依赖（解析服务、节点服务）不可用时不盲目执行。
4. 维护常态化：输出文件清理、健康巡检、错误复位由后台监督线程执行。
5. 可观测性：状态栏可读，聊天区关键事件可追踪，避免噪声日志淹没问题。

## TODO 总览

### P0（必须）

- [x] 任务阶段状态机（phase）落盘并显示在状态栏
- [x] 启动阶段显式“预处理 -> 规划 -> 执行”提示
- [x] 运行失败自动重试（可恢复错误）
- [x] 指数退避 + 抖动（避免重试风暴）
- [x] 每次重试前做运行预检（OmniParser/节点 probe）
- [x] 输出目录自动清理（按保留时长和数量上限）
- [x] 后台 supervisor（无前端交互也持续巡检/维护）

### P1（建议）

- [ ] 任务队列持久化（进程重启后恢复待执行队列）
- [ ] 任务调度器（周期任务/错峰执行/广播策略）
- [ ] 节点熔断（连续失败后临时摘除，冷却后再试）
- [ ] 发布前“人工确认门”（高风险动作二次确认）

### P2（增强）

- [ ] 指标上报（成功率、平均时长、重试次数、节点健康）
- [ ] 告警通道（飞书/企业微信/邮件）
- [ ] 会话压缩与归档策略（长期运行降低上下文膨胀）

## 本次已落地项（对应代码）

1. 运行策略常量（重试、健康巡检、维护清理）
   - `omnitool/gradio/app.py`
2. 节点状态扩展（run_attempt / next_retry_at / health）
   - `omnitool/gradio/app.py`
3. 自动重试执行器（_run_task attempt loop）
   - `omnitool/gradio/app.py`
4. 运行预检 + 健康检查
   - `omnitool/gradio/app.py`
5. 输出目录清理（`./tmp/outputs/<node_id>`）
   - `omnitool/gradio/app.py`
6. 后台监督线程（supervisor）
   - `omnitool/gradio/app.py`

## 推荐默认参数（环境变量）

- `OMNITOOL_RUN_MAX_ATTEMPTS=3`
- `OMNITOOL_RUN_RETRY_BASE_SEC=4`
- `OMNITOOL_RUN_RETRY_MAX_SEC=45`
- `OMNITOOL_RUN_RETRY_JITTER=0.15`
- `OMNITOOL_HEALTH_CHECK_INTERVAL_SEC=12`
- `OMNITOOL_MAINTENANCE_INTERVAL_SEC=300`
- `OMNITOOL_OUTPUT_RETENTION_HOURS=24`
- `OMNITOOL_OUTPUT_MAX_FILES_PER_NODE=2000`
- `OMNITOOL_SUPERVISOR_INTERVAL_SEC=2.5`

## 验收标准（最小）

1. 任一节点出现瞬时网络错误时，任务可自动重试并在状态栏显示倒计时重试。
2. 无前端操作时，后台仍持续进行健康巡检与文件清理。
3. 连续重试失败后，任务转为失败并保留可读错误，不出现无限重试。
4. 状态栏可看到：节点健康、当前阶段、重试尝试次数、计划进度。

