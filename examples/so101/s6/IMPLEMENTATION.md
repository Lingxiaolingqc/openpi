# SO-101 S6 第一批实现记录

日期：2026-08-20

## 本批范围

本批实现 S6 的协议与有限等待基础，只使用 fake policy server 和无 Isaac 单元测试。没有启动 Isaac Sim，
没有连接远程服务器，也没有接触真实机械臂。S5 checkpoint 闭环成功率验收仍是独立工作，本批 S6 结果不能
替代 S5 验收。

本批没有修改 `red_cube_to_box_policy_rollout.py` 的动作执行循环、任务物理逻辑、joint limits、real recorder、
camera capture 或数据 converter。action queue 中止、仿真 safe hold、camera freeze 检测和 LeIsaac fault injection
属于下一批。

## 代码改动

### 协议 envelope

新增 `packages/openpi-client/src/openpi_client/websocket_policy_protocol.py`：

- 定义协议版本 1；
- request 包含 `request_id`、`client_session_id`、`connection_epoch`、`observation_id` 和
  `request_created_unix_ns`；
- response 回传全部 request 关联字段，并增加 `response_id`、`server_received_unix_ns` 和
  `server_completed_unix_ns`；
- 校验字段类型、非空 ID、非负 epoch、正 timestamp 和服务端 timestamp 顺序；
- 支持不含 envelope 的 legacy request；
- protocol error response 只返回异常类型和消息，不把服务端 traceback 暴露给 v1 client。

### 有界、fail-closed WebSocket client

修改 `packages/openpi-client/src/openpi_client/websocket_client_policy.py`：

- connect 总等待从无限循环改为默认最多 60 秒；
- 单次 connect、metadata、send、inference recv、heartbeat 和 close 都有正数有限上界；通用 client 的默认
  inference 上界为 120 秒，以容纳首次模型编译，S6 probe 使用更短的显式 case timeout；
- send 在独立 daemon thread 中执行；超时后直接 shutdown/close 底层 socket，使阻塞发送不能无限遗留；
- 区分 connect timeout、metadata timeout、send timeout、inference timeout、heartbeat timeout、disconnect、
  invalid request、invalid response、stale response、duplicate response 和 server error；
- transport、protocol 或 server response fault 后进入 `FAULTED` 并关闭当前连接；
- fault 后 `infer()` 不会自动重试，也不会复用当前连接；
- 只有显式调用 `reconnect()` 才建立新连接，同时递增 `connection_epoch`；
- response 必须匹配当前 request、session、epoch、observation 和 request timestamp；
- 已接受的 `response_id` 会被记录，重复 ID 被拒绝；
- 增加显式 `health_check()`、`ensure_ready()` 和有界 `close()`；
- 默认仍发送 legacy payload；只有显式指定 `protocol_version=1` 才启用严格 envelope，保留旧调用形式。

### 通用 policy server

修改 `src/openpi/serving/websocket_policy_server.py`：

- metadata 宣告 `openpi_protocol_versions: [1]`；
- 同时接受 legacy request 和 v1 envelope；
- v1 response 自动回传关联字段；
- policy 输出必须是 mapping；
- server send 和错误后的 close handshake 有默认 5 秒上界；
- v1 policy exception 返回关联 error response；legacy client 继续收到原有文本异常帧；
- 不修改 policy inference 语义，也不尝试在线程中取消 GPU inference。

### fault injection 和证据日志

新增 `examples/so101/s6/`：

- `fake_policy_server.py`：支持 normal、固定延迟、jitter、inference timeout、丢 response、断连、主动退出、
  action shape、NaN、Inf、越界、重复 response 和 stale response；
- `probe_client.py`：运行 protocol v1 client，检查 response 关联及 action shape/finite/范围，按预期 fault 自动
  判定通过或失败；任何非法 action 都是拒绝，不做 clipping；
- `safety_log.py`：定义 `S6_EVENT` JSON schema，包括状态转换、fault type、request/response/observation/action
  关联、connection epoch、queue 清空计数、检测延迟、安全动作和恢复条件；
- `README.md`：只记录新工具的安装、运行、fault matrix、测试和日志提取方法。

## 安全不变量

1. 所有 client 网络等待都有有限上界，timeout 参数拒绝 `None`、非有限值和非正数。
2. timeout、disconnect、invalid、stale 或 duplicate response 后，当前连接只能处于 `FAULTED`。
3. client 不自动 reconnect；显式 reconnect 后 connection epoch 必须变化。
4. v1 response 不能跨 request、session 或 connection epoch 使用。
5. fake/probe 对非法 action 只拒绝和记录，绝不修复或 clip。
6. 本批 probe 不调用仿真 `env.step()`，因此 fault case 的 safe action 是 `no_action_emitted`。
7. 本批没有声称 action queue 已可中止；该证据必须在 rollout 集成后单独验收。

## 验证结果

- Ruff check：通过。
- Ruff format check：通过。
- `websockets 15.0.1`：client/protocol/fake/log/probe 37 项通过，通用 server 3 项通过。
- Windows LeIsaac Python、`websockets 12.0`：相同 client/protocol/fake/log/probe 37 项通过。
- 本机真实 WebSocket normal case：action shape `(10, 6)`，request ID 与 response request ID 一致。
- 修改后的通用 `WebsocketPolicyServer` 与 protocol v1 client 完成真实本机连接，metadata 宣告 `[1]`、原始
  policy payload 未混入 envelope、request 关联一致且保留 `server_timing`。
- 固定 300 ms 延迟、50 ms inference timeout：约 62 ms 检出 `inference_timeout`，client 状态为
  `faulted`，底层 socket 已关闭。
- bad shape `(10, 7)`：probe 输出 `invalid_action_shape`、`response_rejected` 和
  `safe_action=no_action_emitted`。
- duplicate response：第一条 response 接受，第二次 request 前遗留的重复 `response_id` 被拒绝为
  `duplicate_response`。

这些验证不包含 Isaac Sim，也没有验证 chunk 执行中断流时的最大旧动作步数。

## 第一批结束时的下一批边界

以下列表记录第一批结束时的计划；1 到 7 均已进入代码，其中第 6 项 camera freeze 仍需 Linux LeIsaac 动态
证据：

1. action chunk 增加来源 request/response/epoch、接收时间和 TTL；
2. 每个 `env.step()` 前检查 client watchdog 状态、chunk epoch 和 TTL；
3. fault 后将剩余 queue 原子清零并记录 `queued_actions_before/after`；
4. 执行 measured-pose hold，随后 pause/terminate simulation；
5. 越界 action 在 S6 模式下直接拒绝，不走现有 S5 clipping 路径；
6. 使用 camera sensor frame counter 和 observation fingerprint 注入/检测冻结；
7. 以 `actions_per_inference=10` 证明 fault 后旧动作继续执行为零步，或给出明确的一步检测上界。

## 直接执行入口修复（2026-08-20）

Linux 按 README 运行 `uv run python examples/so101/s6/probe_client.py` 时，Python 只把脚本目录加入
`sys.path`，导致 `from examples.so101.s6 import safety_log` 在导入阶段失败。`probe_client.py` 现在沿用仓库
其他 SO-101 命令行脚本的入口方式：仅当作为文件直接执行时，根据 `__file__` 将仓库根目录加入
`sys.path`；作为 package 导入或使用 `python -m` 时不修改路径。新增子进程回归测试，在没有仓库根目录
`PYTHONPATH` 的条件下执行 `probe_client.py --help`，验证直接文件入口可以完成所有导入。

## 阶段 C：action chunk fail-closed 执行（2026-08-20）

新增 `action_safety.py`，将每个计划执行的 action chunk 绑定 protocol v1 的 request ID、response ID、
observation ID、client session、connection epoch、服务端时间戳、接收时间和有限 TTL。queue 不允许在旧 chunk
仍有剩余 action 时被新 chunk 覆盖，不允许 epoch 变化或 TTL 过期后的 action 离开队列；fault cancellation
将剩余数量原子变为零，检测后任何 action 标记都会作为独立安全不变量违规。

`red_cube_to_box_policy_rollout.py` 新增 opt-in `--s6-safety`：

- 强制 WebSocket protocol v1，并向 adapter 传入有限 connect/send/recv/heartbeat/close timeout；
- 每个 `env.step()` 前同步执行 heartbeat，再检查 chunk epoch 和 TTL；
- shape、empty、NaN/Inf 和 soft-limit 越界全部拒绝；S6 不进入原有 clip 路径；
- 允许与 `--match_expert_dynamics` 同时启用：保留 gravity、damping 和合法 raw target 执行语义，但严格拒绝
  越界 raw target 后才允许动作进入 queue；
- 每个 chunk 和 action step 输出 request/response/observation/chunk ID 及时间关联 `S6_EVENT`；
- 任意 fault 后记录 `fault_detected → action_queue_cancelled → safe_hold → simulation_terminated →
  recovery_required`，queue 清零后执行一次 measured-pose hold，再关闭环境和 SimulationApp；
- 不调用 reconnect。恢复只能重新启动进程并再次显式传入 `--s6-safety`。

fake server 新增 `disconnect-after-response`：先返回合法 10 步 chunk，再经过有限延迟关闭连接，用于在下一步
heartbeat 验证 queue cancellation。由于同步 heartbeat 与 `env.step()` 之间仍存在竞态，物理断流到检测的
理论上界是一帧；fault 检测后旧 action 执行上界为零步。

无 Isaac 测试覆盖 TTL、epoch、correlation、queue replacement、严格越界拒绝和完整 safe-state 顺序。
本机真实 WebSocket 验证得到：服务端返回 10 步 chunk 后 50 ms 断开，下一次 heartbeat 检测
`disconnected`，client 状态为 `faulted`，queue `10 → 0`，`post_fault_old_action_steps=0`。本批未启动 Isaac
Sim；Windows 原生 LeIsaac 动态证据仍需按 `s6/README.md` 命令生成。

提交前验证：Ruff check 和 format check 通过；`websockets 16.0` 的完整 client/server/S6/rollout 相关测试
`65 passed`；Windows LeIsaac Python 下不启动 Isaac 的新增 action-safety/rollout 测试 `23 passed`。这些结果
不能替代 Windows 原生 LeIsaac 中途断流动态验收。

## Linux 测试入口与同动力学 A/B 修正（2026-08-21）

Linux 从仓库根目录直接执行 `uv run pytest` 时，pytest console script 所在目录可能排在仓库根目录之前，导致
测试收集阶段无法导入 namespace package `examples.so101`。README 中的 Linux 和通用 server 测试入口统一改为
`uv run python -m pytest`，由 Python 把当前仓库根目录放入 `sys.path`；不通过全局 `PYTHONPATH` 或修改任务包
结构掩盖入口问题。

删除 `--s6-safety` 与 `--match_expert_dynamics` 的参数互斥。组合模式仍关闭 robot gravity、写入 joint damping
`10.0`，并对通过 S6 shape/finite/soft-limit 检查的 raw target 原值执行；任何越界 target 先抛出
`invalid_action_out_of_range`，不会 clip、入 queue 或调用 `env.step()`。新增纯 Python 回归测试覆盖组合模式的
合法 raw target 不变，以及 `reject / execute_raw / clip` 三种启动日志语义。

## 阶段 D：camera freeze observation safety（2026-08-21）

新增 `camera_safety.py`，不修改 LeIsaac camera 或任务物理：

- 从当前 IsaacLab 2.x `SensorBase._timestamp_last_update` 读取最后一次完成 camera buffer 更新的 token；若未来
  sensor 提供 public frame counter，则优先读取 public counter；缺失、空值或非有限 token 都 fail closed；
- 对每个 policy camera 取固定 16×16 RGB grid 并计算 BLAKE2 fingerprint，避免每个控制步把完整 640×480
  GPU frame 搬到 CPU；只有 update token 与 fingerprint 同时不变才累计 stale step；
- LeIsaac camera 为 30 FPS、control 为 60 FPS，`--s6-camera-max-stale-steps` 默认允许一次重复；第二次连续
  重复触发 `camera_freeze`；`detection_latency_ms` 从第一个可观测 stale observation 起算，检测前旧动作
  保守上界为 2 步，检测后旧动作为 0；
- 每个 episode reset 后建立 camera baseline；每次 `env.step()` 返回并标记当前 action 已执行后立即检查新
  observation，检查完成前不会授权下一步 action；
- `--s6-inject-camera-freeze-after-step N` 只在 simulation rollout 中冻结送给 policy/freshness monitor 的
  camera tensor 和监控 token，保留 joint、subtask 等 observation，不改变真实 sensor buffer、render 或物理；
- 正常路径记录 `camera_monitor_initialized` 和 `camera_observation_checked`；注入记录
  `camera_freeze_injected`；检测后复用既有 queue clear、measured-pose hold、terminate 和 manual recovery 链。

纯 Python 测试覆盖 token 提取、缺失 token fail-closed、fingerprint、正常 30/60 FPS cadence、两信号联合判定、
注入范围、有限步检测以及 camera 特定旧动作上界进入统一 fault 日志。Linux LeIsaac normal transport 和
camera-freeze 注入均已生成会话日志摘录：normal 最大 stale step 为 1 且无误报，freeze case 在 22.036 ms 检测并
将 queue `8 -> 0`，检测后旧 action 为 0。完整 Phase D 状态、correlation key 和待补案例见
[`PHASE_D_EVIDENCE.md`](PHASE_D_EVIDENCE.md)。
