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

## 下一批边界

下一批才修改 `red_cube_to_box_policy_rollout.py`，建议只以可选 `--s6-safety` 模式接入：

1. action chunk 增加来源 request/response/epoch、接收时间和 TTL；
2. 每个 `env.step()` 前检查 client watchdog 状态、chunk epoch 和 TTL；
3. fault 后将剩余 queue 原子清零并记录 `queued_actions_before/after`；
4. 执行 measured-pose hold，随后 pause/terminate simulation；
5. 越界 action 在 S6 模式下直接拒绝，不走现有 S5 clipping 路径；
6. 使用 camera sensor frame counter 和 observation fingerprint 注入/检测冻结；
7. 以 `actions_per_inference=10` 证明 fault 后旧动作继续执行为零步，或给出明确的一步检测上界。
