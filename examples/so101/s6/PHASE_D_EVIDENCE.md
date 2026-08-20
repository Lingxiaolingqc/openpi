# S6 Phase D Linux LeIsaac 动态证据矩阵

更新日期：2026-08-21。

## 证据口径

本文件把 2026-08-21 会话中提供的 Linux LeIsaac 日志摘录整理成统一矩阵。当前状态标记含义：

- **通过（摘录）**：摘录包含该案例所需的 fault type、检测延迟、queue clear、检测后旧 action 计数和安全终态；
- **部分通过**：核心拒绝或 safe hold 已出现，但当前摘录缺少完整终态或来自修复前运行；
- **待补**：尚未提供该注入模式的 Linux 动态日志；
- 这些摘录不是仓库内保存的原始日志。正式验收时仍应保留 `$S6_LOG` 完整文件、运行命令、commit 和
  `rollout_status`，避免只依赖终端复制文本。

除真实 policy baseline 外，以下 rollout 使用 fake policy server。fake server 的确定性 action 不完成任务，因此
`success_rate=0` 不表示 S6 回退，也不能用于证明真实 policy 的任务成功率。

安全终态缩写 `F -> Q -> H -> T -> R` 依次表示 `rollout_fault_detected`、`action_queue_cancelled`、
`safe_hold_applied`、`simulation_terminated` 和 `recovery_required`。

后续补测统一使用 `CUDA_VISIBLE_DEVICES=6` 映射进程内 `--device cuda:0`，保持单环境、headless、
`actions_per_inference=10`，且不添加 `--renderer_device`。历史摘录没有完整保留 GPU 启动行，因此本矩阵不反推
旧运行的物理 GPU 编号。

## 正常路径

| Case | 观测结果 | Camera freshness | Action/queue | 状态 |
| --- | --- | --- | --- | --- |
| normal transport + camera monitor | `RED_CUBE_TO_BOX_POLICY_ROLLOUT_OK`；1 episode、20 steps、2 次 inference，mean inference 4.726 ms | injection 为 `null`；允许 1 个 stale step，实际最大值 1；末步 frame token 为 11、stale count 为 0 | 2 个 10-step chunk 正常执行；末步 queue `1 -> 0`；clip、soft-limit violation 和 post-fault old action 均为 0 | **通过（摘录）**；摘录未包含外层 `rollout_status=0` 行 |
| fixed delay 0.20 s | 两次 inference latency 为 207.3/205.2 ms；20 steps 后 `ROLLOUT_OK`，无 fault event | 当时摘录未包含 camera monitor 字段 | 两个 chunk 均完整执行 | **通过（摘录）**；server case 命令未包含在摘录中 |
| jitter | 两次 inference latency 为 216.9/61.6 ms；20 steps 后 `ROLLOUT_OK`，无 fault event | 当时摘录未包含 camera monitor 字段 | 两个 chunk 均完整执行 | **通过（摘录）**；server case 命令未包含在摘录中 |

normal fake-server smoke 只证明 protocol、watchdog、queue 和 camera freshness 集成没有误报。Phase D 的“正常网络
baseline 成功率没有明显回退”仍必须连接真实 policy server/checkpoint，以固定 seed、episode 和
`actions_per_inference=10` 与 S6 前基线比较。

## Fault rollout

| 注入案例 | 实际 fault type | Detection latency | Queue before/after | Post-fault old action | 安全终态 | 状态 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| camera freeze after step 1 | `camera_freeze` | 22.036 ms | `8 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）** |
| heartbeat timeout with active chunk | `heartbeat_timeout` | 104.382 ms | `8 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）** |
| inference/recv timeout | `inference_timeout` | 103.114 ms | `0 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）** |
| disconnect before response | `disconnected` | 6.833 ms | `0 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）** |
| server exit | `disconnected` | 5.065 ms | `0 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）** |
| stale response | `stale_response` | 6.243 ms | `0 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）** |
| duplicate response | `duplicate_response` | 4.277 ms | `0 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）**；使用 defer-until-next-request 修复后的运行 |
| bad action shape | `invalid_action_shape` | 6.804 ms | `0 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）** |
| NaN action | `invalid_action_nonfinite` | 5.318 ms | `0 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）** |
| out-of-range action | `invalid_action_out_of_range` | 6.421 ms | `0 -> 0` | 0 | `F -> Q -> H -> T -> R` | **通过（摘录）**；60 个越界值被拒绝，没有 clip 或 `env.step()` |
| disconnect after response（修复前摘录） | `disconnected` | 0.521 ms | `8 -> 0` | 0 | 摘录到 `safe_hold_applied`，未包含 `T -> R` | **部分通过**；应在当前 commit 重跑完整终态 |

queue 非空清除已经由 camera freeze 和 heartbeat timeout 两个当前完整案例证明：检测时均剩余 8 个 action，取消后
立即变为 0，且 `rollout_fault_detected` 后旧 chunk 的执行数为 0。queue 原本为空的案例证明 response/action 在
进入执行队列之前被拒绝，不能替代非空 queue cancellation 证据。

## Correlation 证据

以下值为日志中 UUID 的前 8 位，仅用于快速定位完整日志；`-` 表示 fault 发生前没有合法 response 被接受，因此
`response_id=null` 是预期结果，不是缺失关联。

| Case | observation | request | response | action chunk | epoch |
| --- | --- | --- | --- | --- | ---: |
| camera freeze | `73a47ec9` | `a6d7d923` | `b2c9bc60` | `b2c9bc60` | 0 |
| heartbeat timeout | `cb073f37` | `2c899a31` | `169ed9d7` | `169ed9d7` | 0 |
| inference timeout | `e2cb6203` | `1628cdaa` | `-` | `-` | 0 |
| disconnect | `0c8415cf` | `fb8e8b1a` | `-` | `-` | 0 |
| server exit | `775a60f4` | `2dcdaea4` | `-` | `-` | 0 |
| stale response | `f7db86fe` | `b2a2ee98` | `-` | `-` | 0 |
| duplicate response | `0378d63e` | `b4e546d7` | `-` | `-` | 0 |
| bad shape | `17cb291f` | `9082bd60` | `a0138f9a` | `-` | 0 |
| NaN action | `9e1e83b3` | `c30dfef8` | `f8c70fdc` | `-` | 0 |
| out-of-range action | `253d58bf` | `8b342581` | `2c1427e9` | `-` | 0 |

stale/duplicate response 的 fault event 保留 in-flight observation/request，错误 message 记录收到但被拒绝的
request/response ID。shape、NaN 和越界案例保留合法 transport response ID，但不会创建 action chunk。

## 尚未关闭的 Phase D 项目

| 项目 | 当前证据 | 下一步 |
| --- | --- | --- |
| 当前 commit 的 `disconnect-after-response` | 修复前运行已证明 queue `8 -> 0` 和 hold，但摘录缺少完整 `T -> R` | 用新 fake-server 进程重跑并保存完整 `$S6_LOG` |
| `drop-response` injector | inference-timeout fault class 已通过，但没有该 injector 的独立 Linux 摘录 | 运行 `--fault drop-response`，预期 `inference_timeout` |
| `inf-action` injector | `invalid_action_nonfinite` fault class 已由 NaN 证明，但没有 Inf injector 的独立 Linux 摘录 | 运行 `--fault inf-action`，预期同一 fault type 且零 action 执行 |
| connect/metadata/send timeout | 纯 Python/localhost 测试覆盖有限等待；当前矩阵只有 recv 和 heartbeat 的 LeIsaac 动态证据 | 若 Phase B 要求逐项 Linux 动态证据，分别保存 connect/metadata/send case 日志 |
| TTL 和旧 epoch chunk | 单元测试覆盖 TTL、epoch 和 reconnect 后旧 chunk 拒绝；当前无 Linux LeIsaac 动态摘录 | 仅在正式验收要求动态注入时补充，不允许自动 reconnect 掩盖旧 epoch |
| 真实 policy baseline | fake-server normal smoke 通过，但任务成功率为 0，不能比较策略能力 | 使用真实 checkpoint、相同 seed/episode/dynamics 对比 S6 前后成功率和 latency |
| 原始日志归档 | 当前矩阵来自会话摘录 | 保存每个 case 的命令、commit、`rollout_status`、server log 和 rollout log |

## 当前结论

camera normal/freeze、非空 queue clear、检测后零旧 action、measured-pose hold、受控仿真终止和人工恢复要求均有
Linux LeIsaac 动态证据。当前不能把 Phase D 标记为完全关闭：至少还缺真实 policy baseline；若验收口径要求
fake server 的每个 injector 都必须独立运行，还需补 `drop-response`、`inf-action` 和当前 commit 的
`disconnect-after-response` 完整日志。
