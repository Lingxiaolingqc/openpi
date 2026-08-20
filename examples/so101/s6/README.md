# SO-101 S6 policy fault harness 和 rollout 安全模式

本目录提供本机 fake server、无 Isaac protocol probe，以及供 Windows 原生 LeIsaac rollout 使用的 action queue
安全核心。不得使用 WSL 或服务器运行 Isaac Sim，也不得把 fault case 用于已上电真实机械臂。`--s6-safety`
仍是显式 opt-in；不传该参数时保留原有 S5 rollout 路径。

## 新文件

- `fake_policy_server.py`：本机 WebSocket fake server 和 fault injector；
- `probe_client.py`：protocol v1 client、action 检查和预期 fault 判定；
- `safety_log.py`：统一 `S6_EVENT` JSON 日志；
- `action_safety.py`：带 request/response/epoch/TTL 的 action chunk 和 fail-closed queue；
- `IMPLEMENTATION.md`：本批代码边界、安全不变量、测试结果和未完成项。

`probe_client.py` 同时支持本文档使用的文件路径直接执行和
`python -m examples.so101.s6.probe_client` 模块执行。直接执行时会从脚本位置解析仓库根目录，不依赖当前
shell 已额外配置 `PYTHONPATH`；命令仍应从 OpenPI 仓库根目录运行，以便路径和环境保持一致。

## Windows 准备

在 PowerShell 中把仓库的 `openpi-client` 安装到指定 LeIsaac 环境。该操作只安装 Python client 依赖，不运行
Isaac Sim：

```powershell
$env:OPENPI_ROOT = 'D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi'
$env:LEISAAC_PYTHON = 'D:\Envs\leisaac-so101-win\Scripts\python.exe'
Set-Location $env:OPENPI_ROOT

uv pip install --python $env:LEISAAC_PYTHON -e .\packages\openpi-client
```

确认环境。`websockets 12.x` 和仓库 lock 的 `15.x` 都受支持：

```powershell
& $env:LEISAAC_PYTHON -c "import msgpack, websockets; print(websockets.__version__)"
```

## Windows normal case

终端 A 启动本机 fake server：

```powershell
$env:OPENPI_ROOT = 'D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi'
$env:LEISAAC_PYTHON = 'D:\Envs\leisaac-so101-win\Scripts\python.exe'
Set-Location $env:OPENPI_ROOT

& $env:LEISAAC_PYTHON examples\so101\s6\fake_policy_server.py `
  --host 127.0.0.1 `
  --port 18001 `
  --fault normal
```

终端 B 运行 probe：

```powershell
$env:OPENPI_ROOT = 'D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi'
$env:LEISAAC_PYTHON = 'D:\Envs\leisaac-so101-win\Scripts\python.exe'
Set-Location $env:OPENPI_ROOT

& $env:LEISAAC_PYTHON examples\so101\s6\probe_client.py `
  --host 127.0.0.1 `
  --port 18001 `
  --requests 3 `
  --expect-fault none
```

成功时每次 inference 直接向终端输出关联后的 `S6_EVENT`，最后输出：

```text
S6_PROBE_OK: expected_fault=none observed_fault=none
```

## Fault matrix

每行都使用两个终端。终端 A 在 normal 命令基础上替换 server 参数；终端 B 在 normal probe 基础上替换 probe
参数。每次 case 使用新的 server 进程，避免 request count 和 socket 状态跨 case 残留。

| Case | 终端 A 的 server 参数 | 终端 B 的 probe 参数 | 预期结果 |
|---|---|---|---|
| 固定延迟但未超时 | `--fault fixed-delay --delay-s 0.20` | `--inference-timeout-s 1 --expect-fault none` | 正常 response，日志含 latency |
| jitter | `--fault jitter --jitter-min-s 0.05 --jitter-max-s 0.30` | `--inference-timeout-s 1 --expect-fault none` | 正常 response，延迟变化 |
| inference timeout | `--fault inference-timeout --stall-s 10` | `--inference-timeout-s 0.10 --expect-fault inference_timeout` | client fault，socket 关闭 |
| 短时丢 response | `--fault drop-response` | `--inference-timeout-s 0.10 --expect-fault inference_timeout` | client fault，不自动重试 |
| 完全断流 | `--fault disconnect` | `--expect-fault disconnected` | disconnect fault |
| chunk 中途断流 | `--fault disconnect-after-response --disconnect-after-response-s 0.05` | rollout `--s6-safety` | 下一步 watchdog 拒绝旧 queue |
| server 主动退出 | `--fault server-exit` | `--expect-fault disconnected` | server 进程结束，client fault |
| shape 错误 | `--fault bad-shape` | `--expect-fault invalid_action_shape` | action 被拒绝 |
| NaN | `--fault nan-action` | `--expect-fault invalid_action_nonfinite` | action 被拒绝 |
| Inf | `--fault inf-action` | `--expect-fault invalid_action_nonfinite` | action 被拒绝 |
| 越界 | `--fault out-of-range-action` | `--max-abs-action 180 --expect-fault invalid_action_out_of_range` | action 被拒绝、不 clip |
| stale response | `--fault stale-response` | `--expect-fault stale_response` | response 被拒绝，client fault |
| duplicate response | `--fault duplicate-response` | `--requests 2 --expect-fault duplicate_response` | 第二次 request 拒绝遗留 duplicate |

示例：验证 inference timeout。

终端 A：

```powershell
& $env:LEISAAC_PYTHON examples\so101\s6\fake_policy_server.py `
  --host 127.0.0.1 --port 18001 `
  --fault inference-timeout --stall-s 10
```

终端 B：

```powershell
& $env:LEISAAC_PYTHON examples\so101\s6\probe_client.py `
  --host 127.0.0.1 --port 18001 `
  --inference-timeout-s 0.10 `
  --expect-fault inference_timeout
```

## Windows 单元测试

这些测试不启动 Isaac Sim：

```powershell
$env:OPENPI_ROOT = 'D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi'
$env:LEISAAC_PYTHON = 'D:\Envs\leisaac-so101-win\Scripts\python.exe'
Set-Location $env:OPENPI_ROOT

& $env:LEISAAC_PYTHON -m pytest -q -p no:cacheprovider `
  --basetemp "$env:OPENPI_ROOT\tmp\s6-pytest" `
  packages\openpi-client\src\openpi_client\websocket_policy_protocol_test.py `
  packages\openpi-client\src\openpi_client\websocket_client_policy_test.py `
  examples\so101\s6\safety_log_test.py `
  examples\so101\s6\action_safety_test.py `
  examples\so101\s6\fake_policy_server_test.py `
  examples\so101\s6\probe_client_test.py `
  examples\so101\red_cube_to_box_policy_rollout_test.py
```

通用 server 测试需要仓库使用的 `websockets 15.x` OpenPI 环境，而不是 LeIsaac 的 12.x 环境：

```powershell
uv run pytest -q -p no:cacheprovider `
  --confcutdir src\openpi\serving `
  src\openpi\serving\websocket_policy_server_test.py
```

## Windows 原生 LeIsaac chunk 中途断流

该案例只连接同一台 Windows 机器上的 fake server，不连接训练服务器。终端 A：

```powershell
$env:OPENPI_ROOT = 'D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi'
$env:LEISAAC_PYTHON = 'D:\Envs\leisaac-so101-win\Scripts\python.exe'
Set-Location $env:OPENPI_ROOT

& $env:LEISAAC_PYTHON examples\so101\s6\fake_policy_server.py `
  --host 127.0.0.1 `
  --port 18001 `
  --fault disconnect-after-response `
  --disconnect-after-response-s 0.05
```

看到 `S6_FAKE_POLICY_SERVER_READY` 后，在终端 B 启动单环境、headless、`cuda:0` rollout。不要添加
`--renderer_device`：

```powershell
$env:OPENPI_ROOT = 'D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi'
$env:LEISAAC_PYTHON = 'D:\Envs\leisaac-so101-win\Scripts\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$runRoot = "D:\Sim\results\s6-rollout\mid-chunk-disconnect-$stamp"
$env:S6_LOG = Join-Path $runRoot 'rollout.log'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
Set-Location $env:OPENPI_ROOT

& $env:LEISAAC_PYTHON examples\so101\red_cube_to_box_policy_rollout.py `
  --headless `
  --enable_cameras `
  --device cuda:0 `
  --rendering_mode performance `
  --assets_root 'D:\Sim\leisaac-v0.4.0' `
  --policy_host 127.0.0.1 `
  --policy_port 18001 `
  --require_policy_scope simulation-only `
  --episodes 1 `
  --seed 42 `
  --maximum_steps 100 `
  --actions_per_inference 10 `
  --s6-safety `
  --s6-inference-timeout-s 1 `
  --s6-watchdog-timeout-s 0.10 `
  --s6-action-chunk-ttl-s 2 `
  2>&1 | Tee-Object -FilePath $env:S6_LOG
```

预期进程以 fault 失败退出，但必须先依次记录 `rollout_fault_detected`、`action_queue_cancelled`、
`safe_hold_applied`、`simulation_terminated` 和 `recovery_required`。验收字段为
`queued_actions_after=0`、`post_fault_old_action_steps=0`，且没有自动 reconnect。heartbeat 通过后到下一次
heartbeat 前存在最多一个仿真 action step 的未检测窗口；fault 被检测后旧 action 上界严格为零步。

要做正常网络 transport smoke，终端 A 改成 `--fault normal`；终端 B 保留 `--s6-safety`，把
`--maximum_steps` 改为 `20` 并增加 `--minimum_success_rate 0.0`。该 smoke 只检查 S6 协议、watchdog 和 queue
没有破坏正常通信，不是 S5 checkpoint 成功率验收。

## Linux / OpenPI 环境

Linux 只运行 fake/probe 和单元测试；不要在服务器或 WSL 启动 Isaac Sim。

安装 client 并运行 fake server：

```bash
cd /path/to/openpi
uv pip install -e packages/openpi-client

uv run python examples/so101/s6/fake_policy_server.py \
  --host 127.0.0.1 \
  --port 18001 \
  --fault normal
```

另一个终端运行 probe：

```bash
cd /path/to/openpi
uv run python examples/so101/s6/probe_client.py \
  --host 127.0.0.1 \
  --port 18001 \
  --requests 3 \
  --expect-fault none
```

Linux 单元测试：

```bash
cd /path/to/openpi
uv run pytest -q \
  packages/openpi-client/src/openpi_client/websocket_policy_protocol_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py \
  src/openpi/serving/websocket_policy_server_test.py \
  examples/so101/s6/safety_log_test.py \
  examples/so101/s6/action_safety_test.py \
  examples/so101/s6/fake_policy_server_test.py \
  examples/so101/s6/probe_client_test.py \
  examples/so101/red_cube_to_box_policy_rollout_test.py
```

## 日志提取

probe 和 fake server 默认直接输出到当前终端。若一次正式验证已经由外层运行流程保存为 `$env:S6_LOG`，以下
命令只把匹配行输出到终端，不生成额外文件。

PowerShell：

```powershell
Select-String -LiteralPath $env:S6_LOG -Pattern `
'S6_EVENT|S6_FAULT_INJECTION|S6_PROBE_|fault_type|state_from|state_to|request_id|response_id|observation_id|action_chunk_id|connection_epoch|queued_actions_before|queued_actions_after|action_step_|post_fault_old_action_steps|detection_latency_ms|safe_action|recovery_condition'
```

Linux：

```bash
grep -nE \
  'S6_EVENT|S6_FAULT_INJECTION|S6_PROBE_|fault_type|state_from|state_to|request_id|response_id|observation_id|action_chunk_id|connection_epoch|queued_actions_before|queued_actions_after|action_step_|post_fault_old_action_steps|detection_latency_ms|safe_action|recovery_condition' \
  "$S6_LOG"
```

## 当前限制

- probe 不会向 LeIsaac 发送 action；`safe_action=no_action_emitted` 只证明非法 response 没有离开 probe。
- `--s6-safety` 已实现 TTL、逐步 watchdog、queue clear、单步 measured-pose hold 和仿真关闭；当前已有无 Isaac
  单元测试及真实 localhost socket 证据，Windows LeIsaac 动态证据仍需按上面的命令生成。
- 当前逐步 heartbeat 是同步检查；网络若恰好在 pong 后、`env.step()` 期间断开，会在下一步前检测，因此
  物理断流到检测最多存在一个 action step，检测后旧 action 为零步。
- camera freeze 尚未进入本批 fake server；后续应使用 LeIsaac camera frame counter 和 observation fingerprint
  验证，而不是修改相机或任务物理逻辑。
- 不允许把本目录的 fault case 直接用于已上电真实机械臂。
