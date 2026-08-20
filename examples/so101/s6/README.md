# SO-101 S6 fake policy server 和 protocol probe

本目录只用于第一阶段的本机 fake server 与无 Isaac 测试。它不会启动 Isaac Sim，不需要连接训练服务器，
也不得用于已上电真实机械臂。当前工具验证协议、有限 timeout、stale/duplicate response 和非法 action 拒绝；
它还没有接入 LeIsaac action queue。

## 新文件

- `fake_policy_server.py`：本机 WebSocket fake server 和 fault injector；
- `probe_client.py`：protocol v1 client、action 检查和预期 fault 判定；
- `safety_log.py`：统一 `S6_EVENT` JSON 日志；
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
  examples\so101\s6\fake_policy_server_test.py `
  examples\so101\s6\probe_client_test.py
```

通用 server 测试需要仓库使用的 `websockets 15.x` OpenPI 环境，而不是 LeIsaac 的 12.x 环境：

```powershell
uv run pytest -q -p no:cacheprovider `
  --confcutdir src\openpi\serving `
  src\openpi\serving\websocket_policy_server_test.py
```

## Linux / OpenPI 环境

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
  examples/so101/s6/fake_policy_server_test.py \
  examples/so101/s6/probe_client_test.py
```

## 日志提取

probe 和 fake server 默认直接输出到当前终端。若一次正式验证已经由外层运行流程保存为 `$env:S6_LOG`，以下
命令只把匹配行输出到终端，不生成额外文件。

PowerShell：

```powershell
Select-String -LiteralPath $env:S6_LOG -Pattern `
'S6_EVENT|S6_FAULT_INJECTION|S6_PROBE_|fault_type|state_from|state_to|request_id|response_id|observation_id|connection_epoch|queued_actions_before|queued_actions_after|detection_latency_ms|safe_action|recovery_condition'
```

Linux：

```bash
grep -nE \
  'S6_EVENT|S6_FAULT_INJECTION|S6_PROBE_|fault_type|state_from|state_to|request_id|response_id|observation_id|connection_epoch|queued_actions_before|queued_actions_after|detection_latency_ms|safe_action|recovery_condition' \
  "$S6_LOG"
```

## 当前限制

- probe 不会向 LeIsaac 发送 action；`safe_action=no_action_emitted` 只证明非法 response 没有离开 probe。
- 还没有 action chunk TTL、逐步 watchdog、queue clear、measured-pose hold 或 simulation terminate 证据。
- camera freeze 尚未进入本批 fake server；后续应使用 LeIsaac camera frame counter 和 observation fingerprint
  验证，而不是修改相机或任务物理逻辑。
- 不允许把本目录的 fault case 直接用于已上电真实机械臂。
