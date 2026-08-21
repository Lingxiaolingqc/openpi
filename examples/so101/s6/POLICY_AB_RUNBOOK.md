# ACT / OpenPI 正常网络 S6 A/B

本 runbook 用同一套 LeIsaac 条件分别比较 ACT 和 OpenPI 在 S6 关闭/开启时的闭环结果：

1. ACT S6 off；
2. ACT S6 on；
3. OpenPI S6 off；
4. OpenPI S6 on。

这不是把 ACT 与 OpenPI 的训练 step 数直接对齐。OpenPI 使用
`so101-redcube-polar-s4-frame0-100-v1/30000`，ACT 使用已经完成闭环历史验证的 `070000` checkpoint；正式的
S6 回退判定只在同一个 policy/checkpoint 的 off/on 两组之间进行，ACT/OpenPI 横向结果仅作补充。

四组固定 `seed=42`、20 episodes、`maximum_steps=2400`、`actions_per_inference=10`、
`reset_camera_refreshes=1`、expert dynamics、单环境、headless 和 performance。所有进程把物理 GPU 6 映射成
进程内 `cuda:0`，不传 `--renderer_device`。不要同时启动 ACT 和 OpenPI 两个 server；完成一对 A/B 后停止当前
server，再启动另一个，避免额外 GPU 竞争改变 latency。

## 为什么保留 simulation-only

`simulation-only` 不改变 observation、action、模型参数或 inference 路径，因此不是 A/B 变量。它是部署边界：

- server 只有收到精确的 `--confirm-simulation-only SIMULATION_ONLY` 才发布该 scope；
- off/on 两组 rollout 都用 `--require_policy_scope simulation-only` 校验 server metadata；
- metadata 同时发布 `real_robot_deployment_allowed=false`；
- 这些命令和 S6 fault 证据不得被解释为真机部署许可。

## 预检

```bash
cd /home/data/xiaoqinchuan/projects/openpi

export OPENPI_AB_CKPT=/home/data/xiaoqinchuan/checkpoints/openpi/pi05_lora_so101_liftcube/so101-redcube-polar-s4-frame0-100-v1/30000
export ACT_AB_CKPT=/home/data/xiaoqinchuan/checkpoints/act/so101-redcube-frame0-100/full-d09398b/checkpoints/070000

test -s "$OPENPI_AB_CKPT/params/_METADATA" || exit 1
test -f "$OPENPI_AB_CKPT/assets/local/so101-redcube-polar-s4-frame0-100/norm_stats.json" || exit 1
test -d "$ACT_AB_CKPT" || exit 1
test -f "$ACT_AB_CKPT/pretrained_model/SIMULATION_ONLY.json" || exit 1
```

## ACT：终端 A 启动 server

```bash
cd /home/data/xiaoqinchuan/projects/openpi
export ACT_AB_CKPT=/home/data/xiaoqinchuan/checkpoints/act/so101-redcube-frame0-100/full-d09398b/checkpoints/070000
stamp=$(date +%Y%m%d-%H%M%S)
export ACT_AB_SERVER_LOG=/home/data/xiaoqinchuan/results/s6-policy-ab/act-server-$stamp.log
mkdir -p "$(dirname "$ACT_AB_SERVER_LOG")"

CUDA_VISIBLE_DEVICES=6 uv run python examples/so101/act/serve_policy.py \
  --checkpoint "$ACT_AB_CKPT" \
  --device cuda \
  --host 127.0.0.1 \
  --port 18000 \
  --actions-per-inference 10 \
  --confirm-simulation-only SIMULATION_ONLY \
  2>&1 | tee "$ACT_AB_SERVER_LOG"
```

看到 `ACT_POLICY_SERVER_SIMULATION_ONLY` 后运行终端 B。ACT 两组结束后用 `Ctrl-C` 停止这个 server，再启动
OpenPI server。

## ACT：终端 B 依次运行 off/on

```bash
cd /home/data/xiaoqinchuan/projects/openpi
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
stamp=$(date +%Y%m%d-%H%M%S)
export ACT_AB_ROOT=/home/data/xiaoqinchuan/results/s6-policy-ab/act-$stamp
mkdir -p "$ACT_AB_ROOT"

CUDA_VISIBLE_DEVICES=6 "$LEISAAC_ENV/bin/python" examples/so101/red_cube_to_box_policy_rollout.py \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --policy_host 127.0.0.1 \
  --policy_port 18000 \
  --require_policy_scope simulation-only \
  --episodes 20 \
  --seed 42 \
  --maximum_steps 2400 \
  --actions_per_inference 10 \
  --reset_camera_refreshes 1 \
  --reset_camera_warmup_steps 0 \
  --match_expert_dynamics \
  --minimum_success_rate 0.0 \
  2>&1 | tee "$ACT_AB_ROOT/s6-off.log"
act_off_status=${PIPESTATUS[0]}
echo "act_s6_off_status=$act_off_status"

CUDA_VISIBLE_DEVICES=6 "$LEISAAC_ENV/bin/python" examples/so101/red_cube_to_box_policy_rollout.py \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --policy_host 127.0.0.1 \
  --policy_port 18000 \
  --require_policy_scope simulation-only \
  --episodes 20 \
  --seed 42 \
  --maximum_steps 2400 \
  --actions_per_inference 10 \
  --reset_camera_refreshes 1 \
  --reset_camera_warmup_steps 0 \
  --match_expert_dynamics \
  --s6-safety \
  --minimum_success_rate 0.0 \
  2>&1 | tee "$ACT_AB_ROOT/s6-on.log"
act_on_status=${PIPESTATUS[0]}
echo "act_s6_on_status=$act_on_status"
```

## OpenPI：终端 A 启动 30000-step server

```bash
cd /home/data/xiaoqinchuan/projects/openpi
export OPENPI_SO101_LIFTCUBE_REPO_ID=local/so101-redcube-polar-s4-frame0-100
export OPENPI_AB_CKPT=/home/data/xiaoqinchuan/checkpoints/openpi/pi05_lora_so101_liftcube/so101-redcube-polar-s4-frame0-100-v1/30000
stamp=$(date +%Y%m%d-%H%M%S)
export OPENPI_AB_SERVER_LOG=/home/data/xiaoqinchuan/results/s6-policy-ab/openpi-30000-server-$stamp.log
mkdir -p "$(dirname "$OPENPI_AB_SERVER_LOG")"

test -s "$OPENPI_AB_CKPT/params/_METADATA" || exit 1
test -f "$OPENPI_AB_CKPT/assets/$OPENPI_SO101_LIFTCUBE_REPO_ID/norm_stats.json" || exit 1

CUDA_VISIBLE_DEVICES=6 uv run scripts/serve_policy.py \
  --default-prompt "Pick up the red cube and place it inside the green box." \
  --port 18000 \
  --deployment-scope simulation-only \
  --confirm-simulation-only SIMULATION_ONLY \
  policy:checkpoint \
  --policy.config pi05_lora_so101_liftcube \
  --policy.dir "$OPENPI_AB_CKPT" \
  2>&1 | tee "$OPENPI_AB_SERVER_LOG"
```

看到 `Serving a simulation-only policy` 和 `Creating server` 后运行终端 B。

## OpenPI：终端 B 依次运行 off/on

如果刚运行过 ACT 组，保留同一套 LeIsaac 环境变量；否则先执行 ACT 终端 B 开头的 LeIsaac 环境初始化。然后：

```bash
cd /home/data/xiaoqinchuan/projects/openpi
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
stamp=$(date +%Y%m%d-%H%M%S)
export OPENPI_AB_ROOT=/home/data/xiaoqinchuan/results/s6-policy-ab/openpi-30000-$stamp
mkdir -p "$OPENPI_AB_ROOT"

CUDA_VISIBLE_DEVICES=6 "$LEISAAC_ENV/bin/python" examples/so101/red_cube_to_box_policy_rollout.py \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --policy_host 127.0.0.1 \
  --policy_port 18000 \
  --require_policy_scope simulation-only \
  --episodes 20 \
  --seed 42 \
  --maximum_steps 2400 \
  --actions_per_inference 10 \
  --reset_camera_refreshes 1 \
  --reset_camera_warmup_steps 0 \
  --match_expert_dynamics \
  --minimum_success_rate 0.0 \
  2>&1 | tee "$OPENPI_AB_ROOT/s6-off.log"
openpi_off_status=${PIPESTATUS[0]}
echo "openpi_s6_off_status=$openpi_off_status"

CUDA_VISIBLE_DEVICES=6 "$LEISAAC_ENV/bin/python" examples/so101/red_cube_to_box_policy_rollout.py \
  --headless \
  --enable_cameras \
  --device cuda:0 \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --policy_host 127.0.0.1 \
  --policy_port 18000 \
  --require_policy_scope simulation-only \
  --episodes 20 \
  --seed 42 \
  --maximum_steps 2400 \
  --actions_per_inference 10 \
  --reset_camera_refreshes 1 \
  --reset_camera_warmup_steps 0 \
  --match_expert_dynamics \
  --s6-safety \
  --minimum_success_rate 0.0 \
  2>&1 | tee "$OPENPI_AB_ROOT/s6-on.log"
openpi_on_status=${PIPESTATUS[0]}
echo "openpi_s6_on_status=$openpi_on_status"
```

## 提取证据

ACT 与 OpenPI 各运行一次以下命令，把目录变量替换成对应的 `ACT_AB_ROOT` 或 `OPENPI_AB_ROOT`：

```bash
grep -nE \
  'policy_server_metadata:|completed_episodes:|successful_episodes:|success_rate:|policy_episode_results:|mean_inference_latency_ms|rollout_fault_detected|action_queue_cancelled|safe_hold_|post_fault_old_action_steps|RED_CUBE_TO_BOX_POLICY_ROLLOUT_' \
  "$ACT_AB_ROOT/s6-off.log" "$ACT_AB_ROOT/s6-on.log"

grep -nE \
  'policy_server_metadata:|completed_episodes:|successful_episodes:|success_rate:|policy_episode_results:|mean_inference_latency_ms|rollout_fault_detected|action_queue_cancelled|safe_hold_|post_fault_old_action_steps|RED_CUBE_TO_BOX_POLICY_ROLLOUT_' \
  "$OPENPI_AB_ROOT/s6-off.log" "$OPENPI_AB_ROOT/s6-on.log"
```

正常网络下，两组都应连接到相同 scope，完成 20 episodes，并且 S6-on 不出现 transport/camera fault。比较每一对的
`success_rate` 与 inference latency。如果 S6-on 报 `invalid_action_out_of_range` 而 off 组继续执行，这不是允许 clip
或放宽关节边界的理由，而是 checkpoint 与安全执行边界不兼容的证据；该 checkpoint 的 S6-on A/B 应记录为安全
拒绝，随后修正数据/训练或 policy 输出。
