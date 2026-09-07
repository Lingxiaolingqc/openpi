# PiPER + MuJoCo + π0.5

这里实现了一个明确限定为仿真的 AgileX PiPER RedCubeToBox 学习链路：Windows 运行 MuJoCo、IK
专家、数据采集和闭环客户端，Linux GPU 服务器运行 LeRobot 转换、π0.5 LoRA 训练和 policy server。
当前工作区是 `D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi-piper`，分支是
`piper-mujoco-pi05`。

模型来自 [MuJoCo Menagerie 的 agilex_piper](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper)，
本仓库不复制其 mesh。复现实次验证建议固定 Menagerie commit
`8161bba264d7fa7c99ca301e91e7fb44737676ad`。

## 已实现与当前证据

- `PiperRedCubeToBoxEnv`：固定第三人称 `224×224` RGB，带 seed 的方块/目标随机化，500 Hz 物理、
  20 Hz 控制，每个控制动作推进 25 个物理步。
- 位置执行器和夹爪映射、应用层软限位、逐步速率限制，以及策略故障后的“取消余下 chunk + 保持
  最新测量姿态”。
- 阻尼最小二乘 IK 和八阶段专家：预抓取、下降、闭合、抬升、搬运、下降、释放、撤离。
- 一条 episode 一个不可变 HDF5 文件；逐帧保存 pre-step 图像、7D 状态、实际执行的 7D 绝对动作、
  单调时间戳和阶段标签。
- HDF5 审计、按 episode 的 160/40 划分、源文件 SHA-256、LeRobot 转换、`SIMULATION_ONLY` 标记。
- PiPER 专用 OpenPI policy adapter 和 `pi05_piper_lora`；模型内部 `action_dim=32`，只让前 7 维离开
  adapter，缺失腕部相机明确 mask 掉。
- policy server 的 simulation-only 元数据校验、首动作离线 MAE/RMSE、10 步 action chunk 闭环和视频。

本地已经完成的行为证据：

- 专家 seeds 1000–1099：`97/100`，达到专家门槛 95%。
- pilot：20 条成功轨迹，审计得到 16 条训练、4 条验证、1977 个训练帧；seed 0 的确定性回放最大
  状态误差为 0。
- full：200 条成功轨迹，采集用了 204 次尝试；审计得到 160 条训练、40 条验证、19899 个训练帧。
  batch size 8 对应 1/3/5 个 equivalent passes 分别是 2488/7464/12440 steps。
- 这些数据在 `D:\Sim\results\piper-mujoco-pi05`，没有提交进 Git。

尚未完成的不是代码编写，而是依赖 Linux GPU 服务器的动态门槛：实际 LeRobot 转换、norm stats、
π0.5 forward/backward、overfit-10、完整 LoRA、policy server 和学得策略的 20-seed 闭环成功率。本地
Windows 没有可用的 OpenPI GPU 环境，当前任务也没有服务器连接信息，因此不能把这些项写成已通过。

## 接口为什么是 7D，而不是原始 qpos 的维度

官方 PiPER 本体有 6 个机械臂关节和两根手指。两根手指通过 equality constraint 联动，执行器只有
6 个机械臂位置执行器加 1 个夹爪执行器，所以公共策略动作是：

```text
state  = [q1, q2, q3, q4, q5, q6, gripper_open_fraction]
action = [q1_target, ..., q6_target, gripper_open_fraction_target]
```

前六维单位是 rad；夹爪把独立手指关节的 `0..0.035 m` 归一化到 `[0,1]`。动作是绝对关节目标，
但训练 transform 只把前六维变为“相对当前状态的增量”；夹爪仍是绝对值。推理输出经过逆 transform
后重新成为 7D 绝对目标。

加入自由方块以后，整个任务模型是 `nq=15`，但多出的 7 个 free-joint qpos 属于方块，不属于机器人
控制接口。`qpos` 描述仿真广义坐标，`ctrl` 是 7 个执行器命令，policy action 是人为定义且校验过的
7D 边界，三者不能按数组长度混为一谈。

## Windows：MuJoCo 环境和专家

建议用 Python 3.11 建一个独立环境，不要把训练环境装到 Windows 挂载的 Linux 目录：

```powershell
cd D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi-piper
py -3.11 -m venv .venv-mujoco
.\.venv-mujoco\Scripts\python.exe -m pip install -r examples\piper\requirements-mujoco.txt
.\.venv-mujoco\Scripts\python.exe -m pip install -e packages\openpi-client

git clone https://github.com/google-deepmind/mujoco_menagerie D:\Sim\mujoco_menagerie
git -C D:\Sim\mujoco_menagerie checkout 8161bba264d7fa7c99ca301e91e7fb44737676ad
$env:MUJOCO_MENAGERIE_PATH = 'D:\Sim\mujoco_menagerie'
```

先打印 joint、actuator 和频率；去掉 `--headless` 会打开 20 秒 Viewer：

```powershell
.\.venv-mujoco\Scripts\python.exe -m examples.piper.inspect_model --headless
.\.venv-mujoco\Scripts\python.exe -m examples.piper.inspect_model --seconds 20
```

验证专家 100 个 seed。只有最后的 `PIPER_EXPERT_EVAL_OK` 才代表达到 `>=95/100`：

```powershell
.\.venv-mujoco\Scripts\python.exe -m examples.piper.evaluate_expert `
  --episodes 100 --start-seed 1000
```

先收 20 条 pilot，再审计、回放和画曲线：

```powershell
$pilot = 'D:\Sim\results\piper-mujoco-pi05\pilot20'
.\.venv-mujoco\Scripts\python.exe -m examples.piper.collect_demos `
  --output-dir $pilot --episodes 20 --start-seed 0
.\.venv-mujoco\Scripts\python.exe -m examples.piper.convert_hdf5_to_lerobot `
  --input-path $pilot --dry-run
.\.venv-mujoco\Scripts\python.exe -m examples.piper.replay_episode `
  --episode "$pilot\episode_000000_seed_00000000.h5" `
  --video "$pilot\replay_seed0.mp4"
.\.venv-mujoco\Scripts\python.exe -m examples.piper.plot_episode `
  --episode "$pilot\episode_000000_seed_00000000.h5" `
  --output "$pilot\episode_seed0.png"
```

确认 pilot 后收 200 条成功轨迹并再次 dry-run。失败尝试默认删除；`collection_summary.json` 仍保留
失败原因：

```powershell
$full = 'D:\Sim\results\piper-mujoco-pi05\full200'
.\.venv-mujoco\Scripts\python.exe -m examples.piper.collect_demos `
  --output-dir $full --episodes 200 --start-seed 2000
.\.venv-mujoco\Scripts\python.exe -m examples.piper.convert_hdf5_to_lerobot `
  --input-path $full --dry-run
```

转换器会拒绝空 episode、失败/未 finalize 文件、NaN/Inf、维度或单位不匹配、越界动作、非单调
时间戳、20 Hz 动作跳变、重复 seed、混用模型 SHA，以及覆盖已有 LeRobot 输出。

## Linux GPU：数据转换和 π0.5 LoRA

先把 raw HDF5 和 `collection_summary.json` 复制到 Linux 原生磁盘，并用原 SHA-256 复核。不要直接在
Windows 挂载目录创建 `uv` 环境或训练 checkpoint。以下路径是示例，按服务器实际目录调整：

```bash
cd /home/data/xiaoqinchuan/projects/openpi
export HF_LEROBOT_HOME=/home/data/xiaoqinchuan/datasets/lerobot
export OPENPI_PIPER_RAW=/home/data/xiaoqinchuan/datasets/piper/full200
export OPENPI_PIPER_REPO_ID=local/piper-red-cube-to-box-v1
```

正式数据转换和 normalization：

```bash
uv run python -m examples.piper.convert_hdf5_to_lerobot \
  --input-path "$OPENPI_PIPER_RAW" \
  --repo-id "$OPENPI_PIPER_REPO_ID"
uv run scripts/compute_norm_stats.py --config-name pi05_piper_lora
```

转换后的目录包含 `episode_split.json`、`dataset_contract.json` 和 `SIMULATION_ONLY.json`。训练/验证
按 episode 分割，验证 episode 不会进入 LeRobot 训练集。

### 1. one-step 和 100-step smoke

先确认本机 base 参数存在；首选路径是以往服务器布局，若已迁移则检查 OpenPI cache，而不要静默
重新下载或假设路径仍有效：

```bash
export OPENPI_BASE_PARAMS=/home/data/xiaoqinchuan/models/openpi-assets/checkpoints/pi05_base/params
export OPENPI_CHECKPOINT_ROOT=/home/data/xiaoqinchuan/checkpoints/openpi
test -d "$OPENPI_BASE_PARAMS" || exit 1
```

一次 train step 会覆盖 forward、backward 和 optimizer step；100 steps 只验链路：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_piper_lora \
  --exp-name piper-one-step --checkpoint-base-dir "$OPENPI_CHECKPOINT_ROOT" \
  --weight-loader.params-path "$OPENPI_BASE_PARAMS" \
  --num-train-steps 1 --save-interval 1 --keep-period 1 \
  --no-wandb-enabled --overwrite

CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_piper_lora \
  --exp-name piper-smoke100 --checkpoint-base-dir "$OPENPI_CHECKPOINT_ROOT" \
  --weight-loader.params-path "$OPENPI_BASE_PARAMS" \
  --num-train-steps 100 --save-interval 100 --keep-period 100 \
  --no-wandb-enabled --overwrite
```

若单卡 OOM，选择能被 global batch 8 整除的 GPU 数，并加 `--fsdp-devices 2`；不要仅因为程序退出码为
0 就宣布效果达标，还要检查初始化、loss/grad norm、checkpoint 保存和错误日志。

### 2. overfit-10

从同一个 episode split 可复现地只转换 10 个训练 episode，写到独立 repo id：

```bash
export OPENPI_PIPER_REPO_ID=local/piper-red-cube-to-box-overfit10-v1
uv run python -m examples.piper.convert_hdf5_to_lerobot \
  --input-path "$OPENPI_PIPER_RAW" \
  --repo-id "$OPENPI_PIPER_REPO_ID" \
  --train-episode-limit 10
uv run scripts/compute_norm_stats.py --config-name pi05_piper_lora
```

先对同一命令加 `--dry-run`，读取实际 `train_frame_count`，再计算：

```text
steps_per_pass = ceil(train_frames / 8)
num_train_steps = steps_per_pass * target_passes
```

当前 full200 的 overfit-10 子集是 1243 帧，1/3/5 passes 分别为 156/468/780 steps；原始数据或
split seed 改变后仍以新 dry-run 输出为准。

训练后，从该数据集的 `episode_split.json` 取 10 个 train seed，用下一节的 server 启动 checkpoint，并在
Windows 运行 `policy_rollout --seeds ... --minimum-success-rate 0.8`。至少 `8/10` 才通过 overfit 行为门槛。

### 3. 正式 LoRA 和恢复

恢复正式 repo id；本次 19899 个训练帧对应 3 passes = 7464 steps：

```bash
export OPENPI_PIPER_REPO_ID=local/piper-red-cube-to-box-v1
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_piper_lora \
  --exp-name piper-full-3pass --checkpoint-base-dir "$OPENPI_CHECKPOINT_ROOT" \
  --weight-loader.params-path "$OPENPI_BASE_PARAMS" \
  --num-train-steps 7464 --save-interval 1000 --keep-period 2000 \
  --no-wandb-enabled --overwrite
```

数据变化后必须重新 dry-run 并重算 steps，不能沿用 7464。中断后保留完全相同的 config、repo id、
exp name 和目录，去掉 `--overwrite` 并追加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py pi05_piper_lora \
  --exp-name piper-full-3pass --checkpoint-base-dir "$OPENPI_CHECKPOINT_ROOT" \
  --weight-loader.params-path "$OPENPI_BASE_PARAMS" \
  --num-train-steps 7464 --save-interval 1000 --keep-period 2000 \
  --no-wandb-enabled --resume
```

训练完成后，把审计文件复制进具体 step checkpoint；若已有内容冲突，工具会拒绝覆盖：

```bash
export PIPER_DATASET_DIR="$HF_LEROBOT_HOME/$OPENPI_PIPER_REPO_ID"
export PIPER_CHECKPOINT="$OPENPI_CHECKPOINT_ROOT/pi05_piper_lora/piper-full-3pass/7463"
uv run python -m examples.piper.mark_simulation_only_checkpoint \
  --checkpoint-dir "$PIPER_CHECKPOINT" --dataset-dir "$PIPER_DATASET_DIR"
```

训练循环从 step 0 计数，因此 7464 次更新的最终 checkpoint 目录名是 `7463`；应同时以训练日志中
最后一次 `Saving` 和磁盘上的实际目录为准。

## Policy server、离线指标与 MuJoCo 闭环

Linux 启动 server 时必须显式确认 simulation-only：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_piper_lora \
  --policy.dir="$PIPER_CHECKPOINT" \
  --deployment-scope=simulation-only \
  --confirm-simulation-only=SIMULATION_ONLY
```

Windows 先在 held-out 40 个 episode 上抽样比较预测 chunk 的第一个动作和专家动作。它只产生 offline
MAE/RMSE，不能证明闭环成功：

```powershell
$serverIp = '192.0.2.10'  # 替换为实际 Linux 服务器地址
.\.venv-mujoco\Scripts\python.exe -m examples.piper.offline_policy_eval `
  --input-path 'D:\Sim\results\piper-mujoco-pi05\full200' `
  --host $serverIp --port 8000 `
  --maximum-frames 400 `
  --output-path 'D:\Sim\results\piper-mujoco-pi05\offline-eval.json'
```

最终用 20 个未见 seed 做真正闭环。客户端每次取并执行恰好 10 个动作；坏 shape、NaN/Inf、越界、
断连或执行异常会清空 chunk、保持最新测量姿态并结束当前 episode，不会继续旧动作：

```powershell
$serverIp = '192.0.2.10'  # 替换为实际 Linux 服务器地址
.\.venv-mujoco\Scripts\python.exe -m examples.piper.policy_rollout `
  --host $serverIp --port 8000 `
  --episodes 20 --start-seed 5000 `
  --minimum-success-rate 0.6 `
  --output-dir 'D:\Sim\results\piper-mujoco-pi05\policy-full-20'
```

`PIPER_POLICY_ROLLOUT_OK` 要求至少 `12/20`；输出包含逐 seed 视频、失败原因和
`rollout_summary.json`。这才是首版仿真行为验收。

## 两周学习顺序

每天建议 2–3 小时：30 分钟概念，45 分钟读代码，60–90 分钟实验，15 分钟记录。

| 天 | 从哪里学 | 动手验收 |
|---|---|---|
| 1 | `mujoco_env.py`：`MjModel/MjData`、MJCF、`qpos/qvel/ctrl/mj_step` | Viewer 中观察并打印 joint、finger、actuator |
| 2 | `ik.py`：world/base/link/site、FK、Jacobian、DLS IK | 解释六轴角度如何产生末端位姿，确认 IK 不 teleport live data |
| 3 | `reset()`、任务 XML、camera/render | 同 seed 完全复现，不同 seed 改变方块和盒子 |
| 4 | `contract.py`、`step()`、500/20 Hz | 理解绝对目标、软限位、rate limit 和 25 substeps |
| 5 | `expert.py`、`evaluate_expert.py` | 100 seeds 至少 95 成功，按阶段解释失败 |
| 6 | `record.py`、`collect_demos.py` | 20 pilot 回放对齐，再收 200 条成功轨迹 |
| 7 | `convert_hdf5_to_lerobot.py` | 审计 shape/dtype/range/hash，画六轴和夹爪曲线 |
| 8 | `piper_policy.py`、`pi05_piper_lora` | 解释 7D→32D、相对六轴/绝对夹爪、camera mask；跑 one/100-step |
| 9 | overfit-10、pass 公式、resume | 训练 seed 闭环至少 8/10，再跑正式 1/3/5 pass 对比 |
| 10 | `offline_policy_eval.py`、`policy_rollout.py` | 区分 offline MAE 与闭环成功；未见 seed 至少 12/20 |

第一周结束应能画出：policy → 7D absolute targets → rate limiter → 7 actuators → 500 Hz physics →
RGB/state。第二周结束应能说明 action chunk 的 10 是未来控制点数量，不是物理 substep 数，也不是
训练 batch size。

## 验收边界

- smoke test 通过：只证明数据加载、forward/backward、保存和服务链路可运行。
- offline loss 或 MAE 下降：只证明动作拟合改善。
- MuJoCo 未见 seed 成功率 `>=12/20`：才算首版仿真行为通过。
- 所有这里生成的数据和 checkpoint 都是 `SIMULATION_ONLY`，不得直接用于真实 PiPER。

真实机械臂控制、CAN/ROS 2、PIKA 夹爪、腕部相机、真实标定数据和硬件安全门禁不在本阶段范围内。
