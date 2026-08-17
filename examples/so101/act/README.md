# SO-101 ACT simulation baseline

This directory implements the S5 ACT baseline with the repository-pinned Hugging Face LeRobot ACT. It does
not vendor or modify the original ALOHA ACT implementation. The complete path is:

```text
local LeRobot dataset
  -> metadata/schema and numeric audit
  -> deterministic episode split
  -> forward/backward/one-step gates
  -> 10-episode overfit gate
  -> full ACT training and resumable checkpoints
  -> held-out offline metrics
  -> simulation-only WebSocket policy server
  -> existing RedCubeToBox LeIsaac rollout
```

## Safety boundary

Every run and checkpoint is marked `simulation-only` with `real_robot_deployment_allowed=false`. The policy
server refuses to start without the exact confirmation `--confirm-simulation-only SIMULATION_ONLY`. Nothing in
this directory is a real-robot deployment path. Action labels and predictions are audited but never silently
clipped. LeIsaac may clip an executed simulation action to its soft limits unless the rollout is explicitly run
with `--match_expert_dynamics`; both raw violations and applied clipping remain logged.

The historical 20-episode pilot reported motor-limit violations only on `wrist_flex`, but that number is not
treated as the audit result for a newer dataset. Run the audit again for every metadata revision and repo ID.

## Configuration and dynamic dataset contract

Defaults live in `configs/red_cube_to_box.toml`. Runtime precedence is CLI, then the named environment variable,
then TOML. Important environment variables are:

```text
HF_LEROBOT_HOME       local LeRobot root
OPENPI_ACT_REPO_ID    repo ID below that root
OPENPI_ACT_EXPERIMENT_ID unique name for one dataset/training-condition run
OPENPI_ACT_OUTPUT_DIR run output directory
OPENPI_ACT_RUN_NAME   checkpoint job name
OPENPI_ACT_CHECKPOINT checkpoint/run path for evaluation or serving
OPENPI_ACT_GPU        free physical GPU selected for ACT
CUDA_VISIBLE_DEVICES  physical GPUs exposed to PyTorch
```

The code resolves a dataset only as `HF_LEROBOT_HOME / repo_id`; absolute repo IDs and `..` traversal are
rejected. Episode/frame counts, FPS, camera features, and normalization statistics come from the selected
dataset metadata. No code assumes 20 episodes, 60 FPS, or one camera. The current TOML requires `front` because
that is the current pilot contract; remove or repeat `--expected-camera-key` when intentionally changing the
camera schema. ACT input features themselves are always derived from metadata/checkpoint features.

The TOML also records the full ACT architecture and optimizer preset used by this baseline: backbone and
pretrained weights, transformer dimensions/heads/layers, VAE dimensions, dropout/KL weight, action chunk and
executed prefix, and AdamW learning rates/weight decay. Every field has a matching training CLI override and is
serialized again by LeRobot inside the checkpoint.

The schema gate requires:

- `observation.state` and `action`, each shaped `(6,)`;
- the exact SO-101 joint order `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
  `gripper`;
- at least one HWC RGB `observation.images.*` feature;
- the configured action contract `absolute_joint_position_target_motor_degrees`;
- positive metadata episode/frame counts and FPS.

The numeric audit additionally checks every state/action/timestamp row, episode coverage, finite values,
per-episode timestamp monotonicity and FPS spacing, frame-0 index/timestamp, per-joint ranges and motor-limit
violations. It decodes frame 0 for every episode by default to catch black, constant, non-finite, or malformed
camera frames. This proves structural row/timestamp alignment, not physical causality; the collection setting
`camera_refreshes_before_recording=1` and conversion setting `start-frame=0` remain part of the upstream S4
contract.

## Linux setup

Use the OpenPI `uv` environment for audit, training, offline evaluation, and the ACT server. Use the separate
Isaac Sim/LeIsaac environment only for closed-loop rollout.

```bash
cd /home/data/xiaoqinchuan/projects/openpi

export HF_LEROBOT_HOME=/home/data/xiaoqinchuan/datasets/lerobot
export OPENPI_ACT_REPO_ID=SELECT_REPO_ID_FROM_OPTIONS_BELOW
export OPENPI_ACT_EXPERIMENT_ID=SET_UNIQUE_EXPERIMENT_ID
export OPENPI_ACT_GPU=SELECT_FREE_PHYSICAL_GPU

export OPENPI_ACT_OUTPUT_DIR="/home/data/xiaoqinchuan/checkpoints/act/$OPENPI_ACT_EXPERIMENT_ID"
export OPENPI_ACT_RUN_NAME="$OPENPI_ACT_EXPERIMENT_ID"

test -d "$HF_LEROBOT_HOME/$OPENPI_ACT_REPO_ID" || echo "Replace OPENPI_ACT_REPO_ID with a valid option below"
printf 'dataset=%s\n' "$HF_LEROBOT_HOME/$OPENPI_ACT_REPO_ID"
printf 'experiment=%s gpu=%s\n' "$OPENPI_ACT_EXPERIMENT_ID" "$OPENPI_ACT_GPU"
```

Replace all three sentinel values before running a command. Currently known `OPENPI_ACT_REPO_ID` options are:

- `local/so101-redcube-polar-s4-frame0-pilot20` (20-episode dataset);
- `local/so101-redcube-polar-s4-frame0-100` (100-episode dataset).

`OPENPI_ACT_EXPERIMENT_ID` is not a dataset field and has no fixed choices. Give every controlled run a unique,
descriptive value such as `so101-redcube-pilot20-step15000-seed42`; it becomes both the output-directory suffix
and checkpoint run name. Do not reuse an experiment ID unless resuming that exact run.

Set `OPENPI_ACT_GPU` to a currently free physical GPU. GPUs 5 and 6 are reserved for OpenPI training and must
not be selected for ACT. `CUDA_VISIBLE_DEVICES="$OPENPI_ACT_GPU"` exposes the selected physical GPU as logical
`cuda:0` to PyTorch. Keep `--device cuda`; do not write physical GPU numbers into source or TOML.

## 1. Dataset audit

```bash
mkdir -p "$OPENPI_ACT_OUTPUT_DIR/logs"

uv run python examples/so101/act/audit_dataset.py \
  --repo-id "$OPENPI_ACT_REPO_ID" \
  --output "$OPENPI_ACT_OUTPUT_DIR/dataset_audit.json" \
  2>&1 | tee "$OPENPI_ACT_OUTPUT_DIR/logs/audit.log"
```

The command prints `ACT_DATASET_AUDIT_OK` only after the full numeric and frame-0 gate. Motor-limit violations
are reported and retained without clipping; for this simulation-only baseline they are a required diagnostic,
not automatically rewritten labels.

## 2. Forward, backward, and one-step gates

Each non-resume command requires an empty output directory, preventing accidental overwrite. Use separate gate
directories:

```bash
CUDA_VISIBLE_DEVICES="$OPENPI_ACT_GPU" uv run python examples/so101/act/train.py \
  --mode forward \
  --repo-id "$OPENPI_ACT_REPO_ID" \
  --output-dir "$OPENPI_ACT_OUTPUT_DIR/gates/forward" \
  --batch-size 2 --num-workers 0 \
  2>&1 | tee "$OPENPI_ACT_OUTPUT_DIR/logs/forward.log"

CUDA_VISIBLE_DEVICES="$OPENPI_ACT_GPU" uv run python examples/so101/act/train.py \
  --mode backward \
  --repo-id "$OPENPI_ACT_REPO_ID" \
  --output-dir "$OPENPI_ACT_OUTPUT_DIR/gates/backward" \
  --batch-size 2 --num-workers 0 \
  2>&1 | tee "$OPENPI_ACT_OUTPUT_DIR/logs/backward.log"

CUDA_VISIBLE_DEVICES="$OPENPI_ACT_GPU" uv run python examples/so101/act/train.py \
  --mode one-step \
  --repo-id "$OPENPI_ACT_REPO_ID" \
  --output-dir "$OPENPI_ACT_OUTPUT_DIR/gates/one-step" \
  --batch-size 2 --num-workers 0 \
  2>&1 | tee "$OPENPI_ACT_OUTPUT_DIR/logs/one-step.log"
```

Expected terminal markers are `ACT_FORWARD_GATE_OK`, `ACT_BACKWARD_GATE_OK`, and `ACT_ONE_STEP_GATE_OK`.

## 3. Ten-episode overfit gate

The gate dynamically chooses 10 episodes from the current training split and requires the fixed evaluation
loss to fall below a configurable ratio of its initial value. Episode selection and the current metadata hash
are saved in `overfit_gate.json`. The runner also expands the repository-pinned LeRobot release's compact
episode-boundary lookup in memory, so non-contiguous original episode IDs remain valid without modifying the
dataset on disk.

```bash
export OPENPI_ACT_OVERFIT_DIR="$OPENPI_ACT_OUTPUT_DIR/overfit10"

CUDA_VISIBLE_DEVICES="$OPENPI_ACT_GPU" uv run python examples/so101/act/train.py \
  --mode overfit \
  --repo-id "$OPENPI_ACT_REPO_ID" \
  --output-dir "$OPENPI_ACT_OVERFIT_DIR" \
  --overfit-episodes 10 \
  --overfit-steps 2000 \
  --overfit-evaluation-batches 16 \
  --overfit-maximum-loss-ratio 0.7 \
  --batch-size 8 --num-workers 4 \
  2>&1 | tee "$OPENPI_ACT_OUTPUT_DIR/logs/overfit10.log"

test -f "$OPENPI_ACT_OVERFIT_DIR/overfit_gate.json" || exit 1
```

Tune steps and the loss-ratio threshold through CLI/TOML based on the dataset, but do not change action/state
semantics or simulation physics to make this gate pass.

## 4. Full training and resume

Full training requires a passed overfit report by default. Use a new output directory for the full run:

```bash
export OPENPI_ACT_TRAIN_DIR="$OPENPI_ACT_OUTPUT_DIR/full"

CUDA_VISIBLE_DEVICES="$OPENPI_ACT_GPU" uv run python examples/so101/act/train.py \
  --mode train \
  --repo-id "$OPENPI_ACT_REPO_ID" \
  --output-dir "$OPENPI_ACT_TRAIN_DIR" \
  --run-name "$OPENPI_ACT_RUN_NAME" \
  --overfit-gate-report "$OPENPI_ACT_OVERFIT_DIR/overfit_gate.json" \
  --steps 100000 \
  --save-frequency 5000 \
  --log-frequency 100 \
  --batch-size 8 \
  --num-workers 4 \
  --gradient-accumulation-steps 1 \
  2>&1 | tee "$OPENPI_ACT_OUTPUT_DIR/logs/train.log"
```

If a 24 GB GPU runs out of memory, first reduce `--batch-size` and `--num-workers`, disable image augmentation,
or increase `--gradient-accumulation-steps`. Do not alter labels, robot units, joint order, or task physics.

LeRobot checkpoints are saved as:

```text
<run>/checkpoints/<step>/
  pretrained_model/
    config.json
    model.safetensors
    train_config.json
    dataset_contract.json
    episode_split.json
    SIMULATION_ONLY.json
  training_state/
    optimizer_state.safetensors
    optimizer_param_groups.json
    rng_state.safetensors
    training_step.json
```

Resume from a numbered checkpoint or `checkpoints/last`. `--output-dir` must be the original run root:

```bash
CUDA_VISIBLE_DEVICES="$OPENPI_ACT_GPU" uv run python examples/so101/act/train.py \
  --mode train \
  --repo-id "$OPENPI_ACT_REPO_ID" \
  --output-dir "$OPENPI_ACT_TRAIN_DIR" \
  --resume-checkpoint "$OPENPI_ACT_TRAIN_DIR/checkpoints/last" \
  --overfit-gate-report "$OPENPI_ACT_OVERFIT_DIR/overfit_gate.json" \
  --steps 100000 \
  2>&1 | tee -a "$OPENPI_ACT_OUTPUT_DIR/logs/train.log"
```

Dataset statistics remain repo-ID-specific: the selected dataset's `meta/stats.json` is fingerprinted in the
run manifest and its normalization buffers are embedded in each ACT checkpoint. Switching repo ID or changing
metadata invalidates the saved split and overfit report and requires fresh gates.

## 5. Held-out offline evaluation

Offline evaluation loads the saved validation split and reports aggregate/per-joint MAE and RMSE, predicted
and target action ranges, non-finite counts, and motor-limit violation counts. Predictions and targets are not
clipped. It uses the same in-memory original-episode-ID compatibility lookup as training for the non-contiguous
held-out split. Checkpoint loading goes through LeRobot's registered `PreTrainedConfig` base so the saved
`type = "act"` discriminator selects `ACTConfig` correctly for evaluation, serving, and resume.

```bash
export OPENPI_ACT_CHECKPOINT="$OPENPI_ACT_TRAIN_DIR/checkpoints/last"

CUDA_VISIBLE_DEVICES="$OPENPI_ACT_GPU" uv run python examples/so101/act/offline_eval.py \
  --checkpoint "$OPENPI_ACT_CHECKPOINT" \
  --repo-id "$OPENPI_ACT_REPO_ID" \
  --output "$OPENPI_ACT_TRAIN_DIR/offline_eval.json" \
  --batch-size 8 --num-workers 4 \
  2>&1 | tee "$OPENPI_ACT_OUTPUT_DIR/logs/offline-eval.log"
```

`--actions-per-inference 0` evaluates the checkpoint's configured executed prefix. Set another positive value
to compare a different prefix up to `chunk_size`. Offline metrics always use the raw deterministic ACT chunk;
they do not carry temporal-ensemble state across unrelated held-out samples.

## 6. ACT server and LeIsaac closed-loop rollout

Start the policy server in the OpenPI environment:

```bash
read -r -p "Free physical GPU for the ACT server: " ACT_POLICY_GPU

CUDA_VISIBLE_DEVICES="$ACT_POLICY_GPU" uv run python examples/so101/act/serve_policy.py \
  --checkpoint "$OPENPI_ACT_CHECKPOINT" \
  --device cuda \
  --host 0.0.0.0 \
  --port 18000 \
  --actions-per-inference 10 \
  --confirm-simulation-only SIMULATION_ONLY \
  2>&1 | tee "$OPENPI_ACT_OUTPUT_DIR/logs/server.log"
```

In the separate LeIsaac terminal, run the existing policy rollout. The runner converts between SO-101 motor
degrees and LeIsaac radians and records grasp, lift, settled placement, success rate, inference latency, raw
soft-limit violations, and applied clipping by joint:

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
read -r -p "Free physical GPU for LeIsaac rollout: " ACT_ROLLOUT_GPU
export ISAAC_DEVICE=cuda:0
export ACT_ROLLOUT_DIR=/home/data/xiaoqinchuan/results/act/so101-redcube
mkdir -p "$ACT_ROLLOUT_DIR"

test -x "$LEISAAC_ENV/bin/python" || echo "missing LeIsaac Python: $LEISAAC_ENV/bin/python"

CUDA_VISIBLE_DEVICES="$ACT_ROLLOUT_GPU" \
"$LEISAAC_ENV/bin/python" "$OPENPI_ROOT/examples/so101/red_cube_to_box_policy_rollout.py" \
  --headless \
  --enable_cameras \
  --device "$ISAAC_DEVICE" \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --policy_host 127.0.0.1 \
  --policy_port 18000 \
  --require_policy_scope simulation-only \
  --episodes 20 \
  --actions_per_inference 10 \
  --reset_camera_refreshes 1 \
  --match_expert_dynamics \
  --minimum_success_rate 0.0 \
  --record_dir "$ACT_ROLLOUT_DIR/records" \
  2>&1 | tee "$ACT_ROLLOUT_DIR/rollout.log"
```

For temporal ensembling, configure ACT with `n_action_steps=1`, serve with `--actions-per-inference 1`, and run
the rollout with `--actions_per_inference 1`. The remote-reset handshake clears stateful ACT inference at the
start of every episode.

## Terminal log extraction

```bash
grep -nE \
  'ACT_DATASET_AUDIT_|ACT_SCHEMA_GATE_|ACT_FORWARD_GATE_|ACT_BACKWARD_GATE_|ACT_ONE_STEP_GATE_|ACT_OVERFIT_GATE_|ACT_TRAIN_|act_train_step:|act_checkpoint_saved:|ACT_OFFLINE_EVAL_|ACT_POLICY_SERVER_|Traceback|RuntimeError|FAILED' \
  "$OPENPI_ACT_OUTPUT_DIR"/logs/*.log

grep -nE \
  'policy_server_metadata:|policy_server_diagnostics:|policy_inference:|policy_episode:|completed_episodes:|successful_episodes:|success_rate:|ever_grasped|ever_lifted|settled_inside|clip_count_by_joint|soft_limit_violation_count_by_joint|RED_CUBE_TO_BOX_POLICY_ROLLOUT_|Traceback|RuntimeError' \
  "$ACT_ROLLOUT_DIR/rollout.log"
```

Loss reduction is only an optimization gate. S5 task success is determined by the held-out offline report and,
most importantly, closed-loop LeIsaac rollout success under the same task initialization and success rules used
for the other policy baselines.

## Tests

Pure contract, audit, inference, safety, and rollout-adapter tests do not require Isaac Sim:

```bash
uv run pytest -q \
  examples/so101/act/tests \
  examples/so101/red_cube_to_box_policy_rollout_test.py
```

The actual ACT forward/backward/checkpoint tests are the GPU gates above because this Windows workspace does
not contain the server dataset or a local installed copy of the pinned LeRobot environment.
