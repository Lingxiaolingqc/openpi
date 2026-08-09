# SO-101 + LeIsaac

This integration fine-tunes a pi0.5 policy on single-arm SO-101 episodes recorded by LeIsaac and serves the
result back to LeIsaac over its OpenPI WebSocket client.

For an audit of every SO-101/OpenPI source change relative to the original integration baseline, see
[0.originalCodeChanges.md](0.originalCodeChanges.md).

For the tested Windows Leader publisher, SSH reverse tunnel, complete server environment restoration, and
bounded LeIsaac validation commands, see [REMOTE_LEADER_RUNBOOK.md](REMOTE_LEADER_RUNBOOK.md).

For headless interactive teleoperation, `remote_leader_web_teleop.py` exposes the policy front camera and
Start/Success/Discard controls through a localhost-only browser page. It remains preview-only by default;
passing `--dataset_file` enables LeIsaac's native streaming HDF5 recorder.

## RedCubeToBox development

S3 starts from the already validated LiftCube geometry instead of introducing another unverified robot or
table asset. Before placing the target box, run the bounded scene audit to capture the robot, end-effector,
cube, camera, and environment coordinates from the installed LeIsaac version:

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export LD_PRELOAD=/home/data/xiaoqinchuan/envs/leisaac-so101/lib/libstdc++.so.6
export OMNI_KIT_ACCEPT_EULA=YES

cd "$OPENPI_ROOT"
timeout --signal=KILL 120s \
  /home/data/xiaoqinchuan/envs/leisaac-so101/bin/python \
  examples/so101/red_cube_to_box_scene_audit.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --renderer_device 6 \
  --assets_root "$LEISAAC_ASSETS_ROOT"
```

Success requires both process exit code `0` and the semantic marker
`RED_CUBE_TO_BOX_SCENE_AUDIT_OK`. The audit does not connect to the physical Leader, change assets, or write a
dataset. Its coordinates are the input to the next S3 change: five static cuboids forming a target tray, an
inside-box success predicate, and a scripted pick-place state machine.

## Data contract

Do not install LeRobot into the Isaac Sim environment. LeRobot 0.4.2 requires `packaging>=24.2`, while the
tested Isaac Sim 5.1 environment requires `packaging==23.0`. Record native LeIsaac HDF5 in the stable simulator
environment, then run `convert_leisaac_hdf5_to_lerobot.py` in OpenPI's isolated `uv` environment.

Native LeIsaac HDF5 stores both state and action in USD radians. The converter reproduces LeIsaac v0.4.0's
official `action_align=True` mapping so both resulting LeRobot fields use the same SO-101 motor coordinates.
Copying the HDF5 arrays directly would mix coordinate contracts and train an invalid policy.

The OpenPI data config consumes these LeRobot fields:

| LeRobot field | OpenPI field | Shape | Meaning |
| --- | --- | --- | --- |
| `observation.images.front` | `images/front` | `H x W x 3` | Required RGB image |
| `observation.state` | `state` | `6` | Five arm joints plus gripper |
| `action` | `actions` | `T x 6` | Absolute SO-101 motor targets |
| `task` | `prompt` | text | Language instruction |

The first five action dimensions are converted to deltas for training; the gripper stays absolute. Model output
is converted back to absolute targets before it is returned to LeIsaac. The pi0.5 model keeps its 32-dimensional
internal action padding, while the policy adapter exposes only the six SO-101 dimensions.

LiftCube currently provides only a front camera. The adapter fills the unused wrist slots with black images and
masks them out. It also accepts an optional `images/wrist` input for future SO-101 tasks; a dataset that records
that camera must also add `"images/wrist": "observation.images.wrist"` to the data config's repack mapping.

## Convert native LeIsaac HDF5

First run the non-writing preflight. Set `--fps` to the actual web teleoperation step rate; the tested recording
used 60 Hz:

```bash
uv run examples/so101/convert_leisaac_hdf5_to_lerobot.py \
  --input-path "$LEISAAC_HDF5_FILE" \
  --fps 60 \
  --dry-run
```

After the preflight ranges have been reviewed, create a local LeRobot dataset. The converter refuses to
overwrite an existing repository ID and does not upload anything unless `--push-to-hub` is explicitly passed:

```bash
uv run examples/so101/convert_leisaac_hdf5_to_lerobot.py \
  --input-path "$LEISAAC_HDF5_FILE" \
  --repo-id local/leisaac-so101-liftcube-smoke-20260808 \
  --task "Lift the cube." \
  --fps 60 \
  --image-mode video
```

Only HDF5 episodes with `success=true` are converted. The source HDF5 remains read-only and should be retained
as the reproducible source of truth.

## Configure the dataset

Both SO-101 configs read the dataset ID from `OPENPI_SO101_LIFTCUBE_REPO_ID`. The default points to the tested
local smoke dataset. Override it when using a larger local dataset or a Hugging Face dataset:

```bash
export HF_LEROBOT_HOME=/home/data/xiaoqinchuan/datasets/lerobot
export OPENPI_SO101_LIFTCUBE_REPO_ID=local/leisaac-so101-liftcube-smoke-20260808
```

The environment variable is read when the config module starts, so export it before each `uv run` command.
The one-episode smoke dataset validates the pipeline but is not sufficient for a useful trained policy.

Two configs are available:

- `pi05_lora_so101_liftcube`: LoRA, batch size 8, intended as the first path on a 24 GB RTX 3090.
- `pi05_so101_liftcube`: full fine-tuning, batch size 32. The repository estimates that full fine-tuning needs
  more than 70 GB of accelerator memory, so this config requires sharding or larger GPUs.

## Compute normalization statistics and train

Run these commands in the OpenPI environment, not in the LeIsaac/Isaac Sim environment:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_lora_so101_liftcube

CUDA_VISIBLE_DEVICES=5,6 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run scripts/train.py pi05_lora_so101_liftcube \
  --exp-name=liftcube_lora \
  --fsdp-devices=2 \
  --batch-size=2 \
  --num-workers=0 \
  --overwrite
```

Replace `5,6` with two GPUs that are actually free. The SO-101 configs default to `num_workers=0` so training
does not leave persistent PyTorch loader workers for interpreter shutdown. Increase the worker count only after
the host's multi-GPU exit path has been proven stable.

Computing normalization statistics is mandatory because the six state/action dimensions have robot-specific
ranges. Inspect the generated `q01`, `q99`, and `std` values before starting a long training run.

## Serve the checkpoint

The example uses port 18000. Any free port may be used as long as
the server and LeIsaac client use the same value.

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_lora_so101_liftcube \
  --policy.dir=checkpoints/pi05_lora_so101_liftcube/liftcube_lora/30000 \
  --port=18000
```

Point LeIsaac's `OpenPIServicePolicyClient` to the same host and port. Keep the policy server and Isaac Sim in
separate processes and environments; only the serialized observation/action contract crosses the WebSocket.
