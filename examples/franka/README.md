# FR3 + Franka Hand OpenPI interface

This directory implements the task-agnostic v1 interface for one Franka Research 3 (FR3) with the original
Franka Hand. It connects 20 Hz OpenPI inference to a separately verified 1 kHz local controller. The policy never
predicts Cartesian poses or grasp force: each action is seven absolute joint targets in radians plus one absolute
gripper width in metres.

The hardware baseline is a 7-DoF arm with a 3 kg payload, 855 mm reach, seven joint torque sensors, and repeatability
better than +/-0.1 mm. See the official [FR3 product page](https://franka.de/franka-research-3),
[FCI limits](https://frankarobotics.github.io/docs/robot_specifications.html), and
[software compatibility matrix](https://frankarobotics.github.io/docs/compatibility.html).

## Interface contract

- Observation: `images/base`, optional `images/wrist`, `state: float32[8]`, and `prompt`.
- State order: `fr3_joint1..7` in radians, followed by `gripper_width_m`.
- Model space: pi0.5 uses 32 padded action dimensions and a 16-step horizon. Only the first eight dimensions leave
  the policy adapter.
- Training space: the first seven absolute targets are converted to state-relative joint deltas. Gripper width stays
  absolute. Inference restores absolute joint targets and returns `actions: float32[16,8]`.
- Cameras: the base camera maps to `base_0_rgb`; the wrist camera maps to `left_wrist_0_rgb`. Missing model slots have
  an explicit false mask for pi0.5.

Every dataset must use one camera layout consistently. A base camera is required; wrist-camera presence and image
shape may not change between episodes. Training and inference must use the same camera layout.

## Mock and fake-server checks

Run read-only observation without a policy server:

```bash
uv run python -m examples.franka.main --mode observe --steps 20
```

Start the protocol-v1 fake server in another terminal, then exercise shadow or mock execution:

```bash
uv run python -m examples.franka.fake_policy_server --fault-mode normal
uv run python -m examples.franka.main --mode shadow --steps 20
uv run python -m examples.franka.main --mode execute --steps 20
```

`execute` above only controls `MockFrankaBackend`. Fault modes `bad-shape`, `nonfinite`, and `joint-limit` are for mock
or simulation validation. A fault cancels the whole chunk, holds a fresh measured pose in execute mode, leaves zero
old actions executable, and requires explicit recovery. Do not run fault injection on powered hardware.

## Recording and conversion

`FrankaEpisodeRecorder` writes one append-only HDF5 file per episode. Each sample is a pre-step RGB/state observation
paired with the absolute target that was actually sent. Metadata includes calibration hashes, joint order, units,
control rate, robot/software versions, and language task. Finalize successful and unsuccessful episodes explicitly;
incomplete files and failed episodes are rejected by default.

Always audit before conversion:

```bash
uv run python -m examples.franka.convert_franka_hdf5_to_lerobot \
  --input-path /data/franka/raw \
  --dry-run
```

The audit checks schema and alignment, monotonic timestamps, image synchronization, finite values, joint/gripper
ranges, first-action continuity, and rectangular velocity limits. It prints the accepted frame count and the suggested
three-pass training step count. Convert only after reviewing that output:

```bash
uv run python -m examples.franka.convert_franka_hdf5_to_lerobot \
  --input-path /data/franka/raw \
  --repo-id local/franka-generic-v1
```

The converter refuses to overwrite an existing LeRobot dataset and writes `franka_conversion_manifest.json` with
source hashes and the actual converted frame count.

## pi0.5 LoRA smoke training

The `pi05_franka_lora` config uses `pi05_base`, `action_dim=32`, horizon 16, batch size 8, and 100 steps. Point it at
the converted dataset with `OPENPI_FRANKA_REPO_ID`, then compute normalization statistics before training:

```bash
export OPENPI_FRANKA_REPO_ID=local/franka-generic-v1
uv run scripts/compute_norm_stats.py --config-name pi05_franka_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_franka_lora \
  --exp-name=franka-smoke --overwrite
```

The 100-step run is only a plumbing check for dataset loading, normalization, checkpoint writing, and serving. For a
reviewed training split containing `train_frames`, use:

```text
num_train_steps = ceil(3 * train_frames / 8)
equivalent_passes = num_train_steps * 8 / train_frames
```

Report the real frame count, batch size, step count, and equivalent passes. Override `--num-train-steps` with the
calculated value; do not describe a fixed step count as convergence.

Serve the smoke checkpoint as simulation-only:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_franka_lora \
  --policy.dir=checkpoints/pi05_franka_lora/franka-smoke/100 \
  --deployment-scope=simulation-only \
  --confirm-simulation-only=SIMULATION_ONLY
```

## ROS 2 boundary

The v1 ROS backend is Ubuntu 24.04 + ROS 2 Jazzy. Startup probes `ROS_DISTRO`, `franka_bringup`, libfranka,
`franka_description`, and the robot-system version supplied from Franka Desk. The accepted floor mirrors the current
official Jazzy matrix row: franka_ros2 3.4.0, libfranka 0.20.4, franka_description 2.8.0, and robot system 5.9.0.
An unsupported combination fails before the backend is created.

For ROS 2 fake hardware or Gazebo, first verify that the local controller really runs at 1 kHz, applies
position-dependent velocity limits every cycle, and owns a local watchdog. Then run shadow before simulated execute:

```bash
uv run python -m examples.franka.main \
  --backend-name ros2 \
  --ros2-fake-hardware \
  --robot-system-version 5.9.0 \
  --mode shadow
```

Real hardware has additional gates: the controller contract must be explicitly confirmed, the policy server must
advertise `real_robot_deployment_allowed=true`, and an approval JSON must match the exact server-metadata digest with
an approved speed scale no greater than 10%. The supplied `pi05_franka_lora` and fake server deliberately advertise
`real_robot_deployment_allowed=false`, so they cannot enter real execute mode without a separately reviewed
checkpoint and approval artifact.

Passing the unit, fake-server, and simulation checks means the interface/data/training plumbing works. It does not
establish task success or real-robot quality acceptance.
