# SO-101 + LeIsaac

This integration fine-tunes a pi0.5 policy on single-arm SO-101 episodes recorded by LeIsaac and serves the
result back to LeIsaac over its OpenPI WebSocket client.

For the tested Windows Leader publisher, SSH reverse tunnel, complete server environment restoration, and
bounded LeIsaac validation commands, see [REMOTE_LEADER_RUNBOOK.md](REMOTE_LEADER_RUNBOOK.md).

For headless interactive teleoperation, `remote_leader_web_teleop.py` exposes the policy front camera and
Start/Success/Discard controls through a localhost-only browser page. It remains preview-only by default;
passing `--dataset_file` enables LeIsaac's native streaming HDF5 recorder.

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
  --repo-id xiaoqinchuan/leisaac_so101_liftcube_smoke \
  --task "Lift the cube." \
  --fps 60 \
  --image-mode video
```

Only HDF5 episodes with `success=true` are converted. The source HDF5 remains read-only and should be retained
as the reproducible source of truth.

## Configure the dataset

Replace `your_hf_username/leisaac_so101_liftcube` in both SO-101 configs in
`src/openpi/training/config.py` with the LeRobot repository ID containing your recorded episodes.

Two configs are available:

- `pi05_lora_so101_liftcube`: LoRA, batch size 8, intended as the first path on a 24 GB RTX 3090.
- `pi05_so101_liftcube`: full fine-tuning, batch size 32. The repository estimates that full fine-tuning needs
  more than 70 GB of accelerator memory, so this config requires sharding or larger GPUs.

## Compute normalization statistics and train

Run these commands in the OpenPI environment, not in the LeIsaac/Isaac Sim environment:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_lora_so101_liftcube

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_lora_so101_liftcube \
  --exp-name=liftcube_lora \
  --overwrite
```

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
