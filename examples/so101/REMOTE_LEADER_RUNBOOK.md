# Remote SO-101 Leader Runbook

This runbook restores the tested Windows Leader to server-side LeIsaac workflow after closing terminals,
leaving `tmux`, or rebooting the server. The physical Leader is passive: LeIsaac reads its joint positions but
does not command its motors. Port `5556` carries Leader joint states; it is separate from the OpenPI policy
server port, which is normally `18000` in this repository.

The required startup order is:

1. Start the Leader publisher on Windows.
2. Start the SSH reverse tunnel in a second Windows terminal.
3. Create or attach to a server `tmux` session and restore the complete runtime environment.
4. Start the simulation or the next LeIsaac/OpenPI task.

## 1. Windows: start the Leader publisher

Connect the calibrated SO-101 Leader on `COM7`. Close FT SCServo Debug first because only one process can own
the serial port. In Windows PowerShell:

```powershell
$OpenPiRoot = 'D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi'
Set-Location $OpenPiRoot

& "$OpenPiRoot\tmp\leisaac-remote-env\python.exe" `
  "$OpenPiRoot\examples\so101\leader_remote_windows.py" `
  publish `
  --port COM7 `
  --id leader_arm `
  --rate 50 `
  --bind tcp://127.0.0.1:5556 `
  --leisaac-root "$OpenPiRoot\tmp\leisaac-v0.4.0"
```

Keep the terminal open after it prints:

```text
Publishing on tcp://127.0.0.1:5556 at 50 Hz
```

Do not pass `--recalibrate` during normal startup. The existing calibration is stored at:

```text
tmp\leisaac-v0.4.0\scripts\environments\teleoperation\.cache\leader_arm.json
```

## 2. Windows: start the SSH reverse tunnel

Open a second PowerShell terminal:

```powershell
ssh -N -T `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=3 `
  -L 127.0.0.1:5557:127.0.0.1:5557 `
  -R 127.0.0.1:5556:127.0.0.1:5556 `
  xiaoqinchuan@10.120.16.48
```

Enter the server password and keep this terminal open. With `-N -T`, silence after authentication is normal.
When using `-v` for diagnostics, these lines confirm success:

```text
remote forward success for: listen 127.0.0.1:5556
forwarding_success: all expected forwarding replies received
```

## 3. Server: create a new tmux session

After logging in to the server:

```bash
tmux new -s leisaac
```

If the session already exists, attach instead:

```bash
tmux attach -t leisaac
```

Inside `tmux`, restore the complete environment. These exports are intentionally explicit so the commands do
not depend on variables inherited from an older shell:

```bash
export MINICONDA_ROOT=/home/data/xiaoqinchuan/tools/miniconda3
source "$MINICONDA_ROOT/etc/profile.d/conda.sh"

export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
conda activate "$LEISAAC_ENV"

export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0

export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export CONDA_PKGS_DIRS=/home/data/xiaoqinchuan/cache/conda/pkgs
export PIP_CACHE_DIR=/home/data/xiaoqinchuan/cache/pip
export HF_HOME=/home/data/xiaoqinchuan/cache/huggingface
export TORCH_HOME=/home/data/xiaoqinchuan/cache/torch
export XDG_CACHE_HOME=/home/data/xiaoqinchuan/cache
export TMPDIR=/home/data/xiaoqinchuan/tmp

export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

unset PRIVACY_CONSENT
unset CUDA_VISIBLE_DEVICES

mkdir -p \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$ISAACSIM_PORTABLE_ROOT" \
  "$LEISAAC_BASE/results/leisaac"

cd /home/data/xiaoqinchuan/projects/openpi
git pull --ff-only
```

Do not set `CUDA_VISIBLE_DEVICES` for Isaac Sim on this machine. Pass the same physical GPU index reported by
`nvidia-smi` to `--device`; this avoids a physical-versus-logical GPU numbering mismatch. GPU `6` is the
previously validated default, but it must be changed if another process is using it.

## 4. Optional preflight checks

These commands are diagnostics, not part of every launch. Run them after a reboot, when sharing the server, or
when the publisher/tunnel is suspect:

```bash
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

ss -ltn 'sport = :5556'
```

The SSH tunnel should create a listener on `127.0.0.1:5556`. Choose a free physical GPU and update `SIM_GPU` in
the launch command below.

The local forward on port `5557` makes a server-side preview available to the Windows browser without exposing
the preview service to the network.

## 5. Bounded remote-Leader simulation validation

This command is a reusable validation, not the eventual dataset-recording command. It never writes to the
physical Leader and does not record an episode. Keep the Leader still until the script prints
`REMOTE_LEADER_SMOKE_PHASE=stepping`, then move one joint slowly and hold the final pose.

```bash
export SIM_GPU=6
export REMOTE_LEADER_RESPONSE_LOG="$LEISAAC_BASE/results/leisaac/remote-leader-response-smoke.log"

timeout --signal=KILL 180s \
  "$CONDA_PREFIX/bin/python" \
  examples/so101/remote_leader_sim_smoke.py \
  --headless \
  --enable_cameras \
  --device "cuda:$SIM_GPU" \
  --task LeIsaac-SO101-LiftCube-v0 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --remote_endpoint tcp://127.0.0.1:5556 \
  --receive_timeout 10 \
  --steps 1200 \
  --kit_args="--portable --portable-root=$ISAACSIM_PORTABLE_ROOT --/telemetry/enableAnonymousData=false --/renderer/multiGpu/enabled=False --/app/fastShutdown=True" \
  2>&1 | tee "$REMOTE_LEADER_RESPONSE_LOG"
```

A successful dynamic run reports finite rewards, changed action targets, changed simulated joint positions, and:

```text
REMOTE_LEADER_SIM_SMOKE_OK
```

The previously validated run changed `shoulder_pan` by about `0.824 rad` and finished with a maximum
target-versus-simulated-joint error of about `0.023 rad` (`1.3 degrees`).

## 6. Interactive browser preview

After the bounded validation has passed, use this preview-only command to control the simulated SO-101 while
watching the same `front` camera observation that will be recorded for OpenPI:

```bash
export SIM_GPU=6
export REMOTE_WEB_TELEOP_LOG="$LEISAAC_BASE/results/leisaac/remote-leader-web-teleop.log"

timeout --signal=KILL 30m \
  "$CONDA_PREFIX/bin/python" \
  examples/so101/remote_leader_web_teleop.py \
  --headless \
  --enable_cameras \
  --device "cuda:$SIM_GPU" \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --remote_endpoint tcp://127.0.0.1:5556 \
  --web_host 127.0.0.1 \
  --web_port 5557 \
  --kit_args="--portable --portable-root=$ISAACSIM_PORTABLE_ROOT --/telemetry/enableAnonymousData=false --/renderer/multiGpu/enabled=False --/app/fastShutdown=True" \
  2>&1 | tee "$REMOTE_WEB_TELEOP_LOG"
```

Wait for `REMOTE_WEB_TELEOP_READY`, then open <http://127.0.0.1:5557> on Windows. `Start / Resume` applies
Leader actions; `Success + Reset` and `Discard + Reset` reset the preview environment; `Stop Server` exits.
Without `--dataset_file`, the status page reports `recording_enabled: false` and no dataset is written.

## 7. Record native LeIsaac HDF5 episodes

Do not install `lerobot==0.4.2` into the Isaac Sim environment. Its requirement
`packaging>=24.2,<26` conflicts with Isaac Sim's tested `packaging==23.0`. Recording therefore stays in the
stable LeIsaac environment and writes native HDF5; conversion to the OpenPI-compatible LeRobot format belongs
in a separate environment later.

For the first dataset, keep the Windows publisher and SSH tunnel running, then launch from the server `tmux`
shell after completing section 3:

```bash
export SIM_GPU=6
export LEISAAC_HDF5_DIR="$LEISAAC_BASE/datasets/leisaac"
export LEISAAC_HDF5_FILE="$LEISAAC_HDF5_DIR/so101_liftcube_first.hdf5"
export REMOTE_WEB_RECORD_LOG="$LEISAAC_BASE/results/leisaac/remote-leader-web-record.log"

mkdir -p "$LEISAAC_HDF5_DIR"

timeout --signal=KILL 60m \
  "$CONDA_PREFIX/bin/python" \
  examples/so101/remote_leader_web_teleop.py \
  --headless \
  --enable_cameras \
  --device "cuda:$SIM_GPU" \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --remote_endpoint tcp://127.0.0.1:5556 \
  --web_host 127.0.0.1 \
  --web_port 5557 \
  --dataset_file "$LEISAAC_HDF5_FILE" \
  --flush_steps 100 \
  --kit_args="--portable --portable-root=$ISAACSIM_PORTABLE_ROOT --/telemetry/enableAnonymousData=false --/renderer/multiGpu/enabled=False --/app/fastShutdown=True" \
  2>&1 | tee "$REMOTE_WEB_RECORD_LOG"
```

The script refuses to overwrite an existing file. To append more episodes later, run the same command with
`--resume`. In the browser:

- `Start / Resume` begins applying Leader actions and accumulating the current episode.
- `Success + Reset` stores the episode with `success=true`, resets the scene, and pauses.
- `Discard + Reset` stores it with `success=false`, resets the scene, and pauses. The later converter will skip
  failed episodes.
- `Stop Server` stores any unconfirmed partial episode as failed, finalizes the HDF5 file, and exits.

A clean exit prints both `REMOTE_WEB_TELEOP_HDF5_FINALIZED` and `REMOTE_WEB_TELEOP_OK`. Preserve the HDF5 file
as the source-of-truth recording; do not convert or train until a short audit confirms its episode count,
success labels, action shape, state shape, and camera shape.

## 8. Preflight native HDF5 conversion

Conversion runs in the OpenPI `uv` environment, not the active LeIsaac Conda environment. The first command is
read-only and does not create a LeRobot dataset:

```bash
cd /home/data/xiaoqinchuan/projects/openpi

uv run examples/so101/convert_leisaac_hdf5_to_lerobot.py \
  --input-path "$LEISAAC_HDF5_FILE" \
  --fps 60 \
  --dry-run
```

Review the reported successful episode count, frame count, image shape, duration, and converted state/action
ranges before running the writing conversion documented in [README.md](README.md).

## 9. Stop or leave safely

To leave the server task running, detach from `tmux` with `Ctrl+B`, then `D`. To stop the Windows side, press
`Ctrl+C` once in the publisher terminal and once in the SSH tunnel terminal. Restarting the publisher normally
does not require recalibration.

Commands such as `grep`, `git status`, repeated package checks, and repeated environment echoes are only needed
when diagnosing a failure or collecting an audit log.
