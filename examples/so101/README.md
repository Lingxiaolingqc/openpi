# SO-101 + LeIsaac

This directory contains the OpenPI SO-101 integration, Windows-native LeIsaac validation tools, dataset conversion,
and the RedCubeToBox scripted expert.

- Source-change record: [0.originalCodeChanges.md](0.originalCodeChanges.md)
- Windows Leader and remote policy workflow: [REMOTE_LEADER_RUNBOOK.md](REMOTE_LEADER_RUNBOOK.md)
- Detailed RedCubeToBox troubleshooting history: [../../selfmade/problem_solving.md](../../selfmade/problem_solving.md)

## Real hardware

[`real/`](real/) is the Windows-native, real-hardware-only SO-101 workflow. It is isolated from RedCubeToBox,
Isaac Sim, LeIsaac simulation experts, simulation HDF5 conversion, and sim-to-real work. The fixed hardware mapping is
Leader `COM7` and Follower `COM8`; the Leader remains torque-disabled during teleoperation.

Current files:

- [`real/calibration/calibration_lock.json`](real/calibration/calibration_lock.json) locks the active Leader and Follower
  calibration hashes, controller roles, ports, and frozen snapshot paths.
- [`real/calibration/frozen/2026-08-13/`](real/calibration/frozen/2026-08-13/) contains the reviewed, read-only calibration
  snapshots. A later intentional recalibration must create a new dated snapshot rather than replacing this one.
- [`real/calibration/frozen/2026-08-14-all-joints-v3/`](real/calibration/frozen/2026-08-14-all-joints-v3/) contains the
  current reviewed range-only calibration; the v1 and v2 directories remain immutable provenance.
- [`real/calibration/verify_frozen_calibrations.py`](real/calibration/verify_frozen_calibrations.py) is a serial-free
  preflight gate. It checks the active hashes, snapshot hashes, parsed JSON equality, joint order, IDs, drive modes, and
  ranges.
- [`real/no_jump_enable_test.py`](real/no_jump_enable_test.py) performs the first bounded Follower torque test. It reads
  the current pose, writes that identical pose as the goal, enables torque for five seconds, monitors the arm, and
  disables torque on every exit path.
- [`real/no_jump_enable_test_test.py`](real/no_jump_enable_test_test.py) tests the goal-before-torque ordering and the
  displacement-fault shutdown path with a fake bus; it never opens a serial port.
- [`real/limited_motion_test.py`](real/limited_motion_test.py) moves one selected Follower joint by a bounded relative
  delta at no more than 15 degrees per second, holds briefly, returns to the measured start pose, and unloads.
- [`real/limited_motion_test_test.py`](real/limited_motion_test_test.py) verifies command-step limiting, return-to-start,
  tracking-fault handling, and final torque disable with a fake bus.
- [`real/dual_arm_alignment_monitor.py`](real/dual_arm_alignment_monitor.py) reads both torque-disabled arms and guides
  manual Leader-only alignment until all six joint differences remain within three degrees for two seconds.
- [`real/bounded_leader_follow_test.py`](real/bounded_leader_follow_test.py) is the first torque-enabled
  Leader-to-Follower gate. It mirrors only one selected joint relative to the validated start pose, limits the Leader
  excursion to three degrees, and holds the other five Follower joints at their measured start positions.
- [`real/bounded_leader_follow_test_test.py`](real/bounded_leader_follow_test_test.py) verifies no writes reach the
  Leader, goal-before-torque ordering, single-joint limits, fault shutdown, and port-close behavior using fake buses.
- [`real/bounded_all_joint_follow_test.py`](real/bounded_all_joint_follow_test.py) performs the next short six-joint
  gate with independent three-degree excursion limits, a 10-degree/second slew limit, automatic return, and unload.
- [`real/bounded_all_joint_follow_test_test.py`](real/bounded_all_joint_follow_test_test.py) covers the six-axis bounded
  path and tracking-fault shutdown with fake buses and never opens a serial port.
- [`real/camera_capture.py`](real/camera_capture.py) opens the command-line-selected `front` and `wrist` camera indices
  on independent threads and exposes timestamped newest RGB frames with 100 ms stale-frame rejection.
- [`real/episode_hdf5.py`](real/episode_hdf5.py) incrementally writes one episode per `.partial.h5` file on a background
  thread, then atomically publishes it under `episodes/` or isolates it under `rejected/`.
- [`real/teleop_recorder.py`](real/teleop_recorder.py) combines the two cameras, 30 Hz Leader/Follower control, safety
  monitoring, keyboard episode controls, and the real-only HDF5 contract.
- [`real/requirements-windows.txt`](real/requirements-windows.txt) pins the three collection dependencies added to the
  isolated Windows hardware environment; it does not install Isaac Sim or modify the simulation environment.
- [`real/all_joint_range_audit.py`](real/all_joint_range_audit.py) records a torque-off, six-joint range candidate for
  one arm without writing servo registers or changing the active calibration. It preserves all Homing Offsets and
  writes the candidate and comparison audit under `real/calibration/candidates/` for review.
- [`real/audits/`](real/audits/) contains dated hardware-gate evidence, measured results, and the calibration identity
  used for each physical test.

Verify the frozen calibration before any motion, recording, or deployment:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\calibration\verify_frozen_calibrations.py
```

The required result is `CALIBRATION_LOCK_OK`. A mismatch is a hard stop and must not be accepted automatically.

Run the bounded no-jump test manually while physically beside the arm, with the workspace clear and the Follower's
matched servo power supply switched on. Keep one hand ready at the physical power cutoff; do not touch the arm during
the five-second hold:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\no_jump_enable_test.py `
  --execute `
  --confirm ENABLE_FOLLOWER_HOLD_CURRENT `
  --duration-s 5
```

The test refuses to enable torque if a calibration hash changed, any joint is within two degrees of a calibrated
endpoint, the initial torque state is not disabled, a motor is not in position-control mode, or the preflight
temperature is unsafe. While holding, a displacement over three degrees, temperature at or above 65 degrees Celsius,
invalid torque state, or communication error causes immediate shutdown. Success ends with both
`NO_JUMP_TEST_PASS` and `TORQUE_DISABLED_AND_COM8_CLOSED`.

`Missing motor IDs: 1..6` is not a permissions failure and occurs before any torque write. It means COM8 opened but no
servos answered at 1 Mbps. Check that the Follower servo power supply is on, then check the controller-to-ID-1 bus cable
and the daisy-chain power/data cables. Do not run motor setup or recalibration merely to address this message. The
optional LeIsaac `No module named 'isaaclab_tasks'` message can be ignored in this Windows remote-hardware environment.

After the no-jump gate passes, begin limited motion with one three-degree shoulder-pan excursion. The controller runs
at 30 Hz and 15 degrees per second, so adjacent goals differ by no more than 0.5 degrees. It returns to the measured
start pose automatically:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\limited_motion_test.py `
  --joint shoulder_pan `
  --delta-deg 3 `
  --speed-deg-s 15 `
  --execute `
  --confirm MOVE_FOLLOWER_SMALL_DELTA
```

The first test is only the positive shoulder-pan direction. Do not batch multiple joints or increase the delta until
the terminal ends with `LIMITED_MOTION_TEST_PASS` and `TORQUE_DISABLED_AND_COM8_CLOSED` and the observed direction has
been checked physically.

Before Leader/Follower mirroring, align the passive Leader to the stationary, torque-disabled Follower. This monitor is
strictly read-only and closes both ports without writing torque registers:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\dual_arm_alignment_monitor.py
```

Move only the Leader by hand. The monitor first converts every Leader joint's calibrated range fraction into the
corresponding absolute Follower target; this is required because the two arms, especially their grippers, have different
encoder spans. In the `target-F` column, a positive value means the Leader angle must decrease and a negative value
means it must increase. Stop when the monitor prints `DUAL_ARM_ALIGNMENT_READY`; do not enable Follower torque from a
mismatched mapped pose. Equal endpoint percentages do not imply identical physical jaw gaps, so gripper aperture is
reported and evaluated separately from the rotary-joint correspondence test.

After a centered alignment passes, run the first bounded Leader-to-Follower test manually. Use a wide, soft,
non-rolling support below the Follower; a bottle or other cylindrical prop must not remain in the motion area. Steady
only that support, keep fingers outside the links and joints, and move only Leader `shoulder_pan` slowly out and back
during the eight-second movement window:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\bounded_leader_follow_test.py `
  --joint shoulder_pan `
  --duration-s 8 `
  --max-excursion-deg 3 `
  --execute `
  --confirm RUN_BOUNDED_LEADER_FOLLOW
```

The Follower starts from its measured pose, is limited to 15 degrees/second and 0.5 degrees per control tick, and
automatically returns to its measured start pose. The final three-second hold is the operator's window to support the
Follower before torque switches off. Do not proceed to multi-joint teleoperation unless the terminal ends with both
`BOUNDED_LEADER_FOLLOW_PASS` and `FOLLOWER_TORQUE_DISABLED_AND_PORTS_CLOSED`.

After all six single-joint gates pass, validate combined relative control at a lower 10-degree/second speed. During the
20-second window, move one Leader joint at a time by roughly one or two degrees, including opening/closing the gripper,
and return every joint near its starting pose before the window ends:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\bounded_all_joint_follow_test.py `
  --duration-s 20 `
  --max-excursion-deg 3 `
  --execute `
  --confirm RUN_BOUNDED_ALL_JOINT_FOLLOW
```

This gate requires every joint to move at least 0.5 degrees, prohibits any joint from exceeding three degrees, and
requires the final Leader pose to be within one degree of its start. It is still bounded validation, not ordinary
workspace-scale teleoperation.

### Real episode collection

Install the collection-only dependencies into the existing Windows hardware environment once:

```powershell
tmp\leisaac-remote-env\python.exe -m pip install `
  -r examples\so101\real\requirements-windows.txt
```

The camera roles are always explicit command-line arguments; neither Windows `LocationPath` nor an OpenCV index is
hard-coded in the program. First run a dry-run, which checks dependencies, arguments, and the frozen calibration but
opens neither cameras nor serial ports:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\teleop_recorder.py `
  --dataset-root D:\SO101RealData `
  --front-camera 2 `
  --wrist-camera 1 `
  --target-id 0 `
  --box-id A
```

Change both camera indices to the operator-verified values. `target-id` is `0` for the earbud case, `1` for the
sponge, and `2` for the capped ballpoint pen. `box-id` intentionally accepts only `A` or `B`; Box-C collection is a
hard error. A custom balanced prompt template can be supplied with `--task`; otherwise the canonical prompt for the
target is used.

After fixing both cameras, reconnecting COM7/COM8, clearing the workspace, and placing both arms in aligned safe
poses, start the recorder explicitly:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\teleop_recorder.py `
  --dataset-root D:\SO101RealData `
  --front-camera 2 `
  --wrist-camera 1 `
  --target-id 0 `
  --box-id A `
  --speed-deg-s 15 `
  --max-episode-s 30 `
  --execute `
  --confirm COLLECT_REAL_EPISODES
```

The recorder opens and validates both cameras first, saves one startup JPEG per role under `session_info/`, and then
waits with both serial ports closed. Keep the PowerShell window focused and use these keys:

- `S`: start an episode, run the alignment/no-jump gates, enable Follower torque, and begin recording;
- `Y`: save a successful episode (ignored until at least 30 frames exist);
- `D`: stop and move the episode to `rejected/`;
- `P`: pause or resume motion and recording with a fresh relative pose anchor;
- `X`: emergency one-second measured-pose hold, unload, reject, and exit;
- `Q`: quit while idle, or reject/unload/exit during an episode.

Every unpaused 30 Hz sample stores `obs/joint_pos`, RGB `obs/front`, RGB `obs/wrist`, the absolute six-motor
`actions` target actually scheduled for COM8, and the three monotonic timestamps. Writes are queued to a background
HDF5 thread; a full queue is a hard fault rather than a dropped or misaligned sample. Completed files are published
only by atomic rename. A crash cannot alter a completed episode and leaves only its current `.partial.h5` under
`staging/`.

Use a local SSD rather than a synchronized cloud folder for `--dataset-root`: two uncompressed 640x480 RGB streams can
produce large episodes even with fast lossless HDF5 compression.

Do not rerun `leader_remote_windows.py calibrate` merely to update joint ranges: that path calls
`set_half_turn_homings()` and writes new Homing Offsets for all six motors. To measure all six ranges while preserving
the frozen zero points, run the range-only audit separately for each torque-disabled arm. Move every joint slowly
through both gentle usable endpoints, do not force a mechanical stop, and press Enter when all six are complete:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\all_joint_range_audit.py `
  --arm leader `
  --duration-s 120 `
  --execute `
  --confirm RECORD_TORQUE_OFF_RANGES
```

Then run the same isolated procedure for the stationary Follower:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\all_joint_range_audit.py `
  --arm follower `
  --duration-s 120 `
  --execute `
  --confirm RECORD_TORQUE_OFF_RANGES
```

Each run checks the frozen hashes, operating mode, torque-disabled state, servo Homing Offsets, temperature, and raw
position validity throughout capture. A successful run ends with `ACTIVE_CALIBRATION_UNCHANGED`; candidate ranges do
not become active until both audits have been reviewed and explicitly frozen as a new calibration version.

For an endpoint repeatability check that must not replace the other five ranges, add `--joints gripper`. The resulting
candidate retains every unselected range exactly as it was:

```powershell
tmp\leisaac-remote-env\python.exe `
  examples\so101\real\all_joint_range_audit.py `
  --arm follower `
  --joints gripper `
  --duration-s 30 `
  --execute `
  --confirm RECORD_TORQUE_OFF_RANGES
```

## RedCubeToBox

The task ID is `OpenPI-LeIsaac-SO101-RedCubeToBox-v0`. Both supported command sets use one environment, headless
rendering, and no `--renderer_device`.

Windows-native setup:

```powershell
Set-Location "D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONUNBUFFERED = "1"
```

Server Linux setup:

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export ISAAC_DEVICE=${ISAAC_DEVICE:-cuda:6}
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
cd "$OPENPI_ROOT"
```

### Scene audit

Windows:

```powershell
& "D:\Envs\leisaac-so101-win\Scripts\python.exe" `
  examples/so101/red_cube_to_box_scene_audit.py `
  --headless `
  --enable_cameras `
  --device cuda:0 `
  --assets_root "D:\Sim\leisaac-v0.4.0"
```

A valid audit ends with `RED_CUBE_TO_BOX_SCENE_AUDIT_OK`.

Server Linux:

```bash
"$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_scene_audit.py \
  --headless \
  --enable_cameras \
  --device "$ISAAC_DEVICE" \
  --assets_root "$LEISAAC_ASSETS_ROOT"
```

### Single-environment smoke

`autogen_polar_retreat_transport` is the recommended scripted expert.

Windows:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = "D:\Sim\results\expert-smoke\polar\red_cube_$stamp"
$env:AUTOGEN_POLAR_LOG = Join-Path $runRoot "smoke.log"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

& "D:\Envs\leisaac-so101-win\Scripts\python.exe" `
  examples/so101/red_cube_to_box_expert_smoke.py `
  --headless `
  --enable_cameras `
  --device cuda:0 `
  --rendering_mode performance `
  --assets_root "D:\Sim\leisaac-v0.4.0" `
  --expert autogen_polar_retreat_transport `
  --seed 42 `
  --record_dir (Join-Path $runRoot "recordings") `
  --record_every 4 `
  --record_fps 15 2>&1 |
  Tee-Object -FilePath $env:AUTOGEN_POLAR_LOG
```

A successful run prints `expert_success: True` and `RED_CUBE_TO_BOX_EXPERT_SMOKE_OK`.

Print the relevant fields directly in the terminal:

```powershell
Select-String -LiteralPath $env:AUTOGEN_POLAR_LOG -Pattern `
'expert_phase:|expert_state:|expert_axis_alignment:|expert_polar_close:|expert_polar_path:|expert_ik_runtime_mode:|actual_focus_w=|motion_target_w=|current_target_w=|target_error=|stable_streak=|jaw_cube_distance=|grasp_relative_position_error=|grasp_relative_position_max_error=|grasp_loss_streak=|grasp_confirmed=|servo_timeout_phase:|servo_abort_reason:|release_block_reason:|completed_steps:|cube_final_pos_w:|cube_offset_from_box:|cube_final_speed:|expert_success:|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' |
ForEach-Object { "$($_.LineNumber):$($_.Line)" }
```

Server Linux:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
run_root="$LEISAAC_BASE/results/leisaac/expert-smoke/polar/red_cube_$stamp"
export AUTOGEN_POLAR_LOG="$run_root/smoke.log"
mkdir -p "$run_root"

"$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device "$ISAAC_DEVICE" \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert autogen_polar_retreat_transport \
  --seed 42 \
  --record_dir "$run_root/recordings" \
  --record_every 4 \
  --record_fps 15 2>&1 | tee "$AUTOGEN_POLAR_LOG"
```

Print the server smoke diagnostics directly in the terminal:

```bash
grep -nE \
  'expert_phase:|expert_state:|expert_axis_alignment:|expert_polar_close:|expert_polar_path:|expert_ik_runtime_mode:|actual_focus_w=|motion_target_w=|current_target_w=|target_error=|stable_streak=|jaw_cube_distance=|grasp_relative_position_error=|grasp_relative_position_max_error=|grasp_loss_streak=|grasp_confirmed=|servo_timeout_phase:|servo_abort_reason:|release_block_reason:|completed_steps:|cube_final_pos_w:|cube_offset_from_box:|cube_final_speed:|expert_success:|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' \
  "$AUTOGEN_POLAR_LOG"
```

### Recorded ten-episode batch

All ten episode recordings are stored below one timestamped root.

Windows:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = "D:\Sim\results\expert-batch\polar\red_cube_$stamp"
$env:AUTOGEN_POLAR_LOG = Join-Path $runRoot "batch.log"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

& "D:\Envs\leisaac-so101-win\Scripts\python.exe" `
  examples/so101/red_cube_to_box_expert_batch.py `
  --headless `
  --enable_cameras `
  --device cuda:0 `
  --rendering_mode performance `
  --assets_root "D:\Sim\leisaac-v0.4.0" `
  --expert autogen_polar_retreat_transport `
  --episodes 10 `
  --minimum_success_rate 0.9 `
  --seed 42 `
  --record_dir (Join-Path $runRoot "recordings") `
  --record_every 4 `
  --record_fps 15 2>&1 |
  Tee-Object -FilePath $env:AUTOGEN_POLAR_LOG
```

Extract the batch summary without creating a second output file:

```powershell
Select-String -LiteralPath $env:AUTOGEN_POLAR_LOG -Pattern `
'camera_frame_stats|episode_record_dir:|episode:|grasp_relative_position_error=|grasp_relative_position_max_error=|grasp_loss_streak=|completed_episodes:|successful_episodes:|failed_episodes:|non_finite_episodes:|reset_episodes:|servo_timeout_episodes:|servo_abort_episodes:|success_rate:|RED_CUBE_TO_BOX_EXPERT_BATCH|Traceback|RuntimeError' |
ForEach-Object { "$($_.LineNumber):$($_.Line)" }
```

Server Linux:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
run_root="$LEISAAC_BASE/results/leisaac/expert-batch/polar/red_cube_$stamp"
export AUTOGEN_POLAR_LOG="$run_root/batch.log"
mkdir -p "$run_root"

"$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_batch.py \
  --headless \
  --enable_cameras \
  --device "$ISAAC_DEVICE" \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert autogen_polar_retreat_transport \
  --episodes 10 \
  --minimum_success_rate 0.9 \
  --seed 42 \
  --record_dir "$run_root/recordings" \
  --record_every 4 \
  --record_fps 15 2>&1 | tee "$AUTOGEN_POLAR_LOG"
```

Print the server batch summary directly in the terminal:

```bash
grep -nE \
  'camera_frame_stats|episode_record_dir:|episode:|grasp_relative_position_error=|grasp_relative_position_max_error=|grasp_loss_streak=|completed_episodes:|successful_episodes:|failed_episodes:|non_finite_episodes:|reset_episodes:|servo_timeout_episodes:|servo_abort_episodes:|success_rate:|RED_CUBE_TO_BOX_EXPERT_BATCH|Traceback|RuntimeError' \
  "$AUTOGEN_POLAR_LOG"
```

## Dataset conversion and training

Keep Isaac Sim/LeIsaac and OpenPI/LeRobot in separate Python environments. The collector stores only successful polar
trajectories in resumable shards; failed attempts retain metadata without RGB frames. The current scene has only the
front camera. If a future scene exposes `policy.wrist`, the collector and converter include it automatically.

Windows collection:

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONUNBUFFERED = "1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = "D:\Sim\results\expert-dataset\polar\red_cube_$stamp"
$env:SO101_DATASET_LOG = Join-Path $runRoot "collection.log"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

& "D:\Envs\leisaac-so101-win\Scripts\python.exe" `
  examples/so101/red_cube_to_box_expert_dataset.py `
  --headless `
  --enable_cameras `
  --device cuda:0 `
  --rendering_mode performance `
  --assets_root "D:\Sim\leisaac-v0.4.0" `
  --dataset_dir (Join-Path $runRoot "dataset") `
  --successful_episodes 20 `
  --maximum_attempts 50 `
  --shard_size 50 `
  --seed 42 2>&1 | Tee-Object -FilePath $env:SO101_DATASET_LOG
```

Windows audit and terminal-only log extraction:

```powershell
& "D:\Envs\leisaac-so101-win\Scripts\python.exe" `
  examples/so101/audit_red_cube_to_box_hdf5.py `
  (Join-Path $runRoot "dataset")

Select-String -LiteralPath $env:SO101_DATASET_LOG -Pattern `
'RED_CUBE_TO_BOX_DATASET_|dataset_attempt:|camera_names:|camera_frame_stats:|step_dt:|recovered_staging_groups:|starting_successful_episodes:|completed_new_attempts:|total_attempts:|total_successful_episodes:|Traceback|RuntimeError' |
ForEach-Object { "$($_.LineNumber):$($_.Line)" }
```

Server Linux collection:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
run_root="$LEISAAC_BASE/results/leisaac/expert-dataset/polar/red_cube_$stamp"
export SO101_DATASET_LOG="$run_root/collection.log"
mkdir -p "$run_root"

"$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_dataset.py \
  --headless \
  --enable_cameras \
  --device "$ISAAC_DEVICE" \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --dataset_dir "$run_root/dataset" \
  --successful_episodes 20 \
  --maximum_attempts 50 \
  --shard_size 50 \
  --seed 42 2>&1 | tee "$SO101_DATASET_LOG"

"$LEISAAC_ENV/bin/python" \
  examples/so101/audit_red_cube_to_box_hdf5.py \
  "$run_root/dataset"

grep -nE \
  'RED_CUBE_TO_BOX_DATASET_|dataset_attempt:|camera_names:|camera_frame_stats:|step_dt:|recovered_staging_groups:|starting_successful_episodes:|completed_new_attempts:|total_attempts:|total_successful_episodes:|Traceback|RuntimeError' \
  "$SO101_DATASET_LOG"
```

`--successful_episodes` is the total target, not an increment. The same command can be rerun with the same
`--dataset_dir`: completed shards are preserved and interrupted staging groups are removed before collection resumes.
One 1793-frame front-camera smoke used about 513 MB of native HDF5, so check free space before a large run.

Convert the native shards from the OpenPI environment:

```powershell
uv run examples/so101/convert_leisaac_hdf5_to_lerobot.py `
  --input-path (Join-Path $runRoot "dataset") `
  --repo-id local/leisaac-so101-dataset `
  --task "Pick up the red cube and place it inside the green box." `
  --fps 60 `
  --image-mode video `
  --start-frame 1
```

`--start-frame` defaults to `0`. Pass `--start-frame 1` only after confirming that frame 0 in this collection is an
unrefreshed camera frame. The converter then skips the aligned image, state, and action sample together; it never
shifts only the image stream.

The converter reports `action_motor_limit_violation_count`. It intentionally does not clip labels: nonzero values mean
the frozen expert commanded beyond the declared physical motor range and must be resolved before deploying the trained
policy to hardware.

Set `OPENPI_SO101_LIFTCUBE_REPO_ID` to the converted dataset, compute normalization statistics, and start training.
The pilot dataset is simulation-only, so the converter's raw expert targets remain unchanged; they are not intended for
direct execution on the real robot.

Minimal Windows commands are:

```powershell
$env:OPENPI_SO101_LIFTCUBE_REPO_ID = "local/leisaac-so101-dataset"
uv run scripts/compute_norm_stats.py --config-name pi05_lora_so101_liftcube
uv run scripts/train.py pi05_lora_so101_liftcube --exp-name=so101_lora --num-workers=0
```

For training on the Linux server, start from a fresh terminal. The two existing local copies of the same pi05 base
checkpoint are:

```text
/home/data/xiaoqinchuan/models/openpi-assets/checkpoints/pi05_base/params
/home/data/xiaoqinchuan/cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

Prefer the first path and use the second as a fallback. This avoids downloading `pi05_base` again. The example below
uses the converted 20-episode pilot, writes named checkpoints below
`/home/data/xiaoqinchuan/checkpoints/openpi`, and first recomputes normalization statistics on one GPU:

```bash
cd /home/data/xiaoqinchuan/projects/openpi

export HF_LEROBOT_HOME="/home/data/xiaoqinchuan/datasets/lerobot"
export OPENPI_SO101_LIFTCUBE_REPO_ID="local/so101-redcube-polar-s4-pilot20"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENPI_BASE_PARAMS="/home/data/xiaoqinchuan/models/openpi-assets/checkpoints/pi05_base/params"
export OPENPI_CHECKPOINT_ROOT="/home/data/xiaoqinchuan/checkpoints/openpi"
export OPENPI_EXP_NAME="so101-redcube-polar-s4-pilot20-v1"

test -d "$OPENPI_BASE_PARAMS" || exit 1
mkdir -p "$OPENPI_CHECKPOINT_ROOT/logs"

CUDA_VISIBLE_DEVICES=6 uv run scripts/compute_norm_stats.py \
  --config-name pi05_lora_so101_liftcube \
  2>&1 | tee "$OPENPI_CHECKPOINT_ROOT/logs/${OPENPI_EXP_NAME}.norm-stats.log"
```

The config has global `batch_size=8`. The number of GPUs visible to JAX must therefore divide 8; exposing seven GPUs
fails during sharded batch creation. A single RTX 3090 previously ran out of memory, while two-card FSDP passed the
training initialization and checkpoint-save gate. Select two free GPUs (the example uses physical GPUs 5 and 6), then
run a 1000-step smoke without W&B:

```bash
CUDA_VISIBLE_DEVICES=5,6 uv run scripts/train.py \
  pi05_lora_so101_liftcube \
  --exp-name "$OPENPI_EXP_NAME-smoke" \
  --checkpoint-base-dir "$OPENPI_CHECKPOINT_ROOT" \
  --weight-loader.params-path "$OPENPI_BASE_PARAMS" \
  --fsdp-devices 2 \
  --num-train-steps 1000 \
  --save-interval 250 \
  --keep-period 500 \
  --no-wandb-enabled \
  2>&1 | tee "$OPENPI_CHECKPOINT_ROOT/logs/${OPENPI_EXP_NAME}-smoke.train.log"
```

After the smoke completes without OOM or checkpoint errors, start the named 30,000-step run:

```bash
CUDA_VISIBLE_DEVICES=5,6 uv run scripts/train.py \
  pi05_lora_so101_liftcube \
  --exp-name "$OPENPI_EXP_NAME" \
  --checkpoint-base-dir "$OPENPI_CHECKPOINT_ROOT" \
  --weight-loader.params-path "$OPENPI_BASE_PARAMS" \
  --fsdp-devices 2 \
  --num-train-steps 30000 \
  --save-interval 1000 \
  --keep-period 5000 \
  2>&1 | tee "$OPENPI_CHECKPOINT_ROOT/logs/${OPENPI_EXP_NAME}.train.log"
```

The checkpoint layout is
`$OPENPI_CHECKPOINT_ROOT/pi05_lora_so101_liftcube/$OPENPI_EXP_NAME/<step>/`; the inference parameters are in the
`params/` child of a step directory. `--exp-name` names the run and `--checkpoint-base-dir` selects its storage root.
Do not pass `--overwrite` for an interrupted run. Restore the same environment variables and append `--resume` to the
same training command instead. Extract the important terminal log lines with:

```bash
grep -nE \
  'Running on:|Initialized data loader|Initialized train state|Step [0-9]+:|loss=|grad_norm=|Saving|Finished saving|Closing|closed|Traceback|ValueError|RuntimeError|RESOURCE_EXHAUSTED|out of memory|OOM' \
  "$OPENPI_CHECKPOINT_ROOT/logs/${OPENPI_EXP_NAME}.train.log"
```

Serve a checkpoint with a free port shared by the policy server and LeIsaac client:

```powershell
uv run scripts/serve_policy.py `
  --default-prompt "Pick up the red cube and place it inside the green box." `
  --port 18000 `
  policy:checkpoint `
  --policy.config pi05_lora_so101_liftcube `
  --policy.dir "D:\path\to\checkpoint"
```

### RedCubeToBox learned-policy rollout

[`red_cube_to_box_policy_rollout.py`](red_cube_to_box_policy_rollout.py) evaluates a served OpenPI checkpoint without
changing the scripted expert runners. It uses the direct six-joint `so101leader` action configuration, sends the front
camera, six motor-coordinate joint positions, and task prompt to OpenPI, then executes at most 10 returned absolute
joint targets before replanning. Simulator soft-limit clipping is counted and printed instead of being silent. A
successful episode must lift the cube by at least 30 mm and keep it settled inside the box for 30 consecutive steps.

Install the lightweight client into the LeIsaac environment once. On the Linux server:

```bash
cd /home/data/xiaoqinchuan/projects/openpi
uv pip install --python "$LEISAAC_ENV/bin/python" -e packages/openpi-client
```

Keep `scripts/serve_policy.py` running on port 18000, then use a different GPU for a recorded one-episode smoke:

```bash
stamp=$(date +%Y%m%d-%H%M%S)
run_root="/home/data/xiaoqinchuan/results/leisaac/policy-rollout/red_cube_$stamp"
export RED_CUBE_POLICY_LOG="$run_root/rollout.log"
mkdir -p "$run_root"

"$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_policy_rollout.py \
  --headless \
  --enable_cameras \
  --device cuda:5 \
  --rendering_mode performance \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --policy_host 127.0.0.1 \
  --policy_port 18000 \
  --episodes 1 \
  --seed 42 \
  --maximum_steps 2400 \
  --actions_per_inference 10 \
  --reset_camera_warmup_steps 1 \
  --record_dir "$run_root/recordings" \
  2>&1 | tee "$RED_CUBE_POLICY_LOG"

grep -nE \
  'RED_CUBE_TO_BOX_POLICY_|policy_endpoint:|policy_reset_camera_warmup_steps:|policy_camera:|policy_inference:|policy_episode:|policy_recording_dir:|completed_episodes:|successful_episodes:|success_rate:|Traceback|ValueError|RuntimeError|out of memory|OOM' \
  "$RED_CUBE_POLICY_LOG"
```

The main rollout defaults to `--reset_camera_warmup_steps 0`. Use `1` only when frame 0 is known to be stale; this
holds the current joint targets for one simulation step and sends the refreshed observation to the policy. The legacy
`red_cube_to_box_policy_rollout_skip_first_frame.py` entrypoint is equivalent to selecting `1` explicitly.

For a ten-episode batch, reuse the command with `--episodes 10`, a new `run_root`, and an explicit acceptance threshold
such as `--minimum_success_rate 0.5`. Every episode gets its own JPEG/JSONL/offline-HTML recording directory below the
common `--record_dir` root.

For Windows-native LeIsaac, install the same local client once:

```powershell
& "D:\Envs\leisaac-so101-win\Scripts\python.exe" -m pip install -e .\packages\openpi-client
```

Then connect to the server IP while keeping Isaac Sim on local `cuda:0` and without `--renderer_device`:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$runRoot = "D:\Sim\results\policy-rollout\red_cube_$stamp"
$env:RED_CUBE_POLICY_LOG = Join-Path $runRoot 'rollout.log'
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

& "D:\Envs\leisaac-so101-win\Scripts\python.exe" `
  examples\so101\red_cube_to_box_policy_rollout.py `
  --headless `
  --enable_cameras `
  --device cuda:0 `
  --rendering_mode performance `
  --assets_root "D:\Sim\leisaac-v0.4.0" `
  --policy_host 10.120.16.48 `
  --policy_port 18000 `
  --episodes 1 `
  --seed 42 `
  --maximum_steps 2400 `
  --actions_per_inference 10 `
  --reset_camera_warmup_steps 1 `
  --record_dir (Join-Path $runRoot 'recordings') `
  2>&1 | Tee-Object -FilePath $env:RED_CUBE_POLICY_LOG

Select-String -LiteralPath $env:RED_CUBE_POLICY_LOG -Pattern `
'RED_CUBE_TO_BOX_POLICY_|policy_endpoint:|policy_reset_camera_warmup_steps:|policy_camera:|policy_inference:|policy_episode:|policy_recording_dir:|completed_episodes:|successful_episodes:|success_rate:|Traceback|ValueError|RuntimeError|out of memory|OOM' |
ForEach-Object { "$($_.LineNumber):$($_.Line)" }
```
