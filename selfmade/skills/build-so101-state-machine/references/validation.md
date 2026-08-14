# Windows-native validation reference

## Contents

- Environment contract
- Validation order
- Static checks
- Dynamic smoke
- Terminal log extraction
- Acceptance criteria

## Environment contract

Use the configured Windows-native environment unless the user supplies replacements:

```text
OpenPI:  D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi
Python:  D:\Envs\leisaac-so101-win\Scripts\python.exe
LeIsaac: D:\Sim\leisaac
Assets:  D:\Sim\leisaac-v0.4.0
Device:  cuda:0
```

Do not use WSL or a server for Isaac Sim. Do not add `--renderer_device`. Start with one environment, headless,
performance rendering, and no recording. Accept the NVIDIA EULA through `OMNI_KIT_ACCEPT_EULA=YES`.

## Validation order

1. Confirm package installation and asset identity without reinstalling working dependencies.
2. Run the scene audit and require its semantic success sentinel.
3. Run targeted AST/unit/registry tests.
4. Run one state-machine smoke in a fresh Isaac process.
5. Inspect phase transitions, abort fields, final object pose, speed, and semantic success sentinel.
6. Repeat before adding camera recording or additional seeds.

If 8 GB VRAM is insufficient, first reduce camera or recording load. Do not change physics, phase gates, or success
criteria to mask OOM.

For every recorded run, print first-frame camera dtype, shape, minimum, maximum, and mean. Treat `max=0` as a renderer
failure and stop immediately. Existing JPEG files or a valid tensor shape do not prove that off-screen RTX rendering is
active. When cameras are enabled and no explicit mode was requested, use `performance`; do not add
`--renderer_device`.

## Static checks

Adapt the test list to the files changed:

```powershell
& "D:\Envs\leisaac-so101-win\Scripts\python.exe" -m pytest -q -p no:cacheprovider `
  examples/so101/autogen_polar_retreat_transport_state_machine_test.py `
  examples/so101/red_cube_to_box_expert_registry_test.py `
  examples/so101/cube_axis_alignment_test.py `
  examples/so101/gripper_pick_latch_test.py

git diff --check
git status --short
```

Run the repository's configured Ruff binary if available. Do not install or redownload it merely for one check without
need or approval.

## Dynamic smoke

Use a timestamped primary log. This is the run log, not a second diagnostics artifact:

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONUNBUFFERED = "1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$env:SO101_EXPERT_LOG = "D:\Sim\results\expert-smoke\polar\polar-seed42-$stamp.stdout.log"

& "D:\Envs\leisaac-so101-win\Scripts\python.exe" `
  examples/so101/red_cube_to_box_expert_smoke.py `
  --headless `
  --enable_cameras `
  --device cuda:0 `
  --rendering_mode performance `
  --assets_root "D:\Sim\leisaac-v0.4.0" `
  --expert autogen_polar_retreat_transport `
  --seed 42 2>&1 | Tee-Object -FilePath $env:SO101_EXPERT_LOG
```

Replace only the expert name, seed, or user-specified paths. Do not add `--renderer_device`.

## Terminal log extraction

Print selected fields directly to the terminal. Do not redirect this extraction into another file:

```powershell
Select-String -LiteralPath $env:SO101_EXPERT_LOG -Pattern `
'servo_parameters|expert_phase:|expert_polar_path:|expert_ik_runtime_mode:|phase_step=|path_segment=|focus_body=|control_body=|actual_focus_w=|motion_start_w=|motion_target_w=|current_target_w=|target_error=|reference_tracking_error=|stable_streak=|retreat_z_error=|retreat_radial_error=|bearing_error=|retreat_safety_reason=|wrist_posture_target=|ik_task_error=|ik_task_singular_values=|ik_primary_delta_joint_pos=|ik_nullspace_delta_joint_pos=|ik_delta_joint_pos=|joint_target_accumulation_step=|controlled_joint_names=|last_joint_position_target=|current_controlled_joint_position=|joint_target_minus_actual=|actual_joint_velocity=|jaw_cube_distance=|grasp_relative_position_error=|grasp_relative_position_max_error=|grasp_loss_streak=|grasp_confirmed=|expert_abort|servo_abort_reason|release_block_reason|completed_steps|cube_final_pos_w|cube_offset_from_box|cube_final_speed|expert_success|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' |
ForEach-Object { "$($_.LineNumber):$($_.Line)" }
```

## Acceptance criteria

Require all applicable conditions:

- no traceback, runtime error, OOM, phase timeout, grasp-loss abort, or release block;
- expected phase order completed;
- grasp remained confirmed until intentional release;
- final object lies inside the task's actual success bounds;
- final speed satisfies the task predicate;
- `expert_success: True`;
- `RED_CUBE_TO_BOX_EXPERT_SMOKE_OK`;
- process exit code is zero;
- a fresh-process repeat succeeds before claiming a stable baseline.

Report exact log path, completed steps, final object pose, box offset, speed, semantic sentinel, and commit hash.

When pickup geometry or randomization behavior changes, follow the smoke with a recorded 10-episode batch. Put all ten
episode directories under one timestamped `recordings` root and require the summary to report completed/successful/
failed episodes, aborts, resets, non-finite episodes, and success rate. Compare the same seed sequence to the previous
implementation so a changed random sample is not mistaken for improvement.

Before interpreting controller failures, report the randomized cube min/max pose and check box clearance plus pickup
reachability. If failures all share one phase/reason, repair that bottleneck before changing later phases.

## Dataset collection smoke

After freezing a dynamically validated expert, collect one successful episode into a new timestamped directory. Require
the collector sentinel, then run the read-only HDF5 audit. Confirm that `actions` and `obs/joint_pos` are `[T, 6]`, front
images are `[T, H, W, 3] uint8`, timestamps are strictly increasing at the environment step interval, and optional wrist
images appear only when the policy observation actually contains that sensor. Report the native HDF5 bytes per episode
before choosing the final collection size.
