# SO-101 + LeIsaac

This directory contains the OpenPI SO-101 integration, Windows-native LeIsaac validation tools, dataset conversion,
and the RedCubeToBox scripted expert.

- Source-change record: [0.originalCodeChanges.md](0.originalCodeChanges.md)
- Windows Leader and remote policy workflow: [REMOTE_LEADER_RUNBOOK.md](REMOTE_LEADER_RUNBOOK.md)
- Detailed RedCubeToBox troubleshooting history: [../../selfmade/problem_solving.md](../../selfmade/problem_solving.md)

## RedCubeToBox

The task ID is `OpenPI-LeIsaac-SO101-RedCubeToBox-v0`. Use Windows-native Isaac Sim with one environment,
`cuda:0`, headless rendering, and the installed assets below. Do not use WSL and do not add `--renderer_device`.

```powershell
Set-Location "D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\openpi"
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONUNBUFFERED = "1"
```

### Scene audit

```powershell
& "D:\Envs\leisaac-so101-win\Scripts\python.exe" `
  examples/so101/red_cube_to_box_scene_audit.py `
  --headless `
  --enable_cameras `
  --device cuda:0 `
  --assets_root "D:\Sim\leisaac-v0.4.0"
```

A valid audit ends with `RED_CUBE_TO_BOX_SCENE_AUDIT_OK`.

### Single-environment smoke

`autogen_polar_retreat_transport` is the recommended scripted expert.

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
'expert_phase:|expert_state:|expert_axis_alignment:|expert_polar_close:|expert_polar_path:|expert_ik_runtime_mode:|actual_focus_w=|motion_target_w=|current_target_w=|target_error=|stable_streak=|jaw_cube_distance=|grasp_confirmed=|servo_timeout_phase:|servo_abort_reason:|release_block_reason:|completed_steps:|cube_final_pos_w:|cube_offset_from_box:|cube_final_speed:|expert_success:|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' |
ForEach-Object { "$($_.LineNumber):$($_.Line)" }
```

### Recorded ten-episode batch

All ten episode recordings are stored below one timestamped root.

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
'camera_frame_stats|episode_record_dir:|episode:|completed_episodes:|successful_episodes:|failed_episodes:|non_finite_episodes:|reset_episodes:|servo_timeout_episodes:|servo_abort_episodes:|success_rate:|RED_CUBE_TO_BOX_EXPERT_BATCH|Traceback|RuntimeError' |
ForEach-Object { "$($_.LineNumber):$($_.Line)" }
```

## Dataset conversion and training

Keep Isaac Sim/LeIsaac and OpenPI/LeRobot in separate Python environments. Record native LeIsaac HDF5 in the simulator
environment, then convert it from the OpenPI environment:

```powershell
uv run examples/so101/convert_leisaac_hdf5_to_lerobot.py `
  --input-path "D:\path\to\recording.hdf5" `
  --repo-id local/leisaac-so101-dataset `
  --task "Pick up the red cube and place it inside the green box." `
  --fps 60 `
  --image-mode video
```

Set `OPENPI_SO101_LIFTCUBE_REPO_ID` to the converted dataset, compute normalization statistics, and start training:

```powershell
$env:OPENPI_SO101_LIFTCUBE_REPO_ID = "local/leisaac-so101-dataset"
uv run scripts/compute_norm_stats.py --config-name pi05_lora_so101_liftcube
uv run scripts/train.py pi05_lora_so101_liftcube --exp-name=so101_lora --num-workers=0
```

Serve a checkpoint with a free port shared by the policy server and LeIsaac client:

```powershell
uv run scripts/serve_policy.py policy:checkpoint `
  --policy.config=pi05_lora_so101_liftcube `
  --policy.dir="D:\path\to\checkpoint" `
  --port=18000
```
