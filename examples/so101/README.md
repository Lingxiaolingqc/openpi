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
table asset. Run the bounded scene audit to capture the robot, end-effector, cube, camera, target-box floor,
and environment coordinates from the installed LeIsaac version:

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 120s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_scene_audit.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT"
```

Success requires both process exit code `0` and the semantic marker
`RED_CUBE_TO_BOX_SCENE_AUDIT_OK`. The audit does not connect to the physical Leader, change assets, or write a
dataset. It reports the floor position both as `target_box_floor_pos_w` and as separate
`target_box_floor_x_w`, `target_box_floor_y_w`, and `target_box_floor_z_w` fields for shell parsing. It also
reports every named joint coordinate in radians, every named robot-body origin in world coordinates, and the
FrameTransformer gripper and offset jaw-detection positions. These fields distinguish joint coordinates from
Cartesian positions and prevent the raw jaw body from being mistaken for the IK end-effector.

The tray center is `(0.20606, -0.40428)` in environment coordinates. The robot root is `(0.35, -0.64)`, and the
zero-joint arm points primarily along world `+Y`. Relative to the preceding `(0.49394, -0.40428)` position, the
complete tray keeps the same forward Y reach and mirrors its lateral X offset across the robot's `x=0.35` sagittal
line: `+0.14394 m` becomes `-0.14394 m`. The tray consists of one floor and four green kinematic walls. Its success
predicate requires the cube to be inside the tray bounds, below the wall top, and moving no faster than `0.15 m/s`.

Validate environment creation and the predicate before developing the expert:

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
export RED_CUBE_TO_BOX_SMOKE_LOG="$LEISAAC_BASE/results/leisaac/red-cube-to-box-env-smoke.log"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 120s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_env_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --steps 10 \
  2>&1 | tee "$RED_CUBE_TO_BOX_SMOKE_LOG"

smoke_status=${PIPESTATUS[0]}
echo "red_cube_to_box_env_smoke_exit=$smoke_status"

grep -nE \
  'RED_CUBE_TO_BOX|task_id|device_id|simulation_device|box_part|cube_|initial_success|teleported_success|completed_steps|rewards_finite|Traceback|Error|RuntimeError' \
  "$RED_CUBE_TO_BOX_SMOKE_LOG" |
tail -n 180
```

The semantic checks require five box pieces, `initial_success: False`, `teleported_success: True`, finite
rewards, and `RED_CUBE_TO_BOX_ENV_SMOKE_OK`. Teleportation is used only to test the predicate; it is not part of
the eventual expert or dataset-generation trajectory.

The environment smoke checks geometry and the success predicate, but teleportation alone does not prove that
the floor dynamically catches the cube. Run the independent drop test before tuning the robot expert:

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
export RED_CUBE_TO_BOX_DROP_LOG="$LEISAAC_BASE/results/leisaac/red-cube-to-box-drop-smoke.log"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 180s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_drop_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --steps 360 \
  2>&1 | tee "$RED_CUBE_TO_BOX_DROP_LOG"

drop_status=${PIPESTATUS[0]}
echo "red_cube_to_box_drop_smoke_exit=$drop_status"

grep -nE \
  'RED_CUBE_TO_BOX_DROP|task_id|device_id|simulation_device|action_space|box_part|drop_|fall_distance|horizontal_offset|cube_final_speed|settled_inside|completed_steps|rewards_finite|unexpected_reset|Traceback|Error|RuntimeError' \
  "$RED_CUBE_TO_BOX_DROP_LOG" |
tail -n 180
```

Success requires a non-trivial `fall_distance`, a settled cube inside the tray, finite rewards, no reset, and
`RED_CUBE_TO_BOX_DROP_SMOKE_OK`. This isolates target geometry and collision from all grasp-controller errors.

The scripted expert uses LeIsaac's `so101_state_machine` absolute-pose IK action configuration. It executes
smooth Cartesian phases for approach, grasp, lift, transfer, release, retract, and settling. Task success and
time-out terminations are disabled during this diagnostic episode so the environment cannot auto-reset before
the final state is inspected. This task overrides the generic state-machine gripper close target from `0.4` to
`0.05` radians: the first diagnostic reached a valid `0.01826 m` jaw-to-cube distance but remained above
LeIsaac's `0.26`-radian grasp threshold with the generic close target. Closing farther moved the jaw frame, so
the measured closed-jaw error is also compensated by moving the grasp target `10 mm` in both horizontal axes
and `20 mm` downward.

All three scripted experts therefore have an 8D action even though the robot has six joints: local-frame
end-effector position `(x, y, z)`, end-effector unit quaternion `(w, x, y, z)`, and one binary gripper command.
The differential IK controller converts the first seven values into targets for the five arm joints; the final
value controls the sixth, gripper joint. The legacy expert fixes the target orientation throughout. The
adaptive expert uses the same fixed orientation through grasping, then commands the currently measured
orientation from lift onward. It remains as an experimental comparison: a seed-42 run grasped the cube but
accumulated orientation drift and large horizontal position error during transport. The servo expert keeps
the exact validated world orientation through grasping and gradual lift. Starting at transfer, its action stays
7D pose plus 1D gripper for API compatibility, but its phase-aware DLS action term solves only the three
translation rows of the Jacobian. This avoids asking the five-joint arm to satisfy a six-dimensional pose while
the rate-limited Cartesian P outer loop transports the cube. Leader control uses a different contract: six
direct joint-position values.

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
export RED_CUBE_TO_BOX_EXPERT_LOG="$LEISAAC_BASE/results/leisaac/red-cube-to-box-expert-smoke.log"
export RED_CUBE_TO_BOX_RECORD_DIR="$LEISAAC_BASE/results/leisaac/red-cube-to-box-expert-recordings"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 240s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert servo \
  --seed 42 \
  --record_dir "$RED_CUBE_TO_BOX_RECORD_DIR" \
  --record_every 4 \
  --record_fps 15 \
  2>&1 | tee "$RED_CUBE_TO_BOX_EXPERT_LOG"

expert_status=${PIPESTATUS[0]}
echo "red_cube_to_box_expert_smoke_exit=$expert_status"

if grep -q '^RED_CUBE_TO_BOX_EXPERT_SMOKE_OK$' "$RED_CUBE_TO_BOX_EXPERT_LOG" &&
   ! grep -q '^RED_CUBE_TO_BOX_EXPERT_SMOKE_FAILED$' "$RED_CUBE_TO_BOX_EXPERT_LOG"; then
  expert_semantic_status=0
else
  expert_semantic_status=1
fi
echo "red_cube_to_box_expert_semantic_exit=$expert_semantic_status"

grep -nE \
  'RED_CUBE_TO_BOX|expert_variant|expert_orientation_policy|expert_ik_action_class|expert_ik_runtime_mode|servo_parameters|diagnostic_record|expert_phase|expert_state|expert_feedback|expert_jaw_anchor|expert_safety|expert_tracking|expert_servo|expert_grasp_event|expert_abort_before_step|task_id|device_id|simulation_device|action_space|cube_|target_box|completed_steps|rewards_finite|unexpected_reset|grasp_confirmed|grasp_lost_before_release|box_aligned_before_release|servo_timeout_phase|servo_abort_reason|expert_success|Traceback|Error|RuntimeError' \
  "$RED_CUBE_TO_BOX_EXPERT_LOG" |
tail -n 220
```

Use `expert_semantic_status` as the authoritative result. Some Isaac Sim fast/skip-cleanup shutdown paths end
the host process with status zero even after the Python diagnostic has printed its failure sentinel.

When `--record_dir` is set, each smoke creates a unique `<expert>-seed<seed>-<timestamp>-pid<pid>` directory.
It contains sampled JPEG frames, `trace.jsonl`, `result.json`, and an offline `index.html` player. Recording one
frame every four 60 Hz control steps produces a 15 FPS diagnostic without requiring FFmpeg in the Isaac
environment. Frames are written incrementally, so already-written evidence remains readable if a later step
fails. Copy one run to Windows and open `index.html` locally:

```powershell
scp -r `
  xiaoqinchuan@10.120.16.48:/home/data/xiaoqinchuan/results/leisaac/red-cube-to-box-expert-recordings/<run-directory> `
  D:\Documents\Xprogram\HuiXIONG\EmbodiedAI\red-cube-failures\
```

The initial expert has `1100` control steps. A successful dynamic episode reports every phase, finite rewards,
no unexpected reset, `expert_success: True`, and `RED_CUBE_TO_BOX_EXPERT_SMOKE_OK`. If it fails, the final cube
offset and speed distinguish grasp/transport errors from placement or settling errors before any recording is
enabled.

The validated seed-42 run acquired the cube at a `0.00412 m` jaw distance with a `0.14649 rad` gripper joint,
kept `pick_cube=True` through transport, and released it at a final box-relative offset of
`(0.03634, 0.00642, 0.01907) m`. The final cube speed was `0.001097 m/s`, and the episode reported
`expert_success: True`.

The validated fixed-offset implementation remains in `red_cube_to_box_task/state_machine.py` as the `legacy`
expert. The separate `red_cube_to_box_task/adaptive_state_machine.py` expert deliberately uses that validated
trajectory for its first grasp attempt. It advances only after a confirmed grasp; otherwise it measures the
closed-jaw error, reopens and retracts, applies a bounded Cartesian correction, and makes one second attempt.
After grasping, it tracks whether the grasp is lost before release. Transport and placement use live
cube-position error instead of assuming that the initial grasp transform remains constant. Because the IK
command is an absolute pose, each Cartesian error component is bounded to `0.12 m`, rather than treating the
bound as a per-step increment. A final
closed-gripper alignment phase requires the cube to remain
within `0.012 m` of the release target for 20 consecutive control steps before release. Select one without
changing either implementation:

The `legacy_dynamic_grasp_offset` expert is a single-variable comparison built directly on the original
`legacy`, not on a jaw-anchor expert. Approach, grasp, lift, fixed-world pose IK, timings, gripper commands and
release Z are unchanged. During the final 40 lift steps it samples `cube_xy - gripper_xy` only after the cube
has risen at least `0.05 m`, then freezes the per-environment median. Its desired cube XY is the audited box
center plus a `0.005 m` bias toward the robot root. Transfer and release gripper XY are calculated as
`desired_cube_xy - measured_gripper_to_cube_xy`; the historical fixed `_PLACE_XY_OFFSET` is therefore unused
only by this expert. Smoke logs expose the samples, measured grasp offset, desired cube point, dynamic gripper
target and resulting box-relative place offset.

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
export DYNAMIC_GRASP_LOG="$LEISAAC_BASE/results/leisaac/legacy-dynamic-grasp-offset-seed42.log"
export DYNAMIC_GRASP_RECORD_DIR="$LEISAAC_BASE/results/leisaac/legacy-dynamic-grasp-offset-recordings"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 240s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert legacy_dynamic_grasp_offset \
  --seed 42 \
  --record_dir "$DYNAMIC_GRASP_RECORD_DIR" \
  --record_every 4 \
  --record_fps 15 \
  2>&1 | tee "$DYNAMIC_GRASP_LOG"

dynamic_grasp_status=${PIPESTATUS[0]}
echo "legacy_dynamic_grasp_offset_exit=$dynamic_grasp_status"

grep -nE \
  'expert_phase|expert_state|expert_dynamic_grasp_offset|completed_steps|cube_final|cube_offset|cube_final_speed|expert_success|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' \
  "$DYNAMIC_GRASP_LOG" |
tail -n 280
```

The separate `legacy_dynamic_grasp_offset_residual_corrected` expert tests the tracking-error hypothesis
without overwriting that baseline. It follows `legacy_dynamic_grasp_offset` unchanged through transfer. On
the last 40 transfer steps it re-samples `cube_xy - gripper_xy`; the first `lower_into_box` step freezes the median and
recomputes `desired_cube_xy - transfer_gripper_to_cube_xy`. The change from the lift-derived target is capped at
`0.10 m`. Thus lift samples guide transfer while transfer samples determine lower/release, without a per-step servo or
an accumulated correction. At the same transition it measures the current vertical gripper-to-jaw separation and
raises the frozen lower target as needed so
that the jaw target remains at least `0.030 m` above the audited box-wall top. At the final lower target, the actual
gripper EE must remain within `0.015 m` of the corrected 3D target for 10 consecutive control steps before release.
Otherwise the gripper remains closed; after 300 additional hold steps the run fails without releasing the cube.

When recording is enabled, a red sphere marks the live gripper EE and a green sphere marks its vertical
projection onto the audited table surface. Both points and the release-gate state are also written to `trace.jsonl`.
Both sphere radii are `0.05 m` for visibility in the headless front-camera recording.

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
export RESIDUAL_CORRECTED_LOG="$LEISAAC_BASE/results/leisaac/legacy-dynamic-residual-corrected-seed42.log"
export RESIDUAL_CORRECTED_RECORD_DIR="$LEISAAC_BASE/results/leisaac/legacy-dynamic-residual-corrected-recordings"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 240s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert legacy_dynamic_grasp_offset_residual_corrected \
  --seed 42 \
  --record_dir "$RESIDUAL_CORRECTED_RECORD_DIR" \
  --record_every 4 \
  --record_fps 15 \
  2>&1 | tee "$RESIDUAL_CORRECTED_LOG"

residual_corrected_status=${PIPESTATUS[0]}
echo "legacy_dynamic_grasp_offset_residual_corrected_exit=$residual_corrected_status"

grep -nE \
  'expert_phase|expert_state|expert_dynamic_grasp_offset|expert_transfer_residual_correction|expert_lower_release_gate|release_block_reason|completed_steps|cube_final|cube_offset|cube_final_speed|expert_success|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' \
  "$RESIDUAL_CORRECTED_LOG" |
tail -n 320
```

The independent `autogen_retreat_transport` expert adapts the movement pattern from
[`haoran1062/so101-autogen`](https://github.com/haoran1062/so101-autogen/) without replacing any existing expert.
It preserves the validated legacy pickup and the safe release-height/release-gate safeguards above. After lift it
captures the live gripper pose, moves to a high safe point whose XY radius about the robot root is `5/7` of the entry
radius, then captures the live pose again and transports directly to the live target-box floor-center XY. Both segments
move their Cartesian reference by at most `0.0025 m` per control step. This comparison disables both inherited
`cube_xy - gripper_xy` samplers and the residual placement correction. It keeps the complete 6D legacy pose constraint
through retreat, transport, lower, and release so yaw cannot drift while the cube is held. This is not a pre-rotation
phase and it does not command `shoulder_pan` directly.

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
export AUTOGEN_RETREAT_LOG="$LEISAAC_BASE/results/leisaac/autogen-retreat-transport-seed42.log"
export AUTOGEN_RETREAT_RECORD_DIR="$LEISAAC_BASE/results/leisaac/autogen-retreat-transport-recordings"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 300s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert autogen_retreat_transport \
  --seed 42 \
  --record_dir "$AUTOGEN_RETREAT_RECORD_DIR" \
  --record_every 4 \
  --record_fps 15 \
  2>&1 | tee "$AUTOGEN_RETREAT_LOG"

autogen_retreat_status=${PIPESTATUS[0]}
echo "autogen_retreat_transport_exit=$autogen_retreat_status"

grep -nE \
  'expert_phase|expert_state|expert_autogen_path|expert_tracking|expert_grasp_event|expert_ik_runtime_mode|expert_lower_release_gate|release_block_reason|completed_steps|cube_final|cube_offset|cube_final_speed|expert_success|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' \
  "$AUTOGEN_RETREAT_LOG" |
tail -n 360
```

The separate `autogen_independent_retreat_transport` expert is a clean-room state-machine comparison. It does not
inherit `RedCubeToBoxStateMachine` or either dynamic-offset expert. The pickup keyframes are repeated locally and the
standalone `lift_cube` phase is removed. `retreat_to_safe` is one combined XYZ Cartesian segment: it raises the
gripper toward `0.22 m` above the target-box floor center while reducing its root-relative XY radius to `5/7` of the
phase-entry radius. Unlike the original `0.0025 m` reference step that dropped the cube, this isolated variant limits
the combined path to `0.0008 m` per control step. The phase finishes when the measured gripper has both reached the
safe-height band and reduced its root-relative XY radius sufficiently; it does not require an unreachable exact XYZ
endpoint. The following transfer hover uses the same `floor + 0.22 m` height, so transfer remains approximately
horizontal. A confirmed grasp whose measured
jaw-to-cube distance exceeds `0.025 m` aborts immediately instead of carrying an already dropped cube through later
phases. Retreat, transport, lower, and retract otherwise advance only after their measured completion conditions stay
true for a configured number of consecutive steps. Placement XY is the live target-box floor center, offsets are
disabled, and the complete 6D pose target remains active for this controlled comparison.

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"

export AUTOGEN_INDEPENDENT_LOG="$LEISAAC_BASE/results/leisaac/autogen-independent-retreat-transport-seed42.log"
export AUTOGEN_INDEPENDENT_RECORD_DIR="$LEISAAC_BASE/results/leisaac/autogen-independent-retreat-transport-recordings"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"

timeout --signal=KILL 420s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert autogen_independent_retreat_transport \
  --seed 42 \
  --record_dir "$AUTOGEN_INDEPENDENT_RECORD_DIR" \
  --record_every 4 \
  --record_fps 15 \
  2>&1 | tee "$AUTOGEN_INDEPENDENT_LOG"

autogen_independent_status=${PIPESTATUS[0]}
echo "autogen_independent_retreat_transport_exit=$autogen_independent_status"

grep -nE \
  'expert_phase|expert_state|expert_independent_path|retreat_subphase|jaw_cube_distance|expert_tracking|expert_grasp_event|expert_abort|expert_ik_runtime_mode|servo_timeout_phase|servo_abort_reason|release_block_reason|completed_steps|cube_final|cube_offset|cube_final_speed|expert_success|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' \
  "$AUTOGEN_INDEPENDENT_LOG" |
tail -n 420
```

The additional `autogen_polar_retreat_transport` expert preserves the independent expert as a comparison and changes
only the grasped-object route. After closing, it first holds measured X/Y and raises the gripper toward
`floor + 0.22 m`. It then holds Z and the entry bearing while shortening the measured root-relative radius by exactly
`0.030 m`. The measured safe height at the arc entry is then frozen. It follows a root-centered arc at constant radius and that frozen height to the live box
bearing; the commanded yaw rotates by the same angle as the arc. A final radial segment moves at fixed box bearing and
the same frozen height to the live box-floor center before the existing safe lower,
release, and retract phases. Retreat, arc, and radial phases require the bounded reference to finish and the measured
gripper XYZ and root bearing to remain within tolerance. This prevents the old false completion where radius and
height matched even though the arm had rotated to the wrong Cartesian point.

Both internal retreat segments use the five-dimensional `xyz_pitch_joint` solve: measured XYZ, base-frame pitch, and
the shoulder-pan coordinate captured at retreat entry are constrained. Roll and yaw remain free. This directly prevents
the base rotation observed under both full-pose and `xyz_tilt` retreat while keeping the principal gripper pitch. Arc
and later phases remain unchanged for a controlled comparison. Retreat aborts before another physics step if measured Z exceeds its target by
more than `0.020 m`, root bearing error exceeds `20 degrees`, or the final-target error remains more than `0.010 m`
above its best observed value for 20 consecutive steps after the reference finishes. The corresponding cause appears
in `retreat_safety_reason`, `servo_abort_reason`, and `release_block_reason`.

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"

export AUTOGEN_POLAR_LOG="$LEISAAC_BASE/results/leisaac/autogen-polar-retreat-transport-seed42.log"
export AUTOGEN_POLAR_RECORD_DIR="$LEISAAC_BASE/results/leisaac/autogen-polar-retreat-transport-recordings"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"

timeout --signal=KILL 480s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert autogen_polar_retreat_transport \
  --seed 42 \
  --record_dir "$AUTOGEN_POLAR_RECORD_DIR" \
  --record_every 4 \
  --record_fps 15 \
  2>&1 | tee "$AUTOGEN_POLAR_LOG"

autogen_polar_status=${PIPESTATUS[0]}
echo "autogen_polar_retreat_transport_exit=$autogen_polar_status"

grep -nE \
  'expert_variant|servo_parameters|expert_phase|expert_state|expert_polar_path|bearing_error|retreat_worsening|retreat_safety_reason|expert_tracking|expert_grasp_event|expert_abort|expert_ik_runtime_mode|servo_timeout_phase|servo_abort_reason|release_block_reason|completed_steps|cube_final|cube_offset|cube_final_speed|expert_success|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' \
  "$AUTOGEN_POLAR_LOG" |
tail -n 520
```

The independent `autogen_reference` expert ports the bundled
`autogen/so101-autogen-main/src/state_machine` implementation as a reference baseline. It does not inherit the
legacy, adaptive, servo, or earlier Autogen-derived experts. The original state order and numerical constants are
kept: approach, descend, grasp, grasp settle, lift, radial retreat, transport, release, and return home; the original
Cartesian step sizes, phase limits, green-ray geometry, gripper timing/range, and effective release height are also
preserved.

Only framework adapters are changed. World targets are expressed in the robot-root frame expected by the source
implementation; the second-back port's XYZ wrist IK plus continuously recomputed `wrist_flex` correction is restored
through the existing phase-aware Isaac Lab action term. Because the bundled URDF's `gripper_frame_link` local axes do
not match the current USD detection-frame axes, the grasp trigger is a finite segment from
`ee_frame.target[0]` (`gripper_frame`) to `ee_frame.target[1]` (`jaw_detection_frame`) evaluated against the live cube
OBB; and the binary
gripper action is replaced by a continuous gripper-joint target so the source openness range is meaningful. The
single-object task uses the live target-box floor center instead of Autogen's multi-object placement manager. A failed
grasp stops safely instead of issuing the source project's direct joint-space return-home recovery. Thus this is a
behavioral port with explicit adapters, not a byte-for-byte runtime transplant.

Run the first dynamic comparison with seed 42. The recorder retains both successful and failed visual evidence. Do
not add `--renderer_device`; this installation uses the selected simulation device.

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"

export AUTOGEN_REFERENCE_LOG="$LEISAAC_BASE/results/leisaac/autogen-reference-seed42.log"
export AUTOGEN_REFERENCE_RECORD_DIR="$LEISAAC_BASE/results/leisaac/autogen-reference-recordings"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac" "$AUTOGEN_REFERENCE_RECORD_DIR"

timeout --signal=KILL 480s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert autogen_reference \
  --seed 42 \
  --record_dir "$AUTOGEN_REFERENCE_RECORD_DIR" \
  --record_every 4 \
  --record_fps 15 \
  2>&1 | tee "$AUTOGEN_REFERENCE_LOG"

autogen_reference_status=${PIPESTATUS[0]}
echo "autogen_reference_transport_exit=$autogen_reference_status"

if grep -q '^RED_CUBE_TO_BOX_EXPERT_SMOKE_OK$' "$AUTOGEN_REFERENCE_LOG" &&
   ! grep -q '^RED_CUBE_TO_BOX_EXPERT_SMOKE_FAILED$' "$AUTOGEN_REFERENCE_LOG"; then
  autogen_reference_semantic_status=0
else
  autogen_reference_semantic_status=1
fi
echo "autogen_reference_semantic_exit=$autogen_reference_semantic_status"

grep -nE \
  'expert_variant|servo_parameters|expert_phase|expert_state|expert_autogen_reference|expert_ik_runtime_mode|expert_grasp_event|expert_abort|completed_steps|cube_final|cube_offset|cube_final_speed|expert_success|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' \
  "$AUTOGEN_REFERENCE_LOG" |
tail -n 520
```

The authoritative result is `autogen_reference_semantic_exit`, not only the transport exit. The
`expert_autogen_reference` records expose the live grasp segment, its length and nearest hit distance, the cube's
projection onto and shortest distance from that segment, the gripper and jaw detection
frames, wrist/gripper/jaw height above the cube, robot-base command, actual wrist world position, descent XY tracking
error, gripper command, and retreat/transport targets so that a failure can be compared directly with the source
state-machine assumptions. The jaw detection frame (`ee_frame.target[1]`) remains the post-close grasp-confirmation
reference; it is not used as the source green-ray frame.

This comparison intentionally combines the posture-correction behavior from commit `c1295cb` with a USD-native finite
gripper-to-jaw grasp corridor. It therefore does not claim to be a bit-for-bit reproduction of the bundled source; the log's
`expert_ik_runtime_mode` and `posture_target_rad` fields make that experimental difference explicit.

The third implementation, `red_cube_to_box_task/servo_state_machine.py`, inherits the adaptive grasp, retry,
and gradual lift logic but does not replace either comparison expert. Its companion
`red_cube_to_box_task/phase_aware_ik_action.py` preserves the 7D pose command shape while dropping the three
orientation rows from the DLS solve during transfer, lowering, alignment, release, and retraction. Starting at
transfer, the outer loop uses `delta = 0.25 * position_error`, limits the Cartesian command to `0.003 m` per
control step, treats errors no larger than `0.0015 m` as arrived, and requires 10 consecutive arrived steps
before advancing. Transfer, lowering, and final alignment each have a 300-step deadline; a missed deadline ends
the episode without opening the gripper and reports `servo_timeout_phase`. A lost grasp also aborts before
another physics step and reports `servo_abort_reason`. The single-episode log prints `expert_ik_runtime_mode`
at every phase transition so the pose-to-position-only switch can be audited.

The `legacy_gripper_anchor` variant preserves legacy pickup, lift, transport, timings, and fixed-world
orientation. Only placement changes: it treats the jaw detection frame as the cube anchor, measures the live
three-dimensional `jaw - gripper` offset, and converts the desired jaw position over the box into an IK gripper
target. It opens only after the jaw remains within the configured XY/Z tolerances for ten consecutive steps. If
alignment times out, it stops before opening and leaves the diagnostic recording intact. The original `legacy`
implementation remains unchanged for comparison.

The independent `legacy_gripper_anchor_align_then_lower` variant changes only the placement order while
retaining legacy full-pose IK and fixed-world orientation. It first aligns the measured jaw X/Y with the box
center while holding the measured high Z from phase entry. After ten consecutive horizontally aligned steps,
it locks X/Y to the box center and descends vertically to the same legacy release height. Release is allowed
only after the final X/Y/Z errors remain within tolerance for ten consecutive steps. The commanded descent
occupies the first 120 steps and has another 60-step confirmation budget. Either stage reports its own timeout
phase, so horizontal reachability and vertical placement failures remain distinguishable.
Its gripper target is written explicitly as `desired_jaw_w + (gripper_pos_w - jaw_pos_w)`: desired jaw pose
plus the currently measured jaw-to-gripper offset. This is algebraically identical to the original feedback
form `gripper_pos_w + (desired_jaw_w - jaw_pos_w)`; the rewrite clarifies the frame interpretation but is not
expected to change millimetre-scale convergence.

The `legacy_gripper_anchor_position_align_then_lower` control isolates the suspected high-alignment
over-constraint. Its phases and targets are identical to `legacy_gripper_anchor_align_then_lower`, but only
`align_over_box` removes all three orientation rows and solves the three XYZ position rows. The Z target in
that phase is the measured high jaw position captured on entry, so it serves as a height hold rather than a
descent command. `lower_into_box` restores the original legacy full-pose IK. Comparing these two variants with
the same seed tests the effect of alignment orientation constraints without also changing the descent solver.

The `legacy_gripper_anchor_weighted_position_align_then_lower` variant keeps those same phases and XYZ
targets, but replaces ordinary minimum-norm position IK during high alignment with weighted damped least
squares. Joint motion penalties are `0.25` for `shoulder_pan`, `1.0` for `shoulder_lift` and `elbow_flex`,
`1.5` for `wrist_flex`, and `2.0` for `wrist_roll`, with damping `0.05`. Lower penalties make a joint cheaper
in redundant XYZ solutions, so the solver prefers base rotation when it helps reduce Cartesian error without
hard-coding a shoulder-pan angle. Descent restores unmodified legacy full-pose IK.
The smoke log emits `expert_weighted_ik` with the current five arm-joint positions and the preceding weighted
IK joint delta, making it possible to distinguish solver preference from actuator or joint-limit clipping.

The `legacy_gripper_anchor_safe_planar_align_then_lower` variant replaces the visually contorted
position-only high align without overwriting that comparison expert. Pickup, transfer, align-first ordering,
and full-pose descent remain identical to the legacy control. High alignment solves jaw X/Y plus all three
orientation rows; Z is absent from the IK error and Jacobian. Z is allowed to move freely while safety gates
require cube-bottom clearance above the box wall to remain at least `0.015 m`, robot geometry clearance at
least `0.010 m`, and filtered box contact below `0.25 N`. A violation
aborts before another action is applied instead of invoking a position-only recovery motion.

The companion `legacy_gripper_anchor_safe_xyz_tilt_align_then_lower` control includes XYZ in the high-align
task while retaining two world-frame orientation rows and leaving world yaw free. It therefore also presents
five task rows to the five arm joints, holds Z at the height measured on align entry, and avoids both the
three-row position-only posture freedom and the six-row full-pose over-constraint. It uses the same physical
clearance and contact gates as the no-Z control.

The independent `jaw_frame_xyz_tilt` expert applies that five-row idea to the entire episode rather than only
high alignment. Every control step recomputes the live gripper-to-jaw transform, makes the jaw detection frame
the IK body offset, and solves jaw XYZ plus world roll/pitch while leaving world yaw free. Its Cartesian
keyframes are expressed directly in jaw coordinates: the grasp point is the cube center and transfer/alignment
use the audited `target_box_floor` center, so neither historical gripper XY compensation is used. The initial
jaw tilt is retained as the two orientation targets. A persistent `0.01 rad` per-application joint-target slew
limit bounds command changes. During `align_over_box`, all five errors must remain below the configured
position (`0.006 m`) and tilt (`0.05 rad`) thresholds for ten steps before descent begins.

Run the new expert from a fresh server terminal with the complete environment contract:

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
export JAW_XYZ_TILT_LOG="$LEISAAC_BASE/results/leisaac/jaw-frame-xyz-tilt-seed42.log"
export JAW_XYZ_TILT_RECORD_DIR="$LEISAAC_BASE/results/leisaac/jaw-frame-xyz-tilt-recordings"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 240s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_smoke.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert jaw_frame_xyz_tilt \
  --seed 42 \
  --record_dir "$JAW_XYZ_TILT_RECORD_DIR" \
  --record_every 4 \
  --record_fps 15 \
  2>&1 | tee "$JAW_XYZ_TILT_LOG"

jaw_xyz_tilt_status=${PIPESTATUS[0]}
echo "jaw_frame_xyz_tilt_exit=$jaw_xyz_tilt_status"

grep -nE \
  'expert_phase|expert_state|expert_ik_runtime_mode|expert_direct_jaw_control|expert_jaw_five_dimensional_task|completed_steps|cube_final|cube_offset|box_aligned_before_release|servo_timeout_phase|expert_success|RED_CUBE_TO_BOX_EXPERT_SMOKE|Traceback|RuntimeError' \
  "$JAW_XYZ_TILT_LOG" |
tail -n 360
```

The separate `legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower` control tests whether shoulder-pan
motion stops because the remaining task leaves no base objective. At align entry it computes the signed
world-XY bearing from the robot root through the live jaw and to the box center. In the audited SO-101 joint
convention, that bearing correction is subtracted from the entry `shoulder_pan` angle, capped at `0.35 rad`,
and clamped inside the soft joint limits. High align then solves exactly five rows: XYZ, base-frame pitch, and
the persistent shoulder-pan joint error. Smoke logs report the target/actual pan angle, requested joint delta,
five-element task error, and Jacobian singular values every 25 control steps.

The independent `legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower` variant keeps the same pickup,
explicit shoulder-pan target, physical safety gates, and legacy descent. It removes hard pitch from high align,
leaving four primary rows: XYZ plus shoulder pan. A damped pseudoinverse solves those rows, while the remaining
one-dimensional nullspace moves the arm away from its normalized soft joint limits. The secondary posture term
uses gain `0.08`, is capped at `0.03 rad` per application, and does not directly modify the primary shoulder-pan
joint. Smoke diagnostics separately report the primary and projected nullspace joint deltas so the effect can be
distinguished from the earlier five-row controller.

The independent `legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower` variant keeps that
controller and all earlier pickup, transport, safety, and descent behavior, but changes the high-align control
frame. When the closed grasp first reaches `align_over_box`, it captures the rigid transform from the configured
gripper body to the live jaw detection frame and installs that transform as the IK action's body offset. The four
primary rows therefore solve the desired jaw XYZ and shoulder-pan objective directly, including the jaw lever arm
in the translational Jacobian, instead of converting each jaw target into a compensated gripper-position target.
The offset is reset to identity outside high align. This is intentionally a separate expert so results remain
comparable with the gripper-frame nullspace variant.

Direct-jaw seed-42 diagnostics showed a well-conditioned four-row Jacobian but unbounded primary IK increments of
roughly `0.15-0.24 rad` per application. The arm crossed the target and entered a high-speed alignment oscillation;
the existing `0.03 rad` cap applied only to the nullspace posture term. During the direct-jaw custom high-align
solve, the variant therefore scales the combined primary-plus-nullspace increment uniformly whenever its largest
joint component exceeds `0.02 rad`. Uniform scaling preserves the requested joint-space direction and applies only
to this new expert; older experts retain their original behavior. Smoke logs expose both the unlimited delta and
the actually applied limited delta.

The separate `legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower` experiment also
removes the legacy transfer calibration from the control path. At the time of that audit, `target_box_floor` was
centered at the former `(0.52, -0.36161)` location; the apparent corner target came from adding the historical gripper-frame
`_PLACE_XY_OFFSET=(-0.084, 0.003)`. Because the gripper-to-jaw world offset changes with arm posture, that fixed
single-trial compensation placed the live jaw near `(0.568, -0.407)` at align entry. The new transfer captures its
actual starting jaw, follows a cubic smoothstep trajectory to the floor-center XY at a safe height, and holds the
last 40 of 160 steps to reduce entry velocity. It then uses direct-jaw alignment with a persistent joint-target
slew limit of `0.01 rad/application`. Unlike limiting the target to remain close to the moving actual joint, the
persistent command can remain behind an overshooting joint and therefore provide braking effort.
Diagnostics report both that persistent `joint_position_target` and the command-to-command
`joint_target_slew_step`; the latter, rather than target-to-actual error, is bounded to `0.01 rad`.

That direct-jaw expert also adds two scene-only markers. A red sphere marks the live jaw point controlled by IK;
a cyan sphere marks the same world X/Y projected vertically onto `TABLE_SURFACE_Z`. The markers do not alter the
observation, action, safety gates, or success predicate. Smoke logs emit the desired jaw, captured offset, live
controlled point, and table projection every 25 steps for numerical comparison with the rendered view.

The separate `legacy_gripper_anchor_relaxed_ik` variant keeps that same jaw target and safety gate but uses the
phase-aware IK action only for constraint weighting. All phases through lowering retain the exact pose solve.
During the first 120 alignment steps it sets orientation weight to `0.1`; if alignment still has not converged,
the remaining alignment steps use position-only IK. The weight that achieved stable alignment remains active
through release and retraction. A detected grasp loss ends the episode before another action is applied.

The `legacy_gripper_anchor_planar_ik` variant instead removes only the Z translation row during final
alignment, producing a five-row task from jaw X/Y plus three orientation rows for the five-joint arm. Cube Z
is a safety inequality rather than an IK target: the cube bottom must remain at least `0.010 m` above the box
wall, and horizontal alignment resumes only after `0.015 m` clearance. Candidate jaw X/Y motion is capped at
`0.002 m` per control step. Conservative bounding spheres guard the lower arm, wrist, gripper, and jaw, while
four filtered contact sensors abort on measured robot-to-box contact above `0.25 N`. An unsafe state receives
only a bounded upward recovery action. Stable alignment captures the measured gripper pose; release holds that
zero-error pose instead of returning to an old fixed Z target.

```bash
# Reproduce the fixed-offset baseline.
--expert legacy

# Keep legacy pickup/transport and align placement through the jaw frame.
--expert legacy_gripper_anchor

# Preserve legacy pose IK, but align above the box before descending vertically.
--expert legacy_gripper_anchor_align_then_lower

# Use position-only XYZ IK for high alignment, then restore legacy pose IK for descent.
--expert legacy_gripper_anchor_position_align_then_lower

# Prefer shoulder-pan motion among the redundant position-only high-alignment solutions.
--expert legacy_gripper_anchor_weighted_position_align_then_lower

# Solve jaw XY plus orientation at high align; observe Z only through safety gates.
--expert legacy_gripper_anchor_safe_planar_align_then_lower

# Control XYZ plus world roll/pitch at high align; leave world yaw free.
--expert legacy_gripper_anchor_safe_xyz_tilt_align_then_lower

# Control XYZ and pitch while continuously driving shoulder_pan toward a bearing-derived target.
--expert legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower

# Control XYZ and shoulder_pan while using the remaining nullspace to avoid joint limits.
--expert legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower

# Directly control the closed-jaw frame during high align and draw jaw/table-projection markers.
--expert legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower

# Smoothly transfer the jaw to the true floor center, then direct-align with persistent target slew limiting.
--expert legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower

# Add staged weak-orientation then position-only IK during final jaw alignment.
--expert legacy_gripper_anchor_relaxed_ik

# Keep Z as a collision-gated inequality and solve jaw XY plus orientation.
--expert legacy_gripper_anchor_planar_ik

# Test the jaw-feedback upgrade.
--expert adaptive

# Test the bounded Cartesian P-servo upgrade.
--expert servo
```

Use separate process invocations with the same `--seed` when comparing them. That restarts Isaac's random
sequence so both experts receive the same randomized episode inputs.

## Randomized expert batch preflight

The inherited LiftCube reset events already randomize cube X/Y by `+/-0.075 m`, cube yaw by `+/-30 degrees`,
and the front-camera pose by `+/-0.005 m` plus small rotations. Before recording, run several resets in one
Isaac process and require the scripted expert to succeed across those existing domain-randomized states. The
place control offset is separate from the grasp offset so the held cube is released above the tray center.

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_ROOT=/home/data/xiaoqinchuan/projects/leisaac
export LEISAAC_ENV=/home/data/xiaoqinchuan/envs/leisaac-so101
export LEISAAC_ASSETS_ROOT=/home/data/xiaoqinchuan/assets/leisaac-v0.4.0
export ISAACSIM_PORTABLE_ROOT=/home/data/xiaoqinchuan/cache/isaacsim-portable
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD="$LEISAAC_ENV/lib/libstdc++.so.6"
export RED_CUBE_TO_BOX_BATCH_LOG="$LEISAAC_BASE/results/leisaac/red-cube-to-box-expert-batch.log"

cd "$OPENPI_ROOT"
mkdir -p "$LEISAAC_BASE/results/leisaac"
timeout --signal=KILL 900s \
  "$LEISAAC_ENV/bin/python" \
  examples/so101/red_cube_to_box_expert_batch.py \
  --headless \
  --enable_cameras \
  --device cuda:6 \
  --assets_root "$LEISAAC_ASSETS_ROOT" \
  --expert servo \
  --episodes 10 \
  --minimum_success_rate 0.9 \
  --seed 42 \
  2>&1 | tee "$RED_CUBE_TO_BOX_BATCH_LOG"

batch_status=${PIPESTATUS[0]}
echo "red_cube_to_box_expert_batch_exit=$batch_status"

grep -nE \
  'RED_CUBE_TO_BOX_BATCH|RED_CUBE_TO_BOX_EXPERT_BATCH|expert_variant|expert_orientation_policy|expert_ik_action_class|servo_parameters|cube_randomization|camera_randomization|episode:|completed_episodes|grasped_episodes|grasped_at_lift_episodes|grasped_at_transfer_episodes|retried_episodes|box_aligned_episodes|successful_episodes|failed_episodes|non_finite_episodes|reset_episodes|servo_timeout_episodes|servo_abort_episodes|success_rate|initial_cube_|final_offset_|Traceback|RuntimeError' \
  "$RED_CUBE_TO_BOX_BATCH_LOG" |
tail -n 260
```

This is deliberately non-recording. A passing preflight reports no numerical failures or unexpected resets,
at least `9/10` successful episodes, and `RED_CUBE_TO_BOX_EXPERT_BATCH_OK`. Only then should the same loop be
connected to the native streaming HDF5 recorder for large-scale generation.

## Data contract

Do not install LeRobot into the Isaac Sim environment. LeRobot 0.4.2 requires `packaging>=24.2`, while the
tested Isaac Sim 5.1 environment requires `packaging==23.0`. Record native LeIsaac HDF5 in the stable simulator
environment, then run `convert_leisaac_hdf5_to_lerobot.py` in OpenPI's isolated `uv` environment.

Native LeIsaac HDF5 stores both state and action in USD radians. The converter reproduces LeIsaac v0.4.0's
official `action_align=True` mapping so both resulting LeRobot fields use the same SO-101 motor coordinates.
Copying the HDF5 arrays directly would mix coordinate contracts and train an invalid policy.

The OpenPI data config consumes these LeRobot fields:

| LeRobot field                | OpenPI field     | Shape         | Meaning                       |
| ---------------------------- | ---------------- | ------------- | ----------------------------- |
| `observation.images.front` | `images/front` | `H x W x 3` | Required RGB image            |
| `observation.state`        | `state`        | `6`         | Five arm joints plus gripper  |
| `action`                   | `actions`      | `T x 6`     | Absolute SO-101 motor targets |
| `task`                     | `prompt`       | text          | Language instruction          |

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
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_HDF5_FILE="$LEISAAC_BASE/datasets/leisaac/so101_liftcube_smoke_20260808-123945.hdf5"

cd "$OPENPI_ROOT"
uv run examples/so101/convert_leisaac_hdf5_to_lerobot.py \
  --input-path "$LEISAAC_HDF5_FILE" \
  --fps 60 \
  --dry-run
```

After the preflight ranges have been reviewed, create a local LeRobot dataset. The converter refuses to
overwrite an existing repository ID and does not upload anything unless `--push-to-hub` is explicitly passed:

```bash
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export LEISAAC_BASE=/home/data/xiaoqinchuan
export LEISAAC_HDF5_FILE="$LEISAAC_BASE/datasets/leisaac/so101_liftcube_smoke_20260808-123945.hdf5"

cd "$OPENPI_ROOT"
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
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export HF_LEROBOT_HOME=/home/data/xiaoqinchuan/datasets/lerobot
export OPENPI_SO101_LIFTCUBE_REPO_ID=local/leisaac-so101-liftcube-smoke-20260808

cd "$OPENPI_ROOT"
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
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export HF_LEROBOT_HOME=/home/data/xiaoqinchuan/datasets/lerobot
export OPENPI_SO101_LIFTCUBE_REPO_ID=local/leisaac-so101-liftcube-smoke-20260808

cd "$OPENPI_ROOT"
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
export OPENPI_ROOT=/home/data/xiaoqinchuan/projects/openpi
export HF_LEROBOT_HOME=/home/data/xiaoqinchuan/datasets/lerobot
export OPENPI_SO101_LIFTCUBE_REPO_ID=local/leisaac-so101-liftcube-smoke-20260808

cd "$OPENPI_ROOT"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_lora_so101_liftcube \
  --policy.dir=checkpoints/pi05_lora_so101_liftcube/liftcube_lora/30000 \
  --port=18000
```

Point LeIsaac's `OpenPIServicePolicyClient` to the same host and port. Keep the policy server and Isaac Sim in
separate processes and environments; only the serialized observation/action contract crosses the WebSocket.
