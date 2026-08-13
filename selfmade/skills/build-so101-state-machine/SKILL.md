---
name: build-so101-state-machine
description: Build, extend, diagnose, and validate a scripted SO-101 pick-transport-place state machine in OpenPI with LeIsaac/IsaacLab. Use for new RedCubeToBox-style experts, phase design, wrist/gripper control-point selection, phase-aware differential IK, grasp and release gates, target-offset geometry, state-machine telemetry, Windows-native scene audits, and single-environment dynamic smoke validation.
---

# Build an SO-101 State Machine

Build from measured scene geometry and explicit control contracts. Treat a successful trajectory as the result of path
design, IK behavior, actuator tracking, grasp contact, and state gates working together.

## Read the relevant references

- Read [references/control-design.md](references/control-design.md) before designing phases, targets, IK modes, or
  completion gates.
- Read [references/validation.md](references/validation.md) before running Isaac Sim, accepting a result, or preparing
  the user-facing run and log-extraction commands.

## Establish scope

1. Inspect `env_cfg.py`, `mdp.py`, the configured action term, smoke runner, and registry tests.
2. Create a new expert instead of overwriting an existing expert unless the user explicitly requests replacement.
3. Keep failed or comparison experts runnable under their existing `--expert` names.
4. State the controlled physical object for every phase: wrist, gripper, jaw detection frame, or cube.
5. Separate measured scene constants from tunable controller parameters. Do not copy seed-specific world coordinates
   when the target can be read from the scene.

## Audit the scene first

Run or repair the scene audit before writing motion logic. Confirm:

- robot root and environment origins;
- arm and gripper joint names, indices, limits, and direction conventions;
- wrist and gripper body names;
- `ee_frame.target[0]` and `target[1]` semantics;
- whether each frame is an IK origin, rigid-body origin, contact point, distal detection point, or grasp-gap center;
- cube center and half-height;
- target-box floor center, wall top, and usable inner bounds;
- action shape and quaternion convention;
- success predicate and reset behavior.

Stop and correct frame semantics if a named point is ambiguous. Never infer the end effector from the last body index.
Before controller tuning, verify that randomized cube poses remain outside box geometry and inside the arm's measured
pickup workspace. Preserve useful yaw variation instead of deleting it to hide a grasp-orientation defect.

## Define the state-machine contract

Implement at least:

- `setup(env)` to resolve bodies, joints, and the phase-aware action term;
- `reset()` to clear every phase-local target, counter, latch, and accumulated reference;
- `get_action(env)` to initialize the active phase once, configure its IK mode, update diagnostics, and return one
  action;
- `advance()` to transition only from measured completion or bounded failure;
- `check_success(env)` and `is_episode_done`;
- `phase_name`, failure reason fields, and a concise `servo_parameters` dictionary.

Use explicit minimum steps, maximum steps, tolerance, and continuous stable-step counts for motion phases. Use fixed
duration only for genuinely time-based interpolation, never as proof of physical arrival.

## Build phases in risk order

Implement and validate incrementally:

1. `approach_cube`: move above the measured cube with the gripper open.
2. `descend_to_cube`: preserve the verified pickup orientation and approach the grasp point.
3. optional axis alignment and recenter: align the closing axis to object geometry, then rotate the calibrated
   gripper-to-grasp offset with the measured aligned axis.
4. `close_gripper`: close with feedback; require contact-compatible aperture and a stable window.
5. vertical lift: gain wall/table clearance before horizontal motion.
6. safe horizontal transport: use a segmented path whose geometry is explainable from robot root and box position.
7. radial/placement alignment: compensate the live held-object offset rather than aligning the gripper origin.
8. vertical lower: freeze the achieved handoff XY and judge completion primarily from actual Z.
9. `release`: hold the actual release pose and open only after placement gates pass.
10. vertical retract and settle.

After each phase becomes dynamically valid, add only the next phase. Preserve the last working version as a comparison
expert or commit.

## Configure IK per phase

- Use the verified pickup pose for approach, descend, and initial grasp.
- Prefer wrist XYZ position-only plus a soft entry-joint posture in the XYZ nullspace for high-clearance lift and arc
  motion.
- Use gripper XYZ position-only plus a soft handoff posture for final radial placement and lower.
- Rebase the control body, posture snapshot, and accumulated joint target whenever phase semantics or controlled body
  changes.
- Accumulate limited differential-IK joint corrections in a persistent joint reference. Do not rebuild every command as
  only `q_actual + delta_q` when persistent braking is required.
- Keep accumulation bounded per control step. Do not freeze correction merely because actuator tracking error is large.

Do not use full world 6D pose as a default transport constraint for the five arm joints. Do not use unconstrained
position-only without soft posture stabilization.

## Preserve pickup geometry through axis alignment

Do not treat a jaw detection point as the center of the grasp gap. In the bundled LeIsaac SO-101 scene,
`ee_frame.target[1]` is an offset point at the distal end of the moving jaw. Driving it to the cube center shifts the
whole grasp to one side.

If an unrotated pickup was calibrated as an offset along gripper local axes, rotate that calibration after wrist-roll
alignment. For a calibrated distance `d` along gripper local `+X`:

```text
closing_axis_xy = normalize(world_direction(gripper_local_+X).xy)
target_gripper_xy = live_cube_xy - d * closing_axis_xy
```

Use the live cube position after alignment because contact or alignment may have moved it. Preserve the verified Z.
Only use a midpoint of fixed- and moving-jaw contact points when both contact-point frames are explicitly defined and
verified; never synthesize it from one distal detection point.

When the object has randomized yaw, align the gripper closing axis before close. Settle the measured Cartesian pickup
position first; a completed time interpolation does not prove arrival. Freeze the other arm joints and give
`wrist_roll` one direct target writer. Choose the nearest reachable signed object X/Y axis, preserve soft limits, and
complete only from measured angular error, joint velocity, and consecutive stable frames. Increase bounded target lead
to improve speed before relaxing measured accuracy.

## Compute placement from held geometry

At the high, grasped handoff:

```text
held_offset_xy = jaw_xy - gripper_xy
desired_gripper_xy = box_center_xy - held_offset_xy
```

Use the live held configuration, not an offset measured with the gripper open or in an earlier posture. If the task
defines success on the cube rather than the jaw, log cube/jaw/gripper geometry and justify the chosen proxy.

Lower and release from the actual achieved XY. Avoid adding a new low-height horizontal correction near box walls.

## Add physical gates and aborts

Require:

- grasp confirmation before lift;
- continuous jaw-to-cube or task-specific grasp monitoring until release;
- actual safe Z before horizontal motion;
- reference completion plus actual position/bearing convergence for path handoffs;
- multiple stable frames rather than a single-frame hit;
- lower completion based on actual safe release Z;
- release permission only after all upstream placement gates pass;
- explicit timeouts and reasons for grasp loss, worsening retreat error, overshoot, IK non-convergence, and blocked
  release.

Never continue issuing transport actions after confirmed grasp loss.

Keep close feedback concepts separate: contact geometry may be sticky once established, task `pick_cube` feedback must
be debounced, and gripper aperture still needs its own stable multi-frame window. A longer bounded timeout can absorb
contact settling; it must not replace those gates.

## Instrument before tuning

Print one field per line and make each field grep-friendly. Distinguish:

```text
motion_target_w          final phase endpoint
current_target_w         reference sent this frame
actual_focus_w           measured controlled point
final_target_error       final endpoint minus actual
reference_tracking_error current reference minus actual
```

Do not call final endpoint error "tracking lag." For retreat and transfer, print wrist fields; for pickup, close, lower,
release, and grasp monitoring, print gripper/jaw/cube fields. Include per-joint target, actual, target-minus-actual,
velocity, IK delta, singular values, accumulated step, stable streak, and abort reason when diagnosing motion direction.

## Validate in layers

Follow this order:

1. AST/unit/registry tests.
2. Windows-native scene audit.
3. One environment, one seed, headless, performance rendering, `cuda:0`.
4. Repeat the successful smoke in a fresh process.
5. Add recording only after the non-recorded run succeeds.
6. Test additional seeds in independent processes before claiming robustness.

Treat semantic sentinels and final physical state as authoritative, not only the process exit code. Do not weaken physics
or success logic to hide OOM or controller failure.

## Finish the change

1. Update the project change record with evidence, not guesses.
2. Run `git diff --check` and inspect the exact staged file list.
3. Preserve unrelated worktree modifications.
4. Commit and push only after relevant static checks and dynamic smoke pass.
5. Always provide the complete Windows-native run command and a terminal-only `Select-String` extraction command. Do
   not create a second diagnostics output file.
