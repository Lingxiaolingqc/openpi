# SO-101 control design reference

## Contents

- Control layers
- Recommended phase architecture
- IK and joint-target rules
- Completion and safety rules
- Placement geometry
- Avoided designs

## Control layers

Keep four layers conceptually separate:

```text
path reference r(k)
  -> Cartesian error from measured control point
  -> differential IK delta_q
  -> persistent joint target and physical actuator
```

A path may be time-parameterized while IK is closed-loop. A closed-loop IK may still lose correction memory if each
joint command is rebuilt from the live joint position. Diagnose each layer independently.

For a linear path:

```text
current_target = start + progress * (final_target - start)
reference_tracking_error = current_target - actual
final_target_error = final_target - actual
```

Use the first error to diagnose tracking and the second to decide final completion.

## Recommended phase architecture

Use this default pick-place sequence:

```text
approach
-> descend
-> feedback close
-> vertical wrist lift
-> root-relative wrist retreat
-> root-centered wrist arc
-> box-bearing gripper radial approach
-> vertical gripper lower
-> release at actual pose
-> vertical retract
-> settle
```

The arc is a Cartesian reference, not a precomputed joint trajectory:

```text
radius = norm(start_xy - root_xy)
start_bearing = bearing(start_xy - root_xy)
target_bearing = bearing(box_xy - root_xy)
bearing(k) = interpolate_shortest_angle(start_bearing, target_bearing)
target_xy(k) = root_xy + radius * [sin(bearing(k)), cos(bearing(k))]
```

Use a smoothstep/smootherstep time law and keep Z at the actual safe handoff height.

For randomized object yaw, insert a measured pickup settle, closing-axis alignment, and recenter before feedback close.
Do not start alignment merely because descend interpolation time elapsed. Prefer one direct `wrist_roll` writer while
holding the other arm joints; do not let Cartesian IK and a second joint writer compete for the same joint.

## IK and joint-target rules

SO-101 has five arm joints. Treat full XYZ plus three-axis world orientation as overconstrained unless a measured test
proves the phase can satisfy it.

Recommended modes:

| Phase | Control body | Hard task | Soft task |
| --- | --- | --- | --- |
| approach/descend | gripper | verified pickup pose | none |
| lift/retreat/arc | wrist | XYZ | entry five-joint posture in XYZ nullspace |
| radial/lower | gripper | XYZ | phase-entry five-joint posture in XYZ nullspace |
| release | gripper | hold measured pose | none |

At every body or phase-mode handoff:

1. capture the actual controlled point;
2. capture the actual controlled arm joints for soft posture;
3. rebase the Cartesian reference;
4. reset the accumulated joint target to actual joints;
5. start the next bounded reference without a target jump.

Prefer:

```text
q_reference(k+1) = q_reference(k) + limited_delta_q(k)
q_target(k+1) = q_reference(k+1)
```

over a memoryless command when persistent correction is required:

```text
q_target(k+1) = q_actual(k) + limited_delta_q(k)
```

The memoryless form is still feedback control, but it discards the prior target. With actuator lag, the moving actual
joint anchor can carry the absolute target forward even after IK requests a small reverse delta.

Limit each accumulated step. Rebase on semantic handoffs. Monitor target-minus-actual rather than freezing the
accumulator at an arbitrary tracking gap.

## Completion and safety rules

Use phase-specific measured criteria:

- close: minimum duration plus contact-compatible aperture and stable consecutive frames;
- lift: reference finished plus actual controlled-point Z in the safe band;
- retreat: actual radius, Z, bearing, reference finished, and stable streak;
- arc/radial: final position, bearing where relevant, reference finished, and stable streak;
- lower/retract: actual Z and stable streak;
- release: placement gate already true, then hold actual position while opening.

Monitor grasp geometry continuously after confirmation. Abort before producing the next unsafe motion when the grasp is
lost. Record timeout and safety reasons separately.

For close, distinguish three signals:

1. contact-compatible jaw/object geometry, which may be latched once established;
2. debounced task grasp feedback, which must not be accepted from one frame;
3. a stable gripper-aperture window, which remains required before lift.

Log all three when close times out. Do not fix a concentrated close failure by editing transport first.

## Placement geometry

Align the held object proxy, not the gripper origin:

```text
jaw_from_gripper_xy = live_jaw_xy - live_gripper_xy
placement_gripper_xy = box_center_xy - jaw_from_gripper_xy
```

Capture this offset after grasp at the high placement handoff. Reusing open-gripper or early-pose offsets introduces a
systematic placement error.

Perform final descent at the actual handoff XY. Release in place. Retract vertically from the actual release XY.

## Pickup frame semantics and rotated calibration

Keep these quantities distinct:

| Quantity | Meaning | Safe use |
| --- | --- | --- |
| gripper frame | IK-controlled rigid-body origin | Cartesian pose target |
| jaw detection frame | distal point attached to the moving jaw | task predicate or diagnostic only |
| grasp-gap center | midpoint between verified opposing contact surfaces | object centering, if both surfaces are known |
| calibrated grasp offset | empirically verified vector from gripper origin to desired grasp center | pickup target after rotating into world coordinates |

This correction is invalid when `jaw_detection` is a moving-jaw endpoint:

```text
target_gripper_xy = actual_gripper_xy + (cube_xy - actual_jaw_xy)
```

It makes the moving jaw endpoint chase the cube center and displaces the actual opening. Instead, express a successful
calibration in gripper-local coordinates and transform it with the aligned gripper orientation. For a `20 mm` local
`+X` calibration:

```text
closing_axis_w = rotate(gripper_quaternion_w, [1, 0, 0])
closing_axis_xy = normalize(closing_axis_w.xy)
target_gripper_xy = live_cube_xy - 0.020 * closing_axis_xy
```

Verify that the zero-yaw case reduces exactly to the old successful target. Log the cube and gripper before/after
recenter, and assess whether close displaces the cube. Use jaw distance only as the task-defined grasp proxy, not as
proof that the grasp opening is centered.

## Avoided designs

Do not repeat these without an explicit new hypothesis and instrumentation:

- full 6D world-pose enforcement throughout transport;
- pure position-only with no soft posture stabilization;
- hard shoulder-pan plus XYZ/orientation tasks that consume all five joints;
- releasing retreat Y and allowing a large sideways IK branch;
- simultaneous low-height XY realignment and lowering near box walls;
- aligning the gripper origin directly to the box center;
- driving a single moving-jaw distal detection point to the cube center;
- retaining a world-fixed pickup XY offset after rotating the gripper closing axis;
- starting wrist-roll alignment from a lagging time-driven descend endpoint;
- loosening measured alignment accuracy merely to reduce alignment duration;
- single-frame phase completion;
- diagnosing tracking from final endpoint error;
- continuing after grasp loss;
- freezing accumulated correction when tracking error becomes large.
