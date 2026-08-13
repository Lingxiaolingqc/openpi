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

## Placement geometry

Align the held object proxy, not the gripper origin:

```text
jaw_from_gripper_xy = live_jaw_xy - live_gripper_xy
placement_gripper_xy = box_center_xy - jaw_from_gripper_xy
```

Capture this offset after grasp at the high placement handoff. Reusing open-gripper or early-pose offsets introduces a
systematic placement error.

Perform final descent at the actual handoff XY. Release in place. Retract vertically from the actual release XY.

## Avoided designs

Do not repeat these without an explicit new hypothesis and instrumentation:

- full 6D world-pose enforcement throughout transport;
- pure position-only with no soft posture stabilization;
- hard shoulder-pan plus XYZ/orientation tasks that consume all five joints;
- releasing retreat Y and allowing a large sideways IK branch;
- simultaneous low-height XY realignment and lowering near box walls;
- aligning the gripper origin directly to the box center;
- single-frame phase completion;
- diagnosing tracking from final endpoint error;
- continuing after grasp loss;
- freezing accumulated correction when tracking error becomes large.
