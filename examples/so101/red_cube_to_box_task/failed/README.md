# Archived RedCubeToBox state machines

This package keeps unsuccessful, superseded, and unverified experiment variants available for comparison. Most moved
modules retain their command-line expert names through imports from `red_cube_to_box_task.failed`. Explicitly retired
ablations may remain source-only when a verified successor makes their active smoke and batch entries unnecessary.

The active task directory retains only the verified `legacy`, `autogen_polar_retreat_transport`, and
`autogen_reference_axis_align_slow_grasp` experts plus runtime support. The successful polar expert uses
`polar_base_state_machine.py`; the archived independent expert is a compatibility subclass of that base so the old
`--expert autogen_independent_retreat_transport` entry remains available without making polar depend on this failed
package.

## Archived groups

- Early Autogen comparisons: `autogen_retreat_transport`, `autogen_independent_retreat_transport`.
- Retired Autogen reference ablation: `autogen_reference_slow_grasp`. Its verified successor inherits
  `autogen_reference_state_machine.py` directly and preserves the same 240-step close.
- Adaptive/servo comparisons: `adaptive`, `servo`, `weighted_servo`, `legacy_weighted_servo`,
  `legacy_position_servo`, `legacy_pd_position_servo`, `legacy_trajectory_pd_servo`.
- Dynamic offset comparisons: `legacy_dynamic_grasp_offset`,
  `legacy_dynamic_grasp_offset_residual_corrected`.
- Gripper-anchor and IK comparisons: all `legacy_gripper_anchor*` variants.
- Jaw-frame comparison: `jaw_frame_xyz_tilt`.

The detailed failure evidence and the route leading to the successful polar implementation are recorded in
`selfmade/problem_solving.md` and `examples/so101/0.originalCodeChanges.md`.
