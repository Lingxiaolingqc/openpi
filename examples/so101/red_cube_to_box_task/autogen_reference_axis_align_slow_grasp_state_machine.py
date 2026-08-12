"""Slow Autogen grasp preceded by direct wrist-roll cube-edge alignment."""

from __future__ import annotations

import math

from isaaclab.utils.math import quat_apply
import torch

from .autogen_reference_slow_grasp_state_machine import RedCubeToBoxAutogenReferenceSlowGraspStateMachine
from .cube_axis_alignment import measure_cube_axis_alignment
from .cube_axis_alignment import select_nearest_cube_axis_alignment


class RedCubeToBoxAutogenReferenceAxisAlignSlowGraspStateMachine(RedCubeToBoxAutogenReferenceSlowGraspStateMachine):
    """Freeze four arm joints and align cube edges with wrist_roll before closing."""

    AXIS_ALIGNMENT_TOLERANCE = math.radians(5.0)
    AXIS_ALIGNMENT_STABLE_STEPS = 10
    AXIS_ALIGNMENT_HOLD_SETTLE_STEPS = 8
    AXIS_ALIGNMENT_TIMEOUT_STEPS = 300
    AXIS_ALIGNMENT_RAY_MISS_LIMIT = 3
    AXIS_ALIGNMENT_WRIST_ROLL_VELOCITY_TOLERANCE = 0.02
    AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE = 0.03
    AXIS_ALIGNMENT_DIRECT_TARGET_TOLERANCE = 0.02
    AXIS_ALIGNMENT_FROZEN_JOINT_TOLERANCE = 0.02
    AXIS_ALIGNMENT_WRIST_POSITION_TOLERANCE = 0.01
    AXIS_ALIGNMENT_TARGET_KP = 0.5
    AXIS_ALIGNMENT_MAX_TARGET_STEP = math.radians(1.0)
    AXIS_ALIGNMENT_MAX_TARGET_LEAD = math.radians(2.0)
    AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN = 0.02

    RAY_HIT_TRACKING_STABLE_STEPS = 8
    RAY_HIT_TRACKING_TIMEOUT_STEPS = 120
    RAY_HIT_TRACKING_POSITION_TOLERANCE = 0.005

    IK_HANDOFF_STABLE_STEPS = 8
    IK_HANDOFF_TIMEOUT_STEPS = 120
    IK_HANDOFF_WRIST_POSITION_TOLERANCE = 0.005

    def reset(self) -> None:
        arm_action_term = getattr(self, "_arm_action_term", None)
        if arm_action_term is not None:
            arm_action_term.clear_direct_joint_position_target()
        super().reset()
        self._axis_alignment_complete = False
        self._axis_alignment_streak = 0
        self._axis_alignment_final_gate_streak = 0
        self._axis_alignment_hold_settle_streak = 0
        self._axis_alignment_ray_miss_streak = 0
        self._axis_alignment_selected_index: torch.Tensor | None = None
        self._axis_alignment_selected_sign: torch.Tensor | None = None
        self._axis_alignment_selected_cube_axis: tuple[str, ...] | None = None
        self._axis_alignment_closing_axis_w: torch.Tensor | None = None
        self._axis_alignment_cube_x_axis_w: torch.Tensor | None = None
        self._axis_alignment_cube_y_axis_w: torch.Tensor | None = None
        self._axis_alignment_desired_axis_w: torch.Tensor | None = None
        self._axis_alignment_signed_error: torch.Tensor | None = None
        self._axis_alignment_error: torch.Tensor | None = None
        self._axis_alignment_wrist_roll_target: torch.Tensor | None = None
        self._axis_alignment_wrist_roll_position: torch.Tensor | None = None
        self._axis_alignment_wrist_roll_velocity: torch.Tensor | None = None
        self._axis_alignment_max_arm_joint_velocity: torch.Tensor | None = None
        self._axis_alignment_wrist_position_error: torch.Tensor | None = None
        self._axis_alignment_direct_target_error: torch.Tensor | None = None
        self._axis_alignment_frozen_joint_drift: torch.Tensor | None = None
        self._axis_alignment_direct_joint_target: torch.Tensor | None = None
        self._axis_alignment_frozen_joint_reference: torch.Tensor | None = None
        self._axis_alignment_entry_wrist_pos_w: torch.Tensor | None = None
        self._axis_alignment_controlled_joint_indices: tuple[int, ...] | None = None
        self._axis_alignment_frozen_control_indices: tuple[int, ...] | None = None
        self._axis_alignment_wrist_roll_control_index: int | None = None
        self._axis_alignment_direct_hold_active = False
        self._axis_alignment_direct_hold_release_reason: str | None = None
        self._ray_hit_tracking_target_b: torch.Tensor | None = None
        self._ray_hit_tracking_target_w: torch.Tensor | None = None
        self._ray_hit_tracking_entry_wrist_pos_w: torch.Tensor | None = None
        self._ray_hit_tracking_wrist_delta_w: torch.Tensor | None = None
        self._ray_hit_tracking_residual_descent: torch.Tensor | None = None
        self._ray_hit_tracking_streak = 0
        self._ray_hit_tracking_ray_miss_streak = 0
        self._ray_hit_tracking_wrist_position_error: torch.Tensor | None = None
        self._ray_hit_tracking_max_arm_joint_velocity: torch.Tensor | None = None
        self._ik_handoff_streak = 0
        self._ik_handoff_target_b: torch.Tensor | None = None
        self._ik_handoff_wrist_flex_target: torch.Tensor | None = None
        self._ik_handoff_wrist_position_error: torch.Tensor | None = None
        self._ik_handoff_max_arm_joint_velocity: torch.Tensor | None = None

    def _on_grasp_pose_reached(self, env) -> None:
        """Freeze the existing IK target and wait for the real wrist to reach it."""

        self._gripper_command = self.GRIPPER_OPEN_POSITION
        assert self._command_pos_b is not None
        assert self._wrist_body_index is not None
        robot = env.scene["robot"]
        self._ray_hit_tracking_target_b = self._command_pos_b.detach().clone()
        self._ray_hit_tracking_target_w = self._base_position_to_world(robot, self._ray_hit_tracking_target_b).detach()
        self._ray_hit_tracking_entry_wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index].detach().clone()
        self._ray_hit_tracking_wrist_delta_w = torch.zeros_like(self._ray_hit_tracking_entry_wrist_pos_w)
        self._ray_hit_tracking_residual_descent = torch.zeros_like(self._ray_hit_tracking_entry_wrist_pos_w[:, 2])
        self._ray_hit_tracking_streak = 0
        self._ray_hit_tracking_ray_miss_streak = 0
        self._transition("ray_hit_tracking_settle")

    def _capture_direct_joint_hold(self, env) -> None:
        assert self._arm_action_term is not None
        robot = env.scene["robot"]
        robot_joint_names = list(robot.data.joint_names)
        controlled_joint_names = self._arm_action_term.controlled_joint_names
        if "wrist_roll" not in controlled_joint_names:
            raise RuntimeError(f"wrist_roll is not controlled by the arm action: {controlled_joint_names}")
        controlled_joint_indices = tuple(robot_joint_names.index(name) for name in controlled_joint_names)
        wrist_roll_control_index = controlled_joint_names.index("wrist_roll")
        frozen_control_indices = tuple(
            index for index, name in enumerate(controlled_joint_names) if name != "wrist_roll"
        )
        direct_target = robot.data.joint_pos[:, list(controlled_joint_indices)].detach().clone()

        self._axis_alignment_controlled_joint_indices = controlled_joint_indices
        self._axis_alignment_wrist_roll_control_index = wrist_roll_control_index
        self._axis_alignment_frozen_control_indices = frozen_control_indices
        self._axis_alignment_entry_wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index].detach().clone()
        self._axis_alignment_direct_hold_active = True
        self._arm_action_term.set_direct_joint_position_target(direct_target)
        applied_target = self._arm_action_term.direct_joint_position_target
        if applied_target is None:
            raise RuntimeError("The arm action did not retain the direct joint-position target")
        # The action term clamps against the articulation's soft limits.  Keep
        # the controller shadow and diagnostics identical to what is applied.
        self._axis_alignment_direct_joint_target = applied_target.detach().clone()
        self._axis_alignment_frozen_joint_reference = applied_target.detach().clone()
        self._axis_alignment_wrist_roll_target = applied_target[:, wrist_roll_control_index].detach().clone()

    def _update_state(self, env) -> None:
        if self._state == "ray_hit_tracking_settle":
            self._update_ray_hit_tracking_settle(env)
            return
        if self._state == "pregrasp_axis_align":
            self._update_pregrasp_axis_alignment(env)
            return
        if self._state == "ik_handoff":
            self._update_ik_handoff(env)
            return
        super()._update_state(env)

    def _update_ray_hit_tracking_settle(self, env) -> None:
        """Let IK finish its already-issued descent before direct alignment."""

        assert self._ray_hit_tracking_target_b is not None
        assert self._arm_action_term is not None
        self._gripper_command = self.GRIPPER_OPEN_POSITION
        self._command_pos_b = self._ray_hit_tracking_target_b.detach().clone()

        self.green_ray_hit = self._green_ray_intersects_cube(env)
        if self.green_ray_hit:
            self._ray_hit_tracking_ray_miss_streak = 0
        else:
            self._ray_hit_tracking_ray_miss_streak += 1
            if self._ray_hit_tracking_ray_miss_streak >= self.AXIS_ALIGNMENT_RAY_MISS_LIMIT:
                self._fail("ray lost while the wrist tracked the frozen pre-alignment target")
                return

        robot = env.scene["robot"]
        target_pos_w = self._base_position_to_world(robot, self._ray_hit_tracking_target_b)
        assert self._wrist_body_index is not None
        wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
        self._ray_hit_tracking_target_w = target_pos_w.detach().clone()
        self._wrist_position_w = wrist_pos_w.detach().clone()
        assert self._ray_hit_tracking_entry_wrist_pos_w is not None
        self._ray_hit_tracking_wrist_delta_w = (wrist_pos_w - self._ray_hit_tracking_entry_wrist_pos_w).detach()
        self._ray_hit_tracking_residual_descent = torch.clamp(
            self._ray_hit_tracking_entry_wrist_pos_w[:, 2] - wrist_pos_w[:, 2], min=0.0
        ).detach()
        self._ray_hit_tracking_wrist_position_error = torch.linalg.vector_norm(
            wrist_pos_w - target_pos_w, dim=-1
        ).detach()
        robot_joint_names = list(robot.data.joint_names)
        controlled_joint_indices = [
            robot_joint_names.index(name) for name in self._arm_action_term.controlled_joint_names
        ]
        controlled_joint_velocity = torch.abs(robot.data.joint_vel[:, controlled_joint_indices])
        self._ray_hit_tracking_max_arm_joint_velocity = torch.amax(controlled_joint_velocity, dim=-1).detach()
        stable = (
            self.green_ray_hit
            and bool(
                (self._ray_hit_tracking_wrist_position_error <= self.RAY_HIT_TRACKING_POSITION_TOLERANCE).all().item()
            )
            and bool(
                (self._ray_hit_tracking_max_arm_joint_velocity <= self.AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE)
                .all()
                .item()
            )
        )
        self._ray_hit_tracking_streak = self._ray_hit_tracking_streak + 1 if stable else 0
        if self._ray_hit_tracking_streak >= self.RAY_HIT_TRACKING_STABLE_STEPS:
            self._capture_direct_joint_hold(env)
            self._rebase_command_to_measured_wrist(env)
            self._transition("pregrasp_axis_align")
        elif self._state_step > self.RAY_HIT_TRACKING_TIMEOUT_STEPS:
            self._fail("actual wrist did not settle at the frozen ray-hit IK target")

    def _update_pregrasp_axis_alignment(self, env) -> None:
        self._gripper_command = self.GRIPPER_OPEN_POSITION
        self.green_ray_hit = self._green_ray_intersects_cube(env)
        if self.green_ray_hit:
            self._axis_alignment_ray_miss_streak = 0
        else:
            self._axis_alignment_ray_miss_streak += 1
            if self._axis_alignment_ray_miss_streak >= self.AXIS_ALIGNMENT_RAY_MISS_LIMIT:
                self._fail("ray lost during direct pre-grasp wrist-roll alignment")
                return

        hold_stable = self._update_direct_hold_diagnostics(env)
        if self._axis_alignment_selected_index is None:
            self._axis_alignment_hold_settle_streak = self._axis_alignment_hold_settle_streak + 1 if hold_stable else 0
            if self._axis_alignment_hold_settle_streak >= self.AXIS_ALIGNMENT_HOLD_SETTLE_STEPS:
                alignment = self._update_axis_alignment_geometry(env, select_axis=True)
                if not bool(alignment.selection_feasible.all().item()):
                    self._fail("no cube X/Y alignment is reachable inside wrist-roll limits")
            elif self._state_step > self.AXIS_ALIGNMENT_TIMEOUT_STEPS:
                self._fail("arm did not settle after entering direct joint hold")
            return

        alignment = self._update_axis_alignment_geometry(env, select_axis=False)
        if not bool(alignment.selection_feasible.all().item()):
            self._fail("selected cube axis became degenerate during direct alignment")
            return
        aligned = (
            hold_stable
            and self.green_ray_hit
            and bool((alignment.absolute_error <= self.AXIS_ALIGNMENT_TOLERANCE).all().item())
        )
        self._axis_alignment_streak = self._axis_alignment_streak + 1 if aligned else 0
        self._axis_alignment_final_gate_streak = self._axis_alignment_streak
        if self._axis_alignment_streak >= self.AXIS_ALIGNMENT_STABLE_STEPS:
            self._axis_alignment_complete = True
            # Do not return to Cartesian descent: the direct five-joint target
            # remains the sole arm writer throughout slow closure and settling.
            self._transition("grasp")
        elif self._state_step > self.AXIS_ALIGNMENT_TIMEOUT_STEPS:
            self._fail("gripper closing axis did not align with a cube X/Y edge")

    def _update_ik_handoff(self, env) -> None:
        robot = env.scene["robot"]
        assert self._axis_alignment_controlled_joint_indices is not None
        velocities = torch.abs(robot.data.joint_vel[:, list(self._axis_alignment_controlled_joint_indices)])
        self._ik_handoff_max_arm_joint_velocity = torch.amax(velocities, dim=-1).detach()
        assert self._command_pos_b is not None
        command_pos_w = self._base_position_to_world(robot, self._command_pos_b)
        wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
        self._ik_handoff_wrist_position_error = torch.linalg.vector_norm(wrist_pos_w - command_pos_w, dim=-1).detach()
        stable = (self._ik_handoff_max_arm_joint_velocity <= self.AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE) & (
            self._ik_handoff_wrist_position_error <= self.IK_HANDOFF_WRIST_POSITION_TOLERANCE
        )
        self._ik_handoff_streak = self._ik_handoff_streak + 1 if bool(stable.all().item()) else 0
        if self._ik_handoff_streak >= self.IK_HANDOFF_STABLE_STEPS:
            self._transition("lift")
        elif self._state_step > self.IK_HANDOFF_TIMEOUT_STEPS:
            self._fail("IK did not reacquire the measured wrist pose after direct grasp hold")

    def _transition(self, state: str, env=None) -> None:
        # The base machine normally enters lift as soon as gripper feedback has
        # settled.  Insert a zero-displacement IK reacquisition first.
        if state == "lift" and getattr(self, "_axis_alignment_direct_hold_active", False):
            state = "ik_handoff"
        super()._transition(state, env)

    def _update_posture_target(self, env) -> None:
        assert self._arm_action_term is not None
        if self._state == "ray_hit_tracking_settle":
            assert self._ray_hit_tracking_target_b is not None
            self._command_pos_b = self._ray_hit_tracking_target_b.detach().clone()
            self._arm_action_term.clear_direct_joint_position_target()
            super()._update_posture_target(env)
            return
        if self._state == "pregrasp_axis_align" and not self._axis_alignment_direct_hold_active:
            # The ray-hit transition happens after posture update in the same
            # get_action call.  Preserve the direct target installed by that
            # hook; this branch only protects against an invalid later re-entry.
            self._fail("pre-grasp alignment lost ownership of the direct arm target")
            return
        if self._axis_alignment_direct_hold_active and self._state in {
            "pregrasp_axis_align",
            "grasp",
            "grasp_settle",
        }:
            self._rebase_command_to_measured_wrist(env)
            if self._state == "pregrasp_axis_align" and self._axis_alignment_selected_index is not None:
                self._advance_direct_wrist_roll_target(env)
            assert self._axis_alignment_direct_joint_target is not None
            self._arm_action_term.set_direct_joint_position_target(self._axis_alignment_direct_joint_target)
            robot = env.scene["robot"]
            wrist_flex_index = list(robot.data.joint_names).index("wrist_flex")
            self._posture_target = robot.data.joint_pos[:, wrist_flex_index].detach().clone()
            return

        if self._state == "ik_handoff":
            robot = env.scene["robot"]
            wrist_flex_index = list(robot.data.joint_names).index("wrist_flex")
            if self._ik_handoff_target_b is None:
                # Capture one fixed, zero-displacement IK target at the exact
                # direct-hold pose.  It must not follow the measured wrist on
                # later frames, otherwise the position gate is vacuous.
                self._rebase_command_to_measured_wrist(env)
                assert self._command_pos_b is not None
                self._ik_handoff_target_b = self._command_pos_b.detach().clone()
                self._ik_handoff_wrist_flex_target = robot.data.joint_pos[:, wrist_flex_index].detach().clone()
                self._release_direct_joint_hold("gripper_feedback_settled")
                self._arm_action_term.set_maximum_joint_target_step(maximum_step=None)
                self._arm_action_term.set_joint_target_slew_limit(maximum_step=None)
                self._arm_action_term.reset_joint_target_slew_reference()
            else:
                self._command_pos_b = self._ik_handoff_target_b.detach().clone()
            assert self._ik_handoff_wrist_flex_target is not None
            self._posture_target = self._ik_handoff_wrist_flex_target
            self._arm_action_term.set_xyz_joint_nullspace_target(
                joint_name="wrist_flex",
                joint_target=self._ik_handoff_wrist_flex_target,
                damping=0.04,
                posture_gain=0.0,
                max_posture_step=0.03,
            )
            return

        self._arm_action_term.clear_direct_joint_position_target()
        super()._update_posture_target(env)

    def _advance_direct_wrist_roll_target(self, env) -> None:
        assert self._axis_alignment_direct_joint_target is not None
        assert self._axis_alignment_wrist_roll_control_index is not None
        assert self._axis_alignment_controlled_joint_indices is not None
        alignment = self._update_axis_alignment_geometry(env, select_axis=False)
        robot = env.scene["robot"]
        wrist_roll_robot_index = self._axis_alignment_controlled_joint_indices[
            self._axis_alignment_wrist_roll_control_index
        ]
        wrist_roll_position = robot.data.joint_pos[:, wrist_roll_robot_index]
        target_step = torch.clamp(
            self.AXIS_ALIGNMENT_TARGET_KP * alignment.signed_error,
            min=-self.AXIS_ALIGNMENT_MAX_TARGET_STEP,
            max=self.AXIS_ALIGNMENT_MAX_TARGET_STEP,
        )
        previous_target = self._axis_alignment_direct_joint_target[:, self._axis_alignment_wrist_roll_control_index]
        wrist_roll_target = previous_target + target_step
        wrist_roll_target = torch.clamp(
            wrist_roll_target,
            min=wrist_roll_position - self.AXIS_ALIGNMENT_MAX_TARGET_LEAD,
            max=wrist_roll_position + self.AXIS_ALIGNMENT_MAX_TARGET_LEAD,
        )
        wrist_roll_limits = robot.data.soft_joint_pos_limits[:, wrist_roll_robot_index]
        wrist_roll_target = torch.clamp(
            wrist_roll_target,
            min=wrist_roll_limits[:, 0] + self.AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN,
            max=wrist_roll_limits[:, 1] - self.AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN,
        )
        self._axis_alignment_direct_joint_target[:, self._axis_alignment_wrist_roll_control_index] = wrist_roll_target
        self._axis_alignment_wrist_roll_target = wrist_roll_target.detach().clone()

    def _rebase_command_to_measured_wrist(self, env) -> None:
        robot = env.scene["robot"]
        wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
        self._command_pos_b = self._world_position_to_base(robot, wrist_pos_w).detach().clone()

    def _release_direct_joint_hold(self, reason: str) -> None:
        if not self._axis_alignment_direct_hold_active:
            return
        assert self._arm_action_term is not None
        self._arm_action_term.clear_direct_joint_position_target()
        self._axis_alignment_direct_hold_active = False
        self._axis_alignment_direct_hold_release_reason = reason

    def _update_axis_alignment_geometry(self, env, *, select_axis: bool):
        gripper_quat_w = env.scene["ee_frame"].data.target_quat_w[:, 0]
        cube_quat_w = env.scene["cube"].data.root_quat_w
        dtype = gripper_quat_w.dtype
        local_x = torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=dtype).repeat(env.num_envs, 1)
        local_y = torch.tensor((0.0, 1.0, 0.0), device=env.device, dtype=dtype).repeat(env.num_envs, 1)
        local_z = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=dtype).repeat(env.num_envs, 1)
        closing_axis_w = quat_apply(gripper_quat_w, local_x)
        roll_axis_w = quat_apply(gripper_quat_w, local_z)
        cube_x_axis_w = quat_apply(cube_quat_w, local_x)
        cube_y_axis_w = quat_apply(cube_quat_w, local_y)
        if select_axis or self._axis_alignment_selected_index is None:
            robot = env.scene["robot"]
            wrist_roll_index = list(robot.data.joint_names).index("wrist_roll")
            alignment = select_nearest_cube_axis_alignment(
                closing_axis_w,
                roll_axis_w,
                cube_x_axis_w,
                cube_y_axis_w,
                joint_position=robot.data.joint_pos[:, wrist_roll_index],
                joint_limits=robot.data.soft_joint_pos_limits[:, wrist_roll_index],
                joint_limit_margin=self.AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN,
            )
            self._axis_alignment_selected_index = alignment.selected_axis_index.detach().clone()
            self._axis_alignment_selected_sign = alignment.selected_axis_sign.detach().clone()
            indices = self._axis_alignment_selected_index.cpu().tolist()
            signs = self._axis_alignment_selected_sign.cpu().tolist()
            self._axis_alignment_selected_cube_axis = tuple(
                f"{'+' if sign >= 0.0 else '-'}cube_local_{'x' if index == 0 else 'y'}"
                for index, sign in zip(indices, signs, strict=True)
            )
        else:
            assert self._axis_alignment_selected_sign is not None
            alignment = measure_cube_axis_alignment(
                closing_axis_w,
                roll_axis_w,
                cube_x_axis_w,
                cube_y_axis_w,
                self._axis_alignment_selected_index,
                self._axis_alignment_selected_sign,
            )
        self._axis_alignment_closing_axis_w = closing_axis_w.detach().clone()
        self._axis_alignment_cube_x_axis_w = cube_x_axis_w.detach().clone()
        self._axis_alignment_cube_y_axis_w = cube_y_axis_w.detach().clone()
        self._axis_alignment_desired_axis_w = alignment.desired_axis_w.detach().clone()
        self._axis_alignment_signed_error = alignment.signed_error.detach().clone()
        self._axis_alignment_error = alignment.absolute_error.detach().clone()
        return alignment

    def _update_direct_hold_diagnostics(self, env) -> bool:
        assert self._axis_alignment_direct_joint_target is not None
        assert self._axis_alignment_frozen_joint_reference is not None
        assert self._axis_alignment_controlled_joint_indices is not None
        assert self._axis_alignment_frozen_control_indices is not None
        assert self._axis_alignment_wrist_roll_control_index is not None
        robot = env.scene["robot"]
        joint_pos = robot.data.joint_pos[:, list(self._axis_alignment_controlled_joint_indices)]
        joint_vel = torch.abs(robot.data.joint_vel[:, list(self._axis_alignment_controlled_joint_indices)])
        self._axis_alignment_direct_target_error = torch.amax(
            torch.abs(joint_pos - self._axis_alignment_direct_joint_target), dim=-1
        ).detach()
        self._axis_alignment_frozen_joint_drift = torch.amax(
            torch.abs(
                joint_pos[:, list(self._axis_alignment_frozen_control_indices)]
                - self._axis_alignment_frozen_joint_reference[:, list(self._axis_alignment_frozen_control_indices)]
            ),
            dim=-1,
        ).detach()
        self._axis_alignment_wrist_roll_position = (
            joint_pos[:, self._axis_alignment_wrist_roll_control_index].detach().clone()
        )
        self._axis_alignment_wrist_roll_velocity = (
            joint_vel[:, self._axis_alignment_wrist_roll_control_index].detach().clone()
        )
        self._axis_alignment_max_arm_joint_velocity = torch.amax(joint_vel, dim=-1).detach()
        assert self._axis_alignment_entry_wrist_pos_w is not None
        wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
        self._axis_alignment_wrist_position_error = torch.linalg.vector_norm(
            wrist_pos_w - self._axis_alignment_entry_wrist_pos_w, dim=-1
        ).detach()
        stable = (
            (self._axis_alignment_direct_target_error <= self.AXIS_ALIGNMENT_DIRECT_TARGET_TOLERANCE)
            & (self._axis_alignment_frozen_joint_drift <= self.AXIS_ALIGNMENT_FROZEN_JOINT_TOLERANCE)
            & (self._axis_alignment_wrist_roll_velocity <= self.AXIS_ALIGNMENT_WRIST_ROLL_VELOCITY_TOLERANCE)
            & (self._axis_alignment_max_arm_joint_velocity <= self.AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE)
            & (self._axis_alignment_wrist_position_error <= self.AXIS_ALIGNMENT_WRIST_POSITION_TOLERANCE)
        )
        return bool(stable.all().item())

    def _fail(self, reason: str) -> None:
        self._release_direct_joint_hold(f"abort:{reason}")
        super()._fail(reason)

    @property
    def axis_alignment_complete(self) -> bool:
        return self._axis_alignment_complete

    @property
    def axis_alignment_streak(self) -> int:
        return self._axis_alignment_streak

    @property
    def axis_alignment_final_gate_streak(self) -> int:
        return self._axis_alignment_final_gate_streak

    @property
    def axis_alignment_selected_cube_axis(self) -> tuple[str, ...] | None:
        return self._axis_alignment_selected_cube_axis

    @property
    def axis_alignment_closing_axis_w(self) -> torch.Tensor | None:
        return self._axis_alignment_closing_axis_w

    @property
    def axis_alignment_cube_x_axis_w(self) -> torch.Tensor | None:
        return self._axis_alignment_cube_x_axis_w

    @property
    def axis_alignment_cube_y_axis_w(self) -> torch.Tensor | None:
        return self._axis_alignment_cube_y_axis_w

    @property
    def axis_alignment_desired_axis_w(self) -> torch.Tensor | None:
        return self._axis_alignment_desired_axis_w

    @property
    def axis_alignment_signed_error(self) -> torch.Tensor | None:
        return self._axis_alignment_signed_error

    @property
    def axis_alignment_error(self) -> torch.Tensor | None:
        return self._axis_alignment_error

    @property
    def axis_alignment_wrist_roll_target(self) -> torch.Tensor | None:
        return self._axis_alignment_wrist_roll_target

    @property
    def axis_alignment_wrist_roll_position(self) -> torch.Tensor | None:
        return self._axis_alignment_wrist_roll_position

    @property
    def axis_alignment_wrist_roll_velocity(self) -> torch.Tensor | None:
        return self._axis_alignment_wrist_roll_velocity

    @property
    def axis_alignment_max_arm_joint_velocity(self) -> torch.Tensor | None:
        return self._axis_alignment_max_arm_joint_velocity

    @property
    def axis_alignment_wrist_position_error(self) -> torch.Tensor | None:
        return self._axis_alignment_wrist_position_error

    @property
    def axis_alignment_direct_target_error(self) -> torch.Tensor | None:
        return self._axis_alignment_direct_target_error

    @property
    def axis_alignment_frozen_joint_drift(self) -> torch.Tensor | None:
        return self._axis_alignment_frozen_joint_drift

    @property
    def axis_alignment_direct_joint_target(self) -> torch.Tensor | None:
        return self._axis_alignment_direct_joint_target

    @property
    def axis_alignment_direct_hold_active(self) -> bool:
        return self._axis_alignment_direct_hold_active

    @property
    def axis_alignment_direct_hold_release_reason(self) -> str | None:
        return self._axis_alignment_direct_hold_release_reason

    @property
    def ray_hit_tracking_target_b(self) -> torch.Tensor | None:
        return self._ray_hit_tracking_target_b

    @property
    def ray_hit_tracking_target_w(self) -> torch.Tensor | None:
        return self._ray_hit_tracking_target_w

    @property
    def ray_hit_tracking_entry_wrist_pos_w(self) -> torch.Tensor | None:
        return self._ray_hit_tracking_entry_wrist_pos_w

    @property
    def ray_hit_tracking_wrist_delta_w(self) -> torch.Tensor | None:
        return self._ray_hit_tracking_wrist_delta_w

    @property
    def ray_hit_tracking_residual_descent(self) -> torch.Tensor | None:
        return self._ray_hit_tracking_residual_descent

    @property
    def ray_hit_tracking_streak(self) -> int:
        return self._ray_hit_tracking_streak

    @property
    def ray_hit_tracking_ray_miss_streak(self) -> int:
        return self._ray_hit_tracking_ray_miss_streak

    @property
    def ray_hit_tracking_wrist_position_error(self) -> torch.Tensor | None:
        return self._ray_hit_tracking_wrist_position_error

    @property
    def ray_hit_tracking_max_arm_joint_velocity(self) -> torch.Tensor | None:
        return self._ray_hit_tracking_max_arm_joint_velocity

    @property
    def ik_handoff_streak(self) -> int:
        return self._ik_handoff_streak

    @property
    def ik_handoff_wrist_position_error(self) -> torch.Tensor | None:
        return self._ik_handoff_wrist_position_error

    @property
    def ik_handoff_max_arm_joint_velocity(self) -> torch.Tensor | None:
        return self._ik_handoff_max_arm_joint_velocity

    @property
    def servo_parameters(self) -> dict[str, object]:
        return {
            **super().servo_parameters,
            "comparison_variant": "autogen_reference_axis_align_slow_grasp",
            "closing_axis": "gripper_local_+x",
            "alignment_target": "nearest_unoriented_cube_local_x_or_y",
            "alignment_rotation_joint": "wrist_roll_only",
            "alignment_controller": "direct_joint_hold(freeze_four_plus_rate_limited_wrist_roll)",
            "alignment_tolerance_deg": math.degrees(self.AXIS_ALIGNMENT_TOLERANCE),
            "alignment_stable_steps": self.AXIS_ALIGNMENT_STABLE_STEPS,
            "alignment_hold_settle_steps": self.AXIS_ALIGNMENT_HOLD_SETTLE_STEPS,
            "alignment_timeout_steps": self.AXIS_ALIGNMENT_TIMEOUT_STEPS,
            "alignment_ray_miss_limit": self.AXIS_ALIGNMENT_RAY_MISS_LIMIT,
            "alignment_target_kp": self.AXIS_ALIGNMENT_TARGET_KP,
            "alignment_max_target_step_rad": self.AXIS_ALIGNMENT_MAX_TARGET_STEP,
            "alignment_max_target_lead_rad": self.AXIS_ALIGNMENT_MAX_TARGET_LEAD,
            "pre_alignment_phase": "freeze_existing_ik_target_then_wait_for_measured_wrist",
            "ray_hit_tracking_position_tolerance_m": self.RAY_HIT_TRACKING_POSITION_TOLERANCE,
            "ray_hit_tracking_arm_joint_velocity_tolerance_rad_s": (self.AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE),
            "ray_hit_tracking_stable_steps": self.RAY_HIT_TRACKING_STABLE_STEPS,
            "ray_hit_tracking_timeout_steps": self.RAY_HIT_TRACKING_TIMEOUT_STEPS,
            "ray_hit_tracking_ray_miss_limit": self.AXIS_ALIGNMENT_RAY_MISS_LIMIT,
            "direct_hold_phases": "pregrasp_axis_align,grasp,grasp_settle",
            "direct_hold_release_gate": "feedback_settled_gripper_then_zero_displacement_ik_handoff",
            "pick_cube_semantics": "jaw_distance_and_gripper_angle_only;not_used_to_release_arm_hold",
            "post_alignment_gate": "direct_to_slow_grasp_without_cartesian_redescent",
            "ik_handoff_stable_steps": self.IK_HANDOFF_STABLE_STEPS,
        }
