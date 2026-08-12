"""Slow Autogen grasp preceded by cube-edge-aware wrist-roll alignment."""

from __future__ import annotations

import math

from isaaclab.utils.math import quat_apply
import torch

from .autogen_reference_slow_grasp_state_machine import RedCubeToBoxAutogenReferenceSlowGraspStateMachine
from .cube_axis_alignment import measure_cube_axis_alignment
from .cube_axis_alignment import select_nearest_cube_axis_alignment


class RedCubeToBoxAutogenReferenceAxisAlignSlowGraspStateMachine(RedCubeToBoxAutogenReferenceSlowGraspStateMachine):
    """Align local gripper X to the nearest cube X/Y line before closing."""

    AXIS_ALIGNMENT_TOLERANCE = math.radians(5.0)
    AXIS_ALIGNMENT_STABLE_STEPS = 10
    AXIS_ALIGNMENT_TIMEOUT_STEPS = 300
    AXIS_ALIGNMENT_WRIST_ROLL_VELOCITY_TOLERANCE = 0.02
    AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE = 0.03
    AXIS_ALIGNMENT_WRIST_POSITION_TOLERANCE = 0.01
    AXIS_ALIGNMENT_TARGET_KP = 0.5
    AXIS_ALIGNMENT_MAX_TARGET_STEP = math.radians(1.0)
    AXIS_ALIGNMENT_MAX_IK_JOINT_STEP = math.radians(0.5)
    AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN = 0.02

    def reset(self) -> None:
        super().reset()
        self._axis_alignment_complete = False
        self._axis_alignment_streak = 0
        self._axis_alignment_final_gate_streak = 0
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

    def _on_grasp_pose_reached(self, env) -> None:
        if self._axis_alignment_complete:
            alignment = self._update_axis_alignment_geometry(env, select_axis=False)
            if not bool(alignment.selection_feasible.all().item()):
                self._fail("selected cube axis became degenerate before grasp")
                return
            if bool((alignment.absolute_error > self.AXIS_ALIGNMENT_TOLERANCE).any().item()):
                # Descent re-centering can move upstream joints and rotate the
                # gripper even while wrist_roll itself is held. Re-enter the
                # bounded alignment phase rather than closing on stale geometry.
                self._axis_alignment_complete = False
                self._axis_alignment_streak = 0
                self._axis_alignment_final_gate_streak = 0
                self._gripper_command = self.GRIPPER_OPEN_POSITION
                self._transition("pregrasp_axis_align")
                return
            if self._update_axis_alignment_stability_diagnostics(env):
                self._axis_alignment_final_gate_streak += 1
            else:
                self._axis_alignment_final_gate_streak = 0
            if self._axis_alignment_final_gate_streak >= self.AXIS_ALIGNMENT_STABLE_STEPS:
                self._transition("grasp")
            elif self._state_step > self.AXIS_ALIGNMENT_TIMEOUT_STEPS:
                self._fail("ray and cube-axis alignment did not settle simultaneously before grasp")
            return
        self._gripper_command = self.GRIPPER_OPEN_POSITION
        self._transition("pregrasp_axis_align")
        self._update_axis_alignment_geometry(env, select_axis=True)

    def _update_state(self, env) -> None:
        if self._state != "pregrasp_axis_align":
            phase_before_update = self._state
            super()._update_state(env)
            if (
                phase_before_update == "descend"
                and self._state == "descend"
                and self._axis_alignment_complete
                and not self.green_ray_hit
            ):
                self._axis_alignment_final_gate_streak = 0
            return

        self._gripper_command = self.GRIPPER_OPEN_POSITION
        self.green_ray_hit = self._green_ray_intersects_cube(env)
        alignment = self._update_axis_alignment_geometry(env, select_axis=False)
        if not bool(alignment.selection_feasible.all().item()):
            self._fail("selected cube axis became degenerate during alignment")
            return
        aligned = self._update_axis_alignment_stability_diagnostics(env)
        self._axis_alignment_streak = self._axis_alignment_streak + 1 if aligned else 0
        if self._axis_alignment_streak >= self.AXIS_ALIGNMENT_STABLE_STEPS:
            self._axis_alignment_complete = True
            self._axis_alignment_final_gate_streak = 0
            robot = env.scene["robot"]
            wrist_roll_index = list(robot.data.joint_names).index("wrist_roll")
            self._axis_alignment_wrist_roll_target = robot.data.joint_pos[:, wrist_roll_index].detach().clone()
            # Rotating local X also rotates the local +X ray-origin offset.  Return
            # to descend so the existing ray controller re-centers and rechecks
            # the OBB/range gate before grasp is allowed.
            self._transition("descend")
        elif self._state_step > self.AXIS_ALIGNMENT_TIMEOUT_STEPS:
            self._fail("gripper closing axis did not align with a cube X/Y edge")

    def _update_posture_target(self, env) -> None:
        assert self._arm_action_term is not None
        if self._state == "pregrasp_axis_align":
            alignment = self._update_axis_alignment_geometry(env, select_axis=False)
            robot = env.scene["robot"]
            wrist_roll_index = list(robot.data.joint_names).index("wrist_roll")
            wrist_roll_position = robot.data.joint_pos[:, wrist_roll_index]
            wrist_roll_limits = robot.data.soft_joint_pos_limits[:, wrist_roll_index]
            target_step = torch.clamp(
                self.AXIS_ALIGNMENT_TARGET_KP * alignment.signed_error,
                min=-self.AXIS_ALIGNMENT_MAX_TARGET_STEP,
                max=self.AXIS_ALIGNMENT_MAX_TARGET_STEP,
            )
            self._axis_alignment_wrist_roll_target = torch.clamp(
                wrist_roll_position + target_step,
                min=wrist_roll_limits[:, 0],
                max=wrist_roll_limits[:, 1],
            ).detach()
            wrist_flex_index = list(robot.data.joint_names).index("wrist_flex")
            self._posture_target = robot.data.joint_pos[:, wrist_flex_index].detach().clone()
            self._arm_action_term.set_maximum_joint_target_step(maximum_step=self.AXIS_ALIGNMENT_MAX_IK_JOINT_STEP)
            self._arm_action_term.set_xyz_joint_nullspace_target(
                joint_name="wrist_roll",
                joint_target=self._axis_alignment_wrist_roll_target,
                damping=0.04,
                posture_gain=0.0,
                max_posture_step=0.03,
            )
            return

        self._arm_action_term.set_maximum_joint_target_step(maximum_step=None)
        if self._axis_alignment_complete and self._state in {"descend", "grasp", "grasp_settle", "lift"}:
            assert self._axis_alignment_wrist_roll_target is not None
            robot = env.scene["robot"]
            wrist_flex_index = list(robot.data.joint_names).index("wrist_flex")
            self._posture_target = robot.data.joint_pos[:, wrist_flex_index].detach().clone()
            self._arm_action_term.set_xyz_joint_nullspace_target(
                joint_name="wrist_roll",
                joint_target=self._axis_alignment_wrist_roll_target,
                damping=0.04,
                posture_gain=0.0,
                max_posture_step=0.03,
            )
            return
        super()._update_posture_target(env)

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
            if not bool(alignment.selection_feasible.all().item()):
                self._fail("no cube X/Y closing-axis alignment is reachable inside wrist-roll limits")
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

    def _update_axis_alignment_stability_diagnostics(self, env) -> bool:
        robot = env.scene["robot"]
        wrist_roll_index = list(robot.data.joint_names).index("wrist_roll")
        self._axis_alignment_wrist_roll_position = robot.data.joint_pos[:, wrist_roll_index].detach().clone()
        self._axis_alignment_wrist_roll_velocity = torch.abs(robot.data.joint_vel[:, wrist_roll_index]).detach()
        arm_joint_indices = [
            index for index, joint_name in enumerate(robot.data.joint_names) if joint_name != "gripper"
        ]
        self._axis_alignment_max_arm_joint_velocity = torch.amax(
            torch.abs(robot.data.joint_vel[:, arm_joint_indices]), dim=-1
        ).detach()
        assert self._command_pos_b is not None
        assert self._wrist_body_index is not None
        command_pos_w = self._base_position_to_world(robot, self._command_pos_b)
        wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
        self._axis_alignment_wrist_position_error = torch.linalg.vector_norm(
            wrist_pos_w - command_pos_w, dim=-1
        ).detach()
        assert self._axis_alignment_error is not None
        aligned = (
            (self._axis_alignment_error <= self.AXIS_ALIGNMENT_TOLERANCE)
            & (self._axis_alignment_wrist_roll_velocity <= self.AXIS_ALIGNMENT_WRIST_ROLL_VELOCITY_TOLERANCE)
            & (self._axis_alignment_max_arm_joint_velocity <= self.AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE)
            & (self._axis_alignment_wrist_position_error <= self.AXIS_ALIGNMENT_WRIST_POSITION_TOLERANCE)
        )
        return bool(aligned.all().item())

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
    def servo_parameters(self) -> dict[str, object]:
        return {
            **super().servo_parameters,
            "comparison_variant": "autogen_reference_axis_align_slow_grasp",
            "closing_axis": "gripper_local_+x",
            "alignment_target": "nearest_unoriented_cube_local_x_or_y",
            "alignment_rotation_axis": "gripper_local_+z",
            "alignment_controller": "wrist_xyz_plus_wrist_roll_joint_row",
            "alignment_tolerance_deg": math.degrees(self.AXIS_ALIGNMENT_TOLERANCE),
            "alignment_stable_steps": self.AXIS_ALIGNMENT_STABLE_STEPS,
            "alignment_timeout_steps": self.AXIS_ALIGNMENT_TIMEOUT_STEPS,
            "alignment_wrist_roll_velocity_tolerance": self.AXIS_ALIGNMENT_WRIST_ROLL_VELOCITY_TOLERANCE,
            "alignment_max_arm_joint_velocity_tolerance": self.AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE,
            "alignment_wrist_position_tolerance": self.AXIS_ALIGNMENT_WRIST_POSITION_TOLERANCE,
            "alignment_target_kp": self.AXIS_ALIGNMENT_TARGET_KP,
            "alignment_max_target_step_rad": self.AXIS_ALIGNMENT_MAX_TARGET_STEP,
            "alignment_max_ik_joint_step_rad": self.AXIS_ALIGNMENT_MAX_IK_JOINT_STEP,
            "alignment_joint_limit_margin_rad": self.AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN,
            "post_alignment_gate": "return_to_descend_for_ray_recentering_and_obb_range_recheck",
            "final_grasp_gate": "ray_hit_and_current_axis_alignment_stable_together",
        }
