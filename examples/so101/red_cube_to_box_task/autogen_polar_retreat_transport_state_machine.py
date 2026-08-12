"""Polar-path retreat and transport expert for RedCubeToBox."""

from __future__ import annotations

import math
from typing import ClassVar

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
import torch

from .autogen_independent_retreat_transport_state_machine import (
    RedCubeToBoxAutogenIndependentRetreatTransportStateMachine,
)
from .env_cfg import STATE_MACHINE_GRIPPER_CLOSE_POSITION

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_RETREAT_RADIAL_SCALE = 5.0 / 7.0
_WRIST_SAFE_HEIGHT_ABOVE_FLOOR_CENTER = 0.255
_RETREAT_REFERENCE_STEP = 0.0004
_ARC_REFERENCE_STEP = 0.0010
_RADIAL_REFERENCE_STEP = 0.0010
_BEARING_TOLERANCE = math.radians(3.0)
_RETREAT_Z_OVERSHOOT_LIMIT = 0.020
_RETREAT_ERROR_WORSENING_MARGIN = 0.010
_RETREAT_ERROR_WORSENING_STEPS = 20
_RETREAT_SEGMENT_STABLE_STEPS = 15
_GRASP_CONFIRM_DISTANCE = 0.015
_GRASP_LOSS_DISTANCE = 0.025
_GRIPPER_CLOSE_MINIMUM_STEPS = 80
_GRIPPER_CLOSE_MAXIMUM_STEPS = 200
_GRIPPER_SETTLE_STABLE_STEPS = 8
_SMOOTHERSTEP_MAX_DERIVATIVE = 1.875


class RedCubeToBoxAutogenPolarRetreatTransportStateMachine(RedCubeToBoxAutogenIndependentRetreatTransportStateMachine):
    """Use a short retreat, root-centered arc, and radial box approach.

    Pickup and placement behavior intentionally reuse the independent expert.
    Only the grasped-object path is replaced, making this a controlled
    comparison against its single-line retreat and transfer trajectory.
    """

    _FIXED_PHASE_STEPS: ClassVar[dict[str, int]] = {
        "approach_cube": 120,
        "descend_to_cube": 120,
        "release_cube": 100,
        "settle": 180,
    }
    _MOTION_PHASE_LIMITS: ClassVar[dict[str, tuple[int, int, float, int]]] = {
        "retreat_to_safe": (160, 700, 0.015, 15),
        "arc_transfer": (120, 700, 0.015, 15),
        "radial_transfer": (120, 700, 0.015, 15),
        "lower_into_box": (120, 600, 0.015, 10),
        "retract_gripper": (100, 400, 0.020, 10),
    }
    _PHASES = (
        "approach_cube",
        "descend_to_cube",
        "close_gripper",
        "retreat_to_safe",
        "arc_transfer",
        "radial_transfer",
        "lower_into_box",
        "release_cube",
        "retract_gripper",
        "settle",
    )
    MAX_STEPS = _GRIPPER_CLOSE_MAXIMUM_STEPS + sum(_FIXED_PHASE_STEPS.values()) + sum(
        maximum_steps for _, maximum_steps, _, _ in _MOTION_PHASE_LIMITS.values()
    )

    def __init__(self) -> None:
        super().__init__()
        self._retreat_bearing: torch.Tensor | None = None
        self._retreat_radial_target_w: torch.Tensor | None = None
        self._wrist_body_index: int | None = None
        self._retreat_actual_w: torch.Tensor | None = None
        self._retreat_z_error: float | None = None
        self._retreat_xz_error: float | None = None
        self._retreat_radial_error: float | None = None
        self._retreat_segment_start_phase_step = 0
        self._retreat_segment_ready = False
        self._arc_radius: torch.Tensor | None = None
        self._arc_start_bearing: torch.Tensor | None = None
        self._arc_target_bearing: torch.Tensor | None = None
        self._transport_height: torch.Tensor | None = None
        self._arc_start_quat_w: torch.Tensor | None = None
        self._placement_quat_w: torch.Tensor | None = None
        self._current_target_quat_w: torch.Tensor | None = None
        self._bearing_error: float | None = None
        self._reference_finished = False
        self._retreat_min_target_error = math.inf
        self._retreat_worsening_streak = 0
        self._retreat_safety_reason: str | None = None
        self._gripper_open_position: torch.Tensor | None = None
        self._gripper_target_error: float | None = None
        self._gripper_joint_velocity: float | None = None
        self._gripper_settle_streak = 0
        self._gripper_settle_reason: str | None = None

    def setup(self, env) -> None:
        super().setup(env)
        self._wrist_body_index = list(env.scene["robot"].data.body_names).index("wrist")

    def reset(self) -> None:
        wrist_body_index = self._wrist_body_index
        if self._arm_action_term is not None:
            self._arm_action_term.restore_configured_control_body()
        super().reset()
        self._wrist_body_index = wrist_body_index

    def get_action(self, env) -> torch.Tensor:
        phase = self.phase_name
        if self._arm_action_term is not None and phase != "retreat_to_safe":
            self._arm_action_term.restore_configured_control_body()
        if phase in {"approach_cube", "descend_to_cube", "close_gripper"}:
            action = super().get_action(env)
            if phase == "close_gripper":
                self._update_gripper_settle(env)
            return action

        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a polar AutoGen expert action")
        if phase != "retreat_to_safe":
            self._arm_action_term.set_orientation_weight(weight=1.0)
        self._initialize_anchors(env)

        if phase == "retreat_to_safe":
            self._initialize_polar_retreat(env)
            self._advance_retreat_segment_if_ready(env)
            self._arm_action_term.set_control_body(body_name="wrist")
            self._arm_action_term.set_position_only(enabled=True)
            if self._detect_polar_grasp_loss(env, phase):
                target_w = self._retreat_control_position_w(env).detach().clone()
            else:
                target_w = self._retreat_linear_reference()
            target_quat_w = self._retreat_control_quaternion_w(env).detach().clone()
            gripper = _GRIPPER_CLOSE
        elif phase == "arc_transfer":
            self._initialize_arc_transfer(env)
            if self._detect_polar_grasp_loss(env, phase):
                target_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
                target_quat_w = env.scene["ee_frame"].data.target_quat_w[:, 0, :].detach().clone()
            else:
                target_w, target_quat_w = self._arc_reference(env)
            gripper = _GRIPPER_CLOSE
        elif phase == "radial_transfer":
            self._initialize_radial_transfer(env)
            if self._detect_polar_grasp_loss(env, phase):
                target_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
            else:
                target_w = self._polar_linear_reference(_RADIAL_REFERENCE_STEP)
            assert self._placement_quat_w is not None
            target_quat_w = self._placement_quat_w
            gripper = _GRIPPER_CLOSE
        elif phase == "lower_into_box":
            self._initialize_lower(env)
            if self._detect_polar_grasp_loss(env, phase):
                target_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
            else:
                target_w = self._motion_reference(phase)
            target_quat_w = self._placement_orientation(env)
            gripper = _GRIPPER_CLOSE
        elif phase == "release_cube":
            self._initialize_release(env)
            assert self._motion_target_w is not None
            target_w = self._motion_target_w
            target_quat_w = self._placement_orientation(env)
            gripper = _GRIPPER_OPEN
        elif phase == "retract_gripper":
            self._initialize_retract(env)
            target_w = self._motion_reference(phase)
            target_quat_w = self._placement_orientation(env)
            gripper = _GRIPPER_OPEN
        else:
            self._initialize_settle(env)
            assert self._motion_target_w is not None
            target_w = self._motion_target_w
            target_quat_w = self._placement_orientation(env)
            gripper = _GRIPPER_OPEN

        self._current_target_w = target_w.detach().clone()
        self._current_target_quat_w = target_quat_w.detach().clone()
        if phase == "retreat_to_safe" and not self._episode_done:
            self._update_retreat_convergence(env)
            self._apply_retreat_safety_guards(env)
        elif phase in {"arc_transfer", "radial_transfer"} and not self._episode_done:
            self._update_polar_convergence(env, phase)
        elif phase in self._MOTION_PHASE_LIMITS and not self._episode_done:
            self._update_motion_convergence(env, phase)
        return self._compose_pose_action_with_quaternion(env, target_w, target_quat_w, gripper)

    def advance(self) -> None:
        """Wait for a physically settled grasp before starting the lift."""

        if self._episode_done:
            return
        if self.phase_name != "close_gripper":
            super().advance()
            return

        self._step_count += 1
        self._phase_step += 1
        if (
            self._phase_step >= _GRIPPER_CLOSE_MINIMUM_STEPS
            and self._gripper_settle_streak >= _GRIPPER_SETTLE_STABLE_STEPS
        ):
            self._advance_phase()
        elif self._phase_step >= _GRIPPER_CLOSE_MAXIMUM_STEPS:
            reason = (
                "gripper_not_settled_before_retreat:"
                f"target_error={self._gripper_target_error}:"
                f"velocity={self._gripper_joint_velocity}:"
                f"jaw_cube_distance={self._jaw_cube_distance}:"
                f"stable_streak={self._gripper_settle_streak}"
            )
            self._servo_abort_reason = reason
            self._release_block_reason = reason
            self._episode_done = True

    def _advance_phase(self) -> None:
        if self.phase_name == "retreat_to_safe" and self._arm_action_term is not None:
            self._arm_action_term.restore_configured_control_body()
        super()._advance_phase()
        self._current_target_quat_w = None
        self._bearing_error = None
        self._reference_finished = False

    def _retreat_control_position_w(self, env) -> torch.Tensor:
        assert self._wrist_body_index is not None
        return env.scene["robot"].data.body_pos_w[:, self._wrist_body_index]

    def _retreat_control_quaternion_w(self, env) -> torch.Tensor:
        assert self._wrist_body_index is not None
        return env.scene["robot"].data.body_quat_w[:, self._wrist_body_index]

    def _update_gripper_settle(self, env) -> None:
        robot = env.scene["robot"]
        gripper_index = robot.data.joint_names.index("gripper")
        gripper_position = robot.data.joint_pos[:, gripper_index]
        gripper_velocity = torch.abs(robot.data.joint_vel[:, gripper_index])
        if self._gripper_open_position is None:
            self._gripper_open_position = gripper_position.detach().clone()

        jaw_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        cube_w = env.scene["cube"].data.root_pos_w
        jaw_cube_distance = torch.linalg.vector_norm(jaw_w - cube_w, dim=-1)
        target_error = torch.abs(gripper_position - STATE_MACHINE_GRIPPER_CLOSE_POSITION)
        halfway_closed = gripper_position <= 0.5 * (
            self._gripper_open_position + STATE_MACHINE_GRIPPER_CLOSE_POSITION
        )
        close_enough_to_cube = jaw_cube_distance <= _GRASP_CONFIRM_DISTANCE
        settled = halfway_closed & close_enough_to_cube

        self._jaw_cube_distance = float(jaw_cube_distance.max().item())
        self._gripper_target_error = float(target_error.max().item())
        self._gripper_joint_velocity = float(gripper_velocity.max().item())
        if self._phase_step >= _GRIPPER_CLOSE_MINIMUM_STEPS and bool(settled.all().item()):
            self._gripper_settle_streak += 1
            self._gripper_settle_reason = "half_closed_near_cube"
        else:
            self._gripper_settle_streak = 0
            self._gripper_settle_reason = None

    def _initialize_polar_retreat(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = self._retreat_control_position_w(env).detach().clone()
        lift_target_w = start_w.clone()
        lift_target_w[:, 2] = torch.maximum(
            start_w[:, 2],
            self._floor_anchor_w[:, 2] + _WRIST_SAFE_HEIGHT_ABOVE_FLOOR_CENTER,
        )

        jaw_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        cube_w = env.scene["cube"].data.root_pos_w
        self._jaw_cube_distance = float(torch.linalg.vector_norm(jaw_w - cube_w, dim=-1).max().item())
        self._grasp_confirmed = self._jaw_cube_distance <= _GRASP_CONFIRM_DISTANCE
        self._retreat_subphase = "vertical_lift"
        self._retreat_safe_z = lift_target_w[:, 2].detach().clone()
        self._retreat_bearing = None
        self._retreat_target_radius = None
        self._retreat_radial_target_w = None
        self._retreat_segment_start_phase_step = self._phase_step
        self._retreat_segment_ready = False
        self._reference_finished = False
        self._set_motion(start_w, lift_target_w)

    def _advance_retreat_segment_if_ready(self, env) -> None:
        if self._retreat_subphase != "vertical_lift" or not self._retreat_segment_ready:
            return
        start_w = self._retreat_control_position_w(env).detach().clone()
        robot_root_w = env.scene["robot"].data.root_pos_w
        delta_xy = start_w[:, :2] - robot_root_w[:, :2]
        bearing = self._bearing(delta_xy)
        start_radius = torch.linalg.vector_norm(delta_xy, dim=-1)
        target_radius = _RETREAT_RADIAL_SCALE * start_radius
        target_w = start_w.clone()
        target_w[:, :2] = robot_root_w[:, :2] + _RETREAT_RADIAL_SCALE * delta_xy
        self._retreat_bearing = bearing.detach().clone()
        self._retreat_target_radius = target_radius.detach().clone()
        self._retreat_radial_target_w = target_w.detach().clone()
        self._retreat_subphase = "root_relative_5_over_7_retreat"
        self._retreat_segment_start_phase_step = self._phase_step
        self._retreat_segment_ready = False
        self._reference_finished = False
        self._target_stable_streak = 0
        self._motion_ready = False
        self._retreat_min_target_error = math.inf
        self._retreat_worsening_streak = 0
        self._set_motion(start_w, target_w)

    def _retreat_linear_reference(self) -> torch.Tensor:
        assert self._motion_start_w is not None
        assert self._motion_target_w is not None
        displacement = self._motion_target_w - self._motion_start_w
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        segment_step = self._phase_step - self._retreat_segment_start_phase_step
        traveled = torch.full_like(distance, (segment_step + 1) * _RETREAT_REFERENCE_STEP)
        progress = torch.clamp(traveled / torch.clamp(distance, min=1.0e-8), max=1.0)
        self._reference_finished = bool(torch.all(progress >= 1.0).item())
        return self._motion_start_w + progress * displacement

    def _update_retreat_convergence(self, env) -> None:
        assert self._motion_target_w is not None
        actual_w = self._retreat_control_position_w(env)
        robot_root_xy = env.scene["robot"].data.root_pos_w[:, :2]
        target_bearing = self._bearing(self._motion_target_w[:, :2] - robot_root_xy)
        actual_bearing = self._bearing(actual_w[:, :2] - robot_root_xy)
        position_error = torch.linalg.vector_norm(self._motion_target_w - actual_w, dim=-1)
        xz_error = torch.linalg.vector_norm((self._motion_target_w - actual_w)[:, [0, 2]], dim=-1)
        z_error = torch.abs(self._motion_target_w[:, 2] - actual_w[:, 2])
        target_radius = torch.linalg.vector_norm(self._motion_target_w[:, :2] - robot_root_xy, dim=-1)
        actual_radius = torch.linalg.vector_norm(actual_w[:, :2] - robot_root_xy, dim=-1)
        radial_error = torch.abs(target_radius - actual_radius)
        bearing_error = torch.abs(self._wrap_angle(target_bearing - actual_bearing))
        self._retreat_actual_w = actual_w.detach().clone()
        self._retreat_z_error = float(z_error.max().item())
        self._retreat_xz_error = float(xz_error.max().item())
        self._retreat_radial_error = float(radial_error.max().item())
        self._bearing_error = float(bearing_error.max().item())
        _, _, tolerance, _ = self._MOTION_PHASE_LIMITS["retreat_to_safe"]
        if self._retreat_subphase == "vertical_lift":
            # The lift is a clearance operation.  Once the commanded reference
            # and actual wrist Z are high enough, preserve the achieved XY and
            # rebase the following 5/7 radial segment from that real position.
            self._target_error = self._retreat_z_error
            reached = self._reference_finished and self._target_error <= tolerance
        else:
            self._target_error = float(position_error.max().item())
            reached = (
                self._reference_finished
                and self._target_error <= tolerance
                and self._bearing_error <= _BEARING_TOLERANCE
            )
        if reached:
            self._target_stable_streak += 1
        else:
            self._target_stable_streak = 0
        segment_complete = self._target_stable_streak >= _RETREAT_SEGMENT_STABLE_STEPS
        if self._retreat_subphase == "vertical_lift":
            self._retreat_segment_ready = segment_complete
            self._motion_ready = False
        else:
            self._motion_ready = segment_complete

    def _initialize_arc_transfer(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        start_quat_w = env.scene["ee_frame"].data.target_quat_w[:, 0, :].detach().clone()
        robot_root_w = env.scene["robot"].data.root_pos_w
        start_delta_xy = start_w[:, :2] - robot_root_w[:, :2]
        box_delta_xy = self._floor_anchor_w[:, :2] - robot_root_w[:, :2]
        radius = torch.linalg.vector_norm(start_delta_xy, dim=-1)
        start_bearing = self._bearing(start_delta_xy)
        target_bearing = self._bearing(box_delta_xy)
        bearing_delta = self._wrap_angle(target_bearing - start_bearing)

        target_w = start_w.clone()
        target_w[:, 0] = robot_root_w[:, 0] + radius * torch.sin(target_bearing)
        target_w[:, 1] = robot_root_w[:, 1] + radius * torch.cos(target_bearing)
        target_w[:, 2] = start_w[:, 2]
        zero = torch.zeros_like(bearing_delta)
        target_quat_w = quat_mul(quat_from_euler_xyz(zero, zero, bearing_delta), start_quat_w)

        self._arc_radius = radius.detach().clone()
        self._arc_start_bearing = start_bearing.detach().clone()
        self._arc_target_bearing = target_bearing.detach().clone()
        self._transport_height = start_w[:, 2].detach().clone()
        self._arc_start_quat_w = start_quat_w
        self._placement_quat_w = target_quat_w.detach().clone()
        self._retreat_subphase = "root_centered_arc"
        self._reference_finished = False
        self._set_motion(start_w, target_w)

    def _initialize_radial_transfer(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        target_w = self._floor_anchor_w.clone()
        if self._transport_height is None:
            self._transport_height = start_w[:, 2].detach().clone()
        target_w[:, 2] = self._transport_height
        if self._placement_quat_w is None:
            self._placement_quat_w = env.scene["ee_frame"].data.target_quat_w[:, 0, :].detach().clone()
        self._retreat_subphase = "box_bearing_radial_approach"
        self._reference_finished = False
        self._set_motion(start_w, target_w)

    def _polar_linear_reference(self, maximum_step: float) -> torch.Tensor:
        assert self._motion_start_w is not None
        assert self._motion_target_w is not None
        displacement = self._motion_target_w - self._motion_start_w
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        traveled = torch.full_like(distance, (self._phase_step + 1) * maximum_step)
        progress = torch.clamp(traveled / torch.clamp(distance, min=1.0e-8), max=1.0)
        self._reference_finished = bool(torch.all(progress >= 1.0).item())
        return self._motion_start_w + progress * displacement

    def _arc_reference(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._arc_radius is not None
        assert self._arc_start_bearing is not None
        assert self._arc_target_bearing is not None
        assert self._arc_start_quat_w is not None
        assert self._motion_start_w is not None
        assert self._motion_target_w is not None
        robot_root_w = env.scene["robot"].data.root_pos_w
        bearing_delta = self._wrap_angle(self._arc_target_bearing - self._arc_start_bearing)
        arc_length = self._arc_radius * torch.abs(bearing_delta)
        normalized_time = torch.clamp(
            (self._phase_step + 1)
            * _ARC_REFERENCE_STEP
            / torch.clamp(_SMOOTHERSTEP_MAX_DERIVATIVE * arc_length, min=1.0e-8),
            max=1.0,
        )
        progress = normalized_time**3 * (normalized_time * (normalized_time * 6.0 - 15.0) + 10.0)
        bearing = self._arc_start_bearing + progress * bearing_delta

        target_w = self._motion_start_w.clone()
        target_w[:, 0] = robot_root_w[:, 0] + self._arc_radius * torch.sin(bearing)
        target_w[:, 1] = robot_root_w[:, 1] + self._arc_radius * torch.cos(bearing)
        target_w[:, 2] = self._motion_start_w[:, 2]
        zero = torch.zeros_like(bearing_delta)
        target_quat_w = quat_mul(
            quat_from_euler_xyz(zero, zero, progress * bearing_delta),
            self._arc_start_quat_w,
        )
        self._reference_finished = bool(torch.all(normalized_time >= 1.0).item())
        return target_w, target_quat_w

    def _update_polar_convergence(self, env, phase: str) -> None:
        assert self._motion_target_w is not None
        actual_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        robot_root_xy = env.scene["robot"].data.root_pos_w[:, :2]
        target_bearing = self._bearing(self._motion_target_w[:, :2] - robot_root_xy)
        actual_bearing = self._bearing(actual_w[:, :2] - robot_root_xy)
        position_error = torch.linalg.vector_norm(self._motion_target_w - actual_w, dim=-1)
        bearing_error = torch.abs(self._wrap_angle(target_bearing - actual_bearing))
        self._target_error = float(position_error.max().item())
        self._bearing_error = float(bearing_error.max().item())
        _, _, tolerance, stable_steps = self._MOTION_PHASE_LIMITS[phase]
        reached = (
            self._reference_finished and self._target_error <= tolerance and self._bearing_error <= _BEARING_TOLERANCE
        )
        if reached:
            self._target_stable_streak += 1
        else:
            self._target_stable_streak = 0
        self._motion_ready = self._target_stable_streak >= stable_steps

    def _detect_polar_grasp_loss(self, env, phase: str) -> bool:
        jaw_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        cube_w = env.scene["cube"].data.root_pos_w
        self._jaw_cube_distance = float(torch.linalg.vector_norm(jaw_w - cube_w, dim=-1).max().item())
        if not self._grasp_confirmed and self._jaw_cube_distance > _GRASP_LOSS_DISTANCE:
            self._servo_abort_reason = (
                f"grasp_not_confirmed_at_{phase}_start:"
                f"jaw_cube_distance={self._jaw_cube_distance}:threshold={_GRASP_LOSS_DISTANCE}"
            )
            self._episode_done = True
            return True
        if self._grasp_confirmed and self._jaw_cube_distance > _GRASP_LOSS_DISTANCE:
            self._grasp_lost_before_release = True
            self._servo_abort_reason = (
                f"grasp_lost_during_{phase}:"
                f"jaw_cube_distance={self._jaw_cube_distance}:threshold={_GRASP_LOSS_DISTANCE}"
            )
            self._episode_done = True
            return True
        return False

    def _apply_retreat_safety_guards(self, env) -> None:
        assert self._motion_target_w is not None
        assert self._target_error is not None
        assert self._bearing_error is not None
        actual_w = self._retreat_control_position_w(env)
        maximum_z_overshoot = float(torch.max(actual_w[:, 2] - self._motion_target_w[:, 2]).item())
        self._retreat_min_target_error = min(self._retreat_min_target_error, self._target_error)
        if (
            self._reference_finished
            and self._target_error > self._retreat_min_target_error + _RETREAT_ERROR_WORSENING_MARGIN
        ):
            self._retreat_worsening_streak += 1
        else:
            self._retreat_worsening_streak = 0

        if maximum_z_overshoot > _RETREAT_Z_OVERSHOOT_LIMIT:
            reason = f"retreat_z_overshoot:overshoot={maximum_z_overshoot}:limit={_RETREAT_Z_OVERSHOOT_LIMIT}"
        elif self._retreat_worsening_streak >= _RETREAT_ERROR_WORSENING_STEPS:
            reason = (
                "retreat_target_error_worsening:"
                f"error={self._target_error}:minimum={self._retreat_min_target_error}:"
                f"streak={self._retreat_worsening_streak}"
            )
        else:
            return

        self._retreat_safety_reason = reason
        self._servo_abort_reason = reason
        self._release_block_reason = reason
        self._episode_done = True

    def _placement_orientation(self, env) -> torch.Tensor:
        if self._placement_quat_w is None:
            self._placement_quat_w = env.scene["ee_frame"].data.target_quat_w[:, 0, :].detach().clone()
        return self._placement_quat_w

    @staticmethod
    def _bearing(delta_xy: torch.Tensor) -> torch.Tensor:
        """Return bearing from the robot's +Y centerline toward +X."""

        return torch.atan2(delta_xy[:, 0], delta_xy[:, 1])

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _compose_pose_action_with_quaternion(
        env,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        gripper: float,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        target_pos_local = quat_apply(
            quat_inv(robot.data.root_quat_w),
            target_pos_w - robot.data.root_pos_w,
        )
        target_quat_local = quat_mul(quat_inv(robot.data.root_quat_w), target_quat_w)
        gripper_command = torch.full((env.num_envs, 1), gripper, device=env.device)
        return torch.cat((target_pos_local, target_quat_local, gripper_command), dim=-1)

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        return {
            "implementation": "polar_path_subclass_of_independent_expert",
            "phase_sequence": (
                "approach,descend,close,vertical_lift,root_relative_5_over_7_retreat,"
                "arc_transfer,radial_transfer,lower,release,retract,settle"
            ),
            "retreat_radial_scale": _RETREAT_RADIAL_SCALE,
            "retreat_reference_step": _RETREAT_REFERENCE_STEP,
            "arc_reference_step": _ARC_REFERENCE_STEP,
            "radial_reference_step": _RADIAL_REFERENCE_STEP,
            "wrist_safe_height_above_floor_center": _WRIST_SAFE_HEIGHT_ABOVE_FLOOR_CENTER,
            "transport_height_policy": "freeze_actual_safe_height_at_arc_entry",
            "bearing_tolerance_rad": _BEARING_TOLERANCE,
            "retreat_control_body": "wrist",
            "retreat_ik_mode": "wrist_xyz_position_only(no_world_orientation,no_wrist_flex_task)",
            "retreat_task_error_order": "wrist_x_m,wrist_y_m,wrist_z_m",
            "retreat_wrist_flex_policy": "unconstrained",
            "retreat_y_policy": "constrained_during_lift_and_root_relative_retreat",
            "retreat_path": "wrist_vertical_lift_then_root_relative_xy_scaled_to_5_over_7",
            "pre_retreat_gripper_gate": "half_closed_and_near_cube_stable",
            "gripper_close_minimum_steps": _GRIPPER_CLOSE_MINIMUM_STEPS,
            "gripper_close_maximum_steps": _GRIPPER_CLOSE_MAXIMUM_STEPS,
            "gripper_settle_stable_steps": _GRIPPER_SETTLE_STABLE_STEPS,
            "retreat_bearing_policy": "preserved_by_xy_scaling_and_checked_for_completion",
            "retreat_z_overshoot_limit": _RETREAT_Z_OVERSHOOT_LIMIT,
            "retreat_error_worsening_margin": _RETREAT_ERROR_WORSENING_MARGIN,
            "retreat_error_worsening_steps": _RETREAT_ERROR_WORSENING_STEPS,
            "orientation_policy": "retreat_has_no_world_orientation_task_then_arc_entry_pose_yaw_co_rotation",
            "completion_policy": (
                "vertical_lift_actual_wrist_z_only_then_"
                "radial_retreat_actual_wrist_xyz_and_root_bearing"
            ),
            "grasp_loss_distance": _GRASP_LOSS_DISTANCE,
        }

    @property
    def bearing_error(self) -> float | None:
        return self._bearing_error

    @property
    def current_target_quat_w(self) -> torch.Tensor | None:
        return self._current_target_quat_w

    @property
    def retreat_worsening_streak(self) -> int:
        return self._retreat_worsening_streak

    @property
    def retreat_safety_reason(self) -> str | None:
        return self._retreat_safety_reason

    @property
    def retreat_actual_w(self) -> torch.Tensor | None:
        return self._retreat_actual_w

    @property
    def retreat_z_error(self) -> float | None:
        return self._retreat_z_error

    @property
    def retreat_xz_error(self) -> float | None:
        return self._retreat_xz_error

    @property
    def retreat_radial_error(self) -> float | None:
        return self._retreat_radial_error

    @property
    def retreat_wrist_flex_target(self) -> torch.Tensor | None:
        """Compatibility diagnostic: position-only retreat has no wrist-flex target."""

        return None

    @property
    def retreat_control_body(self) -> str:
        if self._arm_action_term is None:
            return "wrist"
        return self._arm_action_term.control_body_name

    @property
    def retreat_shoulder_pan_target(self) -> None:
        """Compatibility diagnostic: retreat no longer constrains shoulder pan."""

        return None

    @property
    def gripper_target_error(self) -> float | None:
        return self._gripper_target_error

    @property
    def gripper_joint_velocity(self) -> float | None:
        return self._gripper_joint_velocity

    @property
    def gripper_settle_streak(self) -> int:
        return self._gripper_settle_streak

    @property
    def gripper_settle_reason(self) -> str | None:
        return self._gripper_settle_reason
