"""Polar-path retreat and transport expert for RedCubeToBox."""

from __future__ import annotations

import math
from typing import ClassVar

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
import torch

from .cube_axis_alignment import measure_cube_axis_alignment
from .cube_axis_alignment import select_nearest_cube_axis_alignment
from .env_cfg import STATE_MACHINE_GRIPPER_CLOSE_POSITION
from .polar_base_state_machine import RedCubeToBoxPolarBaseStateMachine

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_RETREAT_RADIAL_SCALE = 5.0 / 7.0
_WRIST_SAFE_HEIGHT_ABOVE_FLOOR_CENTER = 0.255
_RETREAT_REFERENCE_STEP = 0.0004
_ARC_REFERENCE_STEP = 0.0010
_RADIAL_REFERENCE_STEP = 0.0010
_BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER = 0.22
_BEARING_TOLERANCE = math.radians(3.0)
_RETREAT_BEARING_TOLERANCE = math.radians(10.0)
_WRIST_POSTURE_DAMPING = 0.04
_WRIST_POSTURE_GAIN = 0.08
_WRIST_POSTURE_MAX_STEP = 0.03
_WRIST_JOINT_TARGET_ACCUMULATION_STEP = 0.005
_RETREAT_Z_OVERSHOOT_LIMIT = 0.020
_RETREAT_ERROR_WORSENING_MARGIN = 0.010
_RETREAT_ERROR_WORSENING_STEPS = 20
_RETREAT_SEGMENT_STABLE_STEPS = 15
_GRASP_CONFIRM_DISTANCE = 0.015
_GRASP_LOSS_DISTANCE = 0.025
_GRIPPER_CLOSE_MINIMUM_STEPS = 80
_GRIPPER_CLOSE_MAXIMUM_STEPS = 320
_GRIPPER_SETTLE_STABLE_STEPS = 8
_GRIPPER_SETTLE_WINDOW_STEPS = 12
_GRIPPER_SETTLE_ANGLE_SPAN_TOLERANCE = 0.01
_PICK_FEEDBACK_STABLE_STEPS = 3
_AXIS_ALIGNMENT_TOLERANCE = math.radians(5.0)
_AXIS_ALIGNMENT_STABLE_STEPS = 10
_AXIS_ALIGNMENT_HOLD_SETTLE_STEPS = 8
_AXIS_ALIGNMENT_TIMEOUT_STEPS = 700
_AXIS_ALIGNMENT_MAX_TARGET_STEP = math.radians(1.0)
_AXIS_ALIGNMENT_MAX_TARGET_LEAD = math.radians(4.0)
_AXIS_ALIGNMENT_TARGET_KP = 0.5
_AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN = 0.02
_AXIS_ALIGNMENT_WRIST_ROLL_VELOCITY_TOLERANCE = 0.03
_AXIS_ALIGNMENT_MAX_ARM_VELOCITY = 0.05
_PREALIGN_SETTLE_TIMEOUT_STEPS = 240
_PREALIGN_POSITION_TOLERANCE = 0.006
_POSTALIGN_RECENTER_TIMEOUT_STEPS = 240
_POSTALIGN_POSITION_TOLERANCE = 0.006
_PICKUP_SETTLE_STABLE_STEPS = 8
_PICK_GRIPPER_OFFSET_ALONG_CLOSING_AXIS = 0.020
_SMOOTHERSTEP_MAX_DERIVATIVE = 1.875


class RedCubeToBoxAutogenPolarRetreatTransportStateMachine(RedCubeToBoxPolarBaseStateMachine):
    """Use a short retreat, root-centered arc, and radial box approach.

    Placement retains the shared polar-base geometry. Pickup adds measured XYZ
    handoffs and feedback-settled closing so domain-randomized poses cannot
    enter retreat merely because a fixed keyframe duration elapsed.
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
        "settle_at_grasp_target",
        "align_gripper_to_cube",
        "recenter_after_alignment",
        "close_gripper",
        "retreat_to_safe",
        "arc_transfer",
        "radial_transfer",
        "lower_into_box",
        "release_cube",
        "retract_gripper",
        "settle",
    )
    MAX_STEPS = (
        _AXIS_ALIGNMENT_TIMEOUT_STEPS
        + _PREALIGN_SETTLE_TIMEOUT_STEPS
        + _POSTALIGN_RECENTER_TIMEOUT_STEPS
        + _GRIPPER_CLOSE_MAXIMUM_STEPS
        + _FIXED_PHASE_STEPS["approach_cube"]
        + _FIXED_PHASE_STEPS["descend_to_cube"]
        + _FIXED_PHASE_STEPS["release_cube"]
        + _FIXED_PHASE_STEPS["settle"]
        + _MOTION_PHASE_LIMITS["retreat_to_safe"][1]
        + _MOTION_PHASE_LIMITS["arc_transfer"][1]
        + _MOTION_PHASE_LIMITS["radial_transfer"][1]
        + _MOTION_PHASE_LIMITS["lower_into_box"][1]
        + _MOTION_PHASE_LIMITS["retract_gripper"][1]
    )

    def __init__(self) -> None:
        super().__init__()
        self._retreat_bearing: torch.Tensor | None = None
        self._retreat_radial_target_w: torch.Tensor | None = None
        self._wrist_body_index: int | None = None
        self._wrist_posture_target: torch.Tensor | None = None
        self._radial_posture_target: torch.Tensor | None = None
        self._lower_posture_target: torch.Tensor | None = None
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
        self._gripper_settle_angle_window: list[float] = []
        self._gripper_settle_angle_span: float | None = None
        self._gripper_settle_streak = 0
        self._gripper_settle_reason: str | None = None
        self._pick_feedback_streak = 0
        self._pick_feedback_confirmed = False
        self._grasp_geometry_latched = False
        self._minimum_jaw_cube_distance: float | None = None
        self._position_joint_target_accumulation_enabled = False
        self._axis_alignment_selected_index: torch.Tensor | None = None
        self._axis_alignment_selected_sign: torch.Tensor | None = None
        self._axis_alignment_selected_cube_axis: tuple[str, ...] | None = None
        self._axis_alignment_closing_axis_w: torch.Tensor | None = None
        self._axis_alignment_cube_x_axis_w: torch.Tensor | None = None
        self._axis_alignment_cube_y_axis_w: torch.Tensor | None = None
        self._axis_alignment_desired_axis_w: torch.Tensor | None = None
        self._axis_alignment_signed_error: torch.Tensor | None = None
        self._axis_alignment_error: torch.Tensor | None = None
        self._axis_alignment_direct_joint_target: torch.Tensor | None = None
        self._axis_alignment_wrist_roll_control_index: int | None = None
        self._axis_alignment_controlled_joint_indices: tuple[int, ...] | None = None
        self._axis_alignment_hold_settle_streak = 0
        self._axis_alignment_streak = 0
        self._axis_alignment_complete = False
        self._axis_alignment_wrist_roll_target: torch.Tensor | None = None
        self._axis_alignment_wrist_roll_position: torch.Tensor | None = None
        self._axis_alignment_wrist_roll_velocity: torch.Tensor | None = None
        self._axis_alignment_max_arm_joint_velocity: torch.Tensor | None = None
        self._pickup_settle_streak = 0
        self._pickup_position_error: float | None = None
        self._aligned_gripper_quat_w: torch.Tensor | None = None
        self._postalign_recenter_target_w: torch.Tensor | None = None

    def setup(self, env) -> None:
        super().setup(env)
        self._wrist_body_index = list(env.scene["robot"].data.body_names).index("wrist")

    def reset(self) -> None:
        wrist_body_index = self._wrist_body_index
        if self._arm_action_term is not None:
            self._disable_position_joint_target_accumulation()
            self._arm_action_term.clear_direct_joint_position_target()
            self._arm_action_term.restore_configured_control_body()
        super().reset()
        self._wrist_body_index = wrist_body_index
        self._wrist_posture_target = None
        self._radial_posture_target = None
        self._lower_posture_target = None

    def get_action(self, env) -> torch.Tensor:
        phase = self.phase_name
        position_only_accumulation_phases = {
            "retreat_to_safe",
            "arc_transfer",
            "radial_transfer",
            "lower_into_box",
        }
        if self._arm_action_term is not None and phase not in position_only_accumulation_phases:
            self._disable_position_joint_target_accumulation()
            self._arm_action_term.restore_configured_control_body()
        if self._arm_action_term is not None and phase not in {"align_gripper_to_cube", "close_gripper"}:
            self._arm_action_term.clear_direct_joint_position_target()
        if phase in {"approach_cube", "descend_to_cube", "close_gripper"}:
            action = super().get_action(env)
            if phase == "close_gripper":
                if self._axis_alignment_direct_joint_target is None:
                    self._capture_axis_alignment_hold(env)
                self._hold_axis_alignment_target()
            if phase == "close_gripper":
                self._update_gripper_settle(env)
            return action

        if phase == "align_gripper_to_cube":
            self._update_axis_alignment(env)
            _, pick_grasp_w = self._pickup_targets()
            self._current_target_w = pick_grasp_w.detach().clone()
            return self._compose_pose_action(env, pick_grasp_w, _GRIPPER_OPEN)

        if phase in {"settle_at_grasp_target", "recenter_after_alignment"}:
            _, pick_grasp_w = self._pickup_targets()
            if phase == "settle_at_grasp_target":
                action = self._compose_pose_action(env, pick_grasp_w, _GRIPPER_OPEN)
            else:
                if self._aligned_gripper_quat_w is None:
                    raise RuntimeError("Aligned gripper quaternion was not captured before recenter")
                if self._postalign_recenter_target_w is None:
                    self._capture_postalign_recenter_target(env, pick_grasp_w)
                assert self._postalign_recenter_target_w is not None
                pick_grasp_w = self._postalign_recenter_target_w
                action = self._compose_pose_action_with_quaternion(
                    env, pick_grasp_w, self._aligned_gripper_quat_w, _GRIPPER_OPEN
                )
            self._update_pickup_position_settle(env, pick_grasp_w)
            return action

        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a polar AutoGen expert action")
        if phase not in position_only_accumulation_phases:
            self._arm_action_term.set_orientation_weight(weight=1.0)
        self._initialize_anchors(env)

        if phase == "retreat_to_safe":
            self._initialize_polar_retreat(env)
            self._advance_retreat_segment_if_ready(env)
            self._configure_wrist_position_posture_mode()
            if self._detect_polar_grasp_loss(env, phase):
                target_w = self._retreat_control_position_w(env).detach().clone()
            else:
                target_w = self._retreat_linear_reference()
            target_quat_w = self._retreat_control_quaternion_w(env).detach().clone()
            gripper = _GRIPPER_CLOSE
        elif phase == "arc_transfer":
            self._initialize_arc_transfer(env)
            self._configure_wrist_position_posture_mode()
            if self._detect_polar_grasp_loss(env, phase):
                target_w = self._retreat_control_position_w(env).detach().clone()
                target_quat_w = self._retreat_control_quaternion_w(env).detach().clone()
            else:
                target_w, target_quat_w = self._arc_reference(env)
            gripper = _GRIPPER_CLOSE
        elif phase == "radial_transfer":
            self._initialize_radial_transfer(env)
            self._configure_gripper_position_posture_mode()
            if self._detect_polar_grasp_loss(env, phase):
                target_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
            else:
                target_w = self._polar_linear_reference(_RADIAL_REFERENCE_STEP)
            assert self._placement_quat_w is not None
            target_quat_w = self._placement_quat_w
            gripper = _GRIPPER_CLOSE
        elif phase == "lower_into_box":
            self._initialize_lower(env)
            self._configure_lower_position_posture_mode()
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
        elif phase == "lower_into_box" and not self._episode_done:
            self._update_lower_convergence(env)
        elif phase == "retract_gripper" and not self._episode_done:
            self._update_vertical_convergence(env, phase)
        elif phase in self._MOTION_PHASE_LIMITS and not self._episode_done:
            self._update_motion_convergence(env, phase)
        return self._compose_pose_action_with_quaternion(env, target_w, target_quat_w, gripper)

    def advance(self) -> None:
        """Wait for a physically settled grasp before starting the lift."""

        if self._episode_done:
            return
        phase = self.phase_name
        if phase in {"settle_at_grasp_target", "recenter_after_alignment"}:
            self._step_count += 1
            self._phase_step += 1
            if self._pickup_settle_streak >= _PICKUP_SETTLE_STABLE_STEPS:
                self._pickup_settle_streak = 0
                self._advance_phase()
            else:
                timeout = (
                    _PREALIGN_SETTLE_TIMEOUT_STEPS
                    if phase == "settle_at_grasp_target"
                    else _POSTALIGN_RECENTER_TIMEOUT_STEPS
                )
                if self._phase_step >= timeout:
                    reason = f"{phase}_timeout:position_error={self._pickup_position_error}"
                    self._servo_abort_reason = reason
                    self._release_block_reason = reason
                    self._episode_done = True
            return
        if phase == "align_gripper_to_cube":
            self._step_count += 1
            self._phase_step += 1
            if self._axis_alignment_complete:
                self._axis_alignment_direct_joint_target = None
                self._advance_phase()
            elif self._phase_step >= _AXIS_ALIGNMENT_TIMEOUT_STEPS:
                reason = (
                    "gripper_axis_alignment_timeout:"
                    f"error={self.axis_alignment_error}:"
                    f"stable_streak={self._axis_alignment_streak}"
                )
                self._servo_abort_reason = reason
                self._release_block_reason = reason
                self._episode_done = True
            return
        if phase != "close_gripper":
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
                f"minimum_jaw_cube_distance={self._minimum_jaw_cube_distance}:"
                f"grasp_geometry_latched={self._grasp_geometry_latched}:"
                f"angle_span={self._gripper_settle_angle_span}:"
                f"pick_feedback_streak={self._pick_feedback_streak}:"
                f"stable_streak={self._gripper_settle_streak}"
            )
            self._servo_abort_reason = reason
            self._release_block_reason = reason
            self._episode_done = True

    def _advance_phase(self) -> None:
        if self.phase_name == "radial_transfer":
            self._disable_position_joint_target_accumulation()
        if self.phase_name == "retreat_to_safe" and self._arm_action_term is not None:
            self._arm_action_term.restore_configured_control_body()
        super()._advance_phase()
        self._current_target_quat_w = None
        self._bearing_error = None
        self._reference_finished = False

    def observe_pick_cube(self, pick_cube: bool | torch.Tensor, env) -> bool:
        """Debounce the task's geometric grasp signal during the close phase."""

        del env
        picked = bool(pick_cube.all().item()) if isinstance(pick_cube, torch.Tensor) else bool(pick_cube)
        if self.phase_name != "close_gripper":
            self._pick_feedback_streak = 0
            return False
        if picked:
            self._pick_feedback_streak += 1
        else:
            self._pick_feedback_streak = 0
        if self._pick_feedback_streak >= _PICK_FEEDBACK_STABLE_STEPS:
            self._pick_feedback_confirmed = True
        # The runner interprets True as "captured a held gripper angle". Polar
        # only consumes the debounced Boolean and keeps commanding nominal close.
        return False

    def _capture_axis_alignment_hold(self, env) -> None:
        assert self._arm_action_term is not None
        robot = env.scene["robot"]
        robot_joint_names = list(robot.data.joint_names)
        controlled_joint_names = self._arm_action_term.controlled_joint_names
        if "wrist_roll" not in controlled_joint_names:
            raise RuntimeError(f"wrist_roll is not controlled by the arm action: {controlled_joint_names}")
        controlled_joint_indices = tuple(robot_joint_names.index(name) for name in controlled_joint_names)
        wrist_roll_control_index = controlled_joint_names.index("wrist_roll")
        direct_target = robot.data.joint_pos[:, list(controlled_joint_indices)].detach().clone()
        self._axis_alignment_controlled_joint_indices = controlled_joint_indices
        self._axis_alignment_wrist_roll_control_index = wrist_roll_control_index
        self._arm_action_term.set_direct_joint_position_target(direct_target)
        applied_target = self._arm_action_term.direct_joint_position_target
        if applied_target is None:
            raise RuntimeError("The arm action did not retain the axis-alignment joint target")
        self._axis_alignment_direct_joint_target = applied_target.detach().clone()
        self._axis_alignment_wrist_roll_target = applied_target[:, wrist_roll_control_index].detach().clone()

    def _hold_axis_alignment_target(self) -> None:
        assert self._arm_action_term is not None
        if self._axis_alignment_direct_joint_target is None:
            raise RuntimeError("Close phase entered without a completed wrist-roll alignment target")
        self._arm_action_term.set_direct_joint_position_target(self._axis_alignment_direct_joint_target)

    def _update_axis_alignment(self, env) -> None:
        assert self._arm_action_term is not None
        if self._axis_alignment_direct_joint_target is None:
            self._capture_axis_alignment_hold(env)
        assert self._axis_alignment_direct_joint_target is not None
        assert self._axis_alignment_controlled_joint_indices is not None
        assert self._axis_alignment_wrist_roll_control_index is not None

        robot = env.scene["robot"]
        joint_indices = list(self._axis_alignment_controlled_joint_indices)
        joint_pos = robot.data.joint_pos[:, joint_indices]
        joint_vel = torch.abs(robot.data.joint_vel[:, joint_indices])
        wrist_roll_position = joint_pos[:, self._axis_alignment_wrist_roll_control_index]
        self._axis_alignment_wrist_roll_position = wrist_roll_position.detach().clone()
        self._axis_alignment_wrist_roll_velocity = joint_vel[:, self._axis_alignment_wrist_roll_control_index].detach()
        self._axis_alignment_max_arm_joint_velocity = torch.amax(joint_vel, dim=-1).detach()

        if self._axis_alignment_selected_index is None:
            hold_stable = bool(
                (
                    (self._axis_alignment_wrist_roll_velocity <= _AXIS_ALIGNMENT_WRIST_ROLL_VELOCITY_TOLERANCE)
                    & (self._axis_alignment_max_arm_joint_velocity <= _AXIS_ALIGNMENT_MAX_ARM_VELOCITY)
                )
                .all()
                .item()
            )
            self._axis_alignment_hold_settle_streak = self._axis_alignment_hold_settle_streak + 1 if hold_stable else 0
            if self._axis_alignment_hold_settle_streak >= _AXIS_ALIGNMENT_HOLD_SETTLE_STEPS:
                alignment = self._measure_axis_alignment(env, select_axis=True)
                if not bool(alignment.selection_feasible.all().item()):
                    reason = "no_cube_xy_axis_reachable_with_wrist_roll"
                    self._servo_abort_reason = reason
                    self._release_block_reason = reason
                    self._episode_done = True
            self._hold_axis_alignment_target()
            return

        alignment = self._measure_axis_alignment(env, select_axis=False)
        target_step = torch.clamp(
            _AXIS_ALIGNMENT_TARGET_KP * alignment.signed_error,
            min=-_AXIS_ALIGNMENT_MAX_TARGET_STEP,
            max=_AXIS_ALIGNMENT_MAX_TARGET_STEP,
        )
        previous_target = self._axis_alignment_direct_joint_target[:, self._axis_alignment_wrist_roll_control_index]
        wrist_roll_target = torch.clamp(
            previous_target + target_step,
            min=wrist_roll_position - _AXIS_ALIGNMENT_MAX_TARGET_LEAD,
            max=wrist_roll_position + _AXIS_ALIGNMENT_MAX_TARGET_LEAD,
        )
        wrist_roll_robot_index = joint_indices[self._axis_alignment_wrist_roll_control_index]
        wrist_roll_limits = robot.data.soft_joint_pos_limits[:, wrist_roll_robot_index]
        wrist_roll_target = torch.clamp(
            wrist_roll_target,
            min=wrist_roll_limits[:, 0] + _AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN,
            max=wrist_roll_limits[:, 1] - _AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN,
        )
        self._axis_alignment_direct_joint_target[:, self._axis_alignment_wrist_roll_control_index] = wrist_roll_target
        self._axis_alignment_wrist_roll_target = wrist_roll_target.detach().clone()
        self._hold_axis_alignment_target()

        aligned = bool((alignment.absolute_error <= _AXIS_ALIGNMENT_TOLERANCE).all().item())
        slow = bool(
            (
                (self._axis_alignment_wrist_roll_velocity <= _AXIS_ALIGNMENT_WRIST_ROLL_VELOCITY_TOLERANCE)
                & (self._axis_alignment_max_arm_joint_velocity <= _AXIS_ALIGNMENT_MAX_ARM_VELOCITY)
            )
            .all()
            .item()
        )
        self._axis_alignment_streak = self._axis_alignment_streak + 1 if aligned and slow else 0
        self._axis_alignment_complete = self._axis_alignment_streak >= _AXIS_ALIGNMENT_STABLE_STEPS
        if self._axis_alignment_complete:
            self._aligned_gripper_quat_w = env.scene["ee_frame"].data.target_quat_w[:, 0].detach().clone()

    def _update_pickup_position_settle(self, env, target_w: torch.Tensor) -> None:
        actual_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        error = torch.linalg.vector_norm(actual_w - target_w, dim=-1)
        self._pickup_position_error = float(error.max().item())
        tolerance = (
            _PREALIGN_POSITION_TOLERANCE
            if self.phase_name == "settle_at_grasp_target"
            else _POSTALIGN_POSITION_TOLERANCE
        )
        stable = bool((error <= tolerance).all().item())
        self._pickup_settle_streak = self._pickup_settle_streak + 1 if stable else 0

    def _capture_postalign_recenter_target(self, env, nominal_gripper_target_w: torch.Tensor) -> None:
        """Rotate the calibrated gripper-to-grasp offset with the aligned closing axis.

        The LeIsaac jaw detection frame is the distal point of the moving jaw,
        not the center of the open grasp gap.  It must therefore not be driven
        to the cube center.  The successful unrotated pickup was calibrated at
        ``cube_xy - 20 mm * gripper_local_+x``.  After wrist-roll alignment,
        rotate that same calibration with the measured local +X closing axis.
        """

        if self._axis_alignment_closing_axis_w is None:
            raise RuntimeError("Closing axis was not measured before post-alignment recenter")
        cube_w = env.scene["cube"].data.root_pos_w
        closing_axis_xy = self._axis_alignment_closing_axis_w[:, :2]
        closing_axis_xy = closing_axis_xy / torch.clamp(
            torch.linalg.vector_norm(closing_axis_xy, dim=-1, keepdim=True), min=1.0e-8
        )
        target_w = nominal_gripper_target_w.clone()
        target_w[:, :2] = cube_w[:, :2] - _PICK_GRIPPER_OFFSET_ALONG_CLOSING_AXIS * closing_axis_xy
        self._postalign_recenter_target_w = target_w.detach().clone()

    def _measure_axis_alignment(self, env, *, select_axis: bool):
        gripper_quat_w = env.scene["ee_frame"].data.target_quat_w[:, 0]
        cube_quat_w = env.scene["cube"].data.root_quat_w
        basis = torch.eye(3, device=env.device, dtype=gripper_quat_w.dtype)
        local_x = basis[0].repeat(env.num_envs, 1)
        local_y = basis[1].repeat(env.num_envs, 1)
        local_z = basis[2].repeat(env.num_envs, 1)
        closing_axis_w = quat_apply(gripper_quat_w, local_x)
        roll_axis_w = quat_apply(gripper_quat_w, local_z)
        cube_x_axis_w = quat_apply(cube_quat_w, local_x)
        cube_y_axis_w = quat_apply(cube_quat_w, local_y)
        if select_axis:
            assert self._axis_alignment_controlled_joint_indices is not None
            assert self._axis_alignment_wrist_roll_control_index is not None
            robot = env.scene["robot"]
            wrist_roll_robot_index = self._axis_alignment_controlled_joint_indices[
                self._axis_alignment_wrist_roll_control_index
            ]
            alignment = select_nearest_cube_axis_alignment(
                closing_axis_w,
                roll_axis_w,
                cube_x_axis_w,
                cube_y_axis_w,
                joint_position=robot.data.joint_pos[:, wrist_roll_robot_index],
                joint_limits=robot.data.soft_joint_pos_limits[:, wrist_roll_robot_index],
                joint_limit_margin=_AXIS_ALIGNMENT_JOINT_LIMIT_MARGIN,
            )
            self._axis_alignment_selected_index = alignment.selected_axis_index.detach().clone()
            self._axis_alignment_selected_sign = alignment.selected_axis_sign.detach().clone()
            self._axis_alignment_selected_cube_axis = tuple(
                f"{'+' if sign >= 0.0 else '-'}cube_local_{'x' if index == 0 else 'y'}"
                for index, sign in zip(
                    self._axis_alignment_selected_index.cpu().tolist(),
                    self._axis_alignment_selected_sign.cpu().tolist(),
                    strict=True,
                )
            )
        else:
            assert self._axis_alignment_selected_index is not None
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

    def _retreat_control_position_w(self, env) -> torch.Tensor:
        assert self._wrist_body_index is not None
        return env.scene["robot"].data.body_pos_w[:, self._wrist_body_index]

    def _retreat_control_quaternion_w(self, env) -> torch.Tensor:
        assert self._wrist_body_index is not None
        return env.scene["robot"].data.body_quat_w[:, self._wrist_body_index]

    def _configure_wrist_position_posture_mode(self) -> None:
        assert self._arm_action_term is not None
        assert self._wrist_posture_target is not None
        self._arm_action_term.set_control_body(body_name="wrist")
        self._arm_action_term.set_position_only_nullspace_posture_target(
            joint_target=self._wrist_posture_target,
            damping=_WRIST_POSTURE_DAMPING,
            posture_gain=_WRIST_POSTURE_GAIN,
            max_posture_step=_WRIST_POSTURE_MAX_STEP,
        )

    def _configure_gripper_position_posture_mode(self) -> None:
        assert self._arm_action_term is not None
        assert self._radial_posture_target is not None
        self._arm_action_term.set_control_body(body_name="gripper")
        self._arm_action_term.set_position_only_nullspace_posture_target(
            joint_target=self._radial_posture_target,
            damping=_WRIST_POSTURE_DAMPING,
            posture_gain=_WRIST_POSTURE_GAIN,
            max_posture_step=_WRIST_POSTURE_MAX_STEP,
        )

    def _configure_lower_position_posture_mode(self) -> None:
        assert self._arm_action_term is not None
        assert self._lower_posture_target is not None
        self._arm_action_term.set_control_body(body_name="gripper")
        self._arm_action_term.set_position_only_nullspace_posture_target(
            joint_target=self._lower_posture_target,
            damping=_WRIST_POSTURE_DAMPING,
            posture_gain=_WRIST_POSTURE_GAIN,
            max_posture_step=_WRIST_POSTURE_MAX_STEP,
        )

    def _enable_position_joint_target_accumulation(self) -> None:
        assert self._arm_action_term is not None
        if self._position_joint_target_accumulation_enabled:
            return
        self._arm_action_term.set_joint_target_accumulation(maximum_step=_WRIST_JOINT_TARGET_ACCUMULATION_STEP)
        self._arm_action_term.reset_joint_target_accumulation_reference()
        self._position_joint_target_accumulation_enabled = True

    def _disable_position_joint_target_accumulation(self) -> None:
        if self._arm_action_term is None or not self._position_joint_target_accumulation_enabled:
            return
        self._arm_action_term.set_joint_target_accumulation(maximum_step=None)
        self._position_joint_target_accumulation_enabled = False

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
        halfway_closed = gripper_position <= 0.5 * (self._gripper_open_position + STATE_MACHINE_GRIPPER_CLOSE_POSITION)
        close_enough_to_cube = jaw_cube_distance <= _GRASP_CONFIRM_DISTANCE
        current_distance = float(jaw_cube_distance.max().item())
        if self._minimum_jaw_cube_distance is None:
            self._minimum_jaw_cube_distance = current_distance
        else:
            self._minimum_jaw_cube_distance = min(self._minimum_jaw_cube_distance, current_distance)
        if bool(close_enough_to_cube.all().item()):
            self._grasp_geometry_latched = True
        self._gripper_settle_angle_window.append(float(gripper_position.max().item()))
        del self._gripper_settle_angle_window[:-_GRIPPER_SETTLE_WINDOW_STEPS]
        window_ready = len(self._gripper_settle_angle_window) >= _GRIPPER_SETTLE_WINDOW_STEPS
        if window_ready:
            self._gripper_settle_angle_span = max(self._gripper_settle_angle_window) - min(
                self._gripper_settle_angle_window
            )
        else:
            self._gripper_settle_angle_span = None
        grasp_geometry_confirmed = self._grasp_geometry_latched or self._pick_feedback_confirmed
        aperture_stable = window_ready and (
            self._gripper_settle_angle_span is not None
            and self._gripper_settle_angle_span <= _GRIPPER_SETTLE_ANGLE_SPAN_TOLERANCE
        )
        settled = halfway_closed & grasp_geometry_confirmed & aperture_stable

        self._jaw_cube_distance = float(jaw_cube_distance.max().item())
        self._gripper_target_error = float(target_error.max().item())
        self._gripper_joint_velocity = float(gripper_velocity.max().item())
        if self._phase_step >= _GRIPPER_CLOSE_MINIMUM_STEPS and bool(settled.all().item()):
            self._gripper_settle_streak += 1
            self._gripper_settle_reason = "confirmed_grasp_and_stable_aperture_window"
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
        self._grasp_confirmed = (
            self._grasp_geometry_latched
            or self._pick_feedback_confirmed
            or self._jaw_cube_distance <= _GRASP_CONFIRM_DISTANCE
        )
        assert self._arm_action_term is not None
        robot = env.scene["robot"]
        controlled_joint_indices = [
            robot.data.joint_names.index(name) for name in self._arm_action_term.controlled_joint_names
        ]
        self._wrist_posture_target = robot.data.joint_pos[:, controlled_joint_indices].detach().clone()
        self._enable_position_joint_target_accumulation()
        self._arm_action_term.reset_joint_target_accumulation_reference()
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
            # Establish the safer radius without accepting a different ray.
            # The relaxed bearing gate allows tracking error but rejects the
            # large sideways branch previously selected by unconstrained IK.
            self._target_error = max(self._retreat_radial_error, self._retreat_z_error)
            reached = (
                self._reference_finished
                and self._retreat_radial_error <= tolerance
                and self._retreat_z_error <= tolerance
                and self._bearing_error <= _RETREAT_BEARING_TOLERANCE
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
        start_w = self._retreat_control_position_w(env).detach().clone()
        robot_root_w = env.scene["robot"].data.root_pos_w
        start_delta_xy = start_w[:, :2] - robot_root_w[:, :2]
        box_delta_xy = self._floor_anchor_w[:, :2] - robot_root_w[:, :2]
        radius = torch.linalg.vector_norm(start_delta_xy, dim=-1)
        start_bearing = self._bearing(start_delta_xy)
        target_bearing = self._bearing(box_delta_xy)

        target_w = start_w.clone()
        target_w[:, 0] = robot_root_w[:, 0] + radius * torch.sin(target_bearing)
        target_w[:, 1] = robot_root_w[:, 1] + radius * torch.cos(target_bearing)
        target_w[:, 2] = start_w[:, 2]

        self._arc_radius = radius.detach().clone()
        self._arc_start_bearing = start_bearing.detach().clone()
        self._arc_target_bearing = target_bearing.detach().clone()
        # Radial transfer returns to gripper control, so it captures the actual
        # gripper height at that handoff instead of reusing the wrist height.
        self._transport_height = None
        self._placement_quat_w = None
        self._retreat_subphase = "root_centered_arc"
        self._reference_finished = False
        self._set_motion(start_w, target_w)

    def _initialize_radial_transfer(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        target_w = self._floor_anchor_w.clone()
        jaw_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :].detach()
        jaw_from_gripper_xy = jaw_w[:, :2] - start_w[:, :2]
        target_w[:, :2] -= jaw_from_gripper_xy
        if self._transport_height is None:
            self._transport_height = start_w[:, 2].detach().clone()
        target_w[:, 2] = self._transport_height
        if self._placement_quat_w is None:
            self._placement_quat_w = env.scene["ee_frame"].data.target_quat_w[:, 0, :].detach().clone()
        assert self._arm_action_term is not None
        robot = env.scene["robot"]
        controlled_joint_indices = [
            robot.data.joint_names.index(name) for name in self._arm_action_term.controlled_joint_names
        ]
        self._radial_posture_target = robot.data.joint_pos[:, controlled_joint_indices].detach().clone()
        self._enable_position_joint_target_accumulation()
        self._arm_action_term.reset_joint_target_accumulation_reference()
        self._retreat_subphase = "box_bearing_radial_approach"
        self._reference_finished = False
        self._set_motion(start_w, target_w)

    def _initialize_lower(self, env) -> None:
        if self._motion_start_w is not None:
            return
        super()._initialize_lower(env)
        assert self._motion_start_w is not None
        assert self._motion_target_w is not None
        # Radial transfer aligned the jaw/cube with the box. Descend from the
        # achieved gripper XY instead of moving the gripper origin to the box
        # center and undoing that alignment.
        self._motion_target_w[:, :2] = self._motion_start_w[:, :2]
        assert self._arm_action_term is not None
        robot = env.scene["robot"]
        controlled_joint_indices = [
            robot.data.joint_names.index(name) for name in self._arm_action_term.controlled_joint_names
        ]
        self._lower_posture_target = robot.data.joint_pos[:, controlled_joint_indices].detach().clone()
        self._enable_position_joint_target_accumulation()
        self._arm_action_term.reset_joint_target_accumulation_reference()

    def _initialize_release(self, env) -> None:
        if self._motion_target_w is not None:
            return
        actual_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        self._set_motion(actual_w, actual_w)

    def _initialize_retract(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        target_w = start_w.clone()
        target_w[:, 2] = self._floor_anchor_w[:, 2] + _BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER
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
        # Arc orientation is deliberately unconstrained.  Keep the unused pose
        # command aligned with the current wrist orientation for clear logging.
        target_quat_w = self._retreat_control_quaternion_w(env).detach().clone()
        self._reference_finished = bool(torch.all(normalized_time >= 1.0).item())
        return target_w, target_quat_w

    def _update_polar_convergence(self, env, phase: str) -> None:
        assert self._motion_target_w is not None
        if phase == "arc_transfer":
            actual_w = self._retreat_control_position_w(env)
            self._retreat_actual_w = actual_w.detach().clone()
        else:
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

    def _update_lower_convergence(self, env) -> None:
        """Accept the safe release height without blocking on an IK-induced XY compromise."""

        self._update_vertical_convergence(env, "lower_into_box")

    def _update_vertical_convergence(self, env, phase: str) -> None:
        """Complete a vertical placement motion from its actual handoff XY."""

        assert self._motion_target_w is not None
        actual_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        z_error = torch.abs(self._motion_target_w[:, 2] - actual_w[:, 2])
        self._target_error = float(z_error.max().item())
        _, _, tolerance, stable_steps = self._MOTION_PHASE_LIMITS[phase]
        if self._target_error <= tolerance:
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
                "approach,descend,measured_grasp_settle,wrist_roll_cube_axis_align,recenter,close,"
                "vertical_lift,root_relative_5_over_7_retreat,"
                "arc_transfer,radial_transfer,lower,release,retract,settle"
            ),
            "retreat_radial_scale": _RETREAT_RADIAL_SCALE,
            "retreat_reference_step": _RETREAT_REFERENCE_STEP,
            "arc_reference_step": _ARC_REFERENCE_STEP,
            "radial_reference_step": _RADIAL_REFERENCE_STEP,
            "wrist_safe_height_above_floor_center": _WRIST_SAFE_HEIGHT_ABOVE_FLOOR_CENTER,
            "transport_height_policy": "freeze_actual_safe_height_at_arc_entry",
            "bearing_tolerance_rad": _BEARING_TOLERANCE,
            "retreat_bearing_tolerance_rad": _RETREAT_BEARING_TOLERANCE,
            "retreat_control_body": "wrist",
            "retreat_ik_mode": "wrist_xyz_plus_soft_entry_joint_posture_in_nullspace",
            "retreat_task_error_order": "wrist_x_m,wrist_y_m,wrist_z_m",
            "retreat_posture_policy": "soft_all_arm_joints_at_retreat_entry_in_xyz_nullspace",
            "retreat_posture_damping": _WRIST_POSTURE_DAMPING,
            "retreat_posture_gain": _WRIST_POSTURE_GAIN,
            "retreat_posture_max_step": _WRIST_POSTURE_MAX_STEP,
            "wrist_joint_target_policy": "accumulate_limited_ik_delta_independent_of_live_joint_drift",
            "wrist_joint_target_accumulation_step": _WRIST_JOINT_TARGET_ACCUMULATION_STEP,
            "retreat_wrist_flex_policy": "soft_nullspace_only_not_a_hard_task_row",
            "retreat_y_policy": "constrained_during_lift_and_root_relative_retreat",
            "retreat_path": "wrist_vertical_lift_then_root_relative_xy_scaled_to_5_over_7",
            "arc_control_body": "wrist",
            "arc_ik_mode": "wrist_xyz_plus_soft_retreat_entry_joint_posture_in_nullspace",
            "arc_path": "root_centered_constant_radius_wrist_arc",
            "radial_control_body": "gripper",
            "radial_ik_mode": "gripper_xyz_plus_soft_handoff_joint_posture_in_nullspace",
            "radial_target_policy": "box_center_minus_live_jaw_from_gripper_xy_offset",
            "lower_control_body": "gripper",
            "lower_ik_mode": "gripper_xyz_plus_soft_lower_entry_joint_posture_in_nullspace",
            "lower_path": "vertical_from_actual_jaw_aligned_gripper_xy",
            "lower_completion_policy": "actual_gripper_z_within_tolerance",
            "release_policy": "open_at_actual_lower_handoff_pose",
            "retract_path": "vertical_from_actual_release_xy",
            "pre_retreat_gripper_gate": "confirmed_grasp_and_stable_aperture_window",
            "pickup_ik_mode": "full_6d_pose_preserving_known_grasp_geometry",
            "pickup_completion_policy": "known_fixed_keyframe_timing",
            "pickup_axis_alignment": "direct_wrist_roll_to_nearest_signed_cube_xy_axis",
            "pickup_axis_alignment_tolerance_rad": _AXIS_ALIGNMENT_TOLERANCE,
            "pickup_axis_alignment_max_target_step_rad": _AXIS_ALIGNMENT_MAX_TARGET_STEP,
            "pickup_recenter_policy": "live_cube_xy_minus_rotated_calibrated_gripper_closing_axis_offset",
            "pickup_gripper_offset_along_closing_axis": _PICK_GRIPPER_OFFSET_ALONG_CLOSING_AXIS,
            "gripper_close_minimum_steps": _GRIPPER_CLOSE_MINIMUM_STEPS,
            "gripper_close_maximum_steps": _GRIPPER_CLOSE_MAXIMUM_STEPS,
            "gripper_settle_stable_steps": _GRIPPER_SETTLE_STABLE_STEPS,
            "gripper_settle_window_steps": _GRIPPER_SETTLE_WINDOW_STEPS,
            "gripper_settle_angle_span_tolerance_rad": _GRIPPER_SETTLE_ANGLE_SPAN_TOLERANCE,
            "pick_feedback_stable_steps": _PICK_FEEDBACK_STABLE_STEPS,
            "retreat_bearing_policy": "preserved_by_xy_scaling_and_checked_with_relaxed_10deg_gate",
            "retreat_z_overshoot_limit": _RETREAT_Z_OVERSHOOT_LIMIT,
            "retreat_error_worsening_margin": _RETREAT_ERROR_WORSENING_MARGIN,
            "retreat_error_worsening_steps": _RETREAT_ERROR_WORSENING_STEPS,
            "orientation_policy": "no_world_orientation_task_with_soft_joint_posture_nullspace",
            "completion_policy": (
                "vertical_lift_actual_wrist_z_only_then_radial_retreat_actual_wrist_radius_z_and_relaxed_bearing"
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
    def wrist_posture_target(self) -> torch.Tensor | None:
        return self._wrist_posture_target

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
    def gripper_settle_angle_span(self) -> float | None:
        return self._gripper_settle_angle_span

    @property
    def gripper_settle_streak(self) -> int:
        return self._gripper_settle_streak

    @property
    def gripper_settle_reason(self) -> str | None:
        return self._gripper_settle_reason

    @property
    def pick_feedback_streak(self) -> int:
        return self._pick_feedback_streak

    @property
    def axis_alignment_complete(self) -> bool:
        return self._axis_alignment_complete

    @property
    def axis_alignment_streak(self) -> int:
        return self._axis_alignment_streak

    @property
    def axis_alignment_final_gate_streak(self) -> int:
        return self._axis_alignment_streak

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
    def axis_alignment_direct_joint_target(self) -> torch.Tensor | None:
        return self._axis_alignment_direct_joint_target

    @property
    def grasp_geometry_latched(self) -> bool:
        return self._grasp_geometry_latched

    @property
    def minimum_jaw_cube_distance(self) -> float | None:
        return self._minimum_jaw_cube_distance
