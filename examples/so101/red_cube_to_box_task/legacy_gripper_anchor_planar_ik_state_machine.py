"""Jaw-anchored legacy placement with a collision-gated planar-pose IK."""

from __future__ import annotations

import torch

from .env_cfg import CUBE_HALF_HEIGHT
from .env_cfg import TARGET_BOX_OUTER_SIZE
from .env_cfg import TARGET_BOX_WALL_TOP_Z
from .legacy_gripper_anchor_state_machine import RedCubeToBoxLegacyGripperAnchorStateMachine
from .phase_aware_ik_action import PhaseAwareDifferentialInverseKinematicsAction
from .phase_aware_ik_action import resolve_action_term

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_MIN_CLEARANCE = 0.010
_RESUME_CLEARANCE = 0.015
_MAX_XY_STEP = 0.002
_MAX_LIFT_STEP = 0.002
_XY_ALIGNMENT_THRESHOLD = 0.006
_MIN_ALIGNMENT_STREAK = 10
_CONTACT_FORCE_THRESHOLD = 0.25
_BOX_CONTACT_SENSOR_NAMES = (
    "lower_arm_box_contact",
    "wrist_box_contact",
    "gripper_box_contact",
    "jaw_box_contact",
)
_BODY_BOUNDING_RADII = {
    "lower_arm": 0.075,
    "wrist": 0.050,
    "gripper": 0.060,
    "jaw": 0.040,
}
_GRASP_DISTANCE_THRESHOLD = 0.02
_GRIPPER_POSITION_THRESHOLD = 0.26
_GRASP_PROTECTED_PHASES = {
    "lift_cube",
    "transfer_to_box",
    "lower_into_box",
    "align_over_box",
}


class RedCubeToBoxLegacyGripperAnchorPlanarIkStateMachine(RedCubeToBoxLegacyGripperAnchorStateMachine):
    """Drop only the Z task row while preserving XY and orientation alignment."""

    def __init__(self) -> None:
        super().__init__()
        self._arm_action_term: PhaseAwareDifferentialInverseKinematicsAction | None = None
        self._grasp_confirmed = False
        self._grasp_lost_before_release = False
        self._servo_abort_reason: str | None = None
        self._safety_mode = "legacy_pose"
        self._cube_clearance = float("inf")
        self._minimum_robot_clearance = float("inf")
        self._maximum_box_contact_force = 0.0
        self._release_hold_pos_w: torch.Tensor | None = None
        self._release_hold_quat_w: torch.Tensor | None = None

    def setup(self, env) -> None:
        super().setup(env)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise TypeError(
                "legacy_gripper_anchor_planar_ik requires "
                "PhaseAwareDifferentialInverseKinematicsAction"
            )
        self._arm_action_term = arm_action_term

    def get_action(self, env) -> torch.Tensor:
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a planar-IK action")

        phase_name, phase_step, phase_duration = self._phase_state()
        self._update_safety_state(env)
        self._update_grasp_status(env, phase_name)
        if self._episode_done:
            return self._hold_current_pose(env, _GRIPPER_CLOSE)

        if phase_name == "align_over_box":
            return self._alignment_action(env)
        if phase_name == "release_cube":
            return self._release_hold_action(env)
        if phase_name in {"retract_gripper", "settle"}:
            return self._retract_action(env, phase_name, phase_step, phase_duration)

        self._arm_action_term.set_orientation_weight(weight=1.0)
        self._safety_mode = "legacy_pose"
        if phase_name == "lower_into_box" and not self._clearance_safe_for_motion():
            return self._raise_clearance_action(env)
        return super().get_action(env)

    def reset(self) -> None:
        super().reset()
        self._grasp_confirmed = False
        self._grasp_lost_before_release = False
        self._servo_abort_reason = None
        self._safety_mode = "legacy_pose"
        self._cube_clearance = float("inf")
        self._minimum_robot_clearance = float("inf")
        self._maximum_box_contact_force = 0.0
        self._release_hold_pos_w = None
        self._release_hold_quat_w = None
        if self._arm_action_term is not None:
            self._arm_action_term.set_orientation_weight(weight=1.0)

    def _alignment_action(self, env) -> torch.Tensor:
        assert self._arm_action_term is not None
        assert self._floor_anchor is not None
        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        gripper_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]

        if not self._clearance_safe_for_motion():
            self._alignment_streak = 0
            self._box_aligned_before_release = False
            return self._raise_clearance_action(env)

        self._arm_action_term.set_planar_pose(enabled=True)
        self._safety_mode = "planar_xy_orientation"
        raw_xy_error = self._floor_anchor[:, :2] - jaw_pos_w[:, :2]
        xy_error_norm = torch.linalg.vector_norm(raw_xy_error, dim=-1, keepdim=True)
        scale = torch.clamp(_MAX_XY_STEP / torch.clamp(xy_error_norm, min=1e-8), max=1.0)
        xy_delta = raw_xy_error * scale

        desired_jaw_w = jaw_pos_w.clone()
        desired_jaw_w[:, :2] += xy_delta
        jaw_error_w = desired_jaw_w - jaw_pos_w
        target_pos_w = gripper_pos_w + jaw_error_w
        self._set_placement_diagnostics(desired_jaw_w, jaw_error_w, target_pos_w)
        self._update_planar_alignment(env, raw_xy_error)
        return self._compose_pose_action(env, robot, target_pos_w, _GRIPPER_CLOSE)

    def _raise_clearance_action(self, env) -> torch.Tensor:
        assert self._arm_action_term is not None
        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        gripper_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]
        cube_lift = max(_RESUME_CLEARANCE - self._cube_clearance, 0.0)
        robot_lift = max(_MIN_CLEARANCE - self._minimum_robot_clearance, 0.0)
        lift_distance = min(max(cube_lift, robot_lift), _MAX_LIFT_STEP)
        desired_jaw_w = jaw_pos_w.clone()
        desired_jaw_w[:, 2] += lift_distance
        jaw_error_w = desired_jaw_w - jaw_pos_w
        target_pos_w = gripper_pos_w + jaw_error_w
        self._set_placement_diagnostics(desired_jaw_w, jaw_error_w, target_pos_w)
        self._arm_action_term.set_position_only(enabled=True)
        self._safety_mode = "raise_clearance"
        return self._compose_pose_action(env, robot, target_pos_w, _GRIPPER_CLOSE)

    def _release_hold_action(self, env) -> torch.Tensor:
        assert self._arm_action_term is not None
        if self._release_hold_pos_w is None or self._release_hold_quat_w is None:
            self._servo_abort_reason = "release_pose_was_not_captured"
            self._episode_done = True
            return self._hold_current_pose(env, _GRIPPER_CLOSE)
        self._arm_action_term.set_orientation_weight(weight=1.0)
        self._safety_mode = "release_hold"
        return self._compose_pose_action(
            env,
            env.scene["robot"],
            self._release_hold_pos_w,
            _GRIPPER_OPEN,
            self._release_hold_quat_w,
        )

    def _retract_action(self, env, phase_name: str, phase_step: int, phase_duration: int) -> torch.Tensor:
        assert self._arm_action_term is not None
        if self._release_hold_pos_w is None:
            self._servo_abort_reason = "release_pose_was_not_captured"
            self._episode_done = True
            return self._hold_current_pose(env, _GRIPPER_OPEN)
        target_pos_w = self._release_hold_pos_w.clone()
        if phase_name == "retract_gripper":
            target_pos_w[:, 2] += 0.12 * min((phase_step + 1) / phase_duration, 1.0)
        else:
            target_pos_w[:, 2] += 0.12
        self._arm_action_term.set_position_only(enabled=True)
        self._safety_mode = "open_retract"
        return self._compose_pose_action(env, env.scene["robot"], target_pos_w, _GRIPPER_OPEN)

    def _hold_current_pose(self, env, gripper: float) -> torch.Tensor:
        assert self._arm_action_term is not None
        ee_frame = env.scene["ee_frame"]
        self._arm_action_term.set_orientation_weight(weight=1.0)
        return self._compose_pose_action(
            env,
            env.scene["robot"],
            ee_frame.data.target_pos_w[:, 0, :].clone(),
            gripper,
            ee_frame.data.target_quat_w[:, 0, :].clone(),
        )

    def _update_safety_state(self, env) -> None:
        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        self._cube_clearance = float(
            (cube.data.root_pos_w[0, 2] - CUBE_HALF_HEIGHT - TARGET_BOX_WALL_TOP_Z).item()
        )
        self._minimum_robot_clearance = self._robot_geometry_clearance(env, floor.data.root_pos_w[0, :2])
        self._maximum_box_contact_force = self._box_contact_force(env)
        if self._maximum_box_contact_force > _CONTACT_FORCE_THRESHOLD:
            self._servo_abort_reason = "robot_box_contact_detected"
            self._episode_done = True

    @staticmethod
    def _robot_geometry_clearance(env, box_xy: torch.Tensor) -> float:
        robot = env.scene["robot"]
        outer_half = TARGET_BOX_OUTER_SIZE / 2.0
        minimum_clearance = float("inf")
        for body_name, radius in _BODY_BOUNDING_RADII.items():
            body_index = robot.data.body_names.index(body_name)
            body_pos_w = robot.data.body_pos_w[0, body_index]
            overlaps_x = abs(float(body_pos_w[0].item() - box_xy[0].item())) <= outer_half + radius
            overlaps_y = abs(float(body_pos_w[1].item() - box_xy[1].item())) <= outer_half + radius
            if overlaps_x and overlaps_y:
                clearance = float(body_pos_w[2].item()) - radius - TARGET_BOX_WALL_TOP_Z
                minimum_clearance = min(minimum_clearance, clearance)
        return minimum_clearance

    @staticmethod
    def _box_contact_force(env) -> float:
        maximum_force = 0.0
        for sensor_name in _BOX_CONTACT_SENSOR_NAMES:
            force_matrix_w = env.scene[sensor_name].data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError(f"Contact sensor {sensor_name} did not produce filtered force data")
            maximum_force = max(
                maximum_force,
                float(torch.linalg.vector_norm(force_matrix_w, dim=-1).max().item()),
            )
        return maximum_force

    def _clearance_safe_for_motion(self) -> bool:
        return (
            self._cube_clearance >= _RESUME_CLEARANCE
            and self._minimum_robot_clearance >= _MIN_CLEARANCE
            and self._maximum_box_contact_force <= _CONTACT_FORCE_THRESHOLD
        )

    def _update_planar_alignment(self, env, raw_xy_error: torch.Tensor) -> None:
        xy_aligned = torch.linalg.vector_norm(raw_xy_error, dim=-1) < _XY_ALIGNMENT_THRESHOLD
        safe = (
            self._cube_clearance >= _MIN_CLEARANCE
            and self._minimum_robot_clearance >= _MIN_CLEARANCE
            and self._maximum_box_contact_force <= _CONTACT_FORCE_THRESHOLD
        )
        if bool(xy_aligned.all().item()) and safe:
            self._alignment_streak += 1
        else:
            self._alignment_streak = 0
        self._box_aligned_before_release = self._alignment_streak >= _MIN_ALIGNMENT_STREAK
        if self._box_aligned_before_release and self._release_hold_pos_w is None:
            ee_frame = env.scene["ee_frame"]
            self._release_hold_pos_w = ee_frame.data.target_pos_w[:, 0, :].clone()
            self._release_hold_quat_w = ee_frame.data.target_quat_w[:, 0, :].clone()

    def _set_placement_diagnostics(
        self,
        desired_jaw_w: torch.Tensor,
        jaw_error_w: torch.Tensor,
        target_pos_w: torch.Tensor,
    ) -> None:
        self._last_desired_jaw_w = desired_jaw_w.clone()
        self._last_jaw_error_w = jaw_error_w.clone()
        self._last_gripper_target_w = target_pos_w.clone()

    def _update_grasp_status(self, env, phase_name: str) -> None:
        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        cube = env.scene["cube"]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]
        jaw_distance = torch.linalg.vector_norm(jaw_pos_w - cube.data.root_pos_w, dim=-1)
        gripper_joint_index = robot.data.joint_names.index("gripper")
        gripper_closed = robot.data.joint_pos[:, gripper_joint_index] < _GRIPPER_POSITION_THRESHOLD
        grasped = bool(torch.logical_and(jaw_distance < _GRASP_DISTANCE_THRESHOLD, gripper_closed).all().item())
        if grasped:
            self._grasp_confirmed = True
        elif self._grasp_confirmed and phase_name in _GRASP_PROTECTED_PHASES:
            self._grasp_lost_before_release = True
            self._servo_abort_reason = "grasp_lost_before_release"
            self._episode_done = True

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "ik_alignment_mode": "planar_pose_xy_plus_orientation",
                "minimum_cube_clearance": _MIN_CLEARANCE,
                "resume_cube_clearance": _RESUME_CLEARANCE,
                "maximum_xy_step": _MAX_XY_STEP,
                "maximum_lift_step": _MAX_LIFT_STEP,
                "contact_force_threshold": _CONTACT_FORCE_THRESHOLD,
                "grasp_loss_policy": "abort_before_release",
            }
        )
        return parameters

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "uninitialized"
        return self._arm_action_term.runtime_mode

    @property
    def grasp_confirmed(self) -> bool:
        return self._grasp_confirmed

    @property
    def grasp_lost_before_release(self) -> bool:
        return self._grasp_lost_before_release

    @property
    def servo_abort_reason(self) -> str | None:
        return self._servo_abort_reason

    @property
    def safety_mode(self) -> str:
        return self._safety_mode

    @property
    def cube_clearance(self) -> float:
        return self._cube_clearance

    @property
    def minimum_robot_clearance(self) -> float:
        return self._minimum_robot_clearance

    @property
    def maximum_box_contact_force(self) -> float:
        return self._maximum_box_contact_force
