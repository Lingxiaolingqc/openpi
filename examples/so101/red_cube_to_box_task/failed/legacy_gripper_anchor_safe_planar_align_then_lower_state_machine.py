"""Align high with XY-plus-orientation IK and physical Z-clearance gates."""

from __future__ import annotations

import torch

from ..env_cfg import CUBE_HALF_HEIGHT
from ..env_cfg import TARGET_BOX_OUTER_SIZE
from ..env_cfg import TARGET_BOX_WALL_TOP_Z
from ..phase_aware_ik_action import PhaseAwareDifferentialInverseKinematicsAction
from ..phase_aware_ik_action import resolve_action_term
from .legacy_gripper_anchor_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine,
)

_GRIPPER_CLOSE = -1.0
_MIN_CUBE_CLEARANCE = 0.015
_MIN_ROBOT_CLEARANCE = 0.010
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


class RedCubeToBoxLegacyGripperAnchorSafePlanarAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine
):
    """Keep legacy pickup/descent but use a safety-gated five-row high align."""

    def __init__(self) -> None:
        super().__init__()
        self._arm_action_term: PhaseAwareDifferentialInverseKinematicsAction | None = None
        self._servo_abort_reason: str | None = None
        self._safety_mode = "legacy_pose"
        self._cube_clearance = float("inf")
        self._minimum_robot_clearance = float("inf")
        self._maximum_box_contact_force = 0.0
        self._align_cube_z_reference: float | None = None
        self._align_cube_z_error = 0.0

    def setup(self, env) -> None:
        super().setup(env)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise TypeError(
                "legacy_gripper_anchor_safe_planar_align_then_lower requires "
                "PhaseAwareDifferentialInverseKinematicsAction"
            )
        self._arm_action_term = arm_action_term

    def get_action(self, env) -> torch.Tensor:
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a safe planar-IK action")

        phase_name, _, _ = self._phase_state()
        if phase_name != "align_over_box":
            self._arm_action_term.set_orientation_weight(weight=1.0)
            self._safety_mode = "legacy_pose"
            return super().get_action(env)

        cube_z = float(env.scene["cube"].data.root_pos_w[0, 2].item())
        if self._align_cube_z_reference is None:
            self._align_cube_z_reference = cube_z
        self._update_align_safety(env, cube_z)
        if self._servo_abort_reason is not None:
            self._episode_done = True
            return self._hold_current_pose(env)

        # The task has exactly five rows for the five arm joints: jaw X/Y and
        # all three orientation rows. Z is observed by the safety gate above,
        # but it is deliberately absent from the IK error and Jacobian.
        self._configure_high_align_ik()
        self._safety_mode = self._high_align_safety_mode
        return super().get_action(env)

    def _configure_high_align_ik(self) -> None:
        assert self._arm_action_term is not None
        self._arm_action_term.set_planar_pose(enabled=True)

    @property
    def _high_align_safety_mode(self) -> str:
        return "planar_xy_orientation_physical_z_gates"

    def reset(self) -> None:
        super().reset()
        self._servo_abort_reason = None
        self._safety_mode = "legacy_pose"
        self._cube_clearance = float("inf")
        self._minimum_robot_clearance = float("inf")
        self._maximum_box_contact_force = 0.0
        self._align_cube_z_reference = None
        self._align_cube_z_error = 0.0
        if self._arm_action_term is not None:
            self._arm_action_term.set_orientation_weight(weight=1.0)

    def _hold_current_pose(self, env) -> torch.Tensor:
        assert self._arm_action_term is not None
        ee_frame = env.scene["ee_frame"]
        self._arm_action_term.set_orientation_weight(weight=1.0)
        self._safety_mode = "abort_hold_pose"
        return self._compose_pose_action(
            env,
            env.scene["robot"],
            ee_frame.data.target_pos_w[:, 0, :].clone(),
            _GRIPPER_CLOSE,
            ee_frame.data.target_quat_w[:, 0, :].clone(),
        )

    def _update_align_safety(self, env, cube_z: float) -> None:
        assert self._align_cube_z_reference is not None
        floor_xy = env.scene["target_box_floor"].data.root_pos_w[0, :2]
        self._cube_clearance = cube_z - CUBE_HALF_HEIGHT - TARGET_BOX_WALL_TOP_Z
        self._align_cube_z_error = abs(cube_z - self._align_cube_z_reference)
        self._minimum_robot_clearance = self._robot_geometry_clearance(env, floor_xy)
        self._maximum_box_contact_force = self._box_contact_force(env)

        if self._maximum_box_contact_force > _CONTACT_FORCE_THRESHOLD:
            self._servo_abort_reason = "align_robot_box_contact_detected"
        elif self._cube_clearance < _MIN_CUBE_CLEARANCE:
            self._servo_abort_reason = "align_cube_clearance_below_safe_range"
        elif self._minimum_robot_clearance < _MIN_ROBOT_CLEARANCE:
            self._servo_abort_reason = "align_robot_clearance_below_safe_range"

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

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "high_align_controller": "planar_pose_xy_plus_orientation",
                "high_align_z_policy": "excluded_from_ik_physical_clearance_only",
                "minimum_cube_clearance": _MIN_CUBE_CLEARANCE,
                "minimum_robot_clearance": _MIN_ROBOT_CLEARANCE,
                "contact_force_threshold": _CONTACT_FORCE_THRESHOLD,
            }
        )
        return parameters

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "uninitialized"
        return self._arm_action_term.runtime_mode

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

    @property
    def align_cube_z_reference(self) -> float | None:
        return self._align_cube_z_reference

    @property
    def align_cube_z_error(self) -> float:
        return self._align_cube_z_error
