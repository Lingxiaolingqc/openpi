"""Align high with XYZ, one tilt row, and an explicit shoulder-pan target."""

from __future__ import annotations

import torch

from .legacy_gripper_anchor_safe_xyz_tilt_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorSafeXyzTiltAlignThenLowerStateMachine,
)

_SHOULDER_PAN_JOINT = "shoulder_pan"
_MAX_BEARING_CORRECTION = 0.35
_JOINT_LIMIT_MARGIN = 0.02


class RedCubeToBoxLegacyGripperAnchorSafeXyzPitchPanAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorSafeXyzTiltAlignThenLowerStateMachine
):
    """Keep a persistent base-angle objective instead of relying on IK weighting."""

    def __init__(self) -> None:
        super().__init__()
        self._shoulder_pan_target: torch.Tensor | None = None
        self._shoulder_pan_entry: torch.Tensor | None = None
        self._bearing_error: torch.Tensor | None = None

    def get_action(self, env) -> torch.Tensor:
        phase_name, _, _ = self._phase_state()
        if phase_name == "align_over_box" and self._shoulder_pan_target is None:
            self._capture_shoulder_pan_target(env)
        return super().get_action(env)

    def reset(self) -> None:
        super().reset()
        self._shoulder_pan_target = None
        self._shoulder_pan_entry = None
        self._bearing_error = None

    def _configure_high_align_ik(self) -> None:
        assert self._arm_action_term is not None
        assert self._shoulder_pan_target is not None
        self._arm_action_term.set_xyz_pitch_joint_target(
            joint_name=_SHOULDER_PAN_JOINT,
            joint_target=self._shoulder_pan_target,
        )

    def _capture_shoulder_pan_target(self, env) -> None:
        self._initialize_anchors(env)
        assert self._floor_anchor is not None
        robot = env.scene["robot"]
        jaw_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        robot_xy = robot.data.root_pos_w[:, :2]
        current_vector = jaw_pos_w[:, :2] - robot_xy
        target_vector = self._floor_anchor[:, :2] - robot_xy
        current_bearing = torch.atan2(current_vector[:, 1], current_vector[:, 0])
        target_bearing = torch.atan2(target_vector[:, 1], target_vector[:, 0])
        raw_bearing_error = target_bearing - current_bearing
        self._bearing_error = torch.atan2(torch.sin(raw_bearing_error), torch.cos(raw_bearing_error))

        joint_index = robot.data.joint_names.index(_SHOULDER_PAN_JOINT)
        self._shoulder_pan_entry = robot.data.joint_pos[:, joint_index].clone()
        correction = torch.clamp(
            self._bearing_error,
            min=-_MAX_BEARING_CORRECTION,
            max=_MAX_BEARING_CORRECTION,
        )
        # The SO-101 shoulder-pan positive direction is opposite the audited
        # world-XY bearing direction, so subtract the geometric correction.
        target = self._shoulder_pan_entry - correction
        soft_limits = robot.data.soft_joint_pos_limits[:, joint_index]
        self._shoulder_pan_target = torch.clamp(
            target,
            min=soft_limits[:, 0] + _JOINT_LIMIT_MARGIN,
            max=soft_limits[:, 1] - _JOINT_LIMIT_MARGIN,
        )

    @property
    def _high_align_safety_mode(self) -> str:
        return "xyz_plus_pitch_plus_shoulder_pan_target_physical_z_gates"

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "high_align_controller": "xyz_plus_base_pitch_plus_shoulder_pan_target",
                "retained_orientation_component": "robot_base_pitch_y",
                "joint_objective": _SHOULDER_PAN_JOINT,
                "joint_target_strategy": "entry_angle_minus_world_bearing_error",
                "maximum_bearing_correction": _MAX_BEARING_CORRECTION,
            }
        )
        return parameters

    @property
    def shoulder_pan_target(self) -> torch.Tensor | None:
        return self._shoulder_pan_target

    @property
    def shoulder_pan_entry(self) -> torch.Tensor | None:
        return self._shoulder_pan_entry

    @property
    def bearing_error(self) -> torch.Tensor | None:
        return self._bearing_error
