"""Align high with XYZ, an explicit shoulder-pan target, and joint-limit avoidance."""

from __future__ import annotations

from .legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorSafeXyzPitchPanAlignThenLowerStateMachine,
)

_NULLSPACE_DAMPING = 0.05
_NULLSPACE_POSTURE_GAIN = 0.08
_NULLSPACE_MAX_POSTURE_STEP = 0.03
_SHOULDER_PAN_JOINT = "shoulder_pan"


class RedCubeToBoxLegacyGripperAnchorSafeXyzPanNullspaceAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorSafeXyzPitchPanAlignThenLowerStateMachine
):
    """Drop hard pitch and spend the remaining DOF keeping joints away from limits."""

    def _configure_high_align_ik(self) -> None:
        assert self._arm_action_term is not None
        assert self._shoulder_pan_target is not None
        self._arm_action_term.set_xyz_joint_nullspace_target(
            joint_name=_SHOULDER_PAN_JOINT,
            joint_target=self._shoulder_pan_target,
            damping=_NULLSPACE_DAMPING,
            posture_gain=_NULLSPACE_POSTURE_GAIN,
            max_posture_step=_NULLSPACE_MAX_POSTURE_STEP,
        )

    @property
    def _high_align_safety_mode(self) -> str:
        return "xyz_plus_shoulder_pan_target_nullspace_joint_limit_avoidance_physical_z_gates"

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "high_align_controller": "xyz_plus_shoulder_pan_target_nullspace_joint_limit_avoidance",
                "retained_orientation_component": "none_hard",
                "secondary_objective": "normalized_soft_joint_limit_centering",
                "nullspace_damping": _NULLSPACE_DAMPING,
                "nullspace_posture_gain": _NULLSPACE_POSTURE_GAIN,
                "nullspace_max_posture_step": _NULLSPACE_MAX_POSTURE_STEP,
            }
        )
        return parameters
