"""Align high with controlled XYZ, controlled tilt, and free world yaw."""

from __future__ import annotations

from .legacy_gripper_anchor_safe_planar_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorSafePlanarAlignThenLowerStateMachine,
)


class RedCubeToBoxLegacyGripperAnchorSafeXyzTiltAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorSafePlanarAlignThenLowerStateMachine
):
    """Control Z without returning to a six-row pose or three-row position task."""

    def _configure_high_align_ik(self) -> None:
        assert self._arm_action_term is not None
        self._arm_action_term.set_xyz_tilt(enabled=True)

    @property
    def _high_align_safety_mode(self) -> str:
        return "xyz_plus_world_tilt_free_yaw_physical_z_gates"

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "high_align_controller": "xyz_plus_world_roll_pitch",
                "high_align_z_policy": "controlled_at_measured_entry_height",
                "free_orientation_component": "world_yaw",
            }
        )
        return parameters
