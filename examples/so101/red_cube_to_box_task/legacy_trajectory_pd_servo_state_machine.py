"""Legacy pickup followed by smooth-reference PD Cartesian transport."""

from __future__ import annotations

import torch

from .legacy_pd_position_servo_state_machine import RedCubeToBoxLegacyPdPositionServoStateMachine

_REFERENCE_HOLD_STEPS = 60


class RedCubeToBoxLegacyTrajectoryPdServoStateMachine(RedCubeToBoxLegacyPdPositionServoStateMachine):
    """Track continuous transfer/lower references instead of endpoint steps."""

    def __init__(self) -> None:
        super().__init__()
        self._transfer_start_cube: torch.Tensor | None = None
        self._lower_start_cube: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self._transfer_start_cube = None
        self._lower_start_cube = None

    def _on_phase_entry(
        self,
        phase_name: str,
        cube_pos_w: torch.Tensor,
        gripper_pos_w: torch.Tensor,
        jaw_pos_w: torch.Tensor,
    ) -> None:
        super()._on_phase_entry(phase_name, cube_pos_w, gripper_pos_w, jaw_pos_w)
        if phase_name == "transfer_to_box":
            self._transfer_start_cube = cube_pos_w.clone()
        elif phase_name == "lower_into_box":
            self._lower_start_cube = cube_pos_w.clone()

    def _desired_transfer_cube(self, phase_step: int, phase_duration: int) -> torch.Tensor:
        assert self._transfer_start_cube is not None
        reference_duration = max(phase_duration - _REFERENCE_HOLD_STEPS, 1)
        return self._interpolate(
            self._transfer_start_cube,
            self._box_hover_cube(),
            phase_step,
            reference_duration,
        )

    def _desired_lower_cube(self, phase_step: int, phase_duration: int) -> torch.Tensor:
        assert self._lower_start_cube is not None
        reference_duration = max(phase_duration - _REFERENCE_HOLD_STEPS, 1)
        return self._interpolate(
            self._lower_start_cube,
            self._box_release_cube(),
            phase_step,
            reference_duration,
        )

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters["transport_controller"] = "trajectory_pd_position_servo"
        parameters["reference_mode"] = "phase_interpolation"
        parameters["reference_hold_steps"] = _REFERENCE_HOLD_STEPS
        return parameters
