"""Dynamic-grasp legacy expert with one frozen transfer-residual correction."""

from __future__ import annotations

import torch

from .legacy_dynamic_grasp_offset_state_machine import (
    RedCubeToBoxLegacyDynamicGraspOffsetStateMachine,
)

_MAXIMUM_RESIDUAL_CORRECTION = 0.10


class RedCubeToBoxLegacyDynamicGraspOffsetResidualCorrectedStateMachine(
    RedCubeToBoxLegacyDynamicGraspOffsetStateMachine
):
    """Correct the placement target once from the measured post-transfer cube error.

    The parent expert is unchanged through transfer. On the first lower-into-box
    control step, this comparison measures ``desired_cube_xy - actual_cube_xy``,
    caps that vector at 10 cm, adds it once to the parent gripper target, and
    freezes the corrected target through release and retraction. It deliberately
    does not accumulate feedback on every step.
    """

    def __init__(self) -> None:
        super().__init__()
        self._raw_transfer_residual_xy: torch.Tensor | None = None
        self._applied_transfer_residual_xy: torch.Tensor | None = None
        self._corrected_gripper_target_xy: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self._raw_transfer_residual_xy = None
        self._applied_transfer_residual_xy = None
        self._corrected_gripper_target_xy = None

    def _placement_gripper_target_xy(self, env, phase_name: str) -> torch.Tensor:
        base_target_xy = super()._placement_gripper_target_xy(env, phase_name)
        if phase_name == "transfer_to_box":
            return base_target_xy

        if self._corrected_gripper_target_xy is None:
            assert self._desired_cube_xy is not None
            actual_cube_xy = env.scene["cube"].data.root_pos_w[:, :2]
            raw_residual_xy = self._desired_cube_xy - actual_cube_xy
            residual_norm = torch.linalg.vector_norm(raw_residual_xy, dim=-1, keepdim=True)
            correction_scale = torch.clamp(
                _MAXIMUM_RESIDUAL_CORRECTION / torch.clamp(residual_norm, min=1e-8),
                max=1.0,
            )
            applied_residual_xy = raw_residual_xy * correction_scale

            self._raw_transfer_residual_xy = raw_residual_xy.detach().clone()
            self._applied_transfer_residual_xy = applied_residual_xy.detach().clone()
            self._corrected_gripper_target_xy = (base_target_xy + applied_residual_xy).detach().clone()

        return self._corrected_gripper_target_xy

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "residual_correction_phase": "lower_into_box_entry",
                "residual_correction_policy": "single_frozen_xy_update",
                "maximum_residual_correction": _MAXIMUM_RESIDUAL_CORRECTION,
            }
        )
        return parameters

    @property
    def raw_transfer_residual_xy(self) -> torch.Tensor | None:
        return self._raw_transfer_residual_xy

    @property
    def applied_transfer_residual_xy(self) -> torch.Tensor | None:
        return self._applied_transfer_residual_xy

    @property
    def corrected_gripper_target_xy(self) -> torch.Tensor | None:
        return self._corrected_gripper_target_xy
