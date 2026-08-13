"""Legacy pickup followed by position-only Cartesian servo transport."""

from __future__ import annotations

from .legacy_weighted_servo_state_machine import RedCubeToBoxLegacyWeightedServoStateMachine


class RedCubeToBoxLegacyPositionServoStateMachine(RedCubeToBoxLegacyWeightedServoStateMachine):
    """Ablate the transport orientation rows while keeping all other logic fixed."""

    @property
    def transport_orientation_weight(self) -> float:
        return 0.0

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters["ik_transport_mode"] = "position_only"
        parameters["transport_orientation_weight"] = self.transport_orientation_weight
        parameters["transport_controller"] = "position_servo"
        return parameters
