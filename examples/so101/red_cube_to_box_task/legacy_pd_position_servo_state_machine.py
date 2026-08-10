"""Legacy pickup followed by velocity-damped position-only servo transport."""

from __future__ import annotations

from .legacy_position_servo_state_machine import RedCubeToBoxLegacyPositionServoStateMachine

_SERVO_KD = 0.02
_SERVO_VELOCITY_THRESHOLD = 0.02


class RedCubeToBoxLegacyPdPositionServoStateMachine(RedCubeToBoxLegacyPositionServoStateMachine):
    """Add Cartesian velocity damping to the isolated position-only transport."""

    @property
    def servo_kd(self) -> float:
        return _SERVO_KD

    @property
    def servo_velocity_threshold(self) -> float:
        return _SERVO_VELOCITY_THRESHOLD

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters["transport_controller"] = "pd_position_servo"
        return parameters
