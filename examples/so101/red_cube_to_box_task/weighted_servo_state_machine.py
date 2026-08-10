"""Translation-priority SO-101 expert with weak orientation regularization."""

from __future__ import annotations

from .servo_state_machine import RedCubeToBoxServoStateMachine

_TRANSPORT_ORIENTATION_WEIGHT = 0.1


class RedCubeToBoxWeightedServoStateMachine(RedCubeToBoxServoStateMachine):
    """Preserve the servo outer loop while damping unconstrained wrist rotation."""

    @property
    def transport_orientation_weight(self) -> float:
        return _TRANSPORT_ORIENTATION_WEIGHT

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters["ik_transport_mode"] = "translation_priority"
        parameters["transport_orientation_weight"] = self.transport_orientation_weight
        return parameters
