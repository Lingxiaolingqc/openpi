"""Legacy pickup followed by translation-priority Cartesian servo transport."""

from __future__ import annotations

import torch

from ..state_machine import RedCubeToBoxStateMachine
from .servo_state_machine import TRANSPORT_IK_PHASES
from .weighted_servo_state_machine import RedCubeToBoxWeightedServoStateMachine

_LEGACY_PICKUP_PHASES = {
    "approach_cube",
    "descend_to_cube",
    "close_gripper",
    "lift_cube",
}


class RedCubeToBoxLegacyWeightedServoStateMachine(RedCubeToBoxWeightedServoStateMachine):
    """Isolate transport control while preserving the verified legacy pickup.

    The first four phases delegate directly to ``RedCubeToBoxStateMachine`` and
    therefore retain its body anchor, targets, timing, orientation, and gripper
    commands. Only transport and placement use the bounded Cartesian outer loop
    and translation-priority IK.
    """

    _PHASES = (
        ("approach_cube", 120),
        ("descend_to_cube", 120),
        ("close_gripper", 80),
        ("lift_cube", 120),
        ("transfer_to_box", 300),
        ("lower_into_box", 300),
        ("align_over_box", 300),
        ("release_cube", 100),
        ("retract_gripper", 100),
        ("settle", 180),
    )
    MAX_STEPS = sum(duration for _, duration in _PHASES)

    def __init__(self) -> None:
        super().__init__()
        self._legacy_pickup = RedCubeToBoxStateMachine()

    def reset(self) -> None:
        super().reset()
        self._legacy_pickup.reset()

    def get_action(self, env) -> torch.Tensor:
        if self.phase_name in _LEGACY_PICKUP_PHASES:
            if self._arm_action_term is None:
                raise RuntimeError("Call setup(env) before requesting a legacy-weighted-servo action")
            self._arm_action_term.set_orientation_weight(weight=1.0)
            return self._legacy_pickup.get_action(env)
        return super().get_action(env)

    def advance(self) -> None:
        if self.phase_name in _LEGACY_PICKUP_PHASES:
            self._legacy_pickup.advance()
            self._step_count += 1
            if self._step_count >= self.MAX_STEPS:
                self._episode_done = True
            return
        super().advance()

    @property
    def abort_on_grasp_loss(self) -> bool:
        # Match legacy's final-outcome evaluation so a one-frame geometric
        # threshold crossing cannot terminate this transport ablation early.
        return False

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters["pickup_controller"] = "legacy_exact"
        parameters["transport_controller"] = "weighted_servo"
        parameters["grasp_loss_policy"] = "diagnostic_only"
        return parameters

    @property
    def ik_runtime_mode(self) -> str:
        if self.phase_name not in TRANSPORT_IK_PHASES:
            return "pose"
        return super().ik_runtime_mode
