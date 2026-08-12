"""Autogen reference ablation with only a slower gripper close trajectory."""

from __future__ import annotations

from .autogen_reference_state_machine import RedCubeToBoxAutogenReferenceStateMachine


class RedCubeToBoxAutogenReferenceSlowGraspStateMachine(RedCubeToBoxAutogenReferenceStateMachine):
    """Reduce pre-contact closing speed while preserving every other behavior."""

    GRASP_DURATION_STEPS = 240

    @property
    def servo_parameters(self) -> dict[str, object]:
        return {
            **super().servo_parameters,
            "comparison_variant": "autogen_reference_slow_grasp",
            "grasp_duration_steps": self.GRASP_DURATION_STEPS,
            "close_reference_speed_vs_autogen_reference": 1.0 / 3.0,
            "only_behavior_change": "grasp_duration_steps:80->240",
        }
