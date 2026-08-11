"""Align-before-descent legacy expert with position-only high alignment."""

from __future__ import annotations

from .legacy_gripper_anchor_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine,
)
from .phase_aware_ik_action import PhaseAwareDifferentialInverseKinematicsAction
from .phase_aware_ik_action import resolve_action_term


class RedCubeToBoxLegacyGripperAnchorPositionAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine
):
    """Remove orientation rows only while aligning XYZ above the box."""

    def __init__(self) -> None:
        super().__init__()
        self._arm_action_term: PhaseAwareDifferentialInverseKinematicsAction | None = None

    def setup(self, env) -> None:
        super().setup(env)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise TypeError(
                "legacy_gripper_anchor_position_align_then_lower requires "
                "PhaseAwareDifferentialInverseKinematicsAction"
            )
        self._arm_action_term = arm_action_term

    def get_action(self, env):
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a position-alignment action")

        if self.phase_name == "align_over_box":
            self._arm_action_term.set_position_only(enabled=True)
        else:
            self._arm_action_term.set_orientation_weight(weight=1.0)
        return super().get_action(env)

    def reset(self) -> None:
        super().reset()
        if self._arm_action_term is not None:
            self._arm_action_term.set_orientation_weight(weight=1.0)

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "high_alignment_ik_mode": "position_only_xyz",
                "lowering_ik_mode": "legacy_full_pose",
                "orientation_rows_removed_during": "align_over_box",
            }
        )
        return parameters

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "uninitialized"
        return self._arm_action_term.runtime_mode
