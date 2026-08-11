"""Align-before-descent legacy expert with shoulder-pan-prioritized XYZ IK."""

from __future__ import annotations

from .legacy_gripper_anchor_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine,
)
from .phase_aware_ik_action import PhaseAwareDifferentialInverseKinematicsAction
from .phase_aware_ik_action import resolve_action_term

_ALIGN_JOINT_PENALTIES = {
    "shoulder_pan": 0.25,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.5,
    "wrist_roll": 2.0,
}
_ALIGN_DAMPING = 0.05


class RedCubeToBoxLegacyGripperAnchorWeightedPositionAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine
):
    """Prefer shoulder pan among redundant XYZ alignment solutions."""

    def __init__(self) -> None:
        super().__init__()
        self._arm_action_term: PhaseAwareDifferentialInverseKinematicsAction | None = None

    def setup(self, env) -> None:
        super().setup(env)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise TypeError(
                "legacy_gripper_anchor_weighted_position_align_then_lower requires "
                "PhaseAwareDifferentialInverseKinematicsAction"
            )
        self._arm_action_term = arm_action_term

    def get_action(self, env):
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a weighted-position action")

        if self.phase_name == "align_over_box":
            self._arm_action_term.set_weighted_position_only(
                joint_penalties=_ALIGN_JOINT_PENALTIES,
                damping=_ALIGN_DAMPING,
            )
        else:
            self._arm_action_term.set_orientation_weight(weight=1.0)
        return super().get_action(env)

    def reset(self) -> None:
        super().reset()
        if self._arm_action_term is not None:
            self._arm_action_term.set_orientation_weight(weight=1.0)

    @property
    def servo_parameters(self) -> dict[str, float | int | str | dict[str, float]]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "high_alignment_ik_mode": "weighted_position_only_xyz",
                "align_joint_penalties": dict(_ALIGN_JOINT_PENALTIES),
                "align_damping": _ALIGN_DAMPING,
                "lowering_ik_mode": "legacy_full_pose",
            }
        )
        return parameters

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "uninitialized"
        return self._arm_action_term.runtime_mode

    @property
    def last_weighted_delta_joint_pos(self):
        if self._arm_action_term is None:
            return None
        return self._arm_action_term.last_weighted_delta_joint_pos

    @property
    def weighted_joint_names(self) -> tuple[str, ...]:
        if self._arm_action_term is None:
            return ()
        return self._arm_action_term.weighted_joint_names
