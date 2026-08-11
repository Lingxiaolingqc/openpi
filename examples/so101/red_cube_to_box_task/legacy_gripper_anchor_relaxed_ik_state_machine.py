"""Jaw-anchored legacy placement with staged IK orientation relaxation."""

from __future__ import annotations

import torch

from .legacy_gripper_anchor_state_machine import RedCubeToBoxLegacyGripperAnchorStateMachine
from .phase_aware_ik_action import PhaseAwareDifferentialInverseKinematicsAction
from .phase_aware_ik_action import resolve_action_term

_WEAK_ORIENTATION_WEIGHT = 0.1
_WEAK_ORIENTATION_STEPS = 120
_GRASP_DISTANCE_THRESHOLD = 0.02
_GRIPPER_POSITION_THRESHOLD = 0.26
_GRASP_PROTECTED_PHASES = {
    "lift_cube",
    "transfer_to_box",
    "lower_into_box",
    "align_over_box",
}


class RedCubeToBoxLegacyGripperAnchorRelaxedIkStateMachine(RedCubeToBoxLegacyGripperAnchorStateMachine):
    """Relax orientation only when fixed-pose jaw alignment cannot converge."""

    def __init__(self) -> None:
        super().__init__()
        self._arm_action_term: PhaseAwareDifferentialInverseKinematicsAction | None = None
        self._release_orientation_weight = _WEAK_ORIENTATION_WEIGHT
        self._grasp_confirmed = False
        self._grasp_lost_before_release = False
        self._servo_abort_reason: str | None = None

    def setup(self, env) -> None:
        super().setup(env)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise TypeError(
                "legacy_gripper_anchor_relaxed_ik requires "
                "PhaseAwareDifferentialInverseKinematicsAction"
            )
        self._arm_action_term = arm_action_term

    def get_action(self, env) -> torch.Tensor:
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a relaxed-IK action")

        phase_name, phase_step, _ = self._phase_state()
        orientation_weight = self._orientation_weight(phase_name, phase_step)
        self._arm_action_term.set_orientation_weight(weight=orientation_weight)
        action = super().get_action(env)
        self._update_grasp_status(env, phase_name)
        return action

    def advance(self) -> None:
        if self._episode_done:
            return
        phase_name = self.phase_name
        if phase_name == "align_over_box" and self.box_aligned_before_release:
            assert self._arm_action_term is not None
            self._release_orientation_weight = self._arm_action_term.orientation_weight
        super().advance()

    def reset(self) -> None:
        super().reset()
        self._release_orientation_weight = _WEAK_ORIENTATION_WEIGHT
        self._grasp_confirmed = False
        self._grasp_lost_before_release = False
        self._servo_abort_reason = None
        if self._arm_action_term is not None:
            self._arm_action_term.set_orientation_weight(weight=1.0)

    def _orientation_weight(self, phase_name: str, phase_step: int) -> float:
        if phase_name == "align_over_box":
            if phase_step < _WEAK_ORIENTATION_STEPS:
                return _WEAK_ORIENTATION_WEIGHT
            return 0.0
        if phase_name in {"release_cube", "retract_gripper", "settle"}:
            return self._release_orientation_weight
        return 1.0

    def _update_grasp_status(self, env, phase_name: str) -> None:
        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        cube = env.scene["cube"]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]
        jaw_distance = torch.linalg.vector_norm(jaw_pos_w - cube.data.root_pos_w, dim=-1)
        gripper_joint_index = robot.data.joint_names.index("gripper")
        gripper_closed = robot.data.joint_pos[:, gripper_joint_index] < _GRIPPER_POSITION_THRESHOLD
        grasped = bool(torch.logical_and(jaw_distance < _GRASP_DISTANCE_THRESHOLD, gripper_closed).all().item())
        if grasped:
            self._grasp_confirmed = True
        elif self._grasp_confirmed and phase_name in _GRASP_PROTECTED_PHASES:
            self._grasp_lost_before_release = True
            self._servo_abort_reason = "grasp_lost_before_release"
            self._episode_done = True

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "ik_alignment_mode": "staged_orientation_relaxation",
                "weak_orientation_weight": _WEAK_ORIENTATION_WEIGHT,
                "weak_orientation_steps": _WEAK_ORIENTATION_STEPS,
                "fallback_orientation_weight": 0.0,
                "grasp_loss_policy": "abort_before_release",
            }
        )
        return parameters

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "uninitialized"
        return self._arm_action_term.runtime_mode

    @property
    def grasp_confirmed(self) -> bool:
        return self._grasp_confirmed

    @property
    def grasp_lost_before_release(self) -> bool:
        return self._grasp_lost_before_release

    @property
    def servo_abort_reason(self) -> str | None:
        return self._servo_abort_reason
