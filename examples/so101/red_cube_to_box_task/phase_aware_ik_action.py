"""Pose-compatible IK action with runtime translation-priority modes."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.math import compute_pose_error
import torch


class PhaseAwareDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Keep the 7D pose command contract while changing transport task weights.

    The SO-101 arm has five non-gripper joints. During grasping, constraining the
    end-effector orientation is useful for aligning the jaw with the cube. During
    transport, however, solving three position rows plus three orientation rows is
    over-constrained. This action term lets the servo expert disable the orientation
    rows at runtime without changing the environment's action dimension. Intermediate
    weights retain weak orientation regularization while prioritizing translation.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._orientation_weight = 1.0

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._orientation_weight = 1.0

    def set_position_only(self, *, enabled: bool) -> None:
        """Select whether the next physics applications solve translation only."""

        self._orientation_weight = 0.0 if enabled else 1.0

    def set_orientation_weight(self, *, weight: float) -> None:
        """Set the transport orientation weight in the closed interval [0, 1]."""

        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Orientation weight must be in [0, 1], received {weight}")
        self._orientation_weight = float(weight)

    @property
    def position_only(self) -> bool:
        return self._orientation_weight == 0.0

    @property
    def orientation_weight(self) -> float:
        return self._orientation_weight

    @property
    def runtime_mode(self) -> str:
        if self._orientation_weight == 1.0:
            return "pose"
        if self._orientation_weight == 0.0:
            return "position_only"
        return f"translation_priority(weight={self._orientation_weight:g})"

    def apply_actions(self) -> None:
        if self._orientation_weight == 1.0:
            super().apply_actions()
            return

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        jacobian = self._compute_frame_jacobian()

        desired_pos = getattr(self._ik_controller, "ee_pos_des", None)
        if desired_pos is None:
            # Compatibility fallback for older IsaacLab controller internals.
            desired_pos = self._ik_controller._ee_pos_des  # noqa: SLF001
        desired_quat = getattr(self._ik_controller, "ee_quat_des", None)
        if desired_quat is None:
            desired_quat = self._ik_controller._ee_quat_des  # noqa: SLF001

        if self.position_only:
            task_error = desired_pos - ee_pos_curr
            task_jacobian = jacobian[:, :3, :]
        else:
            position_error, orientation_error = compute_pose_error(
                ee_pos_curr,
                ee_quat_curr,
                desired_pos,
                desired_quat,
                rot_error_type="axis_angle",
            )
            task_error = torch.cat((position_error, self._orientation_weight * orientation_error), dim=1)
            task_jacobian = jacobian.clone()
            task_jacobian[:, 3:, :] *= self._orientation_weight

        # Reuse IsaacLab's exact configured IK method instead of duplicating its math.
        delta_joint_pos = self._ik_controller._compute_delta_joint_pos(  # noqa: SLF001
            task_error, task_jacobian
        )
        joint_pos_des = joint_pos + delta_joint_pos

        if not bool(torch.isfinite(joint_pos_des).all()):
            raise RuntimeError("Translation-priority differential IK produced a non-finite joint target")
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)


def configure_servo_ik_action(env_cfg) -> None:
    """Use the phase-aware action term for one servo-expert environment config."""

    env_cfg.actions.arm_action.class_type = PhaseAwareDifferentialInverseKinematicsAction


def resolve_action_term(action_manager, term_name: str):
    """Resolve an action term across the public and older IsaacLab APIs."""

    get_term = getattr(action_manager, "get_term", None)
    if callable(get_term):
        return get_term(term_name)

    # IsaacLab 0.47 predates ActionManager.get_term on some installations.
    terms = getattr(action_manager, "_terms", None)
    if not isinstance(terms, dict) or term_name not in terms:
        raise RuntimeError(f"Unable to resolve the {term_name} term from IsaacLab's ActionManager")
    return terms[term_name]
