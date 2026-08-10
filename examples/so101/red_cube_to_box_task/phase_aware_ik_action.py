"""Pose-compatible IK action with a runtime position-only transport mode."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
import torch


class PhaseAwareDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Keep the 7D pose command contract while optionally solving only translation.

    The SO-101 arm has five non-gripper joints. During grasping, constraining the
    end-effector orientation is useful for aligning the jaw with the cube. During
    transport, however, solving three position rows plus three orientation rows is
    over-constrained. This action term lets the servo expert disable the orientation
    rows at runtime without changing the environment's action dimension.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._position_only = False

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._position_only = False

    def set_position_only(self, *, enabled: bool) -> None:
        """Select whether the next physics applications solve translation only."""

        self._position_only = bool(enabled)

    @property
    def position_only(self) -> bool:
        return self._position_only

    def apply_actions(self) -> None:
        if not self._position_only:
            super().apply_actions()
            return

        ee_pos_curr, _ = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        jacobian_pos = self._compute_frame_jacobian()[:, :3, :]

        desired_pos = getattr(self._ik_controller, "ee_pos_des", None)
        if desired_pos is None:
            # Compatibility fallback for older IsaacLab controller internals.
            desired_pos = self._ik_controller._ee_pos_des  # noqa: SLF001
        position_error = desired_pos - ee_pos_curr
        # Reuse IsaacLab's exact configured IK method instead of duplicating its math.
        delta_joint_pos = self._ik_controller._compute_delta_joint_pos(  # noqa: SLF001
            position_error, jacobian_pos
        )
        joint_pos_des = joint_pos + delta_joint_pos

        if not bool(torch.isfinite(joint_pos_des).all()):
            raise RuntimeError("Position-only differential IK produced a non-finite joint target")
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
