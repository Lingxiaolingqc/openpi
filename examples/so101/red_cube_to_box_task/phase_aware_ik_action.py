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
        self._planar_pose = False
        self._xyz_tilt = False
        self._xyz_pitch_joint_name: str | None = None
        self._xyz_pitch_joint_index: int | None = None
        self._xyz_pitch_joint_target: torch.Tensor | None = None
        self._last_delta_joint_pos: torch.Tensor | None = None
        self._last_task_error: torch.Tensor | None = None
        self._last_task_singular_values: torch.Tensor | None = None
        self._weighted_position_penalties: torch.Tensor | None = None
        self._weighted_position_damping: float | None = None
        self._weighted_joint_names: tuple[str, ...] = ()
        self._last_weighted_delta_joint_pos: torch.Tensor | None = None

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._orientation_weight = 1.0
        self._planar_pose = False
        self._xyz_tilt = False
        self._clear_xyz_pitch_joint_target()
        self._clear_solver_diagnostics()
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None

    def set_position_only(self, *, enabled: bool) -> None:
        """Select whether the next physics applications solve translation only."""

        self._planar_pose = False
        self._xyz_tilt = False
        self._clear_xyz_pitch_joint_target()
        self._orientation_weight = 0.0 if enabled else 1.0
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None

    def set_orientation_weight(self, *, weight: float) -> None:
        """Set the transport orientation weight in the closed interval [0, 1]."""

        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Orientation weight must be in [0, 1], received {weight}")
        self._planar_pose = False
        self._xyz_tilt = False
        self._clear_xyz_pitch_joint_target()
        self._orientation_weight = float(weight)
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None

    def set_planar_pose(self, *, enabled: bool) -> None:
        """Solve X/Y translation and all three orientation rows while leaving Z free."""

        self._planar_pose = enabled
        self._xyz_tilt = False
        self._clear_xyz_pitch_joint_target()
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None
        if enabled:
            self._orientation_weight = 1.0

    def set_xyz_tilt(self, *, enabled: bool) -> None:
        """Solve XYZ and world-frame roll/pitch rows while leaving yaw free."""

        self._xyz_tilt = enabled
        self._planar_pose = False
        self._clear_xyz_pitch_joint_target()
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None
        if enabled:
            self._orientation_weight = 1.0

    def set_xyz_pitch_joint_target(
        self,
        *,
        joint_name: str,
        joint_target: torch.Tensor,
    ) -> None:
        """Solve XYZ, base-frame pitch, and one named joint-position row."""

        if joint_target.ndim != 1 or joint_target.shape[0] != self._asset.data.joint_pos.shape[0]:
            raise ValueError(
                f"joint_target must have one value per environment; received shape {tuple(joint_target.shape)}"
            )
        all_joint_names = self._asset.data.joint_names
        if isinstance(self._joint_ids, slice):
            selected_joint_names = list(all_joint_names[self._joint_ids])
        else:
            selected_joint_names = [all_joint_names[joint_id] for joint_id in self._joint_ids]
        if joint_name not in selected_joint_names:
            raise ValueError(f"Joint {joint_name!r} is not controlled by this IK action: {selected_joint_names}")

        self._planar_pose = False
        self._xyz_tilt = False
        self._orientation_weight = 1.0
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None
        self._xyz_pitch_joint_name = joint_name
        self._xyz_pitch_joint_index = selected_joint_names.index(joint_name)
        self._xyz_pitch_joint_target = joint_target.detach().clone()

    def set_weighted_position_only(
        self,
        *,
        joint_penalties: dict[str, float],
        damping: float,
    ) -> None:
        """Solve XYZ while preferring joints with lower positive motion penalties."""

        if damping <= 0.0:
            raise ValueError(f"Weighted DLS damping must be positive, received {damping}")
        all_joint_names = self._asset.data.joint_names
        if isinstance(self._joint_ids, slice):
            selected_joint_names = all_joint_names[self._joint_ids]
        else:
            selected_joint_names = [all_joint_names[joint_id] for joint_id in self._joint_ids]
        missing_joint_names = [name for name in selected_joint_names if name not in joint_penalties]
        if missing_joint_names:
            raise ValueError(f"Missing weighted-DLS penalties for joints: {missing_joint_names}")
        penalties = [float(joint_penalties[name]) for name in selected_joint_names]
        if any(penalty <= 0.0 for penalty in penalties):
            raise ValueError(f"Weighted-DLS penalties must be positive, received {penalties}")

        self._planar_pose = False
        self._xyz_tilt = False
        self._clear_xyz_pitch_joint_target()
        self._orientation_weight = 0.0
        self._weighted_position_penalties = torch.tensor(
            penalties,
            device=self._asset.data.joint_pos.device,
            dtype=self._asset.data.joint_pos.dtype,
        )
        self._weighted_position_damping = float(damping)
        self._weighted_joint_names = tuple(selected_joint_names)

    @property
    def position_only(self) -> bool:
        return not self._planar_pose and self._orientation_weight == 0.0

    @property
    def planar_pose(self) -> bool:
        return self._planar_pose

    @property
    def xyz_tilt(self) -> bool:
        return self._xyz_tilt

    @property
    def orientation_weight(self) -> float:
        return self._orientation_weight

    @property
    def runtime_mode(self) -> str:
        if self._planar_pose:
            return "planar_pose(xy+orientation)"
        if self._xyz_tilt:
            return "xyz_tilt(xyz+orientation_xy)"
        if self._xyz_pitch_joint_target is not None:
            return f"xyz_pitch_joint(xyz+orientation_y+{self._xyz_pitch_joint_name})"
        if self._weighted_position_penalties is not None:
            return f"weighted_position_only(damping={self._weighted_position_damping:g})"
        if self._orientation_weight == 1.0:
            return "pose"
        if self._orientation_weight == 0.0:
            return "position_only"
        return f"translation_priority(weight={self._orientation_weight:g})"

    def apply_actions(self) -> None:
        if (
            not self._planar_pose
            and not self._xyz_tilt
            and self._xyz_pitch_joint_target is None
            and self._orientation_weight == 1.0
        ):
            self._clear_solver_diagnostics()
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

        if self._planar_pose:
            position_error, orientation_error = compute_pose_error(
                ee_pos_curr,
                ee_quat_curr,
                desired_pos,
                desired_quat,
                rot_error_type="axis_angle",
            )
            task_error = torch.cat((position_error[:, :2], orientation_error), dim=1)
            task_jacobian = torch.cat((jacobian[:, :2, :], jacobian[:, 3:, :]), dim=1)
        elif self._xyz_tilt:
            position_error, orientation_error = compute_pose_error(
                ee_pos_curr,
                ee_quat_curr,
                desired_pos,
                desired_quat,
                rot_error_type="axis_angle",
            )
            task_error = torch.cat((position_error, orientation_error[:, :2]), dim=1)
            task_jacobian = torch.cat((jacobian[:, :3, :], jacobian[:, 3:5, :]), dim=1)
        elif self._xyz_pitch_joint_target is not None:
            assert self._xyz_pitch_joint_index is not None
            position_error, orientation_error = compute_pose_error(
                ee_pos_curr,
                ee_quat_curr,
                desired_pos,
                desired_quat,
                rot_error_type="axis_angle",
            )
            joint_error = (self._xyz_pitch_joint_target - joint_pos[:, self._xyz_pitch_joint_index]).unsqueeze(-1)
            joint_row = torch.zeros(
                (jacobian.shape[0], 1, jacobian.shape[2]),
                device=jacobian.device,
                dtype=jacobian.dtype,
            )
            joint_row[:, 0, self._xyz_pitch_joint_index] = 1.0
            task_error = torch.cat((position_error, orientation_error[:, 1:2], joint_error), dim=1)
            task_jacobian = torch.cat((jacobian[:, :3, :], jacobian[:, 4:5, :], joint_row), dim=1)
        elif self.position_only:
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

        if self._weighted_position_penalties is not None:
            assert self._weighted_position_damping is not None
            inverse_penalties = torch.diag_embed(
                (1.0 / self._weighted_position_penalties).expand(task_jacobian.shape[0], -1)
            )
            jacobian_transpose = task_jacobian.transpose(1, 2)
            task_identity = torch.eye(
                task_jacobian.shape[1],
                device=task_jacobian.device,
                dtype=task_jacobian.dtype,
            ).unsqueeze(0)
            damped_system = (
                task_jacobian @ inverse_penalties @ jacobian_transpose
                + self._weighted_position_damping**2 * task_identity
            )
            task_solution = torch.linalg.solve(damped_system, task_error.unsqueeze(-1))
            delta_joint_pos = (inverse_penalties @ jacobian_transpose @ task_solution).squeeze(-1)
            self._last_weighted_delta_joint_pos = delta_joint_pos.detach().clone()
        else:
            # Reuse IsaacLab's exact configured IK method for every unweighted mode.
            delta_joint_pos = self._ik_controller._compute_delta_joint_pos(  # noqa: SLF001
                task_error, task_jacobian
            )
        if self._xyz_pitch_joint_target is not None:
            self._last_delta_joint_pos = delta_joint_pos.detach().clone()
            self._last_task_error = task_error.detach().clone()
            self._last_task_singular_values = torch.linalg.svdvals(task_jacobian).detach().clone()
        else:
            self._clear_solver_diagnostics()
        joint_pos_des = joint_pos + delta_joint_pos

        if not bool(torch.isfinite(joint_pos_des).all()):
            raise RuntimeError("Translation-priority differential IK produced a non-finite joint target")
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    @property
    def last_weighted_delta_joint_pos(self) -> torch.Tensor | None:
        return self._last_weighted_delta_joint_pos

    @property
    def weighted_joint_names(self) -> tuple[str, ...]:
        return self._weighted_joint_names

    @property
    def last_delta_joint_pos(self) -> torch.Tensor | None:
        return self._last_delta_joint_pos

    @property
    def last_task_error(self) -> torch.Tensor | None:
        return self._last_task_error

    @property
    def last_task_singular_values(self) -> torch.Tensor | None:
        return self._last_task_singular_values

    @property
    def controlled_joint_names(self) -> tuple[str, ...]:
        all_joint_names = self._asset.data.joint_names
        if isinstance(self._joint_ids, slice):
            return tuple(all_joint_names[self._joint_ids])
        return tuple(all_joint_names[joint_id] for joint_id in self._joint_ids)

    def _clear_xyz_pitch_joint_target(self) -> None:
        self._xyz_pitch_joint_name = None
        self._xyz_pitch_joint_index = None
        self._xyz_pitch_joint_target = None

    def _clear_solver_diagnostics(self) -> None:
        self._last_delta_joint_pos = None
        self._last_task_error = None
        self._last_task_singular_values = None


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
