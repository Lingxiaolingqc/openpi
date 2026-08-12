"""Pose-compatible IK action with runtime translation-priority modes."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.math import compute_pose_error
from isaaclab.utils.math import matrix_from_quat
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
        self._xyz_joint_nullspace_name: str | None = None
        self._xyz_joint_nullspace_index: int | None = None
        self._xyz_joint_nullspace_target: torch.Tensor | None = None
        self._xyz_joint_nullspace_position_axes = (0, 1, 2)
        self._xyz_joint_nullspace_position_axes_are_world_frame = False
        self._xyz_joint_nullspace_damping = 0.05
        self._xyz_joint_nullspace_posture_gain = 0.08
        self._xyz_joint_nullspace_max_step = 0.03
        self._position_nullspace_posture_target: torch.Tensor | None = None
        self._position_nullspace_damping = 0.05
        self._position_nullspace_posture_gain = 0.08
        self._position_nullspace_max_step = 0.03
        self._last_delta_joint_pos: torch.Tensor | None = None
        self._last_primary_delta_joint_pos: torch.Tensor | None = None
        self._last_nullspace_delta_joint_pos: torch.Tensor | None = None
        self._last_task_error: torch.Tensor | None = None
        self._last_task_singular_values: torch.Tensor | None = None
        self._maximum_joint_target_step: float | None = None
        self._last_unlimited_delta_joint_pos: torch.Tensor | None = None
        self._joint_target_slew_max_step: float | None = None
        self._joint_target_slew_reference: torch.Tensor | None = None
        self._last_joint_target_slew_step: torch.Tensor | None = None
        self._last_joint_position_target: torch.Tensor | None = None
        self._weighted_position_penalties: torch.Tensor | None = None
        self._weighted_position_damping: float | None = None
        self._weighted_joint_names: tuple[str, ...] = ()
        self._last_weighted_delta_joint_pos: torch.Tensor | None = None
        self._direct_joint_position_target: torch.Tensor | None = None
        self._configured_body_name = self._body_name

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._orientation_weight = 1.0
        self._planar_pose = False
        self._xyz_tilt = False
        self._clear_xyz_pitch_joint_target()
        self._clear_xyz_joint_nullspace_target()
        self._clear_position_nullspace_posture_target()
        self._clear_solver_diagnostics()
        self._joint_target_slew_reference = None
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None
        self.clear_direct_joint_position_target()

    def set_direct_joint_position_target(self, joint_target: torch.Tensor) -> None:
        """Bypass IK and hold the complete controlled-joint vector directly.

        A complete target is required so the action term remains the sole writer
        for every arm joint.  This avoids racing a one-joint state-machine write
        against the normal IK write performed later in the same environment step.
        """

        expected_shape = (self._asset.data.joint_pos.shape[0], len(self.controlled_joint_names))
        if joint_target.shape != expected_shape:
            raise ValueError(
                f"Direct joint target must have shape {expected_shape}; received {tuple(joint_target.shape)}"
            )
        target = joint_target.to(
            device=self._asset.data.joint_pos.device,
            dtype=self._asset.data.joint_pos.dtype,
        ).detach()
        if not bool(torch.isfinite(target).all()):
            raise ValueError("Direct joint target contains a non-finite value")
        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._direct_joint_position_target = torch.clamp(
            target,
            min=soft_limits[..., 0],
            max=soft_limits[..., 1],
        ).clone()

    def clear_direct_joint_position_target(self) -> None:
        """Return action execution to the configured differential-IK mode."""

        self._direct_joint_position_target = None

    def set_position_only(self, *, enabled: bool) -> None:
        """Select whether the next physics applications solve translation only."""

        self._planar_pose = False
        self._xyz_tilt = False
        self._clear_xyz_pitch_joint_target()
        self._clear_xyz_joint_nullspace_target()
        self._clear_position_nullspace_posture_target()
        self._orientation_weight = 0.0 if enabled else 1.0
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None

    def set_position_only_nullspace_posture_target(
        self,
        *,
        joint_target: torch.Tensor,
        damping: float,
        posture_gain: float,
        max_posture_step: float,
    ) -> None:
        """Solve XYZ and softly preserve a complete joint posture in its nullspace."""

        expected_shape = (self._asset.data.joint_pos.shape[0], len(self.controlled_joint_names))
        if joint_target.shape != expected_shape:
            raise ValueError(f"joint_target must have shape {expected_shape}; received {tuple(joint_target.shape)}")
        if damping <= 0.0:
            raise ValueError(f"Position-nullspace DLS damping must be positive, received {damping}")
        if posture_gain < 0.0:
            raise ValueError(f"Position-nullspace posture gain must be non-negative, received {posture_gain}")
        if max_posture_step <= 0.0:
            raise ValueError(f"Maximum position-nullspace posture step must be positive, received {max_posture_step}")

        self.set_position_only(enabled=True)
        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._position_nullspace_posture_target = (
            torch.clamp(
                joint_target.to(device=soft_limits.device, dtype=soft_limits.dtype),
                min=soft_limits[..., 0],
                max=soft_limits[..., 1],
            )
            .detach()
            .clone()
        )
        self._position_nullspace_damping = float(damping)
        self._position_nullspace_posture_gain = float(posture_gain)
        self._position_nullspace_max_step = float(max_posture_step)

    def set_orientation_weight(self, *, weight: float) -> None:
        """Set the transport orientation weight in the closed interval [0, 1]."""

        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Orientation weight must be in [0, 1], received {weight}")
        self._planar_pose = False
        self._xyz_tilt = False
        self._clear_xyz_pitch_joint_target()
        self._clear_xyz_joint_nullspace_target()
        self._clear_position_nullspace_posture_target()
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
        self._clear_xyz_joint_nullspace_target()
        self._clear_position_nullspace_posture_target()
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
        self._clear_xyz_joint_nullspace_target()
        self._clear_position_nullspace_posture_target()
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
        self._clear_xyz_joint_nullspace_target()
        self._clear_position_nullspace_posture_target()
        self._xyz_pitch_joint_name = joint_name
        self._xyz_pitch_joint_index = selected_joint_names.index(joint_name)
        self._xyz_pitch_joint_target = joint_target.detach().clone()

    def set_xyz_joint_nullspace_target(
        self,
        *,
        joint_name: str,
        joint_target: torch.Tensor,
        damping: float,
        posture_gain: float,
        max_posture_step: float,
    ) -> None:
        """Solve XYZ plus one joint row and use the remaining nullspace to avoid joint limits."""

        if joint_target.ndim != 1 or joint_target.shape[0] != self._asset.data.joint_pos.shape[0]:
            raise ValueError(
                f"joint_target must have one value per environment; received shape {tuple(joint_target.shape)}"
            )
        if damping <= 0.0:
            raise ValueError(f"Nullspace DLS damping must be positive, received {damping}")
        if posture_gain < 0.0:
            raise ValueError(f"Nullspace posture gain must be non-negative, received {posture_gain}")
        if max_posture_step <= 0.0:
            raise ValueError(f"Maximum nullspace posture step must be positive, received {max_posture_step}")

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
        self._clear_xyz_pitch_joint_target()
        self._clear_position_nullspace_posture_target()
        self._weighted_position_penalties = None
        self._weighted_position_damping = None
        self._weighted_joint_names = ()
        self._last_weighted_delta_joint_pos = None
        self._xyz_joint_nullspace_name = joint_name
        self._xyz_joint_nullspace_index = selected_joint_names.index(joint_name)
        self._xyz_joint_nullspace_target = joint_target.detach().clone()
        self._xyz_joint_nullspace_position_axes = (0, 1, 2)
        self._xyz_joint_nullspace_position_axes_are_world_frame = False
        self._xyz_joint_nullspace_damping = float(damping)
        self._xyz_joint_nullspace_posture_gain = float(posture_gain)
        self._xyz_joint_nullspace_max_step = float(max_posture_step)

    def set_xz_joint_nullspace_target(
        self,
        *,
        joint_name: str,
        joint_target: torch.Tensor,
        damping: float,
        posture_gain: float,
        max_posture_step: float,
    ) -> None:
        """Solve X/Z plus one joint row, deliberately leaving Cartesian Y free."""

        self.set_xyz_joint_nullspace_target(
            joint_name=joint_name,
            joint_target=joint_target,
            damping=damping,
            posture_gain=posture_gain,
            max_posture_step=max_posture_step,
        )
        self._xyz_joint_nullspace_position_axes = (0, 2)
        self._xyz_joint_nullspace_position_axes_are_world_frame = True

    def set_control_frame_offset(self, *, position: torch.Tensor, orientation: torch.Tensor) -> None:
        """Update the configured gripper-relative virtual control frame."""

        if self.cfg.body_offset is None or self._offset_pos is None or self._offset_rot is None:
            raise RuntimeError("Dynamic control-frame offsets require a non-None body_offset configuration")
        if position.shape != self._offset_pos.shape:
            raise ValueError(
                f"Control-frame offset position shape must be {self._offset_pos.shape}, got {position.shape}"
            )
        if orientation.shape != self._offset_rot.shape:
            raise ValueError(
                f"Control-frame offset orientation shape must be {self._offset_rot.shape}, got {orientation.shape}"
            )
        self._offset_pos.copy_(position)
        self._offset_rot.copy_(orientation)

    def set_identity_control_frame_offset(self) -> None:
        """Restore the underlying gripper body as the virtual control frame."""

        if self.cfg.body_offset is None or self._offset_pos is None or self._offset_rot is None:
            return
        self._offset_pos.zero_()
        self._offset_rot.zero_()
        self._offset_rot[:, 0] = 1.0

    def set_control_body(self, *, body_name: str) -> None:
        """Switch the rigid body and corresponding Jacobian used by differential IK."""

        body_ids, body_names = self._asset.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body named {body_name!r}; found {body_names}")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1 if self._asset.is_fixed_base else self._body_idx

    def restore_configured_control_body(self) -> None:
        """Restore the rigid body selected by the environment configuration."""

        self.set_control_body(body_name=self._configured_body_name)

    def set_maximum_joint_target_step(self, *, maximum_step: float | None) -> None:
        """Limit each IK application without changing the requested joint-space direction."""

        if maximum_step is not None and maximum_step <= 0.0:
            raise ValueError(f"Maximum joint target step must be positive, received {maximum_step}")
        self._maximum_joint_target_step = maximum_step

    def set_joint_target_slew_limit(self, *, maximum_step: float | None) -> None:
        """Rate-limit the persistent joint target rather than its distance from the live joint position."""

        if maximum_step is not None and maximum_step <= 0.0:
            raise ValueError(f"Joint target slew step must be positive, received {maximum_step}")
        self._joint_target_slew_max_step = maximum_step
        self._joint_target_slew_reference = None
        self._last_joint_target_slew_step = None

    def reset_joint_target_slew_reference(self) -> None:
        """Start the next limited target trajectory from the then-current joint state."""

        self._joint_target_slew_reference = None
        self._last_joint_target_slew_step = None

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
        self._clear_xyz_joint_nullspace_target()
        self._clear_position_nullspace_posture_target()
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
        if self._direct_joint_position_target is not None:
            return "direct_joint_hold"
        if self._planar_pose:
            return "planar_pose(xy+orientation)"
        if self._xyz_tilt:
            return "xyz_tilt(xyz+orientation_xy)"
        if self._xyz_pitch_joint_target is not None:
            return f"xyz_pitch_joint(xyz+orientation_y+{self._xyz_pitch_joint_name})"
        if self._xyz_joint_nullspace_target is not None:
            if self._xyz_joint_nullspace_position_axes == (0, 2):
                return f"xz_joint_nullspace(xz+{self._xyz_joint_nullspace_name},y_free)"
            return f"xyz_joint_nullspace(xyz+{self._xyz_joint_nullspace_name},joint_limit_avoidance)"
        if self._position_nullspace_posture_target is not None:
            return f"position_only_nullspace_posture(damping={self._position_nullspace_damping:g})"
        if self._weighted_position_penalties is not None:
            return f"weighted_position_only(damping={self._weighted_position_damping:g})"
        if self._orientation_weight == 1.0:
            return "pose"
        if self._orientation_weight == 0.0:
            return "position_only"
        return f"translation_priority(weight={self._orientation_weight:g})"

    def apply_actions(self) -> None:
        if self._direct_joint_position_target is not None:
            joint_pos_des = self._direct_joint_position_target
            if not bool(torch.isfinite(joint_pos_des).all()):
                raise RuntimeError("Direct joint hold contains a non-finite target")
            self._clear_solver_diagnostics()
            self._last_joint_position_target = joint_pos_des.detach().clone()
            self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)
            return

        if (
            not self._planar_pose
            and not self._xyz_tilt
            and self._xyz_pitch_joint_target is None
            and self._xyz_joint_nullspace_target is None
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
        elif self._xyz_joint_nullspace_target is not None:
            assert self._xyz_joint_nullspace_index is not None
            position_error = desired_pos - ee_pos_curr
            joint_error = (self._xyz_joint_nullspace_target - joint_pos[:, self._xyz_joint_nullspace_index]).unsqueeze(
                -1
            )
            joint_row = torch.zeros(
                (jacobian.shape[0], 1, jacobian.shape[2]),
                device=jacobian.device,
                dtype=jacobian.dtype,
            )
            joint_row[:, 0, self._xyz_joint_nullspace_index] = 1.0
            position_axes = list(self._xyz_joint_nullspace_position_axes)
            position_jacobian = jacobian[:, :3, :]
            if self._xyz_joint_nullspace_position_axes_are_world_frame:
                root_rotation_w = matrix_from_quat(self._asset.data.root_quat_w)
                position_error = (root_rotation_w @ position_error.unsqueeze(-1)).squeeze(-1)
                position_jacobian = root_rotation_w @ position_jacobian
            task_error = torch.cat((position_error[:, position_axes], joint_error), dim=1)
            task_jacobian = torch.cat((position_jacobian[:, position_axes, :], joint_row), dim=1)
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

        if self._xyz_joint_nullspace_target is not None:
            task_jacobian_transpose = task_jacobian.transpose(1, 2)
            task_identity = torch.eye(
                task_jacobian.shape[1], device=task_jacobian.device, dtype=task_jacobian.dtype
            ).unsqueeze(0)
            damped_system = (
                task_jacobian @ task_jacobian_transpose + self._xyz_joint_nullspace_damping**2 * task_identity
            )
            damped_pseudoinverse = task_jacobian_transpose @ torch.linalg.solve(
                damped_system, task_identity.expand(task_jacobian.shape[0], -1, -1)
            )
            primary_delta = (damped_pseudoinverse @ task_error.unsqueeze(-1)).squeeze(-1)

            soft_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            joint_midpoints = soft_limits.mean(dim=-1)
            joint_half_ranges = 0.5 * (soft_limits[..., 1] - soft_limits[..., 0])
            normalized_offset = (joint_pos - joint_midpoints) / torch.clamp(joint_half_ranges, min=1.0e-6)
            distance_to_limit = torch.clamp(1.0 - torch.abs(normalized_offset), min=0.1)
            posture_delta = -self._xyz_joint_nullspace_posture_gain * normalized_offset / distance_to_limit
            posture_delta[:, self._xyz_joint_nullspace_index] = 0.0
            posture_delta = torch.clamp(
                posture_delta,
                min=-self._xyz_joint_nullspace_max_step,
                max=self._xyz_joint_nullspace_max_step,
            )

            joint_identity = torch.eye(
                task_jacobian.shape[2], device=task_jacobian.device, dtype=task_jacobian.dtype
            ).unsqueeze(0)
            nullspace_projector = joint_identity - damped_pseudoinverse @ task_jacobian
            nullspace_delta = (nullspace_projector @ posture_delta.unsqueeze(-1)).squeeze(-1)
            delta_joint_pos = primary_delta + nullspace_delta
            self._last_primary_delta_joint_pos = primary_delta.detach().clone()
            self._last_nullspace_delta_joint_pos = nullspace_delta.detach().clone()
        elif self._position_nullspace_posture_target is not None:
            task_jacobian_transpose = task_jacobian.transpose(1, 2)
            task_identity = torch.eye(
                task_jacobian.shape[1], device=task_jacobian.device, dtype=task_jacobian.dtype
            ).unsqueeze(0)
            damped_system = (
                task_jacobian @ task_jacobian_transpose + self._position_nullspace_damping**2 * task_identity
            )
            damped_pseudoinverse = task_jacobian_transpose @ torch.linalg.solve(
                damped_system, task_identity.expand(task_jacobian.shape[0], -1, -1)
            )
            primary_delta = (damped_pseudoinverse @ task_error.unsqueeze(-1)).squeeze(-1)
            posture_delta = self._position_nullspace_posture_gain * (
                self._position_nullspace_posture_target - joint_pos
            )
            posture_delta = torch.clamp(
                posture_delta,
                min=-self._position_nullspace_max_step,
                max=self._position_nullspace_max_step,
            )
            joint_identity = torch.eye(
                task_jacobian.shape[2], device=task_jacobian.device, dtype=task_jacobian.dtype
            ).unsqueeze(0)
            nullspace_projector = joint_identity - damped_pseudoinverse @ task_jacobian
            nullspace_delta = (nullspace_projector @ posture_delta.unsqueeze(-1)).squeeze(-1)
            delta_joint_pos = primary_delta + nullspace_delta
            self._last_primary_delta_joint_pos = primary_delta.detach().clone()
            self._last_nullspace_delta_joint_pos = nullspace_delta.detach().clone()
        elif self._weighted_position_penalties is not None:
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
        self._last_unlimited_delta_joint_pos = delta_joint_pos.detach().clone()
        if self._maximum_joint_target_step is not None:
            maximum_component = torch.amax(torch.abs(delta_joint_pos), dim=-1, keepdim=True)
            scale = torch.clamp(
                self._maximum_joint_target_step / torch.clamp(maximum_component, min=1.0e-8),
                max=1.0,
            )
            delta_joint_pos = delta_joint_pos * scale
        solver_diagnostics_active = (
            self._xyz_pitch_joint_target is not None
            or self._xyz_joint_nullspace_target is not None
            or self._position_nullspace_posture_target is not None
        )
        if solver_diagnostics_active:
            self._last_task_error = task_error.detach().clone()
            self._last_task_singular_values = torch.linalg.svdvals(task_jacobian).detach().clone()
        else:
            self._clear_solver_diagnostics()
        joint_pos_des = joint_pos + delta_joint_pos

        if self._joint_target_slew_max_step is not None:
            if self._joint_target_slew_reference is None:
                self._joint_target_slew_reference = joint_pos.detach().clone()
            target_step = joint_pos_des - self._joint_target_slew_reference
            maximum_component = torch.amax(torch.abs(target_step), dim=-1, keepdim=True)
            scale = torch.clamp(
                self._joint_target_slew_max_step / torch.clamp(maximum_component, min=1.0e-8),
                max=1.0,
            )
            limited_target_step = target_step * scale
            joint_pos_des = self._joint_target_slew_reference + limited_target_step
            soft_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            joint_pos_des = torch.clamp(joint_pos_des, min=soft_limits[..., 0], max=soft_limits[..., 1])
            self._last_joint_target_slew_step = joint_pos_des - self._joint_target_slew_reference
            self._joint_target_slew_reference = joint_pos_des.detach().clone()
            delta_joint_pos = joint_pos_des - joint_pos

        if solver_diagnostics_active:
            self._last_delta_joint_pos = delta_joint_pos.detach().clone()

        if not bool(torch.isfinite(joint_pos_des).all()):
            raise RuntimeError("Translation-priority differential IK produced a non-finite joint target")
        self._last_joint_position_target = joint_pos_des.detach().clone()
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
    def last_primary_delta_joint_pos(self) -> torch.Tensor | None:
        return self._last_primary_delta_joint_pos

    @property
    def last_nullspace_delta_joint_pos(self) -> torch.Tensor | None:
        return self._last_nullspace_delta_joint_pos

    @property
    def last_task_error(self) -> torch.Tensor | None:
        return self._last_task_error

    @property
    def last_task_singular_values(self) -> torch.Tensor | None:
        return self._last_task_singular_values

    @property
    def maximum_joint_target_step(self) -> float | None:
        return self._maximum_joint_target_step

    @property
    def last_unlimited_delta_joint_pos(self) -> torch.Tensor | None:
        return self._last_unlimited_delta_joint_pos

    @property
    def joint_target_slew_max_step(self) -> float | None:
        return self._joint_target_slew_max_step

    @property
    def last_joint_position_target(self) -> torch.Tensor | None:
        return self._last_joint_position_target

    @property
    def last_joint_target_slew_step(self) -> torch.Tensor | None:
        return self._last_joint_target_slew_step

    @property
    def controlled_joint_names(self) -> tuple[str, ...]:
        all_joint_names = self._asset.data.joint_names
        if isinstance(self._joint_ids, slice):
            return tuple(all_joint_names[self._joint_ids])
        return tuple(all_joint_names[joint_id] for joint_id in self._joint_ids)

    @property
    def direct_joint_position_target(self) -> torch.Tensor | None:
        return self._direct_joint_position_target

    @property
    def control_body_name(self) -> str:
        return self._body_name

    def _clear_xyz_pitch_joint_target(self) -> None:
        self._xyz_pitch_joint_name = None
        self._xyz_pitch_joint_index = None
        self._xyz_pitch_joint_target = None

    def _clear_xyz_joint_nullspace_target(self) -> None:
        self._xyz_joint_nullspace_name = None
        self._xyz_joint_nullspace_index = None
        self._xyz_joint_nullspace_target = None
        self._xyz_joint_nullspace_position_axes = (0, 1, 2)
        self._xyz_joint_nullspace_position_axes_are_world_frame = False

    def _clear_position_nullspace_posture_target(self) -> None:
        self._position_nullspace_posture_target = None

    def _clear_solver_diagnostics(self) -> None:
        self._last_delta_joint_pos = None
        self._last_primary_delta_joint_pos = None
        self._last_nullspace_delta_joint_pos = None
        self._last_task_error = None
        self._last_task_singular_values = None
        self._last_unlimited_delta_joint_pos = None
        self._last_joint_position_target = None
        self._last_joint_target_slew_step = None


def configure_servo_ik_action(env_cfg) -> None:
    """Use the phase-aware action term for one servo-expert environment config."""

    env_cfg.actions.arm_action.class_type = PhaseAwareDifferentialInverseKinematicsAction


def configure_dynamic_control_frame_offset(env_cfg) -> None:
    """Allocate an identity body offset that a phase-aware action may update at runtime."""

    env_cfg.actions.arm_action.body_offset = DifferentialInverseKinematicsActionCfg.OffsetCfg()


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
