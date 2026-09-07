"""Damped least-squares inverse kinematics for the PiPER gripper site."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from examples.piper import contract


@dataclass(frozen=True)
class IKResult:
    q_rad: np.ndarray
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float


def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Small-angle world-frame error between two 3x3 rotation matrices."""

    return 0.5 * sum(np.cross(current[:, axis], target[:, axis]) for axis in range(3))


def solve_site_ik(
    env,
    target_position_m: np.ndarray,
    *,
    target_rotation: np.ndarray | None = None,
    initial_q_rad: np.ndarray | None = None,
    damping: float = 0.04,
    orientation_weight: float = 0.25,
    step_size: float = 0.7,
    max_iterations: int = 120,
    position_tolerance_m: float = 0.003,
    orientation_tolerance_rad: float = 0.08,
) -> IKResult:
    """Solve one absolute end-effector target while respecting Menagerie joint limits."""

    target_position_m = np.asarray(target_position_m, dtype=np.float64)
    if target_position_m.shape != (3,) or not np.isfinite(target_position_m).all():
        raise ValueError("target_position_m must be finite shape (3,)")
    if target_rotation is not None:
        target_rotation = np.asarray(target_rotation, dtype=np.float64)
        if target_rotation.shape != (3, 3) or not np.isfinite(target_rotation).all():
            raise ValueError("target_rotation must be finite shape (3, 3)")
    q = np.asarray(initial_q_rad if initial_q_rad is not None else env.state()[:6], dtype=np.float64).copy()
    if q.shape != (6,):
        raise ValueError("initial_q_rad must have shape (6,)")

    mujoco = env.mujoco
    original_data = env.data
    work_data = mujoco.MjData(env.model)
    mujoco.mj_copyData(work_data, env.model, original_data)
    env.data = work_data
    jacobian_position = np.zeros((3, env.model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, env.model.nv), dtype=np.float64)
    orientation_error = np.zeros(3, dtype=np.float64)
    position_error = np.zeros(3, dtype=np.float64)
    converged = False
    iterations = 0

    try:
        for _iteration in range(1, max_iterations + 1):
            iterations = _iteration
            env.set_arm_configuration(q)
            position_error = target_position_m - env.gripper_position
            orientation_error = (
                np.zeros(3, dtype=np.float64)
                if target_rotation is None
                else _orientation_error(env.gripper_rotation, target_rotation)
            )
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(orientation_error))
            if position_norm <= position_tolerance_m and (
                target_rotation is None or orientation_norm <= orientation_tolerance_rad
            ):
                converged = True
                break

            jacobian_position.fill(0.0)
            jacobian_rotation.fill(0.0)
            mujoco.mj_jacSite(
                env.model,
                env.data,
                jacobian_position,
                jacobian_rotation,
                env.gripper_site_id,
            )
            arm_dofs = env.arm_dof_ids
            if target_rotation is None:
                jacobian = jacobian_position[:, arm_dofs]
                error = position_error
            else:
                jacobian = np.vstack(
                    [jacobian_position[:, arm_dofs], orientation_weight * jacobian_rotation[:, arm_dofs]]
                )
                error = np.concatenate([position_error, orientation_weight * orientation_error])
            regularized = jacobian @ jacobian.T + (damping**2) * np.eye(jacobian.shape[0])
            dq = jacobian.T @ np.linalg.solve(regularized, error)
            maximum = float(np.max(np.abs(dq)))
            if maximum > 0.20:
                dq *= 0.20 / maximum
            q = np.clip(q + step_size * dq, contract.ARM_Q_MIN_APP_RAD, contract.ARM_Q_MAX_APP_RAD)

        env.set_arm_configuration(q)
        position_error = target_position_m - env.gripper_position
        orientation_error = (
            np.zeros(3, dtype=np.float64)
            if target_rotation is None
            else _orientation_error(env.gripper_rotation, target_rotation)
        )
    finally:
        env.data = original_data
    return IKResult(
        q_rad=q.astype(np.float32),
        converged=converged,
        iterations=iterations,
        position_error_m=float(np.linalg.norm(position_error)),
        orientation_error_rad=float(np.linalg.norm(orientation_error)),
    )
