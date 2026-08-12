"""Pure tensor geometry for square-cube gripper-axis alignment."""

from __future__ import annotations

from typing import NamedTuple

import torch

_MIN_PROJECTION_NORM = 1.0e-6


class CubeAxisAlignment(NamedTuple):
    """Nearest unoriented cube edge and its signed roll correction."""

    desired_axis_w: torch.Tensor
    signed_error: torch.Tensor
    absolute_error: torch.Tensor
    selected_axis_index: torch.Tensor
    selected_axis_sign: torch.Tensor
    selection_feasible: torch.Tensor


def _project_to_rotation_plane(
    vector_w: torch.Tensor,
    rotation_axis_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_axis_norm = torch.linalg.vector_norm(rotation_axis_w, dim=-1, keepdim=True)
    rotation_axis_valid = rotation_axis_norm.squeeze(-1) >= _MIN_PROJECTION_NORM
    rotation_axis_w = rotation_axis_w / torch.clamp(rotation_axis_norm, min=_MIN_PROJECTION_NORM)
    projected = vector_w - torch.sum(vector_w * rotation_axis_w, dim=-1, keepdim=True) * rotation_axis_w
    projected_norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
    projection_valid = rotation_axis_valid & (projected_norm.squeeze(-1) >= _MIN_PROJECTION_NORM)
    normalized = projected / torch.clamp(projected_norm, min=_MIN_PROJECTION_NORM)
    return normalized, projection_valid


def measure_cube_axis_alignment(
    closing_axis_w: torch.Tensor,
    rotation_axis_w: torch.Tensor,
    cube_x_axis_w: torch.Tensor,
    cube_y_axis_w: torch.Tensor,
    selected_axis_index: torch.Tensor,
    selected_axis_sign: torch.Tensor,
    selection_feasible: torch.Tensor | None = None,
) -> CubeAxisAlignment:
    """Measure alignment to one previously selected signed cube X/Y axis."""

    closing_axis_w, closing_axis_valid = _project_to_rotation_plane(closing_axis_w, rotation_axis_w)
    cube_x_axis_w, cube_x_axis_valid = _project_to_rotation_plane(cube_x_axis_w, rotation_axis_w)
    cube_y_axis_w, cube_y_axis_valid = _project_to_rotation_plane(cube_y_axis_w, rotation_axis_w)
    cube_axes_w = torch.stack(
        (cube_x_axis_w, cube_y_axis_w),
        dim=1,
    )
    cube_axes_valid = torch.stack((cube_x_axis_valid, cube_y_axis_valid), dim=1)
    batch_indices = torch.arange(closing_axis_w.shape[0], device=closing_axis_w.device)
    selected_axis_w = cube_axes_w[batch_indices, selected_axis_index]
    selected_axis_valid = cube_axes_valid[batch_indices, selected_axis_index]
    desired_axis_w = selected_axis_w * selected_axis_sign.unsqueeze(-1)
    rotation_axis_w = rotation_axis_w / torch.clamp(
        torch.linalg.vector_norm(rotation_axis_w, dim=-1, keepdim=True), min=1.0e-8
    )
    cross = torch.linalg.cross(closing_axis_w, desired_axis_w, dim=-1)
    sin_error = torch.sum(rotation_axis_w * cross, dim=-1)
    cos_error = torch.sum(closing_axis_w * desired_axis_w, dim=-1)
    signed_error = torch.atan2(sin_error, cos_error)
    if selection_feasible is None:
        selection_feasible = torch.ones_like(signed_error, dtype=torch.bool)
    selection_feasible = selection_feasible & closing_axis_valid & selected_axis_valid
    return CubeAxisAlignment(
        desired_axis_w=desired_axis_w,
        signed_error=signed_error,
        absolute_error=torch.abs(signed_error),
        selected_axis_index=selected_axis_index,
        selected_axis_sign=selected_axis_sign,
        selection_feasible=selection_feasible,
    )


def select_nearest_cube_axis_alignment(
    closing_axis_w: torch.Tensor,
    rotation_axis_w: torch.Tensor,
    cube_x_axis_w: torch.Tensor,
    cube_y_axis_w: torch.Tensor,
    *,
    joint_position: torch.Tensor | None = None,
    joint_limits: torch.Tensor | None = None,
    joint_limit_margin: float = 0.0,
) -> CubeAxisAlignment:
    """Select the signed cube X/Y axis needing the smallest wrist-roll rotation.

    A square edge is unoriented: ``+X`` and ``-X`` describe the same gripping
    line, as do ``+Y`` and ``-Y``.  Maximizing the absolute dot product first
    chooses X or Y, and its sign then avoids an unnecessary 180-degree turn.
    """

    closing_axis_plane_w, closing_axis_valid = _project_to_rotation_plane(closing_axis_w, rotation_axis_w)
    cube_x_axis_plane_w, cube_x_axis_valid = _project_to_rotation_plane(cube_x_axis_w, rotation_axis_w)
    cube_y_axis_plane_w, cube_y_axis_valid = _project_to_rotation_plane(cube_y_axis_w, rotation_axis_w)
    cube_axes_plane_w = torch.stack(
        (cube_x_axis_plane_w, cube_y_axis_plane_w),
        dim=1,
    )
    cube_axes_valid = torch.stack((cube_x_axis_valid, cube_y_axis_valid), dim=1)
    candidate_axes_w = torch.cat((cube_axes_plane_w, -cube_axes_plane_w), dim=1)
    candidate_axes_valid = torch.cat((cube_axes_valid, cube_axes_valid), dim=1)
    rotation_axis_w = rotation_axis_w / torch.clamp(
        torch.linalg.vector_norm(rotation_axis_w, dim=-1, keepdim=True), min=1.0e-8
    )
    cross = torch.linalg.cross(closing_axis_plane_w[:, None, :], candidate_axes_w, dim=-1)
    candidate_sin = torch.sum(rotation_axis_w[:, None, :] * cross, dim=-1)
    candidate_cos = torch.sum(closing_axis_plane_w[:, None, :] * candidate_axes_w, dim=-1)
    candidate_errors = torch.atan2(candidate_sin, candidate_cos)
    feasible = candidate_axes_valid & closing_axis_valid[:, None]
    if (joint_position is None) != (joint_limits is None):
        raise ValueError("joint_position and joint_limits must either both be provided or both be omitted")
    if joint_position is not None and joint_limits is not None:
        lower = joint_limits[:, 0:1] + joint_limit_margin
        upper = joint_limits[:, 1:2] - joint_limit_margin
        candidate_targets = joint_position[:, None] + candidate_errors
        feasible &= (candidate_targets >= lower) & (candidate_targets <= upper)
    feasible_selection = torch.any(feasible, dim=-1)
    selection_cost = torch.where(feasible, torch.abs(candidate_errors), torch.full_like(candidate_errors, torch.inf))
    fallback_cost = torch.where(
        candidate_axes_valid & closing_axis_valid[:, None],
        torch.abs(candidate_errors),
        torch.full_like(candidate_errors, torch.inf),
    )
    candidate_index = torch.argmin(
        torch.where(feasible_selection[:, None], selection_cost, fallback_cost),
        dim=-1,
    )
    selected_axis_index = candidate_index % 2
    batch_indices = torch.arange(closing_axis_w.shape[0], device=closing_axis_w.device)
    selected_axis_sign = torch.where(
        candidate_index < 2,
        torch.ones_like(candidate_errors[batch_indices, candidate_index]),
        -torch.ones_like(candidate_errors[batch_indices, candidate_index]),
    )
    return measure_cube_axis_alignment(
        closing_axis_w,
        rotation_axis_w,
        cube_x_axis_w,
        cube_y_axis_w,
        selected_axis_index,
        selected_axis_sign,
        feasible_selection,
    )
