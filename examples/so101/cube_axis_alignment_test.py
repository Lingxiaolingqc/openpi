from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch

MODULE_PATH = Path(__file__).with_name("red_cube_to_box_task") / "cube_axis_alignment.py"
MODULE_SPEC = importlib.util.spec_from_file_location("cube_axis_alignment", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load cube-axis helper from {MODULE_PATH}")
cube_axis_alignment = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = cube_axis_alignment
MODULE_SPEC.loader.exec_module(cube_axis_alignment)
select_nearest_cube_axis_alignment = cube_axis_alignment.select_nearest_cube_axis_alignment


def _axis(degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    return torch.tensor([[math.cos(radians), math.sin(radians), 0.0]], dtype=torch.float64)


def test_selects_nearest_unoriented_cube_axis() -> None:
    alignment = select_nearest_cube_axis_alignment(
        _axis(80.0),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        _axis(30.0),
        _axis(120.0),
    )

    assert alignment.selected_axis_index.tolist() == [1]
    assert alignment.selected_axis_sign.tolist() == [1.0]
    torch.testing.assert_close(alignment.signed_error, torch.tensor([math.radians(40.0)], dtype=torch.float64))


def test_antiparallel_cube_axis_is_already_aligned() -> None:
    alignment = select_nearest_cube_axis_alignment(
        _axis(180.0),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        _axis(0.0),
        _axis(90.0),
    )

    assert alignment.selected_axis_index.tolist() == [0]
    assert alignment.selected_axis_sign.tolist() == [-1.0]
    torch.testing.assert_close(alignment.absolute_error, torch.zeros(1, dtype=torch.float64), atol=1.0e-12, rtol=0.0)


def test_alignment_uses_plane_orthogonal_to_roll_axis() -> None:
    roll_axis = torch.tensor([[0.0, math.sin(math.radians(17.0)), math.cos(math.radians(17.0))]], dtype=torch.float64)
    alignment = select_nearest_cube_axis_alignment(
        _axis(10.0),
        roll_axis,
        _axis(30.0),
        _axis(120.0),
    )

    torch.testing.assert_close(
        torch.sum(alignment.desired_axis_w * roll_axis, dim=-1),
        torch.zeros(1, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert bool((alignment.absolute_error <= math.pi / 4.0 + 1.0e-12).all())


def test_selects_longer_rotation_when_nearest_axis_crosses_joint_limit() -> None:
    alignment = select_nearest_cube_axis_alignment(
        _axis(10.0),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        _axis(30.0),
        _axis(120.0),
        joint_position=torch.tensor([[0.95]], dtype=torch.float64).squeeze(-1),
        joint_limits=torch.tensor([[-1.0, 1.0]], dtype=torch.float64),
        joint_limit_margin=0.02,
    )

    assert alignment.selection_feasible.tolist() == [True]
    assert alignment.selected_axis_index.tolist() == [1]
    assert alignment.selected_axis_sign.tolist() == [-1.0]
    torch.testing.assert_close(alignment.signed_error, torch.tensor([-math.radians(70.0)], dtype=torch.float64))


def test_ignores_cube_axis_parallel_to_roll_axis() -> None:
    alignment = select_nearest_cube_axis_alignment(
        _axis(90.0),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
    )

    assert alignment.selection_feasible.tolist() == [True]
    assert alignment.selected_axis_index.tolist() == [1]
    torch.testing.assert_close(alignment.absolute_error, torch.zeros(1, dtype=torch.float64), atol=1.0e-12, rtol=0.0)


def test_reports_infeasible_when_closing_axis_is_parallel_to_roll_axis() -> None:
    alignment = select_nearest_cube_axis_alignment(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        _axis(0.0),
        _axis(90.0),
    )

    assert alignment.selection_feasible.tolist() == [False]
