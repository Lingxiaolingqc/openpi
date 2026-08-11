"""Keep the single-episode and randomized expert registries in sync."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENTRY_POINTS = (
    ROOT / "red_cube_to_box_expert_smoke.py",
    ROOT / "red_cube_to_box_expert_batch.py",
)


def _expert_choices(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--expert":
            continue
        choices = next(keyword.value for keyword in node.keywords if keyword.arg == "choices")
        return tuple(ast.literal_eval(choices))
    raise AssertionError(f"No --expert choices found in {path}")


def _expert_mapping_keys(path: Path) -> list[set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mappings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Dict):
            continue
        if not isinstance(node.slice, ast.Attribute) or node.slice.attr != "expert":
            continue
        if not isinstance(node.slice.value, ast.Name) or node.slice.value.id != "args":
            continue
        if all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in node.value.keys):
            mappings.append({key.value for key in node.value.keys})
    return mappings


def test_smoke_and_batch_expert_choices_match() -> None:
    smoke_choices, batch_choices = (_expert_choices(path) for path in ENTRY_POINTS)
    assert batch_choices == smoke_choices


def test_every_entry_point_maps_every_expert_choice() -> None:
    for path in ENTRY_POINTS:
        choices = set(_expert_choices(path))
        assert choices in _expert_mapping_keys(path), f"No complete expert mapping in {path}"
