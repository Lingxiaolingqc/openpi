"""Keep the single-episode and randomized expert registries in sync."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT / "red_cube_to_box_task"
ENTRY_POINTS = (
    ROOT / "red_cube_to_box_expert_smoke.py",
    ROOT / "red_cube_to_box_expert_batch.py",
)
SLOW_GRASP_CLASS = "RedCubeToBoxAutogenReferenceSlowGraspStateMachine"
AXIS_ALIGN_CLASS = "RedCubeToBoxAutogenReferenceAxisAlignSlowGraspStateMachine"


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


def _class_definition(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)


def _expert_class_mappings(path: Path) -> list[dict[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mappings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Dict):
            continue
        mapping = {
            key.value: value.id
            for key, value in zip(node.value.keys, node.value.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Name)
        }
        if "autogen_reference_slow_grasp" in mapping:
            mappings.append(mapping)
    return mappings


def test_smoke_and_batch_expert_choices_match() -> None:
    smoke_choices, batch_choices = (_expert_choices(path) for path in ENTRY_POINTS)
    assert batch_choices == smoke_choices


def test_every_entry_point_maps_every_expert_choice() -> None:
    for path in ENTRY_POINTS:
        choices = set(_expert_choices(path))
        mappings = _expert_mapping_keys(path)
        assert mappings, f"No expert mapping found in {path}"
        assert all(mapping == choices for mapping in mappings), f"An expert mapping is incomplete in {path}"


def test_slow_grasp_expert_is_a_single_variable_subclass() -> None:
    path = TASK_ROOT / "autogen_reference_slow_grasp_state_machine.py"
    class_node = _class_definition(path, SLOW_GRASP_CLASS)

    assert [ast.unparse(base) for base in class_node.bases] == ["RedCubeToBoxAutogenReferenceStateMachine"]
    class_assignments = {
        target.id: ast.literal_eval(statement.value)
        for statement in class_node.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    assert class_assignments == {"GRASP_DURATION_STEPS": 240}
    assert {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)} == {"servo_parameters"}


def test_axis_alignment_expert_inherits_the_same_slow_close() -> None:
    path = TASK_ROOT / "autogen_reference_axis_align_slow_grasp_state_machine.py"
    class_node = _class_definition(path, AXIS_ALIGN_CLASS)
    assert [ast.unparse(base) for base in class_node.bases] == [SLOW_GRASP_CLASS]

    for entry_point in ENTRY_POINTS:
        mappings = _expert_class_mappings(entry_point)
        assert mappings
        assert all(mapping["autogen_reference_slow_grasp"] == SLOW_GRASP_CLASS for mapping in mappings)
        assert all(mapping["autogen_reference_axis_align_slow_grasp"] == AXIS_ALIGN_CLASS for mapping in mappings)
