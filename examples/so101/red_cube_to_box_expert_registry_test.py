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
PHASE_AWARE_IK_CLASS = "PhaseAwareDifferentialInverseKinematicsAction"


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


def _method_definition(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _attribute_calls(node: ast.AST) -> set[str]:
    return {
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


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


def test_direct_joint_hold_action_writes_the_complete_controlled_joint_vector() -> None:
    path = TASK_ROOT / "phase_aware_ik_action.py"
    class_node = _class_definition(path, PHASE_AWARE_IK_CLASS)
    method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
    assert {
        "set_direct_joint_position_target",
        "clear_direct_joint_position_target",
        "direct_joint_position_target",
    } <= method_names

    setter = _method_definition(class_node, "set_direct_joint_position_target")
    setter_source = ast.unparse(setter)
    assert "controlled_joint_names" in setter_source or "_joint_ids" in setter_source

    runtime_mode = _method_definition(class_node, "runtime_mode")
    assert "direct_joint_hold" in {child.value for child in ast.walk(runtime_mode) if isinstance(child, ast.Constant)}

    apply_actions = _method_definition(class_node, "apply_actions")
    direct_branches = [
        child
        for child in ast.walk(apply_actions)
        if isinstance(child, ast.If) and "direct_joint_position_target" in ast.unparse(child.test)
    ]
    assert direct_branches
    assert any(
        "set_joint_position_target" in _attribute_calls(branch)
        and "_joint_ids" in ast.unparse(branch)
        and any(isinstance(child, ast.Return) for child in ast.walk(branch))
        for branch in direct_branches
    )

    reset = _method_definition(class_node, "reset")
    assert "clear_direct_joint_position_target" in _attribute_calls(reset)


def test_only_axis_alignment_variant_uses_and_clears_direct_joint_hold() -> None:
    slow_path = TASK_ROOT / "autogen_reference_slow_grasp_state_machine.py"
    slow_tree = ast.parse(slow_path.read_text(encoding="utf-8"), filename=str(slow_path))
    slow_calls = _attribute_calls(slow_tree)
    assert "set_direct_joint_position_target" not in slow_calls
    assert "clear_direct_joint_position_target" not in slow_calls

    axis_path = TASK_ROOT / "autogen_reference_axis_align_slow_grasp_state_machine.py"
    axis_class = _class_definition(axis_path, AXIS_ALIGN_CLASS)
    axis_calls = _attribute_calls(axis_class)
    assert "set_direct_joint_position_target" in axis_calls
    assert "clear_direct_joint_position_target" in axis_calls

    axis_source = ast.unparse(axis_class)
    assert "controlled_joint_names" in axis_source
    assert "joint_pos" in axis_source
    assert "wrist_roll" in axis_source

    reset = _method_definition(axis_class, "reset")
    posture_update = _method_definition(axis_class, "_update_posture_target")
    cleanup_calls = _attribute_calls(reset) | _attribute_calls(posture_update)
    assert "clear_direct_joint_position_target" in cleanup_calls

    posture_source = ast.unparse(posture_update)
    assert "if self._ik_handoff_target_b is None" in posture_source
    assert "self._ik_handoff_target_b = self._command_pos_b.detach().clone()" in posture_source
    assert "self._command_pos_b = self._ik_handoff_target_b.detach().clone()" in posture_source
    handoff_branch = next(
        child
        for child in ast.walk(posture_update)
        if isinstance(child, ast.If) and ast.unparse(child.test) == "self._state == 'ik_handoff'"
    )
    assert (
        sum(
            1
            for child in ast.walk(handoff_branch)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "_rebase_command_to_measured_wrist"
        )
        == 1
    )
