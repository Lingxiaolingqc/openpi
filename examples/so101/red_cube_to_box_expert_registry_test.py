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
    assert "self._ik_handoff_joint_posture_target = robot.data.joint_pos" in posture_source
    assert "set_direct_joint_position_target(self._ik_handoff_joint_posture_target)" in posture_source
    assert "_release_direct_joint_hold('measured_joint_handoff_settled')" in posture_source
    assert "set_position_only_nullspace_posture_target" not in posture_source
    assert "set_xyz_joint_nullspace_target" not in ast.unparse(
        next(
            child
            for child in ast.walk(posture_update)
            if isinstance(child, ast.If) and ast.unparse(child.test) == "self._state == 'ik_handoff'"
        )
    )
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
    transition_source = ast.unparse(_method_definition(axis_class, "_transition"))
    assert "self._state != 'ik_handoff'" in transition_source


def test_axis_alignment_waits_for_the_measured_wrist_before_direct_hold() -> None:
    path = TASK_ROOT / "autogen_reference_axis_align_slow_grasp_state_machine.py"
    class_node = _class_definition(path, AXIS_ALIGN_CLASS)

    grasp_hook = _method_definition(class_node, "_on_grasp_pose_reached")
    grasp_hook_source = ast.unparse(grasp_hook)
    assert "self._ray_hit_tracking_target_b = self._command_pos_b.detach().clone()" in grasp_hook_source
    assert "self._transition('ray_hit_tracking_settle')" in grasp_hook_source
    assert "_capture_direct_joint_hold" not in _attribute_calls(grasp_hook)

    state_update = _method_definition(class_node, "_update_state")
    tracking_branch = next(
        child
        for child in ast.walk(state_update)
        if isinstance(child, ast.If) and ast.unparse(child.test) == "self._state == 'ray_hit_tracking_settle'"
    )
    assert "_update_ray_hit_tracking_settle" in _attribute_calls(tracking_branch)
    assert any(isinstance(child, ast.Return) for child in ast.walk(tracking_branch))

    tracking_update = _method_definition(class_node, "_update_ray_hit_tracking_settle")
    tracking_source = ast.unparse(tracking_update)
    assert "self._command_pos_b = self._ray_hit_tracking_target_b.detach().clone()" in tracking_source
    assert "body_pos_w" in tracking_source
    assert "joint_vel" in tracking_source
    assert "self._ray_hit_tracking_streak" in tracking_source
    assert "DESCEND_STEP" not in tracking_source
    assert "_update_move" not in _attribute_calls(tracking_update)
    stable_assignment = next(
        child
        for child in tracking_update.body
        if isinstance(child, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "stable" for target in child.targets)
    )
    stable_source = ast.unparse(stable_assignment.value)
    assert "self.green_ray_hit" in stable_source
    assert "self.RAY_HIT_TRACKING_POSITION_TOLERANCE" in stable_source
    assert "self.AXIS_ALIGNMENT_ARM_JOINT_VELOCITY_TOLERANCE" in stable_source

    stable_gate = next(
        child
        for child in tracking_update.body
        if isinstance(child, ast.If)
        and "self._ray_hit_tracking_streak >= self.RAY_HIT_TRACKING_STABLE_STEPS" in ast.unparse(child.test)
    )
    stable_gate_source = ast.unparse(stable_gate)
    capture_index = stable_gate_source.index("self._capture_direct_joint_hold(env)")
    rebase_index = stable_gate_source.index("self._rebase_command_to_measured_wrist(env)")
    transition_index = stable_gate_source.index("self._transition('pregrasp_axis_align')")
    assert capture_index < rebase_index < transition_index


def test_autogen_pick_hold_preserves_nominal_target_and_uses_velocity_feedback() -> None:
    path = TASK_ROOT / "autogen_reference_state_machine.py"
    class_node = _class_definition(path, "RedCubeToBoxAutogenReferenceStateMachine")
    observer = _method_definition(class_node, "observe_pick_cube")
    observer_source = ast.unparse(observer)

    assert "joint_vel" in observer_source
    assert "allow_capture=self._state in self.PICK_HOLD_CAPTURE_PHASES" in observer_source
    assert "track_loss=self._state in self.PICK_HOLD_LOSS_TRACKING_PHASES" in observer_source
    assert "self._grasp_end_position = self._held_gripper_angle" not in observer_source
    assert "self._gripper_command = self._grasp_end_position" not in observer_source
    assert "elif self._gripper_pick_latch.held_angle is None" not in observer_source
    assert observer_source.count("self._gripper_command =") == 2

    transition_source = ast.unparse(_method_definition(class_node, "_transition"))
    assert transition_source.count("self._gripper_pick_latch.release()") == 1
    assert "if state == 'release'" in transition_source

    state_source = ast.unparse(_method_definition(class_node, "_update_state"))
    assert "env.scene['cube'].data.root_lin_vel_w" in state_source
    assert "self._gripper_settle_angle_span <= self.GRIPPER_SETTLE_ANGLE_SPAN_TOLERANCE" in state_source
    assert "self._cube_settle_max_speed <= self.CUBE_SETTLE_SPEED" not in state_source
    assert "latched_gripper_window_stable_cube_speed_diagnostic_only" in state_source
    assert "angle_span_rad=" in state_source
    assert "cube_max_speed_m_s=" in state_source


def test_smoke_recorder_uses_the_post_action_phase_and_forces_phase_boundaries() -> None:
    smoke_path = ROOT / "red_cube_to_box_expert_smoke.py"
    tree = ast.parse(smoke_path.read_text(encoding="utf-8"), filename=str(smoke_path))
    source = ast.unparse(tree)
    assert "recorded_phase = state_machine.phase_name" in source
    assert "force=recorded_phase != phase" in source


def test_axis_handoff_smoke_trace_exposes_joint_level_ik_diagnostics() -> None:
    source = (ROOT / "red_cube_to_box_expert_smoke.py").read_text(encoding="utf-8")

    assert '"expert_axis_handoff"' in source
    assert "state_machine.ik_handoff_joint_posture_target" in source
    assert "arm_action_term.last_joint_position_target" in source
    assert "arm_action_term.last_primary_delta_joint_pos" in source
    assert "arm_action_term.last_nullspace_delta_joint_pos" in source
    assert "joint_target_minus_actual" in source
    assert "actual_joint_velocity" in source
