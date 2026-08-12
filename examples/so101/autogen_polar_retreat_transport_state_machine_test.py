"""Static regression checks for the polar retreat transport expert."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parent / "red_cube_to_box_task" / (
    "autogen_polar_retreat_transport_state_machine.py"
)
IK_ACTION_PATH = Path(__file__).resolve().parent / "red_cube_to_box_task" / "phase_aware_ik_action.py"
CLASS_NAME = "RedCubeToBoxAutogenPolarRetreatTransportStateMachine"


def _class_node() -> ast.ClassDef:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME)


def _method(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _class_node().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_names(node: ast.AST) -> set[str]:
    return {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }


def test_retreat_controls_wrist_with_position_only_xyz() -> None:
    calls = _call_names(_method("get_action"))
    assert "set_control_body" in calls
    assert "set_position_only" in calls
    assert "set_xyz_joint_nullspace_target" not in calls
    assert "set_xz_joint_nullspace_target" not in calls
    assert "set_xyz_tilt" not in calls
    assert "set_xyz_pitch_joint_target" not in calls


def test_wrist_retreat_rebases_radial_target_after_actual_lift() -> None:
    initialize_source = ast.unparse(_method("_initialize_polar_retreat"))
    advance_source = ast.unparse(_method("_advance_retreat_segment_if_ready"))
    reference_source = ast.unparse(_method("_retreat_linear_reference"))
    convergence_source = ast.unparse(_method("_update_retreat_convergence"))

    assert "_retreat_control_position_w" in initialize_source
    assert "wrist_flex" not in initialize_source
    assert "_retreat_radial_target_w = None" in initialize_source
    assert "start_w = self._retreat_control_position_w(env)" in advance_source
    assert "target_radius = _RETREAT_RADIAL_SCALE * start_radius" in advance_source
    assert "target_w[:, :2] = robot_root_w[:, :2] + _RETREAT_RADIAL_SCALE * delta_xy" in advance_source
    assert "torch.clamp" not in advance_source
    assert "return self._motion_start_w + progress * displacement" in reference_source
    assert "_retreat_control_position_w" not in reference_source
    assert "if self._retreat_subphase == 'vertical_lift'" in convergence_source
    assert "self._target_error = float(position_error.max().item())" in convergence_source
    assert "self._bearing_error <= _BEARING_TOLERANCE" in convergence_source


def test_xz_joint_mode_omits_y_position_and_jacobian_rows() -> None:
    tree = ast.parse(IK_ACTION_PATH.read_text(encoding="utf-8"), filename=str(IK_ACTION_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PhaseAwareDifferentialInverseKinematicsAction"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_xz_joint_nullspace_target"
    )
    source = ast.unparse(method)
    assert "self._xyz_joint_nullspace_position_axes = (0, 2)" in source
    assert "self._xyz_joint_nullspace_position_axes_are_world_frame = True" in source

    apply_method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_actions"
    )
    apply_source = ast.unparse(apply_method)
    assert "root_rotation_w = matrix_from_quat(self._asset.data.root_quat_w)" in apply_source
    assert "position_jacobian = root_rotation_w @ position_jacobian" in apply_source


def test_runtime_control_body_switch_updates_pose_and_jacobian_indices() -> None:
    tree = ast.parse(IK_ACTION_PATH.read_text(encoding="utf-8"), filename=str(IK_ACTION_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PhaseAwareDifferentialInverseKinematicsAction"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_control_body"
    )
    source = ast.unparse(method)
    assert "self._body_idx = body_ids[0]" in source
    assert "self._body_name = body_names[0]" in source
    assert "self._jacobi_body_idx" in source


def test_close_phase_has_feedback_settle_gate() -> None:
    class_node = _class_node()
    fixed_steps = next(
        ast.literal_eval(statement.value)
        for statement in class_node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "_FIXED_PHASE_STEPS"
    )
    assert "close_gripper" not in fixed_steps

    advance_source = ast.unparse(_method("advance"))
    assert "_GRIPPER_CLOSE_MINIMUM_STEPS" in advance_source
    assert "_GRIPPER_CLOSE_MAXIMUM_STEPS" in advance_source
    assert "_GRIPPER_SETTLE_STABLE_STEPS" in advance_source

    settle_source = ast.unparse(_method("_update_gripper_settle"))
    assert "_GRASP_CONFIRM_DISTANCE" in settle_source
    assert "halfway_closed & close_enough_to_cube" in settle_source
    assert "_GRIPPER_SETTLE_VELOCITY_TOLERANCE" not in settle_source
