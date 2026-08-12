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


def test_retreat_controls_wrist_xyz_with_one_joint_posture_row() -> None:
    calls = _call_names(_method("get_action"))
    assert "set_control_body" in calls
    assert "set_xyz_joint_nullspace_target" in calls
    assert "set_xyz_tilt" not in calls
    assert "set_xyz_pitch_joint_target" not in calls


def test_wrist_retreat_rebases_radial_target_after_actual_lift() -> None:
    initialize_source = ast.unparse(_method("_initialize_polar_retreat"))
    advance_source = ast.unparse(_method("_advance_retreat_segment_if_ready"))
    convergence_source = ast.unparse(_method("_update_retreat_convergence"))

    assert "_retreat_control_position_w" in initialize_source
    assert "_retreat_radial_target_w = None" in initialize_source
    assert "start_w = self._retreat_control_position_w(env)" in advance_source
    assert "target_radius = torch.clamp(start_radius - _RETREAT_DISTANCE" in advance_source
    assert "if self._retreat_subphase == 'vertical_lift'" in convergence_source
    assert "self._target_error = self._retreat_z_error" in convergence_source


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
