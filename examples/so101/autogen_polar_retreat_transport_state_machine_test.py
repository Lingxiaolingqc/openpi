"""Static regression checks for the polar retreat transport expert."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_PATH = (
    Path(__file__).resolve().parent / "red_cube_to_box_task" / ("autogen_polar_retreat_transport_state_machine.py")
)
IK_ACTION_PATH = Path(__file__).resolve().parent / "red_cube_to_box_task" / "phase_aware_ik_action.py"
POLAR_BASE_PATH = Path(__file__).resolve().parent / "red_cube_to_box_task" / "polar_base_state_machine.py"
FAILED_ROOT = Path(__file__).resolve().parent / "red_cube_to_box_task" / "failed"
BATCH_PATH = Path(__file__).resolve().parent / "red_cube_to_box_expert_batch.py"
ENV_CFG_PATH = Path(__file__).resolve().parent / "red_cube_to_box_task" / "env_cfg.py"
CLASS_NAME = "RedCubeToBoxAutogenPolarRetreatTransportStateMachine"


def _class_node() -> ast.ClassDef:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME)


def _method(name: str) -> ast.FunctionDef:
    return next(node for node in _class_node().body if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_names(node: ast.AST) -> set[str]:
    return {
        call.func.attr for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }


def test_polar_uses_active_base_while_independent_remains_an_archived_compatibility_expert() -> None:
    polar_class = _class_node()
    assert [ast.unparse(base) for base in polar_class.bases] == ["RedCubeToBoxPolarBaseStateMachine"]

    base_tree = ast.parse(POLAR_BASE_PATH.read_text(encoding="utf-8"), filename=str(POLAR_BASE_PATH))
    base_classes = {node.name for node in base_tree.body if isinstance(node, ast.ClassDef)}
    assert "RedCubeToBoxPolarBaseStateMachine" in base_classes
    assert "RedCubeToBoxAutogenIndependentRetreatTransportStateMachine" not in base_classes

    compatibility_path = FAILED_ROOT / "autogen_independent_retreat_transport_state_machine.py"
    compatibility_tree = ast.parse(
        compatibility_path.read_text(encoding="utf-8"),
        filename=str(compatibility_path),
    )
    compatibility_class = next(node for node in compatibility_tree.body if isinstance(node, ast.ClassDef))
    assert compatibility_class.name == "RedCubeToBoxAutogenIndependentRetreatTransportStateMachine"
    assert [ast.unparse(base) for base in compatibility_class.bases] == ["RedCubeToBoxPolarBaseStateMachine"]


def test_max_steps_does_not_use_a_class_scope_comprehension() -> None:
    class_node = _class_node()
    max_steps = next(
        statement
        for statement in class_node.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MAX_STEPS" for target in statement.targets)
    )

    assert not any(isinstance(node, ast.GeneratorExp | ast.ListComp) for node in ast.walk(max_steps.value))


def test_failed_experts_keep_their_existing_smoke_and_batch_cli_registrations() -> None:
    root = Path(__file__).resolve().parent
    for runner_name in ("red_cube_to_box_expert_smoke.py", "red_cube_to_box_expert_batch.py"):
        source = (root / runner_name).read_text(encoding="utf-8")
        assert "red_cube_to_box_task.failed.autogen_independent_retreat_transport_state_machine" in source
        assert '"autogen_independent_retreat_transport"' in source
        assert "red_cube_to_box_task.failed.servo_state_machine" in source
        assert '"servo"' in source


def test_retreat_controls_wrist_with_position_only_xyz() -> None:
    calls = _call_names(_method("get_action"))
    mode_calls = _call_names(_method("_configure_wrist_position_posture_mode"))
    assert "_configure_wrist_position_posture_mode" in calls
    assert "set_control_body" in mode_calls
    assert "set_position_only_nullspace_posture_target" in mode_calls
    assert "set_xyz_joint_nullspace_target" not in calls
    assert "set_xz_joint_nullspace_target" not in calls
    assert "set_xyz_tilt" not in calls
    assert "set_xyz_pitch_joint_target" not in calls


def test_arc_controls_wrist_with_position_only_xyz_and_wrist_feedback() -> None:
    get_action_source = ast.unparse(_method("get_action"))
    initialize_source = ast.unparse(_method("_initialize_arc_transfer"))
    reference_source = ast.unparse(_method("_arc_reference"))
    convergence_source = ast.unparse(_method("_update_polar_convergence"))

    assert "position_only_accumulation_phases" in get_action_source
    for phase in ("retreat_to_safe", "arc_transfer", "radial_transfer", "lower_into_box"):
        assert repr(phase) in get_action_source
    assert "self._configure_wrist_position_posture_mode()" in get_action_source
    assert "start_w = self._retreat_control_position_w(env)" in initialize_source
    assert "self._transport_height = None" in initialize_source
    assert "target_quat_w = self._retreat_control_quaternion_w(env)" in reference_source
    assert "if phase == 'arc_transfer'" in convergence_source
    assert "actual_w = self._retreat_control_position_w(env)" in convergence_source


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
    vertical_lift_branch, radial_retreat_branch = convergence_source.split("else:", maxsplit=1)
    assert "if self._retreat_subphase == 'vertical_lift'" in vertical_lift_branch
    assert "self._target_error = self._retreat_z_error" in vertical_lift_branch
    assert "self._bearing_error <= _BEARING_TOLERANCE" not in vertical_lift_branch
    assert "self._target_error = max(self._retreat_radial_error, self._retreat_z_error)" in radial_retreat_branch
    assert "self._retreat_radial_error <= tolerance" in radial_retreat_branch
    assert "self._retreat_z_error <= tolerance" in radial_retreat_branch
    assert "self._bearing_error <= _RETREAT_BEARING_TOLERANCE" in radial_retreat_branch


def test_position_only_posture_is_projected_into_xyz_nullspace() -> None:
    tree = ast.parse(IK_ACTION_PATH.read_text(encoding="utf-8"), filename=str(IK_ACTION_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PhaseAwareDifferentialInverseKinematicsAction"
    )
    setter = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_position_only_nullspace_posture_target"
    )
    apply_method = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "apply_actions"
    )
    setter_source = ast.unparse(setter)
    apply_source = ast.unparse(apply_method)
    assert "self.set_position_only(enabled=True)" in setter_source
    assert "self._position_nullspace_posture_target - joint_pos" in apply_source
    assert "nullspace_projector = joint_identity - damped_pseudoinverse @ task_jacobian" in apply_source
    assert "delta_joint_pos = primary_delta + nullspace_delta" in apply_source


def test_wrist_ik_accumulates_limited_deltas_without_reanchoring_to_live_joints() -> None:
    initialize_source = ast.unparse(_method("_initialize_polar_retreat"))
    enable_source = ast.unparse(_method("_enable_position_joint_target_accumulation"))
    disable_source = ast.unparse(_method("_disable_position_joint_target_accumulation"))
    get_action_source = ast.unparse(_method("get_action"))

    assert "self._enable_position_joint_target_accumulation()" in initialize_source
    assert "set_joint_target_accumulation" in enable_source
    assert "reset_joint_target_accumulation_reference" in enable_source
    assert "maximum_step=_WRIST_JOINT_TARGET_ACCUMULATION_STEP" in enable_source
    assert "set_joint_target_accumulation(maximum_step=None)" in disable_source
    assert "self._disable_position_joint_target_accumulation()" in get_action_source

    tree = ast.parse(IK_ACTION_PATH.read_text(encoding="utf-8"), filename=str(IK_ACTION_PATH))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PhaseAwareDifferentialInverseKinematicsAction"
    )
    apply_method = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "apply_actions"
    )
    apply_source = ast.unparse(apply_method)
    assert "joint_pos_des = self._joint_target_accumulation_reference + accumulated_step" in apply_source
    assert "accumulated_step = delta_joint_pos * scale" in apply_source
    assert "self._joint_target_accumulation_reference = joint_pos_des.detach().clone()" in apply_source
    assert "joint_pos_des = joint_pos + delta_joint_pos" in apply_source
    assert "tracking_limit_reached" not in apply_source
    assert "freeze_accumulation" not in apply_source


def test_radial_transfer_rebases_accumulator_and_uses_gripper_position_only() -> None:
    get_action_source = ast.unparse(_method("get_action"))
    initialize_source = ast.unparse(_method("_initialize_radial_transfer"))
    configure_source = ast.unparse(_method("_configure_gripper_position_posture_mode"))

    assert "self._configure_gripper_position_posture_mode()" in get_action_source
    assert "self._radial_posture_target" in initialize_source
    assert "self._enable_position_joint_target_accumulation()" in initialize_source
    assert "reset_joint_target_accumulation_reference" in initialize_source
    assert "jaw_from_gripper_xy = jaw_w[:, :2] - start_w[:, :2]" in initialize_source
    assert "target_w[:, :2] -= jaw_from_gripper_xy" in initialize_source
    assert "set_control_body(body_name='gripper')" in configure_source
    assert "set_position_only_nullspace_posture_target" in configure_source


def test_lower_rebases_position_only_accumulator_and_completes_on_z() -> None:
    get_action_source = ast.unparse(_method("get_action"))
    initialize_source = ast.unparse(_method("_initialize_lower"))
    configure_source = ast.unparse(_method("_configure_lower_position_posture_mode"))
    convergence_source = ast.unparse(_method("_update_lower_convergence"))
    vertical_convergence_source = ast.unparse(_method("_update_vertical_convergence"))

    assert "self._configure_lower_position_posture_mode()" in get_action_source
    assert "self._update_lower_convergence(env)" in get_action_source
    assert "self._lower_posture_target" in initialize_source
    assert "self._enable_position_joint_target_accumulation()" in initialize_source
    assert "reset_joint_target_accumulation_reference" in initialize_source
    assert "self._motion_target_w[:, :2] = self._motion_start_w[:, :2]" in initialize_source
    assert "set_control_body(body_name='gripper')" in configure_source
    assert "set_position_only_nullspace_posture_target" in configure_source
    assert "self._update_vertical_convergence(env, 'lower_into_box')" in convergence_source
    assert "torch.abs(self._motion_target_w[:, 2] - actual_w[:, 2])" in vertical_convergence_source
    assert "vector_norm" not in vertical_convergence_source


def test_release_holds_actual_pose_and_retracts_vertically() -> None:
    release_source = ast.unparse(_method("_initialize_release"))
    retract_source = ast.unparse(_method("_initialize_retract"))
    get_action_source = ast.unparse(_method("get_action"))

    assert "self._set_motion(actual_w, actual_w)" in release_source
    assert "target_w = start_w.clone()" in retract_source
    assert "target_w[:, 2] = self._floor_anchor_w[:, 2] + _BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER" in retract_source
    assert "self._update_vertical_convergence(env, phase)" in get_action_source


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
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "apply_actions"
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
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "set_control_body"
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
    assert "_GRIPPER_SETTLE_WINDOW_STEPS" in settle_source
    assert "_GRIPPER_SETTLE_ANGLE_SPAN_TOLERANCE" in settle_source
    assert "halfway_closed & grasp_geometry_confirmed & aperture_stable" in settle_source
    assert "_GRIPPER_SETTLE_VELOCITY_TOLERANCE" not in settle_source


def test_randomized_pickup_preserves_known_full_pose_fixed_keyframes() -> None:
    get_action_source = ast.unparse(_method("get_action"))
    retreat_source = ast.unparse(_method("_initialize_polar_retreat"))

    pickup_branch = get_action_source.split("if phase in {'approach_cube', 'descend_to_cube', 'close_gripper'}:", 1)[1]
    pickup_branch = pickup_branch.split("if self._arm_action_term is None:", 1)[0]
    assert "_configure_pickup_position_posture_mode" not in pickup_branch
    assert "_update_pickup_convergence" not in pickup_branch
    assert "reset_joint_target_accumulation_reference" in retreat_source


def test_pickup_aligns_closing_axis_with_cube_before_close() -> None:
    class_node = _class_node()
    phases = next(
        ast.literal_eval(statement.value)
        for statement in class_node.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_PHASES" for target in statement.targets)
    )
    assert phases.index("descend_to_cube") < phases.index("settle_at_grasp_target")
    assert phases.index("settle_at_grasp_target") < phases.index("align_gripper_to_cube")
    assert phases.index("align_gripper_to_cube") < phases.index("recenter_after_alignment")
    assert phases.index("recenter_after_alignment") < phases.index("close_gripper")

    align_source = ast.unparse(_method("_update_axis_alignment"))
    geometry_source = ast.unparse(_method("_measure_axis_alignment"))
    close_hold_source = ast.unparse(_method("_hold_axis_alignment_target"))
    assert "set_direct_joint_position_target" in close_hold_source
    assert "_AXIS_ALIGNMENT_MAX_TARGET_STEP" in align_source
    assert "_AXIS_ALIGNMENT_MAX_TARGET_LEAD" in align_source
    assert "_AXIS_ALIGNMENT_TOLERANCE" in align_source
    assert "select_nearest_cube_axis_alignment" in geometry_source
    assert "measure_cube_axis_alignment" in geometry_source
    assert "self._aligned_gripper_quat_w" in align_source


def test_axis_alignment_speedup_keeps_closed_loop_accuracy_gate() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "_AXIS_ALIGNMENT_MAX_TARGET_LEAD = math.radians(4.0)" in source
    assert "_AXIS_ALIGNMENT_TOLERANCE = math.radians(5.0)" in source
    assert "_AXIS_ALIGNMENT_STABLE_STEPS = 10" in source
    assert "_AXIS_ALIGNMENT_MAX_TARGET_STEP = math.radians(1.0)" in source


def test_scene_keeps_box_clear_of_reachable_cube_randomization() -> None:
    source = ENV_CFG_PATH.read_text(encoding="utf-8")
    assert "TARGET_BOX_CENTER_XY = (0.18, -0.43)" in source
    assert "CUBE_RANDOMIZATION_X_RANGE = (-0.02, 0.05)" in source
    assert "CUBE_RANDOMIZATION_Y_RANGE = (-0.06, -0.04)" in source
    assert 'self.events.domain_randomize_0.params["pose_range"]' in source


def test_close_gate_debounces_pick_feedback_without_bypassing_aperture_stability() -> None:
    observe_source = ast.unparse(_method("observe_pick_cube"))
    settle_source = ast.unparse(_method("_update_gripper_settle"))
    retreat_source = ast.unparse(_method("_initialize_polar_retreat"))

    assert "_PICK_FEEDBACK_STABLE_STEPS" in observe_source
    assert "self._pick_feedback_streak = 0" in observe_source
    assert "self._grasp_geometry_latched = True" in settle_source
    assert "self._grasp_geometry_latched or self._pick_feedback_confirmed" in settle_source
    assert "aperture_stable" in settle_source
    assert "self._grasp_geometry_latched" in retreat_source


def test_batch_can_record_each_randomized_episode_under_one_root() -> None:
    source = BATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BATCH_PATH))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_source = ast.unparse(main)

    assert '"--record_dir"' in source
    assert '"--record_every"' in source
    assert "_DiagnosticRecorder" in main_source
    assert "episode{episode_index:03d}" in main_source
    assert "recorder.finish" in main_source
