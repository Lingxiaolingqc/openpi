"""Run randomized RedCubeToBox expert episodes and report success rate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets_root", default=os.environ.get("LEISAAC_ASSETS_ROOT"))
    parser.add_argument(
        "--expert",
        choices=(
            "legacy",
            "legacy_dynamic_grasp_offset",
            "legacy_dynamic_grasp_offset_residual_corrected",
            "autogen_reference",
            "legacy_gripper_anchor",
            "legacy_gripper_anchor_align_then_lower",
            "legacy_gripper_anchor_position_align_then_lower",
            "legacy_gripper_anchor_weighted_position_align_then_lower",
            "legacy_gripper_anchor_safe_planar_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
            "jaw_frame_xyz_tilt",
            "legacy_gripper_anchor_relaxed_ik",
            "legacy_gripper_anchor_planar_ik",
            "adaptive",
            "servo",
            "weighted_servo",
        ),
        default="legacy",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--minimum_success_rate", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _rounded_row(values, digits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in values.detach().cpu().tolist())


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        parser.error("--minimum_success_rate must be between 0 and 1")
    if not args.headless:
        parser.error("This batch test requires --headless")
    if not args.enable_cameras:
        parser.error("The environment requires --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")

    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    print("RED_CUBE_TO_BOX_BATCH_PHASE=before_launcher", flush=True)
    print(f"assets_root: {assets_root}", flush=True)
    print(f"requested_device: {args.device}", flush=True)
    print(f"requested_episodes: {args.episodes}", flush=True)
    print(f"minimum_success_rate: {args.minimum_success_rate:.3f}", flush=True)
    print(f"seed: {args.seed}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac Sim must be launched before importing the remaining simulation modules.
    # isort: off
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    import leisaac.tasks  # noqa: F401
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
    import red_cube_to_box_task
    from red_cube_to_box_task.adaptive_state_machine import RedCubeToBoxAdaptiveStateMachine
    from red_cube_to_box_task.autogen_reference_state_machine import (
        RedCubeToBoxAutogenReferenceStateMachine,
        configure_autogen_reference_action,
    )
    from red_cube_to_box_task.env_cfg import configure_planar_safety_sensors
    from red_cube_to_box_task.legacy_gripper_anchor_state_machine import (
        RedCubeToBoxLegacyGripperAnchorStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_relaxed_ik_state_machine import (
        RedCubeToBoxLegacyGripperAnchorRelaxedIkStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_planar_ik_state_machine import (
        RedCubeToBoxLegacyGripperAnchorPlanarIkStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_position_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorPositionAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_weighted_position_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorWeightedPositionAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_planar_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafePlanarAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeJawTrajectoryDirectXyzPanNullspaceAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeXyzPanNullspaceAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeXyzPitchPanAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_xyz_tilt_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeXyzTiltAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.jaw_frame_xyz_tilt_state_machine import RedCubeToBoxJawFrameXyzTiltStateMachine
    from red_cube_to_box_task.legacy_dynamic_grasp_offset_state_machine import (
        RedCubeToBoxLegacyDynamicGraspOffsetStateMachine,
    )
    from red_cube_to_box_task.legacy_dynamic_grasp_offset_residual_corrected_state_machine import (
        RedCubeToBoxLegacyDynamicGraspOffsetResidualCorrectedStateMachine,
    )
    from red_cube_to_box_task.phase_aware_ik_action import (
        configure_dynamic_control_frame_offset,
        configure_servo_ik_action,
        resolve_action_term,
    )
    from red_cube_to_box_task.servo_state_machine import RedCubeToBoxServoStateMachine
    from red_cube_to_box_task.state_machine import RedCubeToBoxStateMachine
    from red_cube_to_box_task.weighted_servo_state_machine import RedCubeToBoxWeightedServoStateMachine
    # isort: on

    status = 1
    try:
        task_id = red_cube_to_box_task.TASK_ID
        print("RED_CUBE_TO_BOX_BATCH_PHASE=app_ready", flush=True)
        print(f"app_launcher_device_id: {app_launcher.device_id}", flush=True)
        print(f"task_id: {task_id}", flush=True)
        print(f"expert_variant: {args.expert}", flush=True)

        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101_state_machine")
        env_cfg.seed = args.seed
        env_cfg.recorders = None
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        if args.expert in {
            "legacy_gripper_anchor_planar_ik",
            "legacy_gripper_anchor_safe_planar_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
            "jaw_frame_xyz_tilt",
        }:
            configure_planar_safety_sensors(env_cfg)
        if args.expert in {
            "legacy_gripper_anchor_relaxed_ik",
            "legacy_gripper_anchor_planar_ik",
            "legacy_gripper_anchor_position_align_then_lower",
            "legacy_gripper_anchor_weighted_position_align_then_lower",
            "legacy_gripper_anchor_safe_planar_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
            "jaw_frame_xyz_tilt",
            "servo",
            "weighted_servo",
        }:
            configure_servo_ik_action(env_cfg)
        if args.expert == "autogen_reference":
            configure_autogen_reference_action(env_cfg)
        if args.expert in {
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
            "jaw_frame_xyz_tilt",
        }:
            configure_dynamic_control_frame_offset(env_cfg)

        cube_randomization = env_cfg.events.domain_randomize_0.params["pose_range"]
        camera_randomization = env_cfg.events.domain_randomize_1.params["pose_range"]
        print(f"cube_randomization: {cube_randomization}", flush=True)
        print(f"camera_randomization: {camera_randomization}", flush=True)

        print("RED_CUBE_TO_BOX_BATCH_PHASE=creating_env", flush=True)
        env = gym.make(task_id, cfg=env_cfg).unwrapped
        state_machine_class = {
            "legacy": RedCubeToBoxStateMachine,
            "legacy_dynamic_grasp_offset": RedCubeToBoxLegacyDynamicGraspOffsetStateMachine,
            "legacy_dynamic_grasp_offset_residual_corrected": (
                RedCubeToBoxLegacyDynamicGraspOffsetResidualCorrectedStateMachine
            ),
            "autogen_reference": RedCubeToBoxAutogenReferenceStateMachine,
            "legacy_gripper_anchor": RedCubeToBoxLegacyGripperAnchorStateMachine,
            "legacy_gripper_anchor_align_then_lower": RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine,
            "legacy_gripper_anchor_position_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorPositionAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_weighted_position_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorWeightedPositionAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_planar_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafePlanarAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeXyzTiltAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeXyzPitchPanAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeXyzPanNullspaceAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeJawTrajectoryDirectXyzPanNullspaceAlignThenLowerStateMachine
            ),
            "jaw_frame_xyz_tilt": RedCubeToBoxJawFrameXyzTiltStateMachine,
            "legacy_gripper_anchor_relaxed_ik": RedCubeToBoxLegacyGripperAnchorRelaxedIkStateMachine,
            "legacy_gripper_anchor_planar_ik": RedCubeToBoxLegacyGripperAnchorPlanarIkStateMachine,
            "adaptive": RedCubeToBoxAdaptiveStateMachine,
            "servo": RedCubeToBoxServoStateMachine,
            "weighted_servo": RedCubeToBoxWeightedServoStateMachine,
        }[args.expert]
        state_machine = state_machine_class()
        state_machine.setup(env)

        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        successful_episodes = 0
        grasped_episodes = 0
        grasped_at_lift_episodes = 0
        grasped_at_transfer_episodes = 0
        retried_episodes = 0
        box_aligned_episodes = 0
        non_finite_episodes: list[int] = []
        reset_episodes: list[int] = []
        servo_timeout_episodes: list[int] = []
        servo_abort_episodes: list[int] = []
        failed_episodes: list[int] = []
        initial_cube_positions: list[torch.Tensor] = []
        final_offsets: list[torch.Tensor] = []

        print("RED_CUBE_TO_BOX_BATCH_ENV_CREATED_OK", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"action_space: {env.action_space}", flush=True)
        print(f"expert_ik_command_type: {env_cfg.actions.arm_action.controller.command_type}", flush=True)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        print(f"expert_ik_action_class: {type(arm_action_term).__name__}", flush=True)
        orientation_policy = {
            "legacy": "fixed_world",
            "legacy_dynamic_grasp_offset": "legacy_fixed_world,dynamic_per-grasp_placement_xy",
            "legacy_dynamic_grasp_offset_residual_corrected": (
                "legacy_fixed_world,dynamic_per-grasp_placement_xy,single_post-transfer_residual_correction"
            ),
            "autogen_reference": (
                "bundled_autogen_state_flow,robot-base_coordinates,original_green_ray_obb,"
                "wrist_xyz_ik_plus_wrist_flex_posture_correction,continuous_gripper"
            ),
            "legacy_gripper_anchor": "legacy_fixed_world,jaw_anchored_placement",
            "legacy_gripper_anchor_align_then_lower": "legacy_fixed_world,align_high_then_descend",
            "legacy_gripper_anchor_position_align_then_lower": "position_only_high_align,legacy_pose_descent",
            "legacy_gripper_anchor_weighted_position_align_then_lower": (
                "shoulder_pan_priority_xyz_align,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_planar_align_then_lower": (
                "xy_plus_orientation_high_align,physical_z_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower": (
                "xyz_plus_world_tilt_high_align,free_yaw,physical_safety_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower": (
                "xyz_plus_pitch_plus_explicit_shoulder_pan_target,physical_safety_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower": (
                "xyz_plus_explicit_shoulder_pan_target,nullspace_joint_limit_avoidance,"
                "physical_safety_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower": (
                "direct_closed_jaw_xyz_plus_explicit_shoulder_pan_target,nullspace_joint_limit_avoidance,"
                "physical_safety_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower": (
                "smooth_jaw_transfer_to_floor_center,direct_closed_jaw_xyz_plus_explicit_shoulder_pan_target,"
                "persistent_joint_target_slew,nullspace_joint_limit_avoidance,physical_safety_gates"
            ),
            "jaw_frame_xyz_tilt": "live_jaw_frame_xyz_plus_world_tilt_all_phases,free_yaw",
            "legacy_gripper_anchor_relaxed_ik": "legacy_fixed_world,jaw_anchor_then_staged_relaxation",
            "legacy_gripper_anchor_planar_ik": "legacy_fixed_world,collision_gated_xy_plus_orientation",
            "adaptive": "fixed_during_grasp,current_after_grasp",
            "servo": "fixed_world_through_lift,position_only_ik_after_lift",
            "weighted_servo": "fixed_world_through_lift,translation_priority_ik_after_lift",
        }[args.expert]
        print(f"expert_orientation_policy: {orientation_policy}", flush=True)
        print(f"servo_parameters: {getattr(state_machine, 'servo_parameters', 'not_applicable')}", flush=True)
        print("RED_CUBE_TO_BOX_BATCH_PHASE=running", flush=True)

        with torch.inference_mode():
            for episode_index in range(args.episodes):
                observations, _ = env.reset()
                state_machine.reset()
                initial_cube_position = cube.data.root_pos_w[0].clone()
                initial_cube_positions.append(initial_cube_position)

                ever_grasped = False
                grasped_at_lift = False
                grasped_at_transfer = False
                lift_phase_seen = False
                transfer_phase_seen = False
                rewards_finite = True
                unexpected_reset = False

                while not state_machine.is_episode_done:
                    phase_name = state_machine.phase_name
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, "so101_state_machine")

                    action = state_machine.get_action(env)
                    expected_action_shape = (env.num_envs, env.action_manager.total_action_dim)
                    if action.shape != expected_action_shape:
                        raise RuntimeError(f"Unexpected expert action shape: {tuple(action.shape)}")
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("Expert produced a non-finite action")
                    if state_machine.is_episode_done:
                        break

                    step_result = env.step(action)
                    observations = step_result[0]
                    rewards_finite = rewards_finite and bool(torch.isfinite(step_result[1]).all())
                    unexpected_reset = unexpected_reset or bool(step_result[2].any()) or bool(step_result[3].any())
                    pick_cube = bool(observations["subtask_terms"]["pick_cube"][0].item())
                    ever_grasped = ever_grasped or pick_cube
                    if phase_name == "lift_cube" and not lift_phase_seen:
                        lift_phase_seen = True
                        grasped_at_lift = pick_cube
                    if phase_name == "transfer_to_box" and not transfer_phase_seen:
                        transfer_phase_seen = True
                        grasped_at_transfer = pick_cube
                    state_machine.advance()

                success = state_machine.check_success(env)
                final_offset = cube.data.root_pos_w[0] - floor.data.root_pos_w[0]
                final_speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w[0])
                final_offsets.append(final_offset.clone())

                if ever_grasped:
                    grasped_episodes += 1
                if grasped_at_lift:
                    grasped_at_lift_episodes += 1
                if grasped_at_transfer:
                    grasped_at_transfer_episodes += 1
                retry_used = bool(getattr(state_machine, "retry_used", False))
                if retry_used:
                    retried_episodes += 1
                box_aligned = bool(getattr(state_machine, "box_aligned_before_release", False))
                if box_aligned:
                    box_aligned_episodes += 1
                servo_timeout_phase = getattr(state_machine, "servo_timeout_phase", None)
                if servo_timeout_phase is not None:
                    servo_timeout_episodes.append(episode_index)
                servo_abort_reason = getattr(state_machine, "servo_abort_reason", None)
                if servo_abort_reason is not None:
                    servo_abort_episodes.append(episode_index)
                if not rewards_finite:
                    non_finite_episodes.append(episode_index)
                if unexpected_reset:
                    reset_episodes.append(episode_index)
                if (
                    success
                    and ever_grasped
                    and rewards_finite
                    and not unexpected_reset
                    and servo_timeout_phase is None
                    and servo_abort_reason is None
                ):
                    successful_episodes += 1
                else:
                    failed_episodes.append(episode_index)

                print(
                    f"episode:{episode_index}:"
                    f"initial_cube_pos_w={_rounded_row(initial_cube_position)}:"
                    f"ever_grasped={ever_grasped}:"
                    f"grasped_at_lift={grasped_at_lift}:"
                    f"grasped_at_transfer={grasped_at_transfer}:"
                    f"retry_used={retry_used}:"
                    f"box_aligned_before_release={box_aligned}:"
                    f"servo_timeout_phase={servo_timeout_phase}:"
                    f"servo_abort_reason={servo_abort_reason}:"
                    f"final_offset={_rounded_row(final_offset)}:"
                    f"final_speed={final_speed.item():.6f}:"
                    f"success={success}:"
                    f"rewards_finite={rewards_finite}:"
                    f"unexpected_reset={unexpected_reset}",
                    flush=True,
                )

        initial_positions = torch.stack(initial_cube_positions)
        offsets = torch.stack(final_offsets)
        success_rate = successful_episodes / args.episodes

        print(f"completed_episodes: {args.episodes}", flush=True)
        print(f"grasped_episodes: {grasped_episodes}", flush=True)
        print(f"grasped_at_lift_episodes: {grasped_at_lift_episodes}", flush=True)
        print(f"grasped_at_transfer_episodes: {grasped_at_transfer_episodes}", flush=True)
        print(f"retried_episodes: {retried_episodes}", flush=True)
        print(f"box_aligned_episodes: {box_aligned_episodes}", flush=True)
        print(f"successful_episodes: {successful_episodes}", flush=True)
        print(f"failed_episodes: {failed_episodes}", flush=True)
        print(f"non_finite_episodes: {non_finite_episodes}", flush=True)
        print(f"reset_episodes: {reset_episodes}", flush=True)
        print(f"servo_timeout_episodes: {servo_timeout_episodes}", flush=True)
        print(f"servo_abort_episodes: {servo_abort_episodes}", flush=True)
        print(f"success_rate: {success_rate:.3f}", flush=True)
        print(f"initial_cube_min_w: {_rounded_row(initial_positions.amin(dim=0))}", flush=True)
        print(f"initial_cube_max_w: {_rounded_row(initial_positions.amax(dim=0))}", flush=True)
        print(f"final_offset_min: {_rounded_row(offsets.amin(dim=0))}", flush=True)
        print(f"final_offset_max: {_rounded_row(offsets.amax(dim=0))}", flush=True)

        if non_finite_episodes:
            raise RuntimeError(f"Non-finite rewards occurred in episodes {non_finite_episodes}")
        if reset_episodes:
            raise RuntimeError(f"Unexpected resets occurred in episodes {reset_episodes}")
        if success_rate < args.minimum_success_rate:
            raise RuntimeError(f"Success rate {success_rate:.3f} is below the required {args.minimum_success_rate:.3f}")

        print("RED_CUBE_TO_BOX_EXPERT_BATCH_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("RED_CUBE_TO_BOX_EXPERT_BATCH_FAILED", flush=True)
    finally:
        print("RED_CUBE_TO_BOX_BATCH_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
