"""Run one bounded scripted-expert episode in RedCubeToBox."""

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
    parser.add_argument("--expert", choices=("legacy", "adaptive", "servo"), default="legacy")
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _rounded_row(values, digits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in values.detach().cpu().tolist())


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.headless:
        parser.error("This expert smoke requires --headless")
    if not args.enable_cameras:
        parser.error("The environment requires --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")

    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    print("RED_CUBE_TO_BOX_EXPERT_PHASE=before_launcher", flush=True)
    print(f"assets_root: {assets_root}", flush=True)
    print(f"requested_device: {args.device}", flush=True)

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
    from red_cube_to_box_task.servo_state_machine import RedCubeToBoxServoStateMachine
    from red_cube_to_box_task.state_machine import RedCubeToBoxStateMachine
    # isort: on

    status = 1
    try:
        task_id = red_cube_to_box_task.TASK_ID
        print("RED_CUBE_TO_BOX_EXPERT_PHASE=app_ready", flush=True)
        print(f"app_launcher_device_id: {app_launcher.device_id}", flush=True)
        print(f"task_id: {task_id}", flush=True)
        print(f"expert_variant: {args.expert}", flush=True)

        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101_state_machine")
        env_cfg.seed = args.seed
        env_cfg.recorders = None
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        print(
            f"state_machine_gripper_close_expr: {env_cfg.actions.gripper_action.close_command_expr}",
            flush=True,
        )
        print(f"expert_ik_command_type: {env_cfg.actions.arm_action.controller.command_type}", flush=True)
        orientation_policy = {
            "legacy": "fixed_world",
            "adaptive": "fixed_during_grasp,current_after_grasp",
            "servo": "fixed_world,servo_after_lift",
        }[args.expert]
        print(f"expert_orientation_policy: {orientation_policy}", flush=True)

        print("RED_CUBE_TO_BOX_EXPERT_PHASE=creating_env", flush=True)
        env = gym.make(task_id, cfg=env_cfg).unwrapped
        observations, _ = env.reset()

        state_machine_class = {
            "legacy": RedCubeToBoxStateMachine,
            "adaptive": RedCubeToBoxAdaptiveStateMachine,
            "servo": RedCubeToBoxServoStateMachine,
        }[args.expert]
        state_machine = state_machine_class()
        state_machine.setup(env)
        state_machine.reset()
        print(f"servo_parameters: {getattr(state_machine, 'servo_parameters', 'not_applicable')}", flush=True)

        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        print("RED_CUBE_TO_BOX_EXPERT_ENV_CREATED_OK", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"action_space: {env.action_space}", flush=True)
        print(f"cube_initial_pos_w: {_rounded_row(cube.data.root_pos_w[0])}", flush=True)
        print(f"target_box_floor_pos_w: {_rounded_row(floor.data.root_pos_w[0])}", flush=True)

        completed_steps = 0
        all_rewards_finite = True
        unexpected_reset = False
        previous_phase = None
        previous_pick_cube = bool(observations["subtask_terms"]["pick_cube"][0].item())

        with torch.inference_mode():
            while not state_machine.is_episode_done:
                phase = state_machine.phase_name
                phase_changed = phase != previous_phase
                if phase_changed:
                    print(f"expert_phase:{phase}:step={state_machine.step_count}", flush=True)
                    gripper_pos = ee_frame.data.target_pos_w[0, 0]
                    jaw_pos = ee_frame.data.target_pos_w[0, 1]
                    jaw_cube_distance = torch.linalg.vector_norm(jaw_pos - cube.data.root_pos_w[0])
                    pick_cube = observations["subtask_terms"]["pick_cube"][0]
                    print(
                        f"expert_state:{phase}:"
                        f"gripper_pos_w={_rounded_row(gripper_pos)}:"
                        f"jaw_pos_w={_rounded_row(jaw_pos)}:"
                        f"cube_pos_w={_rounded_row(cube.data.root_pos_w[0])}:"
                        f"jaw_cube_distance={jaw_cube_distance.item():.5f}:"
                        f"gripper_joint={robot.data.joint_pos[0, -1].item():.5f}:"
                        f"pick_cube={bool(pick_cube.item())}",
                        flush=True,
                    )
                    previous_phase = phase

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, "so101_state_machine")

                action = state_machine.get_action(env)
                expected_action_shape = (env.num_envs, env.action_manager.total_action_dim)
                if action.shape != expected_action_shape:
                    raise RuntimeError(f"Unexpected expert action shape: {tuple(action.shape)}")
                if not bool(torch.isfinite(action).all()):
                    raise RuntimeError("Expert produced a non-finite action")
                if phase_changed:
                    print(f"expert_action:{phase}:{_rounded_row(action[0])}", flush=True)
                    desired_cube = getattr(state_machine, "last_desired_cube_w", None)
                    cube_error = getattr(state_machine, "last_cube_error_w", None)
                    gripper_target = getattr(state_machine, "last_gripper_target_w", None)
                    if desired_cube is not None and cube_error is not None and gripper_target is not None:
                        print(
                            f"expert_feedback:{phase}:"
                            f"desired_cube_w={_rounded_row(desired_cube[0])}:"
                            f"cube_error_w={_rounded_row(cube_error[0])}:"
                            f"gripper_target_w={_rounded_row(gripper_target[0])}",
                            flush=True,
                        )
                if args.expert == "servo" and phase in {
                    "lift_cube",
                    "transfer_to_box",
                    "lower_into_box",
                    "align_over_box",
                } and (phase_changed or state_machine.step_count % 25 == 0):
                    gripper_pos = ee_frame.data.target_pos_w[0, 0]
                    jaw_pos = ee_frame.data.target_pos_w[0, 1]
                    cube_pos = cube.data.root_pos_w[0]
                    print(
                        f"expert_tracking:{phase}:step={state_machine.step_count}:"
                        f"gripper_pos_w={_rounded_row(gripper_pos)}:"
                        f"gripper_quat_w={_rounded_row(ee_frame.data.target_quat_w[0, 0])}:"
                        f"jaw_pos_w={_rounded_row(jaw_pos)}:"
                        f"cube_pos_w={_rounded_row(cube_pos)}:"
                        f"jaw_cube_distance={torch.linalg.vector_norm(jaw_pos - cube_pos).item():.6f}:"
                        f"joint_pos={_rounded_row(robot.data.joint_pos[0])}:"
                        f"pick_cube={bool(observations['subtask_terms']['pick_cube'][0].item())}",
                        flush=True,
                    )
                if args.expert == "servo" and phase in {
                    "transfer_to_box",
                    "lower_into_box",
                    "align_over_box",
                } and (phase_changed or state_machine.step_count % 50 == 0):
                    servo_delta = state_machine.last_servo_delta_w
                    servo_error_norm = state_machine.last_servo_error_norm
                    if servo_delta is not None and servo_error_norm is not None:
                        print(
                            f"expert_servo:{phase}:"
                            f"error_norm={servo_error_norm[0, 0].item():.6f}:"
                            f"delta_w={_rounded_row(servo_delta[0])}:"
                            f"stable_streak={state_machine.servo_stable_streak}",
                            flush=True,
                        )

                if state_machine.is_episode_done:
                    print(
                        f"expert_abort_before_step:{getattr(state_machine, 'servo_abort_reason', None)}",
                        flush=True,
                    )
                    break

                step_result = env.step(action)
                observations = step_result[0]
                all_rewards_finite = all_rewards_finite and bool(torch.isfinite(step_result[1]).all())
                unexpected_reset = unexpected_reset or bool(step_result[2].any()) or bool(step_result[3].any())
                pick_cube_after = bool(observations["subtask_terms"]["pick_cube"][0].item())
                if previous_pick_cube and not pick_cube_after:
                    print(
                        f"expert_grasp_event:lost:phase={phase}:"
                        f"state_step={state_machine.step_count}:control_step={completed_steps + 1}",
                        flush=True,
                    )
                previous_pick_cube = pick_cube_after
                state_machine.advance()
                completed_steps += 1

        success = state_machine.check_success(env)
        cube_offset = cube.data.root_pos_w - floor.data.root_pos_w
        cube_speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w, dim=-1)

        print(f"completed_steps: {completed_steps}", flush=True)
        print(f"all_rewards_finite: {all_rewards_finite}", flush=True)
        print(f"unexpected_reset: {unexpected_reset}", flush=True)
        print(f"cube_final_pos_w: {_rounded_row(cube.data.root_pos_w[0])}", flush=True)
        print(f"cube_offset_from_box: {_rounded_row(cube_offset[0])}", flush=True)
        print(f"cube_final_speed: {cube_speed[0].item():.6f}", flush=True)
        print(f"pick_cube_final: {bool(observations['subtask_terms']['pick_cube'][0].item())}", flush=True)
        print(f"grasp_confirmed: {getattr(state_machine, 'grasp_confirmed', 'not_tracked')}", flush=True)
        print(
            f"grasp_lost_before_release: {getattr(state_machine, 'grasp_lost_before_release', 'not_tracked')}",
            flush=True,
        )
        print(f"retry_used: {getattr(state_machine, 'retry_used', 'not_tracked')}", flush=True)
        print(
            f"box_aligned_before_release: {getattr(state_machine, 'box_aligned_before_release', 'not_tracked')}",
            flush=True,
        )
        servo_timeout_phase = getattr(state_machine, "servo_timeout_phase", None)
        servo_abort_reason = getattr(state_machine, "servo_abort_reason", None)
        print(f"servo_timeout_phase: {servo_timeout_phase}", flush=True)
        print(f"servo_abort_reason: {servo_abort_reason}", flush=True)
        print(f"expert_success: {success}", flush=True)

        if not all_rewards_finite:
            raise RuntimeError("A non-finite reward was observed")
        if unexpected_reset:
            raise RuntimeError("The environment reset before the expert episode completed")
        if servo_abort_reason is not None:
            raise RuntimeError(f"The servo expert aborted: {servo_abort_reason}")
        if servo_timeout_phase is not None:
            raise RuntimeError(f"The servo expert timed out in phase: {servo_timeout_phase}")
        if not success:
            raise RuntimeError("The scripted expert did not place a settled cube inside the target box")

        print("RED_CUBE_TO_BOX_EXPERT_SMOKE_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("RED_CUBE_TO_BOX_EXPERT_SMOKE_FAILED", flush=True)
    finally:
        print("RED_CUBE_TO_BOX_EXPERT_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
