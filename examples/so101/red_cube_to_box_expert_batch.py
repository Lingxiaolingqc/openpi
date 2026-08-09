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
    from red_cube_to_box_task.state_machine import RedCubeToBoxStateMachine
    # isort: on

    status = 1
    try:
        task_id = red_cube_to_box_task.TASK_ID
        print("RED_CUBE_TO_BOX_BATCH_PHASE=app_ready", flush=True)
        print(f"app_launcher_device_id: {app_launcher.device_id}", flush=True)
        print(f"task_id: {task_id}", flush=True)

        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101_state_machine")
        env_cfg.seed = args.seed
        env_cfg.recorders = None
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None

        cube_randomization = env_cfg.events.domain_randomize_0.params["pose_range"]
        camera_randomization = env_cfg.events.domain_randomize_1.params["pose_range"]
        print(f"cube_randomization: {cube_randomization}", flush=True)
        print(f"camera_randomization: {camera_randomization}", flush=True)

        print("RED_CUBE_TO_BOX_BATCH_PHASE=creating_env", flush=True)
        env = gym.make(task_id, cfg=env_cfg).unwrapped
        state_machine = RedCubeToBoxStateMachine()
        state_machine.setup(env)

        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        successful_episodes = 0
        grasped_episodes = 0
        non_finite_episodes: list[int] = []
        reset_episodes: list[int] = []
        failed_episodes: list[int] = []
        initial_cube_positions: list[torch.Tensor] = []
        final_offsets: list[torch.Tensor] = []

        print("RED_CUBE_TO_BOX_BATCH_ENV_CREATED_OK", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"action_space: {env.action_space}", flush=True)
        print("RED_CUBE_TO_BOX_BATCH_PHASE=running", flush=True)

        with torch.inference_mode():
            for episode_index in range(args.episodes):
                observations, _ = env.reset()
                state_machine.reset()
                initial_cube_position = cube.data.root_pos_w[0].clone()
                initial_cube_positions.append(initial_cube_position)

                ever_grasped = False
                rewards_finite = True
                unexpected_reset = False

                while not state_machine.is_episode_done:
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, "so101_state_machine")

                    action = state_machine.get_action(env)
                    if action.shape != (env.num_envs, 8):
                        raise RuntimeError(f"Unexpected expert action shape: {tuple(action.shape)}")
                    if not bool(torch.isfinite(action).all()):
                        raise RuntimeError("Expert produced a non-finite action")

                    step_result = env.step(action)
                    observations = step_result[0]
                    rewards_finite = rewards_finite and bool(torch.isfinite(step_result[1]).all())
                    unexpected_reset = unexpected_reset or bool(step_result[2].any()) or bool(step_result[3].any())
                    ever_grasped = ever_grasped or bool(observations["subtask_terms"]["pick_cube"][0].item())
                    state_machine.advance()

                success = state_machine.check_success(env)
                final_offset = cube.data.root_pos_w[0] - floor.data.root_pos_w[0]
                final_speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w[0])
                final_offsets.append(final_offset.clone())

                if ever_grasped:
                    grasped_episodes += 1
                if not rewards_finite:
                    non_finite_episodes.append(episode_index)
                if unexpected_reset:
                    reset_episodes.append(episode_index)
                if success and ever_grasped and rewards_finite and not unexpected_reset:
                    successful_episodes += 1
                else:
                    failed_episodes.append(episode_index)

                print(
                    f"episode:{episode_index}:"
                    f"initial_cube_pos_w={_rounded_row(initial_cube_position)}:"
                    f"ever_grasped={ever_grasped}:"
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
        print(f"successful_episodes: {successful_episodes}", flush=True)
        print(f"failed_episodes: {failed_episodes}", flush=True)
        print(f"non_finite_episodes: {non_finite_episodes}", flush=True)
        print(f"reset_episodes: {reset_episodes}", flush=True)
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
