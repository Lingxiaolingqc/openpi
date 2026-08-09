"""Bounded environment and success-predicate smoke for RedCubeToBox."""

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
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _rounded_row(values, digits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in values.detach().cpu().tolist())


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if not args.headless:
        parser.error("This smoke test requires --headless")
    if not args.enable_cameras:
        parser.error("The environment requires --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")

    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    print("RED_CUBE_TO_BOX_SMOKE_PHASE=before_launcher", flush=True)
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
    import red_cube_to_box_task
    from red_cube_to_box_task import mdp
    from red_cube_to_box_task.env_cfg import TARGET_BOX_FLOOR_THICKNESS
    # isort: on

    status = 1
    try:
        task_id = red_cube_to_box_task.TASK_ID
        print("RED_CUBE_TO_BOX_SMOKE_PHASE=app_ready", flush=True)
        print(f"app_launcher_device_id: {app_launcher.device_id}", flush=True)
        print(f"task_id: {task_id}", flush=True)

        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        env_cfg.seed = args.seed
        env_cfg.recorders = None

        print("RED_CUBE_TO_BOX_SMOKE_PHASE=creating_env", flush=True)
        env = gym.make(task_id, cfg=env_cfg).unwrapped
        observations, _ = env.reset()

        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        box_part_names = tuple(name for name in env.scene.rigid_objects if name.startswith("target_box_"))
        initial_success = bool(mdp.cube_inside_target_box(env).all().item())

        print("RED_CUBE_TO_BOX_ENV_CREATED_OK", flush=True)
        print(f"environment_type: {type(env).__name__}", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"box_part_names: {box_part_names}", flush=True)
        for name in box_part_names:
            print(f"{name}_pos_w: {_rounded_row(env.scene[name].data.root_pos_w[0])}", flush=True)
        print(f"cube_initial_pos_w: {_rounded_row(cube.data.root_pos_w[0])}", flush=True)
        print(f"initial_success: {initial_success}", flush=True)
        print(f"front_shape: {tuple(observations['policy']['front'].shape)}", flush=True)

        action = torch.zeros(env.action_space.shape, device=env.device)
        all_rewards_finite = True
        for _ in range(args.steps):
            step_result = env.step(action)
            all_rewards_finite = all_rewards_finite and bool(torch.isfinite(step_result[1]).all())

        target_position = floor.data.root_pos_w.clone()
        target_position[:, 2] += TARGET_BOX_FLOOR_THICKNESS / 2.0 + 0.025
        target_pose = torch.cat([target_position, cube.data.root_quat_w.clone()], dim=-1)
        cube.write_root_pose_to_sim(target_pose)
        cube.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
        env.scene.update(dt=env.physics_dt)
        teleported_success = bool(mdp.cube_inside_target_box(env).all().item())

        print(f"completed_steps: {args.steps}", flush=True)
        print(f"all_rewards_finite: {all_rewards_finite}", flush=True)
        print(f"cube_test_pos_w: {_rounded_row(cube.data.root_pos_w[0])}", flush=True)
        print(f"teleported_success: {teleported_success}", flush=True)

        if initial_success:
            raise RuntimeError("Initial cube pose incorrectly satisfies the target-box predicate")
        if not all_rewards_finite:
            raise RuntimeError("A non-finite reward was observed")
        if not teleported_success:
            raise RuntimeError("Cube inside the target box did not satisfy the success predicate")
        if len(box_part_names) != 5:
            raise RuntimeError(f"Expected five target-box pieces, found {len(box_part_names)}")

        print("RED_CUBE_TO_BOX_ENV_SMOKE_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("RED_CUBE_TO_BOX_ENV_SMOKE_FAILED", flush=True)
    finally:
        print("RED_CUBE_TO_BOX_SMOKE_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
