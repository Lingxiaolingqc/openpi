"""Inspect the tested OpenPI RedCubeToBox scene geometry.

This bounded, read-only scene audit launches one headless environment, resets it
once, and prints the world-space robot, cube, camera, and target-box geometry.
It does not connect to a physical Leader and does not record data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="OpenPI-LeIsaac-SO101-RedCubeToBox-v0")
    parser.add_argument("--assets_root", default=os.environ.get("LEISAAC_ASSETS_ROOT"))
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _rounded_row(values, digits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in values.detach().cpu().tolist())


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.headless:
        parser.error("This scene audit requires --headless")
    if not args.enable_cameras:
        parser.error("The LiftCube environment requires --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")

    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    print("RED_CUBE_TO_BOX_AUDIT_PHASE=before_launcher", flush=True)
    print(f"task_id: {args.task}", flush=True)
    print(f"assets_root: {assets_root}", flush=True)
    print(f"requested_device: {args.device}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac Sim must be launched before importing the remaining simulation modules.
    # isort: off
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    import leisaac.tasks  # noqa: F401
    import red_cube_to_box_task  # noqa: F401
    # isort: on

    status = 1
    try:
        print("RED_CUBE_TO_BOX_AUDIT_PHASE=app_ready", flush=True)
        print(f"app_launcher_device_id: {app_launcher.device_id}", flush=True)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        env_cfg.seed = args.seed
        env_cfg.recorders = None

        print("RED_CUBE_TO_BOX_AUDIT_PHASE=creating_env", flush=True)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        observations, _ = env.reset()

        robot = env.scene["robot"]
        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        front = observations["policy"]["front"]
        floor_pos_w = _rounded_row(floor.data.root_pos_w[0])

        print("RED_CUBE_TO_BOX_SCENE_CREATED_OK", flush=True)
        print(f"environment_type: {type(env).__name__}", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"environment_origin_w: {_rounded_row(env.scene.env_origins[0])}", flush=True)
        print(f"robot_root_pos_w: {_rounded_row(robot.data.root_pos_w[0])}", flush=True)
        print(f"robot_root_quat_w: {_rounded_row(robot.data.root_quat_w[0])}", flush=True)
        print(f"robot_joint_names: {tuple(robot.data.joint_names)}", flush=True)
        print(f"robot_body_names: {tuple(robot.data.body_names)}", flush=True)
        print(f"end_effector_pos_w: {_rounded_row(robot.data.body_pos_w[0, -1])}", flush=True)
        print(f"cube_pos_w: {_rounded_row(cube.data.root_pos_w[0])}", flush=True)
        print(f"cube_quat_w: {_rounded_row(cube.data.root_quat_w[0])}", flush=True)
        print(f"cube_default_pos_w: {_rounded_row(cube.data.default_root_state[0, :3])}", flush=True)
        print(f"target_box_floor_pos_w: {floor_pos_w}", flush=True)
        print(f"target_box_floor_x_w: {floor_pos_w[0]}", flush=True)
        print(f"target_box_floor_y_w: {floor_pos_w[1]}", flush=True)
        print(f"target_box_floor_z_w: {floor_pos_w[2]}", flush=True)
        print(f"scene_rigid_objects: {tuple(env.scene.rigid_objects.keys())}", flush=True)
        print(f"scene_sensors: {tuple(env.scene.sensors.keys())}", flush=True)
        print(f"front_shape: {tuple(front.shape)}", flush=True)
        print(f"front_dtype: {front.dtype}", flush=True)
        print("RED_CUBE_TO_BOX_SCENE_AUDIT_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("RED_CUBE_TO_BOX_SCENE_AUDIT_FAILED", flush=True)
    finally:
        print("RED_CUBE_TO_BOX_AUDIT_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
