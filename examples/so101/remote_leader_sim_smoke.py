"""Bounded headless smoke test for a remote SO-101 Leader and LeIsaac.

Run this script inside the server's LeIsaac environment. It receives the
normalized Leader joint state through LeIsaac's ``SO101LeaderRemote``, maps it
with the same action path as the official teleoperation script, and advances a
single simulation environment for a small, fixed number of steps.

The script never writes to the physical Leader and never records a dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="LeIsaac-SO101-LiftCube-v0")
    parser.add_argument("--remote_endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--assets_root", default=os.environ.get("LEISAAC_ASSETS_ROOT"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--receive_timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _wait_for_leader_state(teleop_interface, timeout: float) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = teleop_interface.get_device_state()
        if any(abs(value) > 1.0e-6 for value in state.values()):
            return state
        time.sleep(0.05)
    raise TimeoutError(f"No non-zero Leader frame received within {timeout:.1f}s")


def _rounded_mapping(values: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 3) for name, value in values.items()}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.receive_timeout <= 0:
        parser.error("--receive_timeout must be positive")
    if not args.headless:
        parser.error("This bounded smoke test requires --headless")
    if not args.enable_cameras:
        parser.error("The LiftCube environment requires --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")

    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    print("REMOTE_LEADER_SMOKE_PHASE=before_launcher", flush=True)
    print(f"task_id: {args.task}", flush=True)
    print(f"remote_endpoint: {args.remote_endpoint}", flush=True)
    print(f"assets_root: {assets_root}", flush=True)
    print(f"requested_device: {args.device}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac Sim must be launched before importing the remaining simulation modules.
    # isort: off
    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    import leisaac.tasks  # noqa: E402, F401
    from leisaac.devices import SO101LeaderRemote  # noqa: E402
    # isort: on

    env = None
    teleop_interface = None
    status = 1

    try:
        print("REMOTE_LEADER_SMOKE_PHASE=app_ready", flush=True)
        print(f"app_launcher_device_id: {app_launcher.device_id}", flush=True)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        env_cfg.seed = args.seed
        env_cfg.recorders = None

        print("REMOTE_LEADER_SMOKE_PHASE=creating_env", flush=True)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset()
        print("REMOTE_LEADER_ENV_CREATED_OK", flush=True)
        print(f"environment_type: {type(env).__name__}", flush=True)
        print(f"simulation_device: {env.device}", flush=True)

        teleop_interface = SO101LeaderRemote(env, endpoint=args.remote_endpoint)
        leader_state = _wait_for_leader_state(teleop_interface, args.receive_timeout)
        print(f"leader_state_first: {_rounded_mapping(leader_state)}", flush=True)

        completed_steps = 0
        all_rewards_finite = True
        first_action = None
        last_action = None

        print("REMOTE_LEADER_SMOKE_PHASE=stepping", flush=True)
        for _ in range(args.steps):
            action_request = teleop_interface.input2action()
            action = env.cfg.preprocess_device_action(action_request, teleop_interface)
            if action.shape != (1, 6):
                raise RuntimeError(f"Unexpected action shape: {tuple(action.shape)}")
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError("Remote Leader produced a non-finite simulation action")

            if first_action is None:
                first_action = action.detach().clone()
            last_action = action.detach().clone()

            step_result = env.step(action)
            rewards = step_result[1]
            all_rewards_finite = all_rewards_finite and bool(torch.isfinite(rewards).all())
            completed_steps += 1

        assert first_action is not None
        assert last_action is not None
        print(f"action_shape: {tuple(first_action.shape)}", flush=True)
        print(f"action_device: {first_action.device}", flush=True)
        print(
            "first_action_rad:",
            tuple(round(value, 5) for value in first_action[0].tolist()),
            flush=True,
        )
        print(
            "last_action_rad:",
            tuple(round(value, 5) for value in last_action[0].tolist()),
            flush=True,
        )
        print(f"completed_steps: {completed_steps}", flush=True)
        print(f"all_rewards_finite: {all_rewards_finite}", flush=True)

        if not all_rewards_finite:
            raise RuntimeError("A non-finite reward was observed")

        print("REMOTE_LEADER_SIM_SMOKE_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("REMOTE_LEADER_SIM_SMOKE_FAILED", flush=True)
    finally:
        if teleop_interface is not None:
            teleop_interface.disconnect()
        print("REMOTE_LEADER_SMOKE_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
