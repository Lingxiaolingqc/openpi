"""Run the task-agnostic Franka OpenPI client in observe, shadow, or execute mode."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Literal

from openpi_client import websocket_client_policy
import tyro

from examples.franka import backend
from examples.franka import contract
from examples.franka import runtime


@dataclasses.dataclass(frozen=True)
class Args:
    mode: Literal["observe", "shadow", "execute"] = "observe"
    backend_name: Literal["mock", "ros2"] = "mock"
    prompt: str = "Perform the instructed manipulation task."
    steps: int = 20
    host: str = "localhost"
    port: int = 8000
    speed_scale: float = 0.1
    approval_path: Path | None = None
    robot_system_version: str | None = None
    base_image_topic: str = "/camera_base/color/image_raw"
    wrist_image_topic: str | None = None
    joint_state_topic: str = "/joint_states"
    trajectory_topic: str = "/fr3_arm_controller/joint_trajectory"
    gripper_action_name: str = "/fr3_gripper/move"
    ros2_fake_hardware: bool = False
    confirm_controller_contract: bool = False


def _make_backend(args: Args) -> backend.FrankaBackend:
    if args.backend_name == "mock":
        return backend.MockFrankaBackend(include_wrist=args.wrist_image_topic is not None)
    if args.backend_name != "ros2":
        raise ValueError("backend_name must be 'mock' or 'ros2'")
    if args.robot_system_version is None:
        raise ValueError("ROS 2 backend requires --robot-system-version from Franka Desk")
    versions = backend.probe_software_versions(robot_system_version=args.robot_system_version)
    logging.info("Validated Franka software versions: %s", versions)
    confirmed = args.confirm_controller_contract or args.ros2_fake_hardware
    return backend.Ros2FrankaBackend(
        base_image_topic=args.base_image_topic,
        wrist_image_topic=args.wrist_image_topic,
        joint_state_topic=args.joint_state_topic,
        trajectory_topic=args.trajectory_topic,
        gripper_action_name=args.gripper_action_name,
        real_hardware=not args.ros2_fake_hardware,
        controller_update_hz=1000,
        controller_has_dynamic_limits=confirmed,
        controller_has_local_watchdog=confirmed,
    )


def main(args: Args) -> None:
    mode = contract.RuntimeMode(args.mode)
    robot_backend = _make_backend(args)
    policy = None
    if mode is not contract.RuntimeMode.OBSERVE:
        policy = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            protocol_version=1,
        )
    policy_runtime = runtime.FrankaRuntime(
        backend=robot_backend,
        mode=mode,
        policy=policy,
        safety_config=contract.SafetyConfig(speed_scale=args.speed_scale),
        approval_path=args.approval_path,
    )
    try:
        policy_runtime.run(prompt=args.prompt, steps=args.steps)
    finally:
        policy_runtime.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
