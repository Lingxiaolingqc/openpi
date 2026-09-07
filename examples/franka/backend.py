"""Robot backend boundary for mock and ROS 2 FR3 control."""

from __future__ import annotations

import abc
from dataclasses import dataclass
import importlib
import os
import platform
import re
import subprocess
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from examples.franka import contract


@dataclass(frozen=True)
class BackendCapabilities:
    control_update_hz: int
    position_dependent_velocity_limits: bool
    local_watchdog: bool
    real_hardware: bool


class FrankaBackend(abc.ABC):
    @property
    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Describe safety capabilities that are outside the model process."""

    @abc.abstractmethod
    def read_snapshot(self) -> contract.FrankaSnapshot:
        """Return one synchronized pre-step observation."""

    @abc.abstractmethod
    def apply_target(self, target: contract.FrankaTarget) -> None:
        """Send one absolute target to the local interpolating controller."""

    @abc.abstractmethod
    def hold_position(self) -> None:
        """Cancel motion and hold a fresh locally measured pose."""

    @abc.abstractmethod
    def recover(self) -> None:
        """Perform an explicit backend recovery, never an automatic one."""

    def close(self) -> None:  # noqa: B027 - stateless backends need no cleanup.
        """Release backend resources."""


class MockFrankaBackend(FrankaBackend):
    def __init__(
        self,
        *,
        q_rad: np.ndarray | None = None,
        gripper_width_m: float = 0.08,
        include_wrist: bool = True,
    ) -> None:
        self._q = np.asarray(q_rad if q_rad is not None else [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0], dtype=np.float64)
        self._dq = np.zeros(7, dtype=np.float64)
        self._gripper_width_m = gripper_width_m
        self._base = np.zeros((64, 96, 3), dtype=np.uint8)
        self._wrist = np.zeros((48, 64, 3), dtype=np.uint8) if include_wrist else None
        self.applied_targets: list[contract.FrankaTarget] = []
        self.hold_count = 0
        self.fail_next_command = False
        self._faulted = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            control_update_hz=1000,
            position_dependent_velocity_limits=True,
            local_watchdog=True,
            real_hardware=False,
        )

    def read_snapshot(self) -> contract.FrankaSnapshot:
        now = time.monotonic_ns()
        return contract.FrankaSnapshot(
            q_rad=self._q.copy(),
            dq_rad_s=self._dq.copy(),
            gripper_width_m=self._gripper_width_m,
            base_rgb=self._base.copy(),
            captured_monotonic_ns=now,
            base_image_monotonic_ns=now,
            wrist_rgb=None if self._wrist is None else self._wrist.copy(),
            wrist_image_monotonic_ns=None if self._wrist is None else now,
        )

    def apply_target(self, target: contract.FrankaTarget) -> None:
        if self._faulted:
            raise RuntimeError("mock backend is faulted")
        if self.fail_next_command:
            self.fail_next_command = False
            self._faulted = True
            raise RuntimeError("injected mock command failure")
        previous = self._q.copy()
        self._q = np.asarray(target.q_target_rad, dtype=np.float64).copy()
        self._dq = (self._q - previous) * contract.CONTROL_HZ
        self._gripper_width_m = float(target.gripper_width_m)
        self.applied_targets.append(
            contract.FrankaTarget(q_target_rad=self._q.copy(), gripper_width_m=self._gripper_width_m)
        )

    def hold_position(self) -> None:
        self.hold_count += 1
        self._dq.fill(0.0)

    def recover(self) -> None:
        self._faulted = False
        self._dq.fill(0.0)


@dataclass(frozen=True)
class FrankaSoftwareVersions:
    ubuntu: str
    ros_distro: str
    franka_ros2: str
    libfranka: str
    franka_description: str
    robot_system: str


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)+", value)
    if match is None:
        raise RuntimeError(f"unable to parse version from {value!r}")
    return tuple(int(part) for part in match.group(0).split("."))


def validate_software_versions(versions: FrankaSoftwareVersions) -> None:
    failures = []
    if versions.ubuntu != "24.04":
        failures.append(f"Ubuntu {versions.ubuntu} is not the v1 target 24.04")
    if versions.ros_distro.lower() != "jazzy":
        failures.append(f"ROS distro {versions.ros_distro!r} is not Jazzy")
    minimums = {
        "franka_ros2": (versions.franka_ros2, (3, 4, 0)),
        "libfranka": (versions.libfranka, (0, 20, 4)),
        "franka_description": (versions.franka_description, (2, 8, 0)),
        "robot_system": (versions.robot_system, (5, 9, 0)),
    }
    for name, (value, minimum) in minimums.items():
        if _version_tuple(value) < minimum:
            failures.append(f"{name} {value} is older than {'.'.join(map(str, minimum))}")
    if failures:
        raise RuntimeError("incompatible Franka software stack: " + "; ".join(failures))


def _command_output(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=10)
    return result.stdout.strip()


def _ros_package_version(package: str) -> str:
    prefix = _command_output("ros2", "pkg", "prefix", package)
    package_xml = ET.parse(f"{prefix}/share/{package}/package.xml")
    version = package_xml.getroot().findtext("version")
    if not version:
        raise RuntimeError(f"ROS package {package} does not declare a version")
    return version


def probe_software_versions(*, robot_system_version: str) -> FrankaSoftwareVersions:
    if platform.system() != "Linux":
        raise RuntimeError("franka_ros2 is supported only on Linux")
    os_release = {}
    with open("/etc/os-release", encoding="utf-8") as file:
        for line in file:
            if "=" in line:
                key, value = line.rstrip().split("=", 1)
                os_release[key] = value.strip('"')
    versions = FrankaSoftwareVersions(
        ubuntu=os_release.get("VERSION_ID", "unknown"),
        ros_distro=os.environ.get("ROS_DISTRO", ""),
        franka_ros2=_ros_package_version("franka_bringup"),
        libfranka=_command_output("pkg-config", "--modversion", "libfranka"),
        franka_description=_ros_package_version("franka_description"),
        robot_system=robot_system_version,
    )
    validate_software_versions(versions)
    return versions


class Ros2FrankaBackend(FrankaBackend):
    """ROS 2 adapter for a local 1 kHz JointTrajectory controller and Franka Hand action server.

    ROS imports are deliberately delayed so OpenPI training and unit tests do not depend on ROS.
    The controller, not this 20 Hz Python process, owns interpolation, the watchdog, and libfranka's
    position-dependent velocity checks.
    """

    def __init__(
        self,
        *,
        base_image_topic: str,
        wrist_image_topic: str | None = None,
        joint_state_topic: str = "/joint_states",
        trajectory_topic: str = "/fr3_arm_controller/joint_trajectory",
        gripper_action_name: str = "/fr3_gripper/move",
        gripper_speed_m_s: float = 0.03,
        real_hardware: bool = True,
        controller_update_hz: int = 1000,
        controller_has_dynamic_limits: bool = False,
        controller_has_local_watchdog: bool = False,
    ) -> None:
        if real_hardware and (
            controller_update_hz != 1000 or not controller_has_dynamic_limits or not controller_has_local_watchdog
        ):
            raise RuntimeError(
                "real FR3 execution requires a 1 kHz controller with position-dependent limits and a local watchdog"
            )
        self._capabilities = BackendCapabilities(
            control_update_hz=controller_update_hz,
            position_dependent_velocity_limits=controller_has_dynamic_limits,
            local_watchdog=controller_has_local_watchdog,
            real_hardware=real_hardware,
        )
        self._rclpy = importlib.import_module("rclpy")
        sensor_msgs = importlib.import_module("sensor_msgs.msg")
        trajectory_msgs = importlib.import_module("trajectory_msgs.msg")
        builtin_interfaces = importlib.import_module("builtin_interfaces.msg")
        franka_actions = importlib.import_module("franka_msgs.action")
        rclpy_action = importlib.import_module("rclpy.action")
        rclpy_executors = importlib.import_module("rclpy.executors")

        self._joint_trajectory_type = trajectory_msgs.JointTrajectory
        self._joint_trajectory_point_type = trajectory_msgs.JointTrajectoryPoint
        self._duration_type = builtin_interfaces.Duration
        self._gripper_move_type = franka_actions.Move
        self._lock = threading.Lock()
        self._q: np.ndarray | None = None
        self._dq: np.ndarray | None = None
        self._gripper_width_m: float | None = None
        self._joint_received_ns: int | None = None
        self._base_rgb: np.ndarray | None = None
        self._base_received_ns: int | None = None
        self._wrist_rgb: np.ndarray | None = None
        self._wrist_received_ns: int | None = None
        self._last_gripper_goal: float | None = None
        self._gripper_speed_m_s = gripper_speed_m_s
        self._control_period_s = 1.0 / contract.CONTROL_HZ

        if not self._rclpy.ok():
            self._rclpy.init()
        self._node = self._rclpy.create_node("openpi_franka_backend")
        self._trajectory_publisher = self._node.create_publisher(self._joint_trajectory_type, trajectory_topic, 1)
        self._gripper_client = rclpy_action.ActionClient(self._node, self._gripper_move_type, gripper_action_name)
        self._node.create_subscription(sensor_msgs.JointState, joint_state_topic, self._on_joint_state, 10)
        self._node.create_subscription(sensor_msgs.Image, base_image_topic, self._on_base_image, 2)
        if wrist_image_topic:
            self._node.create_subscription(sensor_msgs.Image, wrist_image_topic, self._on_wrist_image, 2)
        self._expect_wrist = wrist_image_topic is not None
        self._executor = rclpy_executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, name="franka-ros2", daemon=True)
        self._spin_thread.start()

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @staticmethod
    def _decode_image(message: Any) -> np.ndarray:
        if message.encoding not in {"rgb8", "bgr8"}:
            raise RuntimeError(f"unsupported ROS image encoding {message.encoding!r}; expected rgb8 or bgr8")
        row_bytes = int(message.step)
        packed = np.frombuffer(message.data, dtype=np.uint8).reshape(int(message.height), row_bytes)
        image = packed[:, : int(message.width) * 3].reshape(int(message.height), int(message.width), 3).copy()
        return image[..., ::-1].copy() if message.encoding == "bgr8" else image

    def _on_joint_state(self, message: Any) -> None:
        positions = dict(zip(message.name, message.position, strict=True))
        velocities = dict(zip(message.name, message.velocity, strict=True)) if message.velocity else {}
        if not all(name in positions for name in contract.JOINT_NAMES):
            return
        q = np.asarray([positions[name] for name in contract.JOINT_NAMES], dtype=np.float64)
        dq = np.asarray([velocities.get(name, 0.0) for name in contract.JOINT_NAMES], dtype=np.float64)
        finger_names = ("fr3_finger_joint1", "fr3_finger_joint2")
        width = (
            sum(float(positions[name]) for name in finger_names)
            if all(name in positions for name in finger_names)
            else None
        )
        with self._lock:
            self._q = q
            self._dq = dq
            if width is not None:
                self._gripper_width_m = width
            self._joint_received_ns = time.monotonic_ns()

    def _on_base_image(self, message: Any) -> None:
        image = self._decode_image(message)
        with self._lock:
            self._base_rgb = image
            self._base_received_ns = time.monotonic_ns()

    def _on_wrist_image(self, message: Any) -> None:
        image = self._decode_image(message)
        with self._lock:
            self._wrist_rgb = image
            self._wrist_received_ns = time.monotonic_ns()

    def read_snapshot(self) -> contract.FrankaSnapshot:
        with self._lock:
            missing = []
            if self._q is None or self._dq is None or self._joint_received_ns is None:
                missing.append("joint state")
            if self._gripper_width_m is None:
                missing.append("Franka Hand state")
            if self._base_rgb is None or self._base_received_ns is None:
                missing.append("base camera")
            if self._expect_wrist and (self._wrist_rgb is None or self._wrist_received_ns is None):
                missing.append("wrist camera")
            if missing:
                raise RuntimeError(f"ROS 2 Franka backend is waiting for: {', '.join(missing)}")
            assert self._q is not None
            assert self._dq is not None
            assert self._joint_received_ns is not None
            assert self._gripper_width_m is not None
            assert self._base_rgb is not None
            assert self._base_received_ns is not None
            return contract.FrankaSnapshot(
                q_rad=self._q.copy(),
                dq_rad_s=self._dq.copy(),
                gripper_width_m=self._gripper_width_m,
                base_rgb=self._base_rgb.copy(),
                captured_monotonic_ns=self._joint_received_ns,
                base_image_monotonic_ns=self._base_received_ns,
                wrist_rgb=None if self._wrist_rgb is None else self._wrist_rgb.copy(),
                wrist_image_monotonic_ns=self._wrist_received_ns,
            )

    def apply_target(self, target: contract.FrankaTarget) -> None:
        trajectory = self._joint_trajectory_type()
        trajectory.joint_names = list(contract.JOINT_NAMES)
        point = self._joint_trajectory_point_type()
        point.positions = np.asarray(target.q_target_rad, dtype=np.float64).tolist()
        nanoseconds = int(self._control_period_s * 1_000_000_000)
        point.time_from_start = self._duration_type(
            sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000
        )
        trajectory.points = [point]
        self._trajectory_publisher.publish(trajectory)

        if self._last_gripper_goal is None or abs(target.gripper_width_m - self._last_gripper_goal) >= 0.001:
            goal = self._gripper_move_type.Goal()
            goal.width = float(target.gripper_width_m)
            goal.speed = float(self._gripper_speed_m_s)
            self._gripper_client.send_goal_async(goal)
            self._last_gripper_goal = float(target.gripper_width_m)

    def hold_position(self) -> None:
        snapshot = self.read_snapshot()
        self.apply_target(
            contract.FrankaTarget(q_target_rad=snapshot.q_rad.copy(), gripper_width_m=snapshot.gripper_width_m)
        )

    def recover(self) -> None:
        raise RuntimeError("real Franka recovery is manual; recreate the backend after clearing the robot fault")

    def close(self) -> None:
        self._executor.shutdown(timeout_sec=2.0)
        self._node.destroy_node()
        self._spin_thread.join(timeout=2.0)
