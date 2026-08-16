"""Collect real SO-101 demonstrations with two USB cameras and safe teleoperation.

Keyboard controls while the program is focused:

* S: start a new episode from the current measured poses.
* Y: mark the active episode successful and publish it.
* D: discard the active episode into the rejected directory.
* P: pause/resume motion and recording.
* A: approve a displayed automatic catch-up above 10 degrees.
* R: capture the current pose and start recording during pre-record follow.
* X: emergency hold for one second, unload, reject, and exit.
* Q: exit while idle, cancel before recording, or reject/unload/exit while recording.
"""

# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

REAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = REAL_DIR.parents[2]
SO101_DIR = REAL_DIR.parent
sys.path.insert(0, str(REAL_DIR))
sys.path.insert(0, str(SO101_DIR))

import bounded_leader_follow_test as single  # noqa: E402
import camera_capture  # noqa: E402
import dual_arm_alignment_monitor as alignment  # noqa: E402
import episode_hdf5  # noqa: E402
import no_jump_enable_test as no_jump  # noqa: E402

CONFIRMATION = "COLLECT_REAL_EPISODES"
CONTROL_RATE_HZ = 30.0
DEFAULT_SPEED_DEG_S = 15.0
MAX_ALLOWED_SPEED_DEG_S = 30.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30.0
MAX_FRAME_AGE_S = 0.100
MAX_CAMERA_SKEW_S = 0.100
MAX_TRACKING_ERROR_DEG = 15.0
TRACKING_ERROR_GRACE_S = 0.5
MAX_TEMPERATURE_C = 65
STARTUP_MARGIN_DEG = 0.0
DEFAULT_MOTION_MARGIN_DEG = 1.0
MIN_MOTION_MARGIN_DEG = 0.5
MAX_MOTION_MARGIN_DEG = 2.0
START_ALIGNMENT_DEG = 3.0
AUTO_ALIGN_STABLE_S = 1.0
AUTO_ALIGN_TIMEOUT_MARGIN_S = 10.0
DEFAULT_MAX_AUTO_ALIGN_DEG = 10.0
MIN_MAX_AUTO_ALIGN_DEG = 0.5
HARD_MAX_AUTO_ALIGN_DEG = 30.0
LARGE_AUTO_ALIGN_THRESHOLD_DEG = 10.0
DEFAULT_LARGE_AUTO_ALIGN_SPEED_DEG_S = 5.0
MIN_LARGE_AUTO_ALIGN_SPEED_DEG_S = 1.0
MAX_LARGE_AUTO_ALIGN_SPEED_DEG_S = 10.0
LARGE_AUTO_ALIGN_CONFIRM_TIMEOUT_S = 30.0
CATCH_UP_ANGLE_GATED_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
)
CATCH_UP_ANGLE_EXEMPT_JOINTS = ("wrist_roll", "gripper")
READY_HOLD_S = 1.0
NORMAL_UNLOAD_HOLD_S = 3.0
FAULT_HOLD_S = 1.0
DEFAULT_MAX_EPISODE_S = 30.0
MIN_SUCCESS_FRAMES = 30
TASKS = {
    0: "Put the earbud case in the box.",
    1: "Put the sponge in the box.",
    2: "Put the ballpoint pen in the box.",
}


class OperatorEmergencyStopError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class RecorderConfig:
    dataset_root: Path
    target_id: int
    box_id: str
    task: str
    operator: str
    object_inventory_version: str
    front_camera: int
    wrist_camera: int
    speed_deg_s: float
    max_episode_s: float
    motion_margin_deg: float
    max_auto_align_deg: float
    large_auto_align_speed_deg_s: float


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    path: Path | None
    success: bool
    abort_reason: str
    exit_requested: bool


class WindowsKeySource:
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("teleop_recorder keyboard controls require Windows")
        import msvcrt

        self._msvcrt = msvcrt

    def poll(self) -> list[str]:
        keys = []
        while self._msvcrt.kbhit():
            key = self._msvcrt.getwch()
            if key in {"\x00", "\xe0"}:
                if self._msvcrt.kbhit():
                    self._msvcrt.getwch()
                continue
            keys.append(key.lower())
        return keys


class RelativeJointController:
    def __init__(
        self,
        *,
        anchor_mapped: dict[str, float],
        anchor_follower: dict[str, float],
        follower_limits: dict[str, tuple[float, float]],
        maximum_step_deg: float,
    ) -> None:
        self.anchor_mapped = anchor_mapped.copy()
        self.anchor_follower = anchor_follower.copy()
        self.follower_limits = follower_limits
        self.maximum_step_deg = maximum_step_deg
        self.command = anchor_follower.copy()

    def reset(self, *, anchor_mapped: dict[str, float], anchor_follower: dict[str, float]) -> None:
        self.anchor_mapped = anchor_mapped.copy()
        self.anchor_follower = anchor_follower.copy()
        self.command = anchor_follower.copy()

    def compute(self, mapped_leader: dict[str, float]) -> dict[str, float]:
        no_jump._check_complete_finite(mapped_leader, "mapped Leader pose")
        desired = {
            name: self.anchor_follower[name] + mapped_leader[name] - self.anchor_mapped[name]
            for name in no_jump.hold.MOTOR_NAMES
        }
        no_jump._check_inside_limits(desired, self.follower_limits)
        command = single._slew(self.command, desired, self.maximum_step_deg)
        no_jump._check_complete_finite(command, "Follower command")
        no_jump._check_inside_limits(command, self.follower_limits)
        self.command = command
        return command.copy()


class TrackingMonitor:
    def __init__(self) -> None:
        self.fault_started_s: float | None = None
        self.maximum_error_deg = 0.0

    def reset(self) -> None:
        self.fault_started_s = None

    def observe(
        self,
        *,
        measured: dict[str, float],
        command: dict[str, float],
        now_s: float,
    ) -> float:
        errors = {name: abs(measured[name] - command[name]) for name in no_jump.hold.MOTOR_NAMES}
        joint = max(errors, key=errors.get)
        maximum = errors[joint]
        self.maximum_error_deg = max(self.maximum_error_deg, maximum)
        if maximum > MAX_TRACKING_ERROR_DEG:
            if self.fault_started_s is None:
                self.fault_started_s = now_s
            elif now_s - self.fault_started_s >= TRACKING_ERROR_GRACE_S:
                raise RuntimeError(
                    f"Follower tracking error remained above {MAX_TRACKING_ERROR_DEG:.1f} deg: "
                    f"{joint}={maximum:.2f} deg"
                )
        else:
            self.fault_started_s = None
        return maximum


def _clamp_pose_to_limits(
    pose: dict[str, float],
    limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    no_jump._check_complete_finite(pose, "pose to clamp")
    return {
        name: min(max(pose[name], limits[name][0]), limits[name][1])
        for name in no_jump.hold.MOTOR_NAMES
    }


def _pose_inside_limits(
    pose: dict[str, float],
    limits: dict[str, tuple[float, float]],
) -> bool:
    return all(
        limits[name][0] - no_jump.LIMIT_EPSILON_DEG
        <= pose[name]
        <= limits[name][1] + no_jump.LIMIT_EPSILON_DEG
        for name in no_jump.hold.MOTOR_NAMES
    )


def _maximum_pose_delta(
    first: dict[str, float],
    second: dict[str, float],
    joint_names: tuple[str, ...] | None = None,
) -> tuple[str, float]:
    selected_joints = no_jump.hold.MOTOR_NAMES if joint_names is None else joint_names
    errors = {name: abs(first[name] - second[name]) for name in selected_joints}
    joint = max(errors, key=errors.get)
    return joint, errors[joint]


def _auto_align_timeout_s(initial_error_deg: float, alignment_speed_deg_s: float) -> float:
    """Budget large catch-up time conservatively even when a faster speed is selected."""
    timeout_reference_speed_deg_s = (
        min(alignment_speed_deg_s, DEFAULT_LARGE_AUTO_ALIGN_SPEED_DEG_S)
        if initial_error_deg > LARGE_AUTO_ALIGN_THRESHOLD_DEG
        else alignment_speed_deg_s
    )
    return max(
        AUTO_ALIGN_TIMEOUT_MARGIN_S,
        initial_error_deg / timeout_reference_speed_deg_s + AUTO_ALIGN_TIMEOUT_MARGIN_S,
    )


def _dict_from_pose(pose: dict[str, float]) -> list[float]:
    return [float(pose[name]) for name in no_jump.hold.MOTOR_NAMES]


def _git_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {"software_commit": "unknown", "software_dirty": True}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result = {"software_commit": commit, "software_dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _episode_id(config: RecorderConfig) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{stamp}_box-{config.box_id}_target-{config.target_id}_{suffix}"


def _load_lock_metadata() -> dict[str, Any]:
    lock_path = REAL_DIR / "calibration" / "calibration_lock.json"
    with lock_path.open("r", encoding="utf-8") as stream:
        lock = json.load(stream)
    return {
        "calibration_freeze_id": lock["freeze_id"],
        "calibration_hashes": {arm["role"]: arm["active_sha256"] for arm in lock["arms"]},
    }


def _build_metadata(
    config: RecorderConfig,
    cameras: camera_capture.DualCameraCapture,
) -> dict[str, Any]:
    metadata = {
        "target_id": config.target_id,
        "task": config.task,
        "box_id": config.box_id,
        "operator": config.operator,
        "object_inventory_version": config.object_inventory_version,
        "camera_config": cameras.config(),
        "control_rate_hz": CONTROL_RATE_HZ,
        "action_mode": "absolute_calibrated_motor_degrees",
        "collection_started_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    metadata.update(_load_lock_metadata())
    metadata.update(_git_metadata())
    return metadata


def _build_buses(
    calibrations: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    leisaac_root = REPO_ROOT / "tmp" / "leisaac-v0.4.0"
    buses = {}
    for role, port, _ in alignment.ARM_CONFIGS:
        buses[role] = no_jump.hold._build_bus(port, leisaac_root, calibrations[role])
    return buses


def _load_calibrations() -> dict[str, dict[str, dict[str, int]]]:
    leisaac_root = REPO_ROOT / "tmp" / "leisaac-v0.4.0"
    calibrations = {}
    for role, _, calibration_id in alignment.ARM_CONFIGS:
        path = no_jump.hold._calibration_path(leisaac_root, calibration_id)
        calibrations[role] = no_jump.hold._load_calibration_data(path)
    return calibrations


class RealTeleopRecorder:
    def __init__(
        self,
        *,
        config: RecorderConfig,
        cameras: camera_capture.DualCameraCapture,
        calibrations: dict[str, dict[str, dict[str, int]]],
        key_source: Any,
        bus_factory: Callable[[], dict[str, Any]],
        writer_factory: Callable[..., episode_hdf5.IncrementalEpisodeWriter] = episode_hdf5.IncrementalEpisodeWriter,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.cameras = cameras
        self.calibrations = calibrations
        self.key_source = key_source
        self.bus_factory = bus_factory
        self.writer_factory = writer_factory
        self.clock = clock
        self.sleep = sleep
        self.startup_limits = no_jump.hold._angle_limits(calibrations["follower"], STARTUP_MARGIN_DEG)
        self.follower_limits = no_jump.hold._angle_limits(
            calibrations["follower"],
            config.motion_margin_deg,
        )

    def _safe_auto_align_target(
        self,
        mapped_leader: dict[str, float],
        initial_follower: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        """Clamp all targets inward and gate catch-up only on the first four joints."""
        no_jump._check_inside_limits(mapped_leader, self.startup_limits)
        target = _clamp_pose_to_limits(mapped_leader, self.follower_limits)
        joint, maximum_gated_delta = _maximum_pose_delta(
            target,
            initial_follower,
            CATCH_UP_ANGLE_GATED_JOINTS,
        )
        if maximum_gated_delta > self.config.max_auto_align_deg:
            raise RuntimeError(
                "Automatic alignment refused before motion: "
                f"{joint} requires {maximum_gated_delta:.2f} deg, exceeding the configured "
                f"{self.config.max_auto_align_deg:.2f} deg catch-up limit. "
                "With both arms torque-disabled, place them closer and retry."
            )
        return target, maximum_gated_delta

    def _preflight_episode(
        self, buses: dict[str, Any]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float]:
        single._check_modes_and_torque(buses["leader"], "Leader", 0)
        single._check_modes_and_torque(buses["follower"], "Follower", 0)
        leader_temperature = single._read_temperatures(buses["leader"], "Leader", MAX_TEMPERATURE_C)
        follower_temperature = single._read_temperatures(buses["follower"], "Follower", MAX_TEMPERATURE_C)
        leader = buses["leader"].sync_read("Present_Position")
        follower = buses["follower"].sync_read("Present_Position")
        no_jump._check_complete_finite(leader, "initial Leader pose")
        no_jump._check_complete_finite(follower, "initial Follower pose")
        # Startup may be exactly at a frozen calibrated endpoint. The first goal
        # remains the measured pose; automatic alignment then moves only toward
        # a target clamped into the normal motion interior.
        no_jump._check_inside_limits(follower, self.startup_limits)
        mapped = alignment.mapped_follower_target(
            leader,
            self.calibrations["leader"],
            self.calibrations["follower"],
        )
        no_jump._check_complete_finite(mapped, "initial mapped Leader pose")
        safe_target, maximum_gated_error = self._safe_auto_align_target(mapped, follower)
        maximum_joint, maximum_all_error = _maximum_pose_delta(safe_target, follower)
        endpoint_clamp_joint, endpoint_clamp = _maximum_pose_delta(mapped, safe_target)
        if endpoint_clamp > 0:
            print(
                f"AUTO_ALIGN_ENDPOINT_CLAMP joint={endpoint_clamp_joint} "
                f"inward_deg={endpoint_clamp:.2f} motion_margin_deg={self.config.motion_margin_deg:.2f}",
                flush=True,
            )
        if maximum_all_error > START_ALIGNMENT_DEG:
            print(
                f"AUTO_ALIGN_NEEDED initial_max_error_deg={maximum_all_error:.2f} "
                f"joint={maximum_joint} gated_max_error_deg={maximum_gated_error:.2f} "
                f"threshold_deg={START_ALIGNMENT_DEG:.1f}",
                flush=True,
            )
        else:
            print(
                f"AUTO_ALIGN_ALREADY_CLOSE initial_max_error_deg={maximum_all_error:.2f} "
                f"gated_max_error_deg={maximum_gated_error:.2f}",
                flush=True,
            )
        self.cameras.get_latest_pair(max_age_s=MAX_FRAME_AGE_S, max_skew_s=MAX_CAMERA_SKEW_S)
        return (
            leader,
            follower,
            mapped,
            max(leader_temperature, follower_temperature),
            maximum_gated_error,
        )

    def _wait_for_large_auto_align_confirmation(
        self,
        buses: dict[str, Any],
        *,
        initial_follower: dict[str, float],
        mapped_leader: dict[str, float],
    ) -> bool:
        """Require an extra operator confirmation while both arms remain torque-disabled."""
        target, maximum_gated_delta = self._safe_auto_align_target(mapped_leader, initial_follower)
        _, maximum_all_delta = _maximum_pose_delta(target, initial_follower)
        deltas = {
            name: target[name] - initial_follower[name]
            for name in no_jump.hold.MOTOR_NAMES
        }
        formatted = " ".join(f"{name}={deltas[name]:+.2f}" for name in no_jump.hold.MOTOR_NAMES)
        print(
            "LARGE_AUTO_ALIGN_CONFIRM_REQUIRED "
            f"max_gated_delta_deg={maximum_gated_delta:.2f} "
            f"max_all_delta_deg={maximum_all_delta:.2f} deltas_deg[{formatted}] "
            "speed_cap_deg_s="
            f"{min(self.config.speed_deg_s, self.config.large_auto_align_speed_deg_s):.1f}; "
            "press A to allow, Q to cancel, X for emergency exit",
            flush=True,
        )

        deadline_s = self.clock() + LARGE_AUTO_ALIGN_CONFIRM_TIMEOUT_S
        next_check_s = self.clock()
        while True:
            keys = self.key_source.poll()
            if "x" in keys:
                raise OperatorEmergencyStopError(
                    "operator requested exit before large automatic alignment"
                )
            if "q" in keys:
                print("LARGE_AUTO_ALIGN_CANCELLED", flush=True)
                return False
            if "a" in keys:
                print("LARGE_AUTO_ALIGN_CONFIRMED: rechecking poses before torque enable", flush=True)
                return True

            now_s = self.clock()
            if now_s >= deadline_s:
                raise RuntimeError(
                    f"large automatic alignment was not confirmed within "
                    f"{LARGE_AUTO_ALIGN_CONFIRM_TIMEOUT_S:.0f} seconds"
                )
            if now_s >= next_check_s:
                single._check_modes_and_torque(buses["leader"], "Leader", 0)
                single._check_modes_and_torque(buses["follower"], "Follower", 0)
                single._read_temperatures(buses["leader"], "Leader", MAX_TEMPERATURE_C)
                single._read_temperatures(buses["follower"], "Follower", MAX_TEMPERATURE_C)
                self.cameras.get_latest_pair(
                    max_age_s=MAX_FRAME_AGE_S,
                    max_skew_s=MAX_CAMERA_SKEW_S,
                )
                next_check_s = now_s + 0.5
            self.sleep(0.05)

    def _ready_hold(
        self,
        buses: dict[str, Any],
        pose: dict[str, float],
    ) -> float:
        maximum_temperature = 0.0
        interval_s = 1.0 / CONTROL_RATE_HZ
        for cycle in range(math.ceil(READY_HOLD_S * CONTROL_RATE_HZ)):
            buses["follower"].sync_write("Goal_Position", pose)
            self.sleep(interval_s)
            measured = buses["follower"].sync_read("Present_Position")
            no_jump._check_complete_finite(measured, "Follower ready-hold pose")
            error = max(abs(measured[name] - pose[name]) for name in no_jump.hold.MOTOR_NAMES)
            if error > 3.0:
                raise RuntimeError(f"Follower moved {error:.2f} deg during no-jump ready hold")
            if cycle % 6 == 0:
                single._check_modes_and_torque(buses["leader"], "Leader", 0)
                single._check_modes_and_torque(buses["follower"], "Follower", 1)
                maximum_temperature = max(
                    maximum_temperature,
                    single._read_temperatures(buses["leader"], "Leader", MAX_TEMPERATURE_C),
                    single._read_temperatures(buses["follower"], "Follower", MAX_TEMPERATURE_C),
                )
            self.cameras.get_latest_pair(max_age_s=MAX_FRAME_AGE_S, max_skew_s=MAX_CAMERA_SKEW_S)
        return maximum_temperature

    def _auto_align_follower_to_leader(
        self,
        buses: dict[str, Any],
        *,
        initial_follower: dict[str, float],
        large_alignment_confirmed: bool,
    ) -> tuple[dict[str, float], dict[str, float], float]:
        """Slowly move Follower onto the live mapped Leader pose before recording."""
        interval_s = 1.0 / CONTROL_RATE_HZ
        required_streak = max(1, math.ceil(AUTO_ALIGN_STABLE_S * CONTROL_RATE_HZ))
        aligned_streak = 0
        tick = 0
        command = initial_follower.copy()
        tracking = TrackingMonitor()
        maximum_temperature = 0.0

        leader = buses["leader"].sync_read("Present_Position")
        no_jump._check_complete_finite(leader, "auto-align Leader pose")
        mapped = alignment.mapped_follower_target(
            leader,
            self.calibrations["leader"],
            self.calibrations["follower"],
        )
        no_jump._check_complete_finite(mapped, "auto-align mapped Leader pose")
        target, initial_gated_error = self._safe_auto_align_target(mapped, initial_follower)
        _, initial_all_error = _maximum_pose_delta(target, initial_follower)
        if initial_gated_error > LARGE_AUTO_ALIGN_THRESHOLD_DEG and not large_alignment_confirmed:
            raise RuntimeError(
                "automatic alignment grew beyond 10 degrees after preflight without large-move confirmation"
            )
        alignment_speed_deg_s = (
            min(self.config.speed_deg_s, self.config.large_auto_align_speed_deg_s)
            if initial_all_error > LARGE_AUTO_ALIGN_THRESHOLD_DEG
            else self.config.speed_deg_s
        )
        timeout_s = _auto_align_timeout_s(initial_all_error, alignment_speed_deg_s)
        deadline_s = self.clock() + timeout_s
        next_report_s = self.clock()

        print(
            f"AUTO_ALIGN_START max_error_deg={initial_all_error:.2f} "
            f"gated_max_error_deg={initial_gated_error:.2f} "
            f"speed_deg_s={alignment_speed_deg_s:.1f} timeout_s={timeout_s:.1f}; "
            "keep Leader steady until AUTO_ALIGN_READY "
            "(X=emergency-unload)",
            flush=True,
        )

        while True:
            now_s = self.clock()
            if now_s >= deadline_s:
                raise RuntimeError(
                    f"automatic Leader/Follower alignment timed out after {timeout_s:.1f} s"
                )

            if "x" in self.key_source.poll():
                raise OperatorEmergencyStopError("operator requested emergency unload during auto-alignment")

            single._check_modes_and_torque(buses["leader"], "Leader", 0)
            if tick % 30 == 0:
                single._check_modes_and_torque(buses["follower"], "Follower", 1)

            leader = buses["leader"].sync_read("Present_Position")
            no_jump._check_complete_finite(leader, "auto-align Leader pose")
            mapped = alignment.mapped_follower_target(
                leader,
                self.calibrations["leader"],
                self.calibrations["follower"],
            )
            target, live_gated_delta = self._safe_auto_align_target(mapped, initial_follower)
            _, live_all_delta = _maximum_pose_delta(target, initial_follower)
            if live_gated_delta > LARGE_AUTO_ALIGN_THRESHOLD_DEG and not large_alignment_confirmed:
                raise RuntimeError(
                    "Leader moved beyond the 10-degree small-alignment gate after torque enable; "
                    "episode stopped"
                )

            live_speed_deg_s = (
                min(self.config.speed_deg_s, self.config.large_auto_align_speed_deg_s)
                if live_all_delta > LARGE_AUTO_ALIGN_THRESHOLD_DEG
                else self.config.speed_deg_s
            )
            maximum_step_deg = live_speed_deg_s / CONTROL_RATE_HZ
            command = single._slew(command, target, maximum_step_deg)
            no_jump._check_complete_finite(command, "auto-align Follower command")
            # The first few commands may still be in the endpoint startup band,
            # but slew only toward the clamped interior target.
            no_jump._check_inside_limits(command, self.startup_limits)
            buses["follower"].sync_write("Goal_Position", command)

            self.sleep(interval_s)

            follower = buses["follower"].sync_read("Present_Position")
            no_jump._check_complete_finite(follower, "auto-align Follower pose")
            no_jump._check_inside_limits(follower, self.startup_limits)
            tracking_error = tracking.observe(
                measured=follower,
                command=command,
                now_s=self.clock(),
            )

            if tick % 6 == 0:
                maximum_temperature = max(
                    maximum_temperature,
                    single._read_temperatures(buses["leader"], "Leader", MAX_TEMPERATURE_C),
                    single._read_temperatures(buses["follower"], "Follower", MAX_TEMPERATURE_C),
                )
                self.cameras.get_latest_pair(
                    max_age_s=MAX_FRAME_AGE_S,
                    max_skew_s=MAX_CAMERA_SKEW_S,
                )

            alignment_error = max(
                abs(target[name] - follower[name]) for name in no_jump.hold.MOTOR_NAMES
            )
            safely_inside = _pose_inside_limits(follower, self.follower_limits)
            aligned_streak = (
                aligned_streak + 1
                if alignment_error <= START_ALIGNMENT_DEG and safely_inside
                else 0
            )

            report_now_s = self.clock()
            if report_now_s >= next_report_s:
                print(
                    f"auto_align_error_deg={alignment_error:.2f} "
                    f"tracking_deg={tracking_error:.2f} "
                    f"stable={aligned_streak}/{required_streak}",
                    flush=True,
                )
                next_report_s = report_now_s + 1.0

            if aligned_streak >= required_streak:
                # Re-anchor teleoperation at the actual poses at handoff so there
                # is no discontinuity between automatic catch-up and normal teleop.
                final_leader = buses["leader"].sync_read("Present_Position")
                final_follower = buses["follower"].sync_read("Present_Position")
                no_jump._check_complete_finite(final_leader, "aligned Leader pose")
                no_jump._check_complete_finite(final_follower, "aligned Follower pose")
                final_mapped = alignment.mapped_follower_target(
                    final_leader,
                    self.calibrations["leader"],
                    self.calibrations["follower"],
                )
                final_target, _ = self._safe_auto_align_target(final_mapped, initial_follower)
                final_error = max(
                    abs(final_target[name] - final_follower[name])
                    for name in no_jump.hold.MOTOR_NAMES
                )
                if final_error <= START_ALIGNMENT_DEG and _pose_inside_limits(
                    final_follower,
                    self.follower_limits,
                ):
                    print(
                        f"AUTO_ALIGN_READY final_max_error_deg={final_error:.2f}; "
                        "entering pre-record follow",
                        flush=True,
                    )
                    return final_follower, final_mapped, maximum_temperature
                aligned_streak = 0

            tick += 1

    def _wait_for_record_start(
        self,
        buses: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float], float] | None:
        """Follow the Leader without recording until R captures fresh relative anchors."""
        leader = buses["leader"].sync_read("Present_Position")
        follower = buses["follower"].sync_read("Present_Position")
        no_jump._check_complete_finite(leader, "pre-record Leader pose")
        no_jump._check_complete_finite(follower, "pre-record Follower pose")
        no_jump._check_inside_limits(follower, self.follower_limits)
        mapped = alignment.mapped_follower_target(
            leader,
            self.calibrations["leader"],
            self.calibrations["follower"],
        )
        no_jump._check_complete_finite(mapped, "pre-record mapped Leader pose")
        controller = RelativeJointController(
            anchor_mapped=mapped,
            anchor_follower=follower,
            follower_limits=self.follower_limits,
            maximum_step_deg=self.config.speed_deg_s / CONTROL_RATE_HZ,
        )
        tracking = TrackingMonitor()
        buses["follower"].sync_write("Goal_Position", follower)

        maximum_temperature = 0.0
        next_tick_s = self.clock()
        next_report_s = self.clock()
        tick = 0

        print(
            "PRE_RECORD_FOLLOW_ACTIVE: move the Leader to the desired start pose; "
            "keep it steady and press R to start recording; "
            "Q=cancel, X=emergency-unload",
            flush=True,
        )

        while True:
            keys = self.key_source.poll()

            if "x" in keys:
                raise OperatorEmergencyStopError(
                    "operator requested emergency unload while waiting to record"
                )

            if "q" in keys:
                print("RECORD_START_CANCELLED", flush=True)
                return None

            if "r" in keys:
                # Re-read both arms at the exact handoff moment. This makes
                # pressing R the zero point of relative teleoperation.
                leader = buses["leader"].sync_read("Present_Position")
                follower = buses["follower"].sync_read("Present_Position")
                no_jump._check_complete_finite(leader, "record-start Leader pose")
                no_jump._check_complete_finite(follower, "record-start Follower pose")
                no_jump._check_inside_limits(follower, self.follower_limits)

                mapped = alignment.mapped_follower_target(
                    leader,
                    self.calibrations["leader"],
                    self.calibrations["follower"],
                )
                no_jump._check_complete_finite(mapped, "record-start mapped Leader pose")

                # Keep the handoff no-jump: the current measured Follower pose
                # becomes both its held goal and the controller anchor.
                buses["follower"].sync_write("Goal_Position", follower)

                print(
                    "RECORD_START_CONFIRMED: relative teleoperation anchor captured",
                    flush=True,
                )
                return follower, mapped, maximum_temperature

            leader = buses["leader"].sync_read("Present_Position")
            follower = buses["follower"].sync_read("Present_Position")
            no_jump._check_complete_finite(leader, "pre-record Leader pose")
            no_jump._check_complete_finite(follower, "pre-record Follower pose")
            no_jump._check_inside_limits(follower, self.follower_limits)
            mapped = alignment.mapped_follower_target(
                leader,
                self.calibrations["leader"],
                self.calibrations["follower"],
            )
            no_jump._check_complete_finite(mapped, "pre-record mapped Leader pose")
            command = controller.compute(mapped)
            tracking_error = tracking.observe(
                measured=follower,
                command=command,
                now_s=self.clock(),
            )
            buses["follower"].sync_write("Goal_Position", command)

            if tick % 6 == 0:
                single._check_modes_and_torque(buses["leader"], "Leader", 0)
                single._check_modes_and_torque(buses["follower"], "Follower", 1)
                maximum_temperature = max(
                    maximum_temperature,
                    single._read_temperatures(
                        buses["leader"], "Leader", MAX_TEMPERATURE_C
                    ),
                    single._read_temperatures(
                        buses["follower"], "Follower", MAX_TEMPERATURE_C
                    ),
                )
            self.cameras.get_latest_pair(
                max_age_s=MAX_FRAME_AGE_S,
                max_skew_s=MAX_CAMERA_SKEW_S,
            )

            now_s = self.clock()
            if now_s >= next_report_s:
                print(
                    f"pre_record_follow_tracking_deg={tracking_error:.2f} "
                    f"hottest_c={maximum_temperature:.0f}; press R when steady",
                    flush=True,
                )
                next_report_s = now_s + 1.0

            tick += 1
            next_tick_s += 1.0 / CONTROL_RATE_HZ
            sleep_s = next_tick_s - self.clock()
            if sleep_s > 0:
                self.sleep(sleep_s)
            elif sleep_s < -0.5:
                raise RuntimeError(
                    f"pre-record follow loop is {-sleep_s * 1000.0:.1f} ms behind schedule"
                )

    def _normal_unload_hold(self, buses: dict[str, Any]) -> None:
        measured = buses["follower"].sync_read("Present_Position")
        no_jump._check_complete_finite(measured, "normal unload pose")
        buses["follower"].sync_write("Goal_Position", measured)
        print(
            f"SUPPORT_FOLLOWER_NOW: torque switches off in {NORMAL_UNLOAD_HOLD_S:.0f} seconds",
            flush=True,
        )
        deadline = self.clock() + NORMAL_UNLOAD_HOLD_S
        while self.clock() < deadline:
            single._check_modes_and_torque(buses["follower"], "Follower", 1)
            single._read_temperatures(buses["follower"], "Follower", MAX_TEMPERATURE_C)
            self.sleep(0.1)

    def collect_one_episode(self) -> EpisodeResult:
        buses = self.bus_factory()
        connected: list[tuple[str, Any]] = []
        follower_enabled = False
        writer: episode_hdf5.IncrementalEpisodeWriter | None = None
        success = False
        abort_reason = "not_started"
        exit_requested = False
        maximum_temperature = 0.0
        maximum_frame_age_ms = 0.0
        maximum_camera_skew_ms = 0.0
        control_overruns = 0
        paused_ticks = 0
        tracking = TrackingMonitor()
        caught: BaseException | None = None
        try:
            buses["leader"].connect()
            connected.append(("Leader", buses["leader"]))
            buses["follower"].connect()
            connected.append(("Follower", buses["follower"]))
            (
                _,
                initial_follower,
                initial_mapped,
                maximum_temperature,
                initial_alignment_error,
            ) = self._preflight_episode(buses)
            large_alignment_confirmed = False
            if initial_alignment_error > LARGE_AUTO_ALIGN_THRESHOLD_DEG:
                large_alignment_confirmed = self._wait_for_large_auto_align_confirmation(
                    buses,
                    initial_follower=initial_follower,
                    mapped_leader=initial_mapped,
                )
                if not large_alignment_confirmed:
                    abort_reason = "operator_cancel_large_auto_align"
                    return EpisodeResult(
                        path=None,
                        success=False,
                        abort_reason=abort_reason,
                        exit_requested=False,
                    )
                (
                    _,
                    initial_follower,
                    initial_mapped,
                    maximum_temperature,
                    initial_alignment_error,
                ) = self._preflight_episode(buses)
            no_jump._print_pose("episode_start_follower_deg", initial_follower)
            buses["follower"].sync_write("Goal_Position", initial_follower)
            buses["follower"].enable_torque(num_retry=2)
            follower_enabled = True
            print("TORQUE_ENABLED_AT_MEASURED_FOLLOWER_POSE", flush=True)
            maximum_temperature = max(maximum_temperature, self._ready_hold(buses, initial_follower))
            initial_follower, initial_mapped, auto_align_temperature = self._auto_align_follower_to_leader(
                buses,
                initial_follower=initial_follower,
                large_alignment_confirmed=large_alignment_confirmed,
            )
            maximum_temperature = max(maximum_temperature, auto_align_temperature)

            record_start = self._wait_for_record_start(buses)
            if record_start is None:
                abort_reason = "operator_cancel_before_recording"
                self._normal_unload_hold(buses)
                return EpisodeResult(
                    path=None,
                    success=False,
                    abort_reason=abort_reason,
                    exit_requested=False,
                )

            initial_follower, initial_mapped, ready_temperature = record_start
            maximum_temperature = max(maximum_temperature, ready_temperature)

            episode_id = _episode_id(self.config)
            writer = self.writer_factory(
                dataset_root=self.config.dataset_root,
                episode_id=episode_id,
                metadata=_build_metadata(self.config, self.cameras),
                image_shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            )
            controller = RelativeJointController(
                anchor_mapped=initial_mapped,
                anchor_follower=initial_follower,
                follower_limits=self.follower_limits,
                maximum_step_deg=self.config.speed_deg_s / CONTROL_RATE_HZ,
            )
            print(
                "EPISODE_RECORDING: Y=success D=discard P=pause/resume X=emergency-unload Q=discard-and-exit",
                flush=True,
            )
            start_s = self.clock()
            next_tick_s = start_s
            next_report_s = start_s
            paused = False
            tick = 0
            hottest = maximum_temperature

            while True:
                now_s = self.clock()
                keys = self.key_source.poll()
                if "x" in keys:
                    exit_requested = True
                    raise OperatorEmergencyStopError("operator requested emergency unload")
                if "q" in keys:
                    abort_reason = "operator_quit"
                    exit_requested = True
                    break
                if "d" in keys:
                    abort_reason = "operator_discard"
                    break
                if "y" in keys:
                    if writer.frames_enqueued < MIN_SUCCESS_FRAMES:
                        print(
                            f"SUCCESS_IGNORED: only {writer.frames_enqueued} frames; "
                            f"at least {MIN_SUCCESS_FRAMES} are required",
                            flush=True,
                        )
                    else:
                        success = True
                        abort_reason = ""
                        break
                if "p" in keys:
                    if not paused:
                        hold_pose = buses["follower"].sync_read("Present_Position")
                        no_jump._check_complete_finite(hold_pose, "pause hold pose")
                        buses["follower"].sync_write("Goal_Position", hold_pose)
                        controller.command = hold_pose.copy()
                        paused = True
                        print("EPISODE_PAUSED: press P to resume", flush=True)
                    else:
                        leader = buses["leader"].sync_read("Present_Position")
                        follower = buses["follower"].sync_read("Present_Position")
                        no_jump._check_complete_finite(leader, "resume Leader pose")
                        no_jump._check_complete_finite(follower, "resume Follower pose")
                        mapped = alignment.mapped_follower_target(
                            leader,
                            self.calibrations["leader"],
                            self.calibrations["follower"],
                        )
                        controller.reset(anchor_mapped=mapped, anchor_follower=follower)
                        tracking.reset()
                        paused = False
                        print("EPISODE_RESUMED", flush=True)

                if now_s - start_s >= self.config.max_episode_s:
                    abort_reason = "episode_timeout"
                    print("EPISODE_TIMEOUT: episode rejected", flush=True)
                    break

                leader = buses["leader"].sync_read("Present_Position")
                follower = buses["follower"].sync_read("Present_Position")
                no_jump._check_complete_finite(leader, "Leader pose")
                no_jump._check_complete_finite(follower, "Follower pose")
                no_jump._check_inside_limits(follower, self.follower_limits)
                if tick % 6 == 0:
                    hottest = max(
                        hottest,
                        single._read_temperatures(buses["leader"], "Leader", MAX_TEMPERATURE_C),
                        single._read_temperatures(buses["follower"], "Follower", MAX_TEMPERATURE_C),
                    )
                if tick % 30 == 0:
                    single._check_modes_and_torque(buses["leader"], "Leader", 0)
                    single._check_modes_and_torque(buses["follower"], "Follower", 1)
                front, wrist = self.cameras.get_latest_pair(max_age_s=MAX_FRAME_AGE_S, max_skew_s=MAX_CAMERA_SKEW_S)
                control_timestamp_s = self.clock()
                front_age_ms = (control_timestamp_s - front.timestamp_s) * 1000.0
                wrist_age_ms = (control_timestamp_s - wrist.timestamp_s) * 1000.0
                skew_ms = abs(front.timestamp_s - wrist.timestamp_s) * 1000.0
                maximum_frame_age_ms = max(maximum_frame_age_ms, front_age_ms, wrist_age_ms)
                maximum_camera_skew_ms = max(maximum_camera_skew_ms, skew_ms)

                if paused:
                    paused_ticks += 1
                    command = controller.command.copy()
                    buses["follower"].sync_write("Goal_Position", command)
                else:
                    mapped = alignment.mapped_follower_target(
                        leader,
                        self.calibrations["leader"],
                        self.calibrations["follower"],
                    )
                    command = controller.compute(mapped)
                    tracking_error = tracking.observe(
                        measured=follower,
                        command=controller.command,
                        now_s=control_timestamp_s,
                    )
                    sample = episode_hdf5.EpisodeSample(
                        joint_pos=_dict_from_pose(follower),
                        front_rgb=front.rgb,
                        wrist_rgb=wrist.rgb,
                        action=_dict_from_pose(command),
                        control_timestamp_s=control_timestamp_s,
                        front_timestamp_s=front.timestamp_s,
                        wrist_timestamp_s=wrist.timestamp_s,
                    )
                    writer.append(sample)
                    buses["follower"].sync_write("Goal_Position", command)

                if now_s >= next_report_s:
                    status_tracking = 0.0 if paused else tracking_error
                    print(
                        f"episode_s={now_s - start_s:.1f} frames={writer.frames_enqueued} "
                        f"paused={int(paused)} queue={writer.queue_depth} "
                        f"frame_age_ms={max(front_age_ms, wrist_age_ms):.1f} "
                        f"camera_skew_ms={skew_ms:.1f} tracking_deg={status_tracking:.2f} "
                        f"hottest_c={hottest:.0f}",
                        flush=True,
                    )
                    next_report_s += 1.0
                maximum_temperature = max(maximum_temperature, hottest)
                tick += 1
                next_tick_s += 1.0 / CONTROL_RATE_HZ
                sleep_s = next_tick_s - self.clock()
                if sleep_s > 0:
                    self.sleep(sleep_s)
                else:
                    control_overruns += 1
                    if sleep_s < -0.5:
                        raise RuntimeError(f"control loop is {-sleep_s * 1000.0:.1f} ms behind schedule")

            self._normal_unload_hold(buses)
        except BaseException as exc:
            caught = exc
            abort_reason = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, OperatorEmergencyStopError):
                exit_requested = True
            if follower_enabled:
                single._fault_hold(buses["follower"], hold_s=FAULT_HOLD_S, sleep=self.sleep)
        finally:
            for role, bus in reversed(connected):
                try:
                    bus.disconnect(disable_torque=(role == "Follower"))
                except Exception as exc:
                    print(
                        f"CRITICAL_{role.upper()}_CLOSE_FAILURE: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if role == "Follower":
                        print(
                            "Disconnect Follower servo power immediately.",
                            file=sys.stderr,
                            flush=True,
                        )
                        if caught is None:
                            caught = exc
                            abort_reason = f"Follower close failure: {exc}"
            if connected:
                print("FOLLOWER_TORQUE_DISABLED_AND_PORTS_CLOSED", flush=True)

        path = None
        if writer is not None:
            path = writer.finish(
                success=success and caught is None,
                abort_reason=abort_reason,
                final_metadata={
                    "max_tracking_error_deg": tracking.maximum_error_deg,
                    "max_temperature_c": maximum_temperature,
                    "max_frame_age_ms": maximum_frame_age_ms,
                    "max_camera_skew_ms": maximum_camera_skew_ms,
                    "control_overrun_count": control_overruns,
                    "paused_tick_count": paused_ticks,
                },
            )
            print(
                f"EPISODE_{'SAVED' if success and caught is None else 'REJECTED'} path={path}",
                flush=True,
            )
        if caught is not None and not isinstance(caught, OperatorEmergencyStopError):
            print(
                f"EPISODE_FAULT: {type(caught).__name__}: {caught}",
                file=sys.stderr,
                flush=True,
            )
        return EpisodeResult(
            path=path,
            success=success and caught is None,
            abort_reason=abort_reason,
            exit_requested=exit_requested,
        )

    def run(self) -> int:
        print(
            "RECORDER_IDLE: S=start episode, Q=quit. "
            f"target_id={self.config.target_id} box_id={self.config.box_id} task={self.config.task!r}",
            flush=True,
        )
        next_camera_check_s = self.clock()
        try:
            while True:
                keys = self.key_source.poll()
                if "q" in keys or "x" in keys:
                    print("RECORDER_EXIT", flush=True)
                    return 0
                if "s" in keys:
                    result = self.collect_one_episode()
                    if result.exit_requested:
                        return 1 if result.abort_reason else 0
                    print("RECORDER_IDLE: S=start another episode, Q=quit", flush=True)
                now_s = self.clock()
                if now_s >= next_camera_check_s:
                    self.cameras.get_latest_pair(
                        max_age_s=MAX_FRAME_AGE_S,
                        max_skew_s=MAX_CAMERA_SKEW_S,
                    )
                    next_camera_check_s = now_s + 0.5
                self.sleep(0.05)
        except KeyboardInterrupt:
            print("RECORDER_STOPPED_BY_OPERATOR", flush=True)
            return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--front-camera", type=int, required=True)
    parser.add_argument("--wrist-camera", type=int, required=True)
    parser.add_argument("--target-id", type=int, choices=sorted(TASKS), required=True)
    parser.add_argument("--box-id", choices=("A", "B"), required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--operator", default=os.environ.get("USERNAME", "unknown"))
    parser.add_argument("--object-inventory-version", default="v1")
    parser.add_argument("--speed-deg-s", type=float, default=DEFAULT_SPEED_DEG_S)
    parser.add_argument("--max-episode-s", type=float, default=DEFAULT_MAX_EPISODE_S)
    parser.add_argument("--motion-margin-deg", type=float, default=DEFAULT_MOTION_MARGIN_DEG)
    parser.add_argument(
        "--large-auto-align-speed-deg-s",
        type=float,
        default=DEFAULT_LARGE_AUTO_ALIGN_SPEED_DEG_S,
        help=(
            "Speed cap used when any automatic-alignment joint delta exceeds 10 degrees "
            f"(allowed {MIN_LARGE_AUTO_ALIGN_SPEED_DEG_S:g}-"
            f"{MAX_LARGE_AUTO_ALIGN_SPEED_DEG_S:g} deg/s)"
        ),
    )
    parser.add_argument(
        "--max-auto-align-deg",
        type=float,
        default=DEFAULT_MAX_AUTO_ALIGN_DEG,
        help=(
            "Catch-up limit for shoulder_pan, shoulder_lift, elbow_flex, and wrist_flex only; "
            "wrist_roll and gripper remain range-, slew-, tracking-, and temperature-limited"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Required: {CONFIRMATION}")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.front_camera == args.wrist_camera:
        raise SystemExit("--front-camera and --wrist-camera must be different")
    if not 1.0 <= args.speed_deg_s <= MAX_ALLOWED_SPEED_DEG_S:
        raise SystemExit(f"--speed-deg-s must be between 1 and {MAX_ALLOWED_SPEED_DEG_S:g}")
    if not 5.0 <= args.max_episode_s <= DEFAULT_MAX_EPISODE_S:
        raise SystemExit(f"--max-episode-s must be between 5 and {DEFAULT_MAX_EPISODE_S:g} seconds")
    if not MIN_MOTION_MARGIN_DEG <= args.motion_margin_deg <= MAX_MOTION_MARGIN_DEG:
        raise SystemExit(
            f"--motion-margin-deg must be between {MIN_MOTION_MARGIN_DEG:g} "
            f"and {MAX_MOTION_MARGIN_DEG:g} degrees"
        )
    if not (
        MIN_LARGE_AUTO_ALIGN_SPEED_DEG_S
        <= args.large_auto_align_speed_deg_s
        <= MAX_LARGE_AUTO_ALIGN_SPEED_DEG_S
    ):
        raise SystemExit(
            "--large-auto-align-speed-deg-s must be between "
            f"{MIN_LARGE_AUTO_ALIGN_SPEED_DEG_S:g} and "
            f"{MAX_LARGE_AUTO_ALIGN_SPEED_DEG_S:g}"
        )
    if not MIN_MAX_AUTO_ALIGN_DEG <= args.max_auto_align_deg <= HARD_MAX_AUTO_ALIGN_DEG:
        raise SystemExit(
            f"--max-auto-align-deg must be between {MIN_MAX_AUTO_ALIGN_DEG:g} "
            f"and {HARD_MAX_AUTO_ALIGN_DEG:g} degrees"
        )
    if args.max_auto_align_deg < args.motion_margin_deg:
        raise SystemExit(
            "--max-auto-align-deg must be at least --motion-margin-deg so an endpoint startup "
            "can reach the motion interior"
        )
    task = args.task or TASKS[args.target_id]
    config = RecorderConfig(
        dataset_root=args.dataset_root,
        target_id=args.target_id,
        box_id=args.box_id,
        task=task,
        operator=args.operator,
        object_inventory_version=args.object_inventory_version,
        front_camera=args.front_camera,
        wrist_camera=args.wrist_camera,
        speed_deg_s=args.speed_deg_s,
        max_episode_s=args.max_episode_s,
        motion_margin_deg=args.motion_margin_deg,
        max_auto_align_deg=args.max_auto_align_deg,
        large_auto_align_speed_deg_s=args.large_auto_align_speed_deg_s,
    )

    lock_path = REAL_DIR / "calibration" / "calibration_lock.json"
    if no_jump.verify(lock_path, REPO_ROOT) != 0:
        return 2
    print("CALIBRATION_GATE_PASS", flush=True)
    import cv2
    import h5py
    import numpy as np

    print(
        f"DEPENDENCY_GATE_PASS numpy={np.__version__} cv2={cv2.__version__} h5py={h5py.__version__}",
        flush=True,
    )
    print(
        f"front_camera={config.front_camera} wrist_camera={config.wrist_camera} "
        f"target_id={config.target_id} box_id={config.box_id} "
        f"speed_deg_s={config.speed_deg_s:.1f} max_episode_s={config.max_episode_s:.1f} "
        f"motion_margin_deg={config.motion_margin_deg:.1f} "
        f"max_auto_align_deg={config.max_auto_align_deg:.1f} "
        f"large_auto_align_speed_deg_s={config.large_auto_align_speed_deg_s:.1f} "
        f"dataset_root={config.dataset_root.resolve()}",
        flush=True,
    )
    if not args.execute or args.confirm != CONFIRMATION:
        print(
            "DRY_RUN_ONLY: cameras and serial ports were not opened. To collect, pass "
            f"--execute --confirm {CONFIRMATION}"
        )
        return 0

    config.dataset_root.mkdir(parents=True, exist_ok=True)
    calibrations = _load_calibrations()
    cameras = camera_capture.DualCameraCapture(
        front_index=config.front_camera,
        wrist_index=config.wrist_camera,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
    )
    try:
        cameras.start()
        session_stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        startup_paths = cameras.save_startup_frames(config.dataset_root / "session_info" / session_stamp)
        print(
            f"CAMERA_GATE_PASS front_startup={startup_paths[0]} wrist_startup={startup_paths[1]}",
            flush=True,
        )
        key_source = WindowsKeySource()
        recorder = RealTeleopRecorder(
            config=config,
            cameras=cameras,
            calibrations=calibrations,
            key_source=key_source,
            bus_factory=lambda: _build_buses(calibrations),
        )
        return recorder.run()
    finally:
        cameras.close()
        print("CAMERAS_CLOSED", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
