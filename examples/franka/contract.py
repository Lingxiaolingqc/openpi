"""Versioned public contract and conservative FR3 application limits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

INTERFACE_VERSION = 1
ACTION_DIM = 8
ACTION_HORIZON = 16
CONTROL_HZ = 20.0
JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
STATE_LAYOUT = (*JOINT_NAMES, "gripper_width_m")
ACTION_LAYOUT = STATE_LAYOUT

# Franka's documented suggested rectangular position/velocity limits for FR3. The lower-level
# controller must additionally enforce the position-dependent limits returned by libfranka.
Q_MIN_RAD = np.array([-2.3476, -1.5454, -2.4937, -2.7714, -2.5100, 0.7773, -2.7045], dtype=np.float64)
Q_MAX_RAD = np.array([2.3476, 1.5454, 2.4937, -0.4226, 2.5100, 4.2841, 2.7045], dtype=np.float64)
QDOT_RECT_MAX_RAD_S = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0], dtype=np.float64)
DEFAULT_MAX_GRIPPER_WIDTH_M = 0.08


class ContractError(ValueError):
    """Raised when data violates the Franka v1 interface contract."""


class RuntimeMode(str, Enum):
    OBSERVE = "observe"
    SHADOW = "shadow"
    EXECUTE = "execute"


@dataclass(frozen=True)
class SafetyConfig:
    control_hz: float = CONTROL_HZ
    action_horizon: int = ACTION_HORIZON
    speed_scale: float = 0.1
    max_gripper_width_m: float = DEFAULT_MAX_GRIPPER_WIDTH_M
    max_gripper_speed_m_s: float = 0.05
    max_image_skew_s: float = 0.05
    max_sensor_age_s: float = 0.25
    max_response_age_s: float = 1.5
    action_chunk_ttl_s: float = 1.0

    def __post_init__(self) -> None:
        finite_positive = {
            "control_hz": self.control_hz,
            "max_gripper_width_m": self.max_gripper_width_m,
            "max_gripper_speed_m_s": self.max_gripper_speed_m_s,
            "max_image_skew_s": self.max_image_skew_s,
            "max_sensor_age_s": self.max_sensor_age_s,
            "max_response_age_s": self.max_response_age_s,
            "action_chunk_ttl_s": self.action_chunk_ttl_s,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"{name} must be finite and positive")
        if self.action_horizon < 1:
            raise ContractError("action_horizon must be positive")
        if not math.isfinite(self.speed_scale) or not 0.0 < self.speed_scale <= 1.0:
            raise ContractError("speed_scale must be in (0, 1]")


@dataclass(frozen=True)
class FrankaSnapshot:
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    gripper_width_m: float
    base_rgb: np.ndarray
    captured_monotonic_ns: int
    base_image_monotonic_ns: int
    wrist_rgb: np.ndarray | None = None
    wrist_image_monotonic_ns: int | None = None

    def policy_observation(self, prompt: str) -> dict[str, Any]:
        observation: dict[str, Any] = {
            "images/base": np.asarray(self.base_rgb),
            "state": np.concatenate(
                [np.asarray(self.q_rad, dtype=np.float32), np.asarray([self.gripper_width_m], dtype=np.float32)]
            ),
            "prompt": prompt,
        }
        if self.wrist_rgb is not None:
            observation["images/wrist"] = np.asarray(self.wrist_rgb)
        return observation


@dataclass(frozen=True)
class FrankaTarget:
    q_target_rad: np.ndarray
    gripper_width_m: float


def _validate_image(image: np.ndarray, name: str) -> None:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ContractError(f"{name} must have HWC RGB shape, got {image.shape}")
    if image.dtype != np.uint8:
        raise ContractError(f"{name} must be uint8, got {image.dtype}")


def validate_snapshot(
    snapshot: FrankaSnapshot,
    config: SafetyConfig,
    *,
    now_monotonic_ns: int | None = None,
) -> None:
    q = np.asarray(snapshot.q_rad, dtype=np.float64)
    dq = np.asarray(snapshot.dq_rad_s, dtype=np.float64)
    if q.shape != (7,) or dq.shape != (7,):
        raise ContractError(f"FR3 q and dq must both have shape (7,), got {q.shape} and {dq.shape}")
    if not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ContractError("FR3 q and dq must be finite")
    if np.any(q < Q_MIN_RAD) or np.any(q > Q_MAX_RAD):
        raise ContractError("FR3 measured joints are outside the configured application position envelope")
    if not math.isfinite(snapshot.gripper_width_m) or not 0.0 <= snapshot.gripper_width_m <= config.max_gripper_width_m:
        raise ContractError("Franka Hand width is outside the configured range")

    _validate_image(snapshot.base_rgb, "base_rgb")
    timestamps = (snapshot.captured_monotonic_ns, snapshot.base_image_monotonic_ns)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in timestamps):
        raise ContractError("snapshot and base image timestamps must be positive monotonic nanoseconds")
    if snapshot.wrist_rgb is None:
        if snapshot.wrist_image_monotonic_ns is not None:
            raise ContractError("wrist timestamp was supplied without a wrist image")
    else:
        _validate_image(snapshot.wrist_rgb, "wrist_rgb")
        if snapshot.wrist_image_monotonic_ns is None or snapshot.wrist_image_monotonic_ns <= 0:
            raise ContractError("wrist image requires a positive monotonic timestamp")

    sample_times = [snapshot.captured_monotonic_ns, snapshot.base_image_monotonic_ns]
    if snapshot.wrist_image_monotonic_ns is not None:
        sample_times.append(snapshot.wrist_image_monotonic_ns)
    max_skew_ns = int(config.max_image_skew_s * 1_000_000_000)
    if max(sample_times) - min(sample_times) > max_skew_ns:
        raise ContractError("Franka state/camera timestamps exceed the configured synchronization skew")

    now_monotonic_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
    oldest_required_sample_ns = min(sample_times)
    newest_required_sample_ns = max(sample_times)
    age_ns = now_monotonic_ns - oldest_required_sample_ns
    if newest_required_sample_ns > now_monotonic_ns or age_ns > int(config.max_sensor_age_s * 1_000_000_000):
        raise ContractError("Franka observation is stale or has a future timestamp")


def validate_target(target: FrankaTarget, config: SafetyConfig) -> None:
    q_target = np.asarray(target.q_target_rad, dtype=np.float64)
    if q_target.shape != (7,) or not np.isfinite(q_target).all():
        raise ContractError("FR3 joint target must be a finite array with shape (7,)")
    if np.any(q_target < Q_MIN_RAD) or np.any(q_target > Q_MAX_RAD):
        raise ContractError("FR3 joint target is outside the application position envelope")
    if not math.isfinite(target.gripper_width_m) or not 0.0 <= target.gripper_width_m <= config.max_gripper_width_m:
        raise ContractError("Franka Hand target width is outside the configured range")


def expected_server_metadata() -> dict[str, Any]:
    return {
        "interface_version": INTERFACE_VERSION,
        "policy_config": "pi05_franka_lora",
        "robot_type": "fr3",
        "end_effector": "franka_hand",
        "control_hz": int(CONTROL_HZ),
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "state_layout": list(STATE_LAYOUT),
        "action_layout": list(ACTION_LAYOUT),
        "action_semantics": "absolute_joint_position_and_gripper_width",
        "camera_roles": {"base": "required", "wrist": "optional"},
    }


def validate_server_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise ContractError("policy server metadata must be a mapping")
    expected = expected_server_metadata()
    mismatches = {key: (value, metadata.get(key)) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ContractError(f"policy server does not implement the Franka v1 contract: {mismatches}")
    protocol_versions = metadata.get("openpi_protocol_versions")
    if protocol_versions != [1]:
        raise ContractError("Franka execution requires OpenPI WebSocket protocol version 1")


def metadata_digest(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_real_robot_approval(
    approval_path: Path,
    server_metadata: dict[str, Any],
    *,
    requested_speed_scale: float,
) -> dict[str, Any]:
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"unable to read real-robot approval {approval_path}: {exc}") from exc
    required = {
        "schema_version": 1,
        "robot_type": "fr3",
        "real_robot_deployment_allowed": True,
        "approved_server_metadata_sha256": metadata_digest(server_metadata),
    }
    mismatches = {key: (value, approval.get(key)) for key, value in required.items() if approval.get(key) != value}
    if mismatches:
        raise ContractError(f"real-robot approval does not match this policy server: {mismatches}")
    maximum = approval.get("max_speed_scale")
    if not isinstance(maximum, int | float) or isinstance(maximum, bool) or not 0.0 < maximum <= 0.1:
        raise ContractError("approval max_speed_scale must be in (0, 0.1]")
    if requested_speed_scale > float(maximum):
        raise ContractError(f"requested speed scale {requested_speed_scale} exceeds approved maximum {maximum}")
    if not isinstance(approval.get("approval_id"), str) or not approval["approval_id"]:
        raise ContractError("approval_id must be a non-empty string")
    return approval


def training_steps_for_passes(train_frames: int, *, batch_size: int = 8, passes: float = 3.0) -> int:
    if train_frames < 1 or batch_size < 1 or not math.isfinite(passes) or passes <= 0.0:
        raise ContractError("train_frames, batch_size, and passes must be positive")
    return math.ceil(passes * train_frames / batch_size)
