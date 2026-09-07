"""Append-only per-episode HDF5 recorder for synchronized FR3 demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import string
from typing import Any

import h5py
import numpy as np

from examples.franka import contract


@dataclass(frozen=True)
class EpisodeMetadata:
    task: str
    camera_calibration_sha256: dict[str, str]
    robot_system_version: str
    software_versions: dict[str, str]
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise contract.ContractError("Franka episode task must be a non-empty string")
        if not isinstance(self.camera_calibration_sha256, dict):
            raise contract.ContractError("camera calibration hashes must be a mapping")
        camera_names = set(self.camera_calibration_sha256)
        if "base" not in camera_names or not camera_names <= {"base", "wrist"}:
            raise contract.ContractError("camera calibration hashes must contain base and may contain wrist")
        for camera_name, digest in self.camera_calibration_sha256.items():
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in string.hexdigits for char in digest):
                raise contract.ContractError(f"{camera_name} camera calibration must be a SHA-256 hex digest")
        if not isinstance(self.robot_system_version, str) or not self.robot_system_version.strip():
            raise contract.ContractError("robot_system_version must be a non-empty string")
        required_versions = {"franka_ros2", "libfranka"}
        if (
            not isinstance(self.software_versions, dict)
            or not required_versions <= self.software_versions.keys()
            or any(not isinstance(value, str) or not value.strip() for value in self.software_versions.values())
        ):
            raise contract.ContractError("software_versions must contain non-empty franka_ros2 and libfranka versions")
        if not isinstance(self.extra, dict):
            raise contract.ContractError("extra Franka episode metadata must be a mapping")
        try:
            json.dumps(self.extra)
        except (TypeError, ValueError) as exc:
            raise contract.ContractError("extra Franka episode metadata must be JSON serializable") from exc


class FrankaEpisodeRecorder:
    """Write one immutable pre-step observation/absolute-target episode."""

    def __init__(
        self,
        output_path: Path,
        metadata: EpisodeMetadata,
        *,
        safety_config: contract.SafetyConfig | None = None,
    ) -> None:
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite Franka episode: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = safety_config or contract.SafetyConfig()
        self._file = h5py.File(output_path, "x")
        self._closed = False
        self._sample_count = 0
        self._has_wrist: bool | None = None
        self._calibration_names = set(metadata.camera_calibration_sha256)
        self._datasets: dict[str, h5py.Dataset] = {}

        attrs = self._file.attrs
        attrs["schema_version"] = contract.INTERFACE_VERSION
        attrs["complete"] = False
        attrs["success"] = False
        attrs["observation_alignment"] = "pre_step"
        attrs["robot_type"] = "fr3"
        attrs["end_effector"] = "franka_hand"
        attrs["joint_names_json"] = json.dumps(contract.JOINT_NAMES)
        attrs["state_units_json"] = json.dumps(["rad"] * 7 + ["m"])
        attrs["action_units_json"] = json.dumps(["rad"] * 7 + ["m"])
        attrs["control_hz"] = self._config.control_hz
        attrs["task"] = metadata.task
        attrs["camera_calibration_sha256_json"] = json.dumps(metadata.camera_calibration_sha256, sort_keys=True)
        attrs["robot_system_version"] = metadata.robot_system_version
        attrs["software_versions_json"] = json.dumps(metadata.software_versions, sort_keys=True)
        attrs["extra_metadata_json"] = json.dumps(metadata.extra, sort_keys=True)
        attrs["num_samples"] = 0

    @staticmethod
    def _resizable_dataset(
        group: h5py.Group,
        name: str,
        sample: np.ndarray,
        *,
        dtype: np.dtype | type,
    ) -> h5py.Dataset:
        sample = np.asarray(sample)
        return group.create_dataset(
            name,
            shape=(0, *sample.shape),
            maxshape=(None, *sample.shape),
            chunks=(1, *sample.shape),
            dtype=dtype,
        )

    def _initialize(self, snapshot: contract.FrankaSnapshot) -> None:
        self._has_wrist = snapshot.wrist_rgb is not None
        expected_calibration_names = {"base", *({"wrist"} if self._has_wrist else set())}
        if self._calibration_names != expected_calibration_names:
            raise contract.ContractError(
                "camera calibration metadata does not match the cameras present in the Franka episode"
            )
        observations = self._file.create_group("observations")
        actions = self._file.create_group("actions")
        images = observations.create_group("images")
        image_timestamps = observations.create_group("image_timestamps_ns")
        self._datasets = {
            "q": self._resizable_dataset(observations, "q", snapshot.q_rad, dtype=np.float32),
            "dq": self._resizable_dataset(observations, "dq", snapshot.dq_rad_s, dtype=np.float32),
            "gripper": self._resizable_dataset(
                observations, "gripper_width_m", np.asarray(snapshot.gripper_width_m), dtype=np.float32
            ),
            "timestamp": self._resizable_dataset(
                observations, "timestamps_monotonic_ns", np.asarray(snapshot.captured_monotonic_ns), dtype=np.int64
            ),
            "base": self._resizable_dataset(images, "base", snapshot.base_rgb, dtype=np.uint8),
            "base_timestamp": self._resizable_dataset(
                image_timestamps,
                "base",
                np.asarray(snapshot.base_image_monotonic_ns),
                dtype=np.int64,
            ),
            "q_target": self._resizable_dataset(actions, "q_target", snapshot.q_rad, dtype=np.float32),
            "gripper_target": self._resizable_dataset(
                actions, "gripper_width_m", np.asarray(snapshot.gripper_width_m), dtype=np.float32
            ),
        }
        if snapshot.wrist_rgb is not None:
            assert snapshot.wrist_image_monotonic_ns is not None
            self._datasets["wrist"] = self._resizable_dataset(images, "wrist", snapshot.wrist_rgb, dtype=np.uint8)
            self._datasets["wrist_timestamp"] = self._resizable_dataset(
                image_timestamps,
                "wrist",
                np.asarray(snapshot.wrist_image_monotonic_ns),
                dtype=np.int64,
            )
        self._file.attrs["has_wrist_camera"] = self._has_wrist

    @staticmethod
    def _append_dataset(dataset: h5py.Dataset, value: object, index: int) -> None:
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    def append(self, snapshot: contract.FrankaSnapshot, target: contract.FrankaTarget) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a finalized Franka episode")
        contract.validate_snapshot(snapshot, self._config)
        contract.validate_target(target, self._config)
        if not self._datasets:
            self._initialize(snapshot)
        if (snapshot.wrist_rgb is not None) is not self._has_wrist:
            raise contract.ContractError("wrist camera presence changed within a Franka episode")

        values: dict[str, object] = {
            "q": np.asarray(snapshot.q_rad, dtype=np.float32),
            "dq": np.asarray(snapshot.dq_rad_s, dtype=np.float32),
            "gripper": np.float32(snapshot.gripper_width_m),
            "timestamp": np.int64(snapshot.captured_monotonic_ns),
            "base": snapshot.base_rgb,
            "base_timestamp": np.int64(snapshot.base_image_monotonic_ns),
            "q_target": np.asarray(target.q_target_rad, dtype=np.float32),
            "gripper_target": np.float32(target.gripper_width_m),
        }
        if snapshot.wrist_rgb is not None:
            assert snapshot.wrist_image_monotonic_ns is not None
            values["wrist"] = snapshot.wrist_rgb
            values["wrist_timestamp"] = np.int64(snapshot.wrist_image_monotonic_ns)
        for name, value in values.items():
            self._append_dataset(self._datasets[name], value, self._sample_count)
        self._sample_count += 1
        self._file.attrs["num_samples"] = self._sample_count
        self._file.flush()

    def finalize(self, *, success: bool) -> None:
        if self._closed:
            raise RuntimeError("Franka episode was already finalized")
        if self._sample_count < 1:
            raise RuntimeError("refusing to finalize an empty Franka episode")
        self._file.attrs["success"] = success
        self._file.attrs["complete"] = True
        self._file.flush()
        self._file.close()
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._file.flush()
            self._file.close()
            self._closed = True

    def __enter__(self) -> FrankaEpisodeRecorder:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            self.abort()
