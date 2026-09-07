"""Append-only HDF5 recording for PiPER MuJoCo expert trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from examples.piper import contract
from examples.piper.expert import Phase


@dataclass(frozen=True)
class EpisodeMetadata:
    seed: int
    model_sha256: str
    menagerie_revision: str
    task: str = contract.TASK_PROMPT
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise contract.ContractError("seed must be non-negative")
        if len(self.model_sha256) != 64:
            raise contract.ContractError("model_sha256 must be a SHA-256 hex digest")
        if not self.menagerie_revision.strip():
            raise contract.ContractError("menagerie_revision must not be empty")
        if not self.task.strip():
            raise contract.ContractError("task must not be empty")
        try:
            json.dumps(self.extra or {})
        except (TypeError, ValueError) as exc:
            raise contract.ContractError("extra metadata must be JSON serializable") from exc


class PiperEpisodeRecorder:
    """Write immutable pre-step observation/action pairs, one file per episode."""

    def __init__(self, output_path: Path, metadata: EpisodeMetadata) -> None:
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite PiPER episode: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(output_path, "x")
        self._closed = False
        self._count = 0
        attrs = self._file.attrs
        attrs["schema_version"] = contract.INTERFACE_VERSION
        attrs["complete"] = False
        attrs["success"] = False
        attrs["robot_type"] = contract.ROBOT_TYPE
        attrs["end_effector"] = contract.END_EFFECTOR
        attrs["observation_alignment"] = "pre_step"
        attrs["state_layout_json"] = json.dumps(contract.STATE_LAYOUT)
        attrs["action_layout_json"] = json.dumps(contract.ACTION_LAYOUT)
        attrs["state_units_json"] = json.dumps(["rad"] * 6 + ["unit_interval"])
        attrs["action_units_json"] = json.dumps(["rad"] * 6 + ["unit_interval"])
        attrs["physics_hz"] = contract.PHYSICS_HZ
        attrs["control_hz"] = contract.CONTROL_HZ
        attrs["task"] = metadata.task
        attrs["seed"] = metadata.seed
        attrs["model_sha256"] = metadata.model_sha256
        attrs["menagerie_revision"] = metadata.menagerie_revision
        attrs["extra_metadata_json"] = json.dumps(metadata.extra or {}, sort_keys=True)
        attrs["num_samples"] = 0
        self._observations = self._file.create_group("observations")
        self._actions = self._file.create_group("actions")
        self._state = None
        self._image = None
        self._timestamp = None
        self._action = None
        self._phase = None

    def _initialize(self, observation: contract.Observation) -> None:
        if observation.image.shape != (contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3):
            raise contract.ContractError(f"unexpected image shape {observation.image.shape}")
        if observation.image.dtype != np.uint8:
            raise contract.ContractError(f"image must be uint8, got {observation.image.dtype}")
        self._state = self._observations.create_dataset(
            "state", shape=(0, contract.ACTION_DIM), maxshape=(None, contract.ACTION_DIM), dtype=np.float32
        )
        self._image = self._observations.create_dataset(
            "image",
            shape=(0, contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3),
            maxshape=(None, contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3),
            chunks=(1, contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3),
            compression="gzip",
            compression_opts=1,
            dtype=np.uint8,
        )
        self._timestamp = self._observations.create_dataset(
            "timestamp_ns", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        self._action = self._actions.create_dataset(
            "absolute_target", shape=(0, contract.ACTION_DIM), maxshape=(None, contract.ACTION_DIM), dtype=np.float32
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._phase = self._actions.create_dataset("phase", shape=(0,), maxshape=(None,), dtype=string_dtype)

    @staticmethod
    def _append(dataset: h5py.Dataset, value, index: int) -> None:
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    def append(self, observation: contract.Observation, action: np.ndarray, phase: Phase | str) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a finalized PiPER episode")
        state = contract.validate_state(observation.state)
        action = contract.validate_action(action)
        if not np.isfinite(observation.timestamp_ns):
            raise contract.ContractError("timestamp must be finite")
        if self._state is None:
            self._initialize(observation)
        assert self._state is not None
        assert self._image is not None
        assert self._timestamp is not None
        assert self._action is not None
        assert self._phase is not None
        values = (
            (self._state, state),
            (self._image, observation.image),
            (self._timestamp, np.int64(observation.timestamp_ns)),
            (self._action, action),
            (self._phase, phase.value if isinstance(phase, Phase) else str(phase)),
        )
        for dataset, value in values:
            self._append(dataset, value, self._count)
        self._count += 1
        self._file.attrs["num_samples"] = self._count
        self._file.flush()

    def finalize(self, *, success: bool, final_phase: str, failure_reason: str | None) -> None:
        if self._closed:
            raise RuntimeError("PiPER episode was already finalized")
        if self._count < 1:
            raise RuntimeError("refusing to finalize an empty PiPER episode")
        self._file.attrs["success"] = bool(success)
        self._file.attrs["final_phase"] = final_phase
        self._file.attrs["failure_reason"] = failure_reason or ""
        self._file.attrs["complete"] = True
        self._file.flush()
        self._file.close()
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._file.flush()
            self._file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._closed:
            self.abort()
