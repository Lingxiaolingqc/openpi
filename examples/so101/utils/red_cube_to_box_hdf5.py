"""Crash-resumable sharded HDF5 storage for SO-101 expert trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import uuid

import h5py
import numpy as np

SCHEMA_VERSION = "so101-red-cube-v1"
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class RecoveryReport:
    removed_staging_groups: int
    successful_episodes: int
    attempts: int


class EpisodeBuffer:
    """One streamed staging episode that is either promoted or discarded."""

    def __init__(self, owner: ShardedDatasetWriter, h5_file: h5py.File, group: h5py.Group, attempt_id: int):
        self._owner = owner
        self._h5_file = h5_file
        self._group = group
        self.attempt_id = attempt_id
        self._closed = False
        self._datasets: dict[str, h5py.Dataset] = {}

    @property
    def num_samples(self) -> int:
        if not self._datasets:
            return 0
        return int(next(iter(self._datasets.values())).shape[0])

    @property
    def is_closed(self) -> bool:
        return self._closed

    def append(self, sample: dict[str, np.ndarray | float]) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed episode")
        required = {"actions", "obs/joint_pos", "obs/front", "timestamps"}
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"Sample is missing required fields: {sorted(missing)}")
        if self._datasets and sample.keys() != self._datasets.keys():
            raise ValueError("Every sample in an episode must contain the same fields")

        for path, raw_value in sample.items():
            value = np.asarray(raw_value)
            if path == "timestamps":
                value = np.asarray(value, dtype=np.float64)
            elif path.startswith("obs/") and path.split("/", 1)[1] in {"front", "wrist"}:
                value = np.asarray(value, dtype=np.uint8)
            else:
                value = np.asarray(value, dtype=np.float32)
            if path not in self._datasets:
                parent_path, name = path.rsplit("/", 1) if "/" in path else ("", path)
                parent = self._group.require_group(parent_path) if parent_path else self._group
                chunks = (1, *value.shape)
                compression = "lzf" if value.dtype == np.uint8 else None
                self._datasets[path] = parent.create_dataset(
                    name,
                    shape=(0, *value.shape),
                    maxshape=(None, *value.shape),
                    chunks=chunks,
                    dtype=value.dtype,
                    compression=compression,
                )
            dataset = self._datasets[path]
            if dataset.shape[1:] != value.shape:
                raise ValueError(f"Field {path} changed shape from {dataset.shape[1:]} to {value.shape}")
            index = dataset.shape[0]
            dataset.resize(index + 1, axis=0)
            dataset[index] = value
        self._h5_file.flush()

    def finish(self, *, success: bool, summary: dict[str, Any]) -> str | None:
        if self._closed:
            raise RuntimeError("Episode was already finalized")
        self._closed = True
        return self._owner._finish_episode(self, success=success, summary=summary)  # noqa: SLF001


class ShardedDatasetWriter:
    """Write successful trajectories in shards and retain all attempt summaries."""

    def __init__(
        self,
        root: Path,
        *,
        shard_size: int = 50,
        metadata: dict[str, Any] | None = None,
        resume: bool = True,
    ) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.manifest_path = self.root / "collection_manifest.json"
        self.metadata = dict(metadata or {})
        self._active: EpisodeBuffer | None = None
        if self.manifest_path.exists() and not resume:
            raise FileExistsError(f"Dataset already exists: {self.root}")
        self.recovery = self._recover()
        self._write_manifest()

    @property
    def successful_episodes(self) -> int:
        return self.recovery.successful_episodes

    @property
    def attempts(self) -> int:
        return self.recovery.attempts

    def _shard_paths(self) -> list[Path]:
        return sorted(self.root.glob("part-*.hdf5"))

    def _recover(self) -> RecoveryReport:
        removed = 0
        successes = 0
        attempts = 0
        for path in self._shard_paths():
            with h5py.File(path, "a") as h5_file:
                staging = h5_file.require_group("_staging")
                for name in list(staging):
                    del staging[name]
                    removed += 1
                data = h5_file.require_group("data")
                attempt_group = h5_file.require_group("attempts")
                successes += sum(name.startswith("demo_") for name in data)
                attempts += sum(name.startswith("attempt_") for name in attempt_group)
                h5_file.flush()
        return RecoveryReport(removed, successes, attempts)

    def _shard_path(self) -> Path:
        return self.root / f"part-{self.successful_episodes // self.shard_size:03d}.hdf5"

    def begin_attempt(self, metadata: dict[str, Any] | None = None) -> EpisodeBuffer:
        if self._active is not None:
            raise RuntimeError("An attempt is already active")
        path = self._shard_path()
        h5_file = h5py.File(path, "a")
        if "schema_version" not in h5_file.attrs:
            h5_file.attrs["schema_version"] = SCHEMA_VERSION
        if "joint_names" not in h5_file.attrs:
            h5_file.attrs["joint_names"] = _json_value(JOINT_NAMES)
        h5_file.require_group("data")
        h5_file.require_group("attempts")
        staging = h5_file.require_group("_staging")
        group = staging.create_group(uuid.uuid4().hex)
        attempt_id = self.attempts
        group.attrs["attempt_id"] = attempt_id
        group.attrs["started_at"] = _utc_now()
        for key, value in (metadata or {}).items():
            group.attrs[key] = _json_value(value)
        h5_file.flush()
        self._active = EpisodeBuffer(self, h5_file, group, attempt_id)
        return self._active

    def _finish_episode(self, episode: EpisodeBuffer, *, success: bool, summary: dict[str, Any]) -> str | None:
        if episode is not self._active:
            raise RuntimeError("Episode does not belong to this active writer")
        h5_file = episode._h5_file  # noqa: SLF001
        staging_path = episode._group.name  # noqa: SLF001
        attempt_name = f"attempt_{episode.attempt_id}"
        attempt = h5_file["attempts"].create_group(attempt_name)
        attempt.attrs["success"] = bool(success)
        attempt.attrs["num_samples"] = episode.num_samples
        attempt.attrs["finished_at"] = _utc_now()
        attempt.attrs["summary_json"] = _json_value(summary)
        demo_name: str | None = None
        if success:
            demo_name = f"demo_{self.successful_episodes}"
            target = f"/data/{demo_name}"
            h5_file.move(staging_path, target)
            demo = h5_file[target]
            demo.attrs["success"] = True
            demo.attrs["num_samples"] = episode.num_samples
            demo.attrs["attempt_id"] = episode.attempt_id
            demo.attrs["summary_json"] = _json_value(summary)
        else:
            del h5_file[staging_path]
        h5_file.flush()
        h5_file.close()
        self.recovery = RecoveryReport(
            self.recovery.removed_staging_groups,
            self.successful_episodes + int(success),
            self.attempts + 1,
        )
        self._active = None
        self._write_manifest()
        return demo_name

    def _write_manifest(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "joint_names": list(JOINT_NAMES),
            "shard_size": self.shard_size,
            "successful_episodes": self.successful_episodes,
            "attempts": self.attempts,
            "recovered_staging_groups": self.recovery.removed_staging_groups,
            "shards": [path.name for path in self._shard_paths()],
            "updated_at": _utc_now(),
            "metadata": self.metadata,
        }
        _atomic_json(self.manifest_path, payload)
