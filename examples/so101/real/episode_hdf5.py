"""Crash-isolated, incremental HDF5 writer for real SO-101 episodes."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import queue
import threading
from typing import Any

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclasses.dataclass(frozen=True)
class EpisodeSample:
    joint_pos: Any
    front_rgb: Any
    wrist_rgb: Any
    action: Any
    control_timestamp_s: float
    front_timestamp_s: float
    wrist_timestamp_s: float


@dataclasses.dataclass(frozen=True)
class _Finalize:
    success: bool
    abort_reason: str
    final_metadata: dict[str, Any]


class IncrementalEpisodeWriter:
    """Write one episode on a background thread and atomically publish it."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        episode_id: str,
        metadata: dict[str, Any],
        image_shape: tuple[int, int, int] = (480, 640, 3),
        queue_size: int = 90,
        flush_every: int = 30,
        compression: str | None = "lzf",
    ) -> None:
        if not episode_id or any(char in episode_id for char in '<>:"/\\|?*'):
            raise ValueError(f"invalid Windows episode id: {episode_id!r}")
        self.dataset_root = Path(dataset_root).resolve()
        self.episode_id = episode_id
        self.metadata = metadata.copy()
        self.image_shape = image_shape
        self.flush_every = flush_every
        self.compression = compression
        self.staging_dir = self.dataset_root / "staging"
        self.episodes_dir = self.dataset_root / "episodes"
        self.rejected_dir = self.dataset_root / "rejected"
        for directory in (self.staging_dir, self.episodes_dir, self.rejected_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.staging_dir / f"{episode_id}.partial.h5"
        self.success_path = self.episodes_dir / f"{episode_id}.h5"
        self.rejected_path = self.rejected_dir / f"{episode_id}.h5"
        for path in (self.partial_path, self.success_path, self.rejected_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite episode path: {path}")

        self._queue: queue.Queue[EpisodeSample | _Finalize] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=f"so101-hdf5-{episode_id}",
            daemon=True,
        )
        self._failure: BaseException | None = None
        self._finished = False
        self._frames_enqueued = 0
        self._frames_written = 0
        self._thread.start()

    @staticmethod
    def _attribute_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, str | bytes | int | float | bool):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _create_datasets(self, file: Any) -> dict[str, Any]:
        height, width, channels = self.image_shape
        obs = file.create_group("obs")
        timestamps = file.create_group("timestamps")
        return {
            "joint_pos": obs.create_dataset(
                "joint_pos",
                shape=(0, 6),
                maxshape=(None, 6),
                chunks=(256, 6),
                dtype="float32",
            ),
            "front": obs.create_dataset(
                "front",
                shape=(0, height, width, channels),
                maxshape=(None, height, width, channels),
                chunks=(1, height, width, channels),
                dtype="uint8",
                compression=self.compression,
            ),
            "wrist": obs.create_dataset(
                "wrist",
                shape=(0, height, width, channels),
                maxshape=(None, height, width, channels),
                chunks=(1, height, width, channels),
                dtype="uint8",
                compression=self.compression,
            ),
            "actions": file.create_dataset(
                "actions",
                shape=(0, 6),
                maxshape=(None, 6),
                chunks=(256, 6),
                dtype="float32",
            ),
            "control": timestamps.create_dataset(
                "control",
                shape=(0,),
                maxshape=(None,),
                chunks=(1024,),
                dtype="float64",
            ),
            "front_timestamp": timestamps.create_dataset(
                "front",
                shape=(0,),
                maxshape=(None,),
                chunks=(1024,),
                dtype="float64",
            ),
            "wrist_timestamp": timestamps.create_dataset(
                "wrist",
                shape=(0,),
                maxshape=(None,),
                chunks=(1024,),
                dtype="float64",
            ),
        }

    @staticmethod
    def _append_dataset(dataset: Any, value: Any, index: int) -> None:
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    def _writer_loop(self) -> None:
        try:
            import h5py

            with h5py.File(self.partial_path, "x") as file:
                file.attrs["schema_version"] = 1
                file.attrs["joint_order"] = json.dumps(JOINT_ORDER)
                file.attrs["success"] = -1
                file.attrs["abort_reason"] = "unfinished"
                file.attrs["created_utc"] = dt.datetime.now(dt.UTC).isoformat()
                for key, value in self.metadata.items():
                    file.attrs[key] = self._attribute_value(value)
                datasets = self._create_datasets(file)

                while True:
                    item = self._queue.get()
                    try:
                        if isinstance(item, _Finalize):
                            file.attrs["success"] = int(item.success)
                            file.attrs["abort_reason"] = item.abort_reason
                            file.attrs["frame_count"] = self._frames_written
                            file.attrs["completed_utc"] = dt.datetime.now(dt.UTC).isoformat()
                            for key, value in item.final_metadata.items():
                                file.attrs[key] = self._attribute_value(value)
                            file.flush()
                            break

                        index = self._frames_written
                        self._append_dataset(datasets["joint_pos"], item.joint_pos, index)
                        self._append_dataset(datasets["front"], item.front_rgb, index)
                        self._append_dataset(datasets["wrist"], item.wrist_rgb, index)
                        self._append_dataset(datasets["actions"], item.action, index)
                        self._append_dataset(datasets["control"], item.control_timestamp_s, index)
                        self._append_dataset(datasets["front_timestamp"], item.front_timestamp_s, index)
                        self._append_dataset(datasets["wrist_timestamp"], item.wrist_timestamp_s, index)
                        self._frames_written += 1
                        if self._frames_written % self.flush_every == 0:
                            file.flush()
                    finally:
                        self._queue.task_done()
        except BaseException as exc:
            self._failure = exc

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"HDF5 writer failed: {self._failure}") from self._failure
        if not self._thread.is_alive() and not self._finished:
            raise RuntimeError("HDF5 writer stopped unexpectedly")

    def append(self, sample: EpisodeSample) -> None:
        if self._finished:
            raise RuntimeError("cannot append to a finished episode")
        self._raise_if_failed()
        import numpy as np

        joint_pos = np.asarray(sample.joint_pos, dtype=np.float32)
        action = np.asarray(sample.action, dtype=np.float32)
        front = np.asarray(sample.front_rgb)
        wrist = np.asarray(sample.wrist_rgb)
        if joint_pos.shape != (6,) or action.shape != (6,):
            raise ValueError(f"joint_pos/action must both have shape (6,), got {joint_pos.shape}/{action.shape}")
        if not np.isfinite(joint_pos).all() or not np.isfinite(action).all():
            raise ValueError("joint_pos/action contains NaN or Inf")
        for role, image in (("front", front), ("wrist", wrist)):
            if image.shape != self.image_shape:
                raise ValueError(f"{role} image shape {image.shape} does not match {self.image_shape}")
            if image.dtype != np.uint8:
                raise ValueError(f"{role} image dtype {image.dtype} is not uint8")
        timestamps = (
            sample.control_timestamp_s,
            sample.front_timestamp_s,
            sample.wrist_timestamp_s,
        )
        if not np.isfinite(timestamps).all():
            raise ValueError("sample contains a non-finite timestamp")

        detached = EpisodeSample(
            joint_pos=joint_pos.copy(),
            front_rgb=front.copy(),
            wrist_rgb=wrist.copy(),
            action=action.copy(),
            control_timestamp_s=float(sample.control_timestamp_s),
            front_timestamp_s=float(sample.front_timestamp_s),
            wrist_timestamp_s=float(sample.wrist_timestamp_s),
        )
        try:
            self._queue.put_nowait(detached)
        except queue.Full as exc:
            raise RuntimeError("HDF5 writer queue is full; refusing to drop or misalign a control sample") from exc
        self._frames_enqueued += 1

    def finish(
        self,
        *,
        success: bool,
        abort_reason: str = "",
        final_metadata: dict[str, Any] | None = None,
        timeout_s: float = 120.0,
    ) -> Path:
        if self._finished:
            raise RuntimeError("episode is already finished")
        self._raise_if_failed()
        self._queue.put(
            _Finalize(
                success=success,
                abort_reason=abort_reason,
                final_metadata=final_metadata or {},
            ),
            timeout=10.0,
        )
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TimeoutError(
                f"HDF5 writer did not finish within {timeout_s:.1f} seconds; "
                f"partial file retained at {self.partial_path}"
            )
        self._raise_if_failed_after_join()
        destination = self.success_path if success else self.rejected_path
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite episode: {destination}")
        os.replace(self.partial_path, destination)
        self._finished = True
        return destination

    def _raise_if_failed_after_join(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                f"HDF5 writer failed; partial file retained at {self.partial_path}: {self._failure}"
            ) from self._failure

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def frames_enqueued(self) -> int:
        return self._frames_enqueued

    @property
    def frames_written(self) -> int:
        return self._frames_written
