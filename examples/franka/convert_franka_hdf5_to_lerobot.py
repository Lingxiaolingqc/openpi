"""Audit and convert canonical FR3 HDF5 episodes into a LeRobot dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import string
from typing import Literal

import h5py
import numpy as np

from examples.franka import contract


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    num_samples: int
    base_image_shape: tuple[int, int, int]
    wrist_image_shape: tuple[int, int, int] | None
    task: str


REQUIRED_DATASETS = (
    "observations/q",
    "observations/dq",
    "observations/gripper_width_m",
    "observations/timestamps_monotonic_ns",
    "observations/images/base",
    "observations/image_timestamps_ns/base",
    "actions/q_target",
    "actions/gripper_width_m",
)


def _source_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    if input_path.is_dir():
        files = sorted({*input_path.glob("*.h5"), *input_path.glob("*.hdf5")})
        if files:
            return [path.resolve() for path in files]
        raise ValueError(f"no .h5 or .hdf5 files found in {input_path}")
    raise ValueError(f"input path does not exist: {input_path}")


def _validate_attrs(path: Path, h5_file: h5py.File) -> None:
    expected = {
        "schema_version": contract.INTERFACE_VERSION,
        "complete": True,
        "observation_alignment": "pre_step",
        "robot_type": "fr3",
        "end_effector": "franka_hand",
        "joint_names_json": json.dumps(contract.JOINT_NAMES),
        "state_units_json": json.dumps(["rad"] * 7 + ["m"]),
        "action_units_json": json.dumps(["rad"] * 7 + ["m"]),
        "control_hz": contract.CONTROL_HZ,
    }
    mismatches = {
        key: (value, h5_file.attrs.get(key)) for key, value in expected.items() if h5_file.attrs.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path} has incompatible Franka metadata: {mismatches}")
    try:
        calibration_hashes = json.loads(h5_file.attrs["camera_calibration_sha256_json"])
        software_versions = json.loads(h5_file.attrs["software_versions_json"])
        extra_metadata = json.loads(h5_file.attrs["extra_metadata_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} has unreadable audit metadata: {exc}") from exc
    expected_cameras = {"base", *({"wrist"} if bool(h5_file.attrs.get("has_wrist_camera", False)) else set())}
    if not isinstance(calibration_hashes, dict) or set(calibration_hashes) != expected_cameras:
        raise ValueError(f"{path} camera calibration hashes do not match the recorded cameras")
    if any(
        not isinstance(digest, str) or len(digest) != 64 or any(char not in string.hexdigits for char in digest)
        for digest in calibration_hashes.values()
    ):
        raise ValueError(f"{path} contains an invalid camera calibration SHA-256")
    required_versions = {"franka_ros2", "libfranka"}
    if (
        not isinstance(software_versions, dict)
        or not required_versions <= software_versions.keys()
        or any(not isinstance(value, str) or not value.strip() for value in software_versions.values())
    ):
        raise ValueError(f"{path} has incomplete Franka software version metadata")
    robot_system_version = h5_file.attrs.get("robot_system_version")
    if not isinstance(robot_system_version, str) or not robot_system_version.strip():
        raise ValueError(f"{path} has no robot system version")
    if not isinstance(extra_metadata, dict):
        raise ValueError(f"{path} extra metadata must be a mapping")


def _validate_episode(path: Path, *, max_image_skew_s: float) -> EpisodeRef:
    with h5py.File(path, "r") as h5_file:
        _validate_attrs(path, h5_file)
        missing = [name for name in REQUIRED_DATASETS if name not in h5_file]
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        count = int(h5_file.attrs.get("num_samples", 0))
        if count < 1:
            raise ValueError(f"{path} has no samples")
        q = np.asarray(h5_file["observations/q"][:], dtype=np.float64)
        dq = np.asarray(h5_file["observations/dq"][:], dtype=np.float64)
        gripper = np.asarray(h5_file["observations/gripper_width_m"][:], dtype=np.float64)
        timestamps = np.asarray(h5_file["observations/timestamps_monotonic_ns"][:], dtype=np.int64)
        q_target = np.asarray(h5_file["actions/q_target"][:], dtype=np.float64)
        gripper_target = np.asarray(h5_file["actions/gripper_width_m"][:], dtype=np.float64)
        base = h5_file["observations/images/base"]
        base_timestamps = np.asarray(h5_file["observations/image_timestamps_ns/base"][:], dtype=np.int64)
        expected_shapes = {
            "q": (count, 7),
            "dq": (count, 7),
            "gripper": (count,),
            "timestamps": (count,),
            "q_target": (count, 7),
            "gripper_target": (count,),
            "base_timestamps": (count,),
        }
        values = {
            "q": q,
            "dq": dq,
            "gripper": gripper,
            "timestamps": timestamps,
            "q_target": q_target,
            "gripper_target": gripper_target,
            "base_timestamps": base_timestamps,
        }
        bad_shapes = {
            name: (expected_shapes[name], value.shape)
            for name, value in values.items()
            if value.shape != expected_shapes[name]
        }
        if bad_shapes:
            raise ValueError(f"{path} has invalid dataset shapes: {bad_shapes}")
        if base.shape[0] != count or base.ndim != 4 or base.shape[-1] != 3 or base.dtype != np.uint8:
            raise ValueError(f"{path} has invalid base images shape/dtype {base.shape}/{base.dtype}")
        if not all(np.isfinite(value).all() for value in (q, dq, gripper, q_target, gripper_target)):
            raise ValueError(f"{path} contains NaN or infinity")
        if np.any(q < contract.Q_MIN_RAD) or np.any(q > contract.Q_MAX_RAD):
            raise ValueError(f"{path} measured q exceeds the FR3 application limits")
        if np.any(q_target < contract.Q_MIN_RAD) or np.any(q_target > contract.Q_MAX_RAD):
            raise ValueError(f"{path} q_target exceeds the FR3 application limits")
        if np.any(gripper < 0.0) or np.any(gripper > contract.DEFAULT_MAX_GRIPPER_WIDTH_M):
            raise ValueError(f"{path} measured gripper width is out of range")
        if np.any(gripper_target < 0.0) or np.any(gripper_target > contract.DEFAULT_MAX_GRIPPER_WIDTH_M):
            raise ValueError(f"{path} gripper target is out of range")
        if np.any(np.diff(timestamps) <= 0) or np.any(np.diff(base_timestamps) <= 0):
            raise ValueError(f"{path} timestamps must be strictly increasing")
        if np.max(np.abs(base_timestamps - timestamps)) > int(max_image_skew_s * 1_000_000_000):
            raise ValueError(f"{path} base camera timestamps exceed the allowed skew")

        commanded_velocity = np.abs(q_target - q) * contract.CONTROL_HZ
        if np.any(commanded_velocity > contract.QDOT_RECT_MAX_RAD_S + 1e-9):
            raise ValueError(f"{path} observation/action pairs violate continuity or alignment")
        q_sequence = np.concatenate([q[:1], q_target], axis=0)
        observed_velocity = np.abs(np.diff(q_sequence, axis=0)) * contract.CONTROL_HZ
        if np.any(observed_velocity > contract.QDOT_RECT_MAX_RAD_S + 1e-9):
            raise ValueError(f"{path} action targets violate the suggested rectangular velocity limits")

        wrist_shape = None
        has_wrist = bool(h5_file.attrs.get("has_wrist_camera", False))
        if has_wrist:
            for name in ("observations/images/wrist", "observations/image_timestamps_ns/wrist"):
                if name not in h5_file:
                    raise ValueError(f"{path} declares a wrist camera but is missing {name}")
            wrist = h5_file["observations/images/wrist"]
            wrist_timestamps = np.asarray(h5_file["observations/image_timestamps_ns/wrist"][:], dtype=np.int64)
            if wrist.shape[0] != count or wrist.ndim != 4 or wrist.shape[-1] != 3 or wrist.dtype != np.uint8:
                raise ValueError(f"{path} has invalid wrist images shape/dtype {wrist.shape}/{wrist.dtype}")
            if wrist_timestamps.shape != (count,) or np.any(np.diff(wrist_timestamps) <= 0):
                raise ValueError(f"{path} wrist timestamps must be one strictly increasing value per sample")
            if np.max(np.abs(wrist_timestamps - base_timestamps)) > int(max_image_skew_s * 1_000_000_000):
                raise ValueError(f"{path} base/wrist camera timestamps exceed the allowed skew")
            if np.max(np.abs(wrist_timestamps - timestamps)) > int(max_image_skew_s * 1_000_000_000):
                raise ValueError(f"{path} wrist camera/state timestamps exceed the allowed skew")
            wrist_shape = tuple(wrist.shape[1:])
        task = h5_file.attrs.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"{path} task must be a non-empty string")
        return EpisodeRef(
            path=path,
            num_samples=count,
            base_image_shape=tuple(base.shape[1:]),
            wrist_image_shape=wrist_shape,
            task=task,
        )


def discover_episodes(
    input_path: Path,
    *,
    include_failures: bool = False,
    max_image_skew_s: float = 0.05,
) -> tuple[list[EpisodeRef], int]:
    episodes = []
    skipped = 0
    for path in _source_files(input_path):
        with h5py.File(path, "r") as h5_file:
            complete = bool(h5_file.attrs.get("complete", False))
            success = bool(h5_file.attrs.get("success", False))
        if not complete:
            raise ValueError(f"{path} is not finalized")
        if not success and not include_failures:
            skipped += 1
            continue
        episodes.append(_validate_episode(path, max_image_skew_s=max_image_skew_s))
    if not episodes:
        raise ValueError("no eligible Franka episodes remain after filtering")
    first = episodes[0]
    for episode in episodes[1:]:
        if (episode.base_image_shape, episode.wrist_image_shape) != (
            first.base_image_shape,
            first.wrist_image_shape,
        ):
            raise ValueError("all Franka episodes must use the same camera presence and image shapes")
    return episodes, skipped


def _lerobot_home() -> Path:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

        return Path(HF_LEROBOT_HOME)
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME

        return Path(LEROBOT_HOME)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_dataset(
    episodes: list[EpisodeRef],
    *,
    repo_id: str,
    fps: int = 20,
    image_mode: Literal["image", "video"] = "video",
    image_writer_processes: int = 4,
    image_writer_threads: int = 4,
    push_to_hub: bool = False,
    private: bool = True,
) -> Path:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if fps != 20:
        raise ValueError("Franka v1 datasets must be converted at 20 fps")
    output_path = _lerobot_home() / repo_id
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing LeRobot dataset: {output_path}")
    first = episodes[0]
    features = {
        "observation.images.base": {
            "dtype": image_mode,
            "shape": first.base_image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (contract.ACTION_DIM,),
            "names": list(contract.STATE_LAYOUT),
        },
        "action": {
            "dtype": "float32",
            "shape": (contract.ACTION_DIM,),
            "names": list(contract.ACTION_LAYOUT),
        },
    }
    if first.wrist_image_shape is not None:
        features["observation.images.wrist"] = {
            "dtype": image_mode,
            "shape": first.wrist_image_shape,
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="fr3",
        fps=fps,
        features=features,
        use_videos=image_mode == "video",
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    converted_frames = 0
    manifest_episodes = []
    for episode in episodes:
        with h5py.File(episode.path, "r") as h5_file:
            q = np.asarray(h5_file["observations/q"][:], dtype=np.float32)
            gripper = np.asarray(h5_file["observations/gripper_width_m"][:], dtype=np.float32)
            q_target = np.asarray(h5_file["actions/q_target"][:], dtype=np.float32)
            gripper_target = np.asarray(h5_file["actions/gripper_width_m"][:], dtype=np.float32)
            base = h5_file["observations/images/base"]
            wrist = h5_file.get("observations/images/wrist")
            for index in range(episode.num_samples):
                frame = {
                    "observation.images.base": base[index],
                    "observation.state": np.concatenate([q[index], gripper[index : index + 1]]),
                    "action": np.concatenate([q_target[index], gripper_target[index : index + 1]]),
                    "task": episode.task,
                }
                if wrist is not None:
                    frame["observation.images.wrist"] = wrist[index]
                dataset.add_frame(frame)
                converted_frames += 1
        dataset.save_episode()
        manifest_episodes.append(
            {"source": str(episode.path), "sha256": _sha256(episode.path), "frames": episode.num_samples}
        )
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    if push_to_hub:
        dataset.push_to_hub(
            tags=["fr3", "franka-hand", "openpi"],
            private=private,
            push_videos=image_mode == "video",
            license="apache-2.0",
        )
    manifest = {
        "schema_version": 1,
        "repo_id": repo_id,
        "robot_type": "fr3",
        "fps": fps,
        "frames": converted_frames,
        "episodes": manifest_episodes,
        "state_layout": list(contract.STATE_LAYOUT),
        "action_layout": list(contract.ACTION_LAYOUT),
        "action_semantics": "absolute_joint_position_and_gripper_width",
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "franka_conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"converted_episode_count:{len(episodes)}")
    print(f"converted_frame_count:{converted_frames}")
    print(f"lerobot_output_path:{output_path}")
    print("FRANKA_HDF5_TO_LEROBOT_OK")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--max-image-skew-s", type=float, default=0.05)
    parser.add_argument("--image-mode", choices=("image", "video"), default="video")
    parser.add_argument("--image-writer-processes", type=int, default=4)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.dry_run and not args.repo_id:
        parser.error("--repo-id is required unless --dry-run is used")
    if args.push_to_hub and args.dry_run:
        parser.error("--push-to-hub cannot be combined with --dry-run")
    if args.max_image_skew_s <= 0.0:
        parser.error("--max-image-skew-s must be positive")
    episodes, skipped = discover_episodes(
        args.input_path,
        include_failures=args.include_failures,
        max_image_skew_s=args.max_image_skew_s,
    )
    frames = sum(episode.num_samples for episode in episodes)
    print(f"source_episode_count:{len(episodes)}")
    print(f"source_frame_count:{frames}")
    print(f"skipped_failed_episodes:{skipped}")
    print(f"recommended_train_steps_batch8_three_passes:{contract.training_steps_for_passes(frames)}")
    if args.dry_run:
        print("FRANKA_HDF5_TO_LEROBOT_DRY_RUN_OK")
        return 0
    convert_dataset(
        episodes,
        repo_id=args.repo_id,
        image_mode=args.image_mode,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        push_to_hub=args.push_to_hub,
        private=args.private,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
