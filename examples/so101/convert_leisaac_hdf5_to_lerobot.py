"""Convert native LeIsaac SO-101 HDF5 recordings to LeRobot format.

Run this script in OpenPI's ``uv`` environment, not in the Isaac Sim Conda
environment. By default only episodes marked successful are converted. The
source HDF5 files are opened read-only and existing LeRobot datasets are never
overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# These limits are copied from LeIsaac v0.4.0
# source/leisaac/leisaac/assets/robots/lerobot.py. LeIsaac stores joint
# positions and actions in USD radians, while its LeRobot integration stores
# them in the physical SO-101 motor coordinate system.
USD_JOINT_LIMITS_DEGREES = np.asarray(
    [
        (-110.0, 110.0),
        (-100.0, 100.0),
        (-100.0, 90.0),
        (-95.0, 95.0),
        (-160.0, 160.0),
        (-10.0, 100.0),
    ],
    dtype=np.float64,
)
MOTOR_LIMITS_DEGREES = np.asarray(
    [
        (-100.0, 100.0),
        (-100.0, 100.0),
        (-100.0, 100.0),
        (-100.0, 100.0),
        (-100.0, 100.0),
        (0.0, 100.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class EpisodeRef:
    """A validated episode inside a native LeIsaac HDF5 file."""

    path: Path
    name: str
    num_samples: int
    image_shape: tuple[int, int, int]


def leisaac_radians_to_motor_degrees(values: np.ndarray) -> np.ndarray:
    """Apply LeIsaac v0.4.0's official SO-101 action-alignment mapping."""
    values = np.asarray(values)
    if values.ndim == 0 or values.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"Expected final dimension {len(JOINT_NAMES)}, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("SO-101 joint values contain NaN or infinity")

    joint_degrees = np.rad2deg(values.astype(np.float64, copy=False))
    joint_low = USD_JOINT_LIMITS_DEGREES[:, 0]
    joint_range = USD_JOINT_LIMITS_DEGREES[:, 1] - joint_low
    motor_low = MOTOR_LIMITS_DEGREES[:, 0]
    motor_range = MOTOR_LIMITS_DEGREES[:, 1] - motor_low
    converted = (joint_degrees - joint_low) / joint_range * motor_range + motor_low
    return converted.astype(np.float32)


def _input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".hdf5":
            raise ValueError(f"Expected an .hdf5 input file: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.hdf5"))
        if files:
            return files
        raise ValueError(f"No .hdf5 files found in directory: {input_path}")
    raise ValueError(f"Input path does not exist: {input_path}")


def _demo_sort_key(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid LeIsaac episode name: {name}") from exc


def _validate_episode(path: Path, name: str, demo: h5py.Group) -> EpisodeRef:
    required_paths = ("actions", "obs/joint_pos", "obs/front")
    missing_paths = [dataset_path for dataset_path in required_paths if dataset_path not in demo]
    if missing_paths:
        raise ValueError(f"{path}:{name} is missing datasets: {missing_paths}")

    actions = demo["actions"]
    state = demo["obs/joint_pos"]
    front = demo["obs/front"]
    if actions.ndim != 2 or actions.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"{path}:{name} has invalid actions shape {actions.shape}")
    if state.shape != actions.shape:
        raise ValueError(f"{path}:{name} state shape {state.shape} does not match actions {actions.shape}")
    if front.ndim != 4 or front.shape[0] != actions.shape[0] or front.shape[-1] != 3:
        raise ValueError(f"{path}:{name} has invalid front camera shape {front.shape}")
    if front.dtype != np.uint8:
        raise ValueError(f"{path}:{name} front camera must be uint8, got {front.dtype}")

    num_samples = int(demo.attrs.get("num_samples", -1))
    if num_samples != actions.shape[0]:
        raise ValueError(f"{path}:{name} num_samples={num_samples} does not match actions length {actions.shape[0]}")
    if not np.isfinite(actions[:]).all() or not np.isfinite(state[:]).all():
        raise ValueError(f"{path}:{name} contains NaN or infinity in state/action data")

    return EpisodeRef(
        path=path,
        name=name,
        num_samples=num_samples,
        image_shape=tuple(front.shape[1:]),
    )


def discover_successful_episodes(input_path: Path) -> tuple[list[EpisodeRef], int, int]:
    """Return validated successful episodes, skipped failures, and file count."""
    files = _input_files(input_path.expanduser().resolve())
    episodes: list[EpisodeRef] = []
    skipped_failures = 0
    for path in files:
        with h5py.File(path, "r") as h5_file:
            if "data" not in h5_file:
                raise ValueError(f"{path} is missing the root data group")
            data = h5_file["data"]
            demo_names = sorted(
                (name for name in data if name.startswith("demo_")),
                key=_demo_sort_key,
            )
            for name in demo_names:
                demo = data[name]
                if not bool(demo.attrs.get("success", False)):
                    skipped_failures += 1
                    continue
                episodes.append(_validate_episode(path, name, demo))

    if not episodes:
        raise ValueError("No successful LeIsaac episodes were found")
    image_shapes = {episode.image_shape for episode in episodes}
    if len(image_shapes) != 1:
        raise ValueError(f"Successful episodes have inconsistent camera shapes: {sorted(image_shapes)}")
    return episodes, skipped_failures, len(files)


def _audit_ranges(episodes: list[EpisodeRef]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_min = np.full(len(JOINT_NAMES), np.inf, dtype=np.float64)
    state_max = np.full(len(JOINT_NAMES), -np.inf, dtype=np.float64)
    action_min = state_min.copy()
    action_max = state_max.copy()
    for episode in episodes:
        with h5py.File(episode.path, "r") as h5_file:
            demo = h5_file["data"][episode.name]
            state = leisaac_radians_to_motor_degrees(demo["obs/joint_pos"][:])
            action = leisaac_radians_to_motor_degrees(demo["actions"][:])
            state_min = np.minimum(state_min, state.min(axis=0))
            state_max = np.maximum(state_max, state.max(axis=0))
            action_min = np.minimum(action_min, action.min(axis=0))
            action_max = np.maximum(action_max, action.max(axis=0))
    return state_min, state_max, action_min, action_max


def _format_range(values: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(value), 4) for value in values)


def print_source_audit(episodes: list[EpisodeRef], *, skipped_failures: int, file_count: int, fps: int) -> None:
    total_frames = sum(episode.num_samples for episode in episodes)
    state_min, state_max, action_min, action_max = _audit_ranges(episodes)
    print("source_file_count:", file_count)
    print("successful_episode_count:", len(episodes))
    print("skipped_failed_episode_count:", skipped_failures)
    print("total_frames:", total_frames)
    print("image_shape:", episodes[0].image_shape)
    print("fps:", fps)
    print("duration_seconds:", round(total_frames / fps, 4))
    print("state_motor_min:", _format_range(state_min))
    print("state_motor_max:", _format_range(state_max))
    print("action_motor_min:", _format_range(action_min))
    print("action_motor_max:", _format_range(action_max))


def _lerobot_home() -> Path:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

        return Path(HF_LEROBOT_HOME)
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME

        return Path(LEROBOT_HOME)


def convert_dataset(
    episodes: list[EpisodeRef],
    *,
    repo_id: str,
    task: str,
    fps: int,
    image_mode: Literal["image", "video"],
    image_writer_processes: int,
    image_writer_threads: int,
    push_to_hub: bool,
    private: bool,
) -> Path:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = _lerobot_home() / repo_id
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing LeRobot dataset: {output_path}")

    image_shape = episodes[0].image_shape
    features = {
        "observation.images.front": {
            "dtype": image_mode,
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="so101_follower",
        fps=fps,
        features=features,
        use_videos=image_mode == "video",
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    converted_frames = 0
    for episode_index, episode in enumerate(episodes):
        with h5py.File(episode.path, "r") as h5_file:
            demo = h5_file["data"][episode.name]
            state = leisaac_radians_to_motor_degrees(demo["obs/joint_pos"][:])
            action = leisaac_radians_to_motor_degrees(demo["actions"][:])
            front = demo["obs/front"]
            for frame_index in range(episode.num_samples):
                dataset.add_frame(
                    {
                        "observation.images.front": front[frame_index],
                        "observation.state": state[frame_index],
                        "action": action[frame_index],
                    }
                )
                converted_frames += 1
                if converted_frames % 250 == 0:
                    print("converted_frames:", converted_frames, flush=True)
        dataset.save_episode(task=task)
        print(
            f"saved_episode:{episode_index}:source={episode.path}:{episode.name}:frames={episode.num_samples}",
            flush=True,
        )

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    if push_to_hub:
        dataset.push_to_hub(
            tags=["so101", "leisaac", "openpi"],
            private=private,
            push_videos=image_mode == "video",
            license="apache-2.0",
        )

    print("converted_episode_count:", len(episodes))
    print("converted_frame_count:", converted_frames)
    print("lerobot_output_path:", output_path)
    print("LEISAAC_HDF5_TO_LEROBOT_OK")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--task", default="Lift the cube.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--image-mode", choices=("image", "video"), default="video")
    parser.add_argument("--image-writer-processes", type=int, default=4)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.image_writer_processes < 0 or args.image_writer_threads <= 0:
        parser.error("Image writer process/thread counts must be non-negative/positive")
    if not args.dry_run and not args.repo_id:
        parser.error("--repo-id is required unless --dry-run is used")
    if args.push_to_hub and args.dry_run:
        parser.error("--push-to-hub cannot be combined with --dry-run")

    episodes, skipped_failures, file_count = discover_successful_episodes(args.input_path)
    print_source_audit(
        episodes,
        skipped_failures=skipped_failures,
        file_count=file_count,
        fps=args.fps,
    )
    if args.dry_run:
        print("LEISAAC_HDF5_TO_LEROBOT_DRY_RUN_OK")
        return 0

    convert_dataset(
        episodes,
        repo_id=args.repo_id,
        task=args.task,
        fps=args.fps,
        image_mode=args.image_mode,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        push_to_hub=args.push_to_hub,
        private=args.private,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
