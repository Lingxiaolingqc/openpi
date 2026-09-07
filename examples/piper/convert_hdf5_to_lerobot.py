"""Audit PiPER MuJoCo HDF5 episodes and convert the training split to LeRobot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from examples.piper import contract
from examples.piper.expert import Phase


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    seed: int
    frames: int
    task: str
    model_sha256: str
    source_sha256: str


REQUIRED_DATASETS = (
    "observations/state",
    "observations/image",
    "observations/timestamp_ns",
    "actions/absolute_target",
    "actions/phase",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    if input_path.is_dir():
        paths = sorted({*input_path.glob("*.h5"), *input_path.glob("*.hdf5")})
        if paths:
            return [path.resolve() for path in paths]
        raise ValueError(f"no HDF5 episodes found in {input_path}")
    raise ValueError(f"input path does not exist: {input_path}")


def _validate_attrs(path: Path, file: h5py.File) -> None:
    expected = {
        "schema_version": contract.INTERFACE_VERSION,
        "complete": True,
        "success": True,
        "robot_type": contract.ROBOT_TYPE,
        "end_effector": contract.END_EFFECTOR,
        "observation_alignment": "pre_step",
        "state_layout_json": json.dumps(contract.STATE_LAYOUT),
        "action_layout_json": json.dumps(contract.ACTION_LAYOUT),
        "state_units_json": json.dumps(["rad"] * 6 + ["unit_interval"]),
        "action_units_json": json.dumps(["rad"] * 6 + ["unit_interval"]),
        "physics_hz": contract.PHYSICS_HZ,
        "control_hz": contract.CONTROL_HZ,
        "task": contract.TASK_PROMPT,
    }
    mismatches = {key: (value, file.attrs.get(key)) for key, value in expected.items() if file.attrs.get(key) != value}
    if mismatches:
        raise ValueError(f"{path} has incompatible PiPER metadata: {mismatches}")
    model_sha256 = file.attrs.get("model_sha256")
    revision = file.attrs.get("menagerie_revision")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise ValueError(f"{path} has no valid model_sha256")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"{path} has no Menagerie revision")
    try:
        extra = json.loads(file.attrs["extra_metadata_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} has unreadable extra metadata") from exc
    if not isinstance(extra, dict):
        raise ValueError(f"{path} extra metadata must be a mapping")


def validate_episode(path: Path) -> EpisodeRef:
    """Validate one finalized successful episode and return its immutable reference."""

    with h5py.File(path, "r") as file:
        _validate_attrs(path, file)
        missing = [name for name in REQUIRED_DATASETS if name not in file]
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        frames = int(file.attrs.get("num_samples", 0))
        if frames < 1:
            raise ValueError(f"{path} is empty")
        state = np.asarray(file["observations/state"][:], dtype=np.float64)
        image = file["observations/image"]
        timestamps = np.asarray(file["observations/timestamp_ns"][:], dtype=np.int64)
        actions = np.asarray(file["actions/absolute_target"][:], dtype=np.float64)
        phases = np.asarray(file["actions/phase"].asstr()[:])
        expected_shapes = {
            "state": (frames, contract.ACTION_DIM),
            "image": (frames, contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3),
            "timestamp": (frames,),
            "action": (frames, contract.ACTION_DIM),
            "phase": (frames,),
        }
        actual_shapes = {
            "state": state.shape,
            "image": image.shape,
            "timestamp": timestamps.shape,
            "action": actions.shape,
            "phase": phases.shape,
        }
        bad_shapes = {
            name: (expected_shapes[name], shape)
            for name, shape in actual_shapes.items()
            if shape != expected_shapes[name]
        }
        if bad_shapes:
            raise ValueError(f"{path} has invalid dataset shapes: {bad_shapes}")
        if image.dtype != np.uint8:
            raise ValueError(f"{path} images must be uint8, got {image.dtype}")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError(f"{path} contains NaN or infinity")
        if np.any(state[:, :6] < contract.ARM_Q_MIN_RAD) or np.any(state[:, :6] > contract.ARM_Q_MAX_RAD):
            raise ValueError(f"{path} measured joints exceed Menagerie hard limits")
        if np.any(actions[:, :6] < contract.ARM_Q_MIN_APP_RAD - 1e-5) or np.any(
            actions[:, :6] > contract.ARM_Q_MAX_APP_RAD + 1e-5
        ):
            raise ValueError(f"{path} action joints exceed application soft limits")
        if np.any(state[:, 6] < 0.0) or np.any(state[:, 6] > 1.0):
            raise ValueError(f"{path} measured gripper is outside [0, 1]")
        if np.any(actions[:, 6] < 0.0) or np.any(actions[:, 6] > 1.0):
            raise ValueError(f"{path} action gripper is outside [0, 1]")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{path} timestamps must be strictly increasing")
        allowed_phases = {phase.value for phase in Phase}
        if not set(phases) <= allowed_phases:
            raise ValueError(f"{path} contains unknown expert phases: {sorted(set(phases) - allowed_phases)}")

        maximum_step = np.concatenate([contract.ARM_MAX_STEP_RAD, [contract.GRIPPER_MAX_STEP]])
        first_step = np.abs(actions[0] - state[0])
        sequence_steps = np.abs(np.diff(actions, axis=0))
        if np.any(first_step > maximum_step + 1e-5) or np.any(sequence_steps > maximum_step + 1e-5):
            raise ValueError(f"{path} action sequence violates the 20 Hz rate limit")
        return EpisodeRef(
            path=path,
            seed=int(file.attrs["seed"]),
            frames=frames,
            task=str(file.attrs["task"]),
            model_sha256=str(file.attrs["model_sha256"]),
            source_sha256=sha256_file(path),
        )


def discover_episodes(input_path: Path) -> list[EpisodeRef]:
    episodes = [validate_episode(path) for path in _source_files(input_path)]
    if len({episode.seed for episode in episodes}) != len(episodes):
        raise ValueError("episode seeds must be unique")
    if len({episode.model_sha256 for episode in episodes}) != 1:
        raise ValueError("all episodes must use the same Menagerie model SHA-256")
    return episodes


def split_episodes(
    episodes: list[EpisodeRef], *, train_fraction: float = 0.8, split_seed: int = 2026
) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    if len(episodes) < 2:
        raise ValueError("at least two episodes are required for an episode-level split")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    order = np.random.default_rng(split_seed).permutation(len(episodes))
    train_count = min(len(episodes) - 1, max(1, round(len(episodes) * train_fraction)))
    train_indices = {int(index) for index in order[:train_count]}
    train = [episode for index, episode in enumerate(episodes) if index in train_indices]
    validation = [episode for index, episode in enumerate(episodes) if index not in train_indices]
    return train, validation


def limit_training_episodes(train: list[EpisodeRef], limit: int | None) -> list[EpisodeRef]:
    """Select a deterministic prefix for small overfit experiments."""

    if limit is None:
        return train
    if limit < 1:
        raise ValueError("train episode limit must be positive")
    if limit > len(train):
        raise ValueError(f"train episode limit {limit} exceeds available training episodes {len(train)}")
    return train[:limit]


def _lerobot_home() -> Path:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

        return Path(HF_LEROBOT_HOME)
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME

        return Path(LEROBOT_HOME)


def _episode_manifest(episode: EpisodeRef) -> dict:
    return {
        "source": str(episode.path),
        "seed": episode.seed,
        "frames": episode.frames,
        "sha256": episode.source_sha256,
    }


def convert_training_split(
    train: list[EpisodeRef],
    validation: list[EpisodeRef],
    *,
    repo_id: str,
    split_seed: int,
    image_mode: Literal["image", "video"] = "video",
    image_writer_processes: int = 4,
    image_writer_threads: int = 4,
) -> Path:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = _lerobot_home() / repo_id
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing LeRobot dataset: {output_path}")
    features = {
        "observation.images.base": {
            "dtype": image_mode,
            "shape": (contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3),
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
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=contract.ROBOT_TYPE,
        fps=contract.CONTROL_HZ,
        features=features,
        use_videos=image_mode == "video",
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    converted_frames = 0
    for episode in train:
        with h5py.File(episode.path, "r") as file:
            states = np.asarray(file["observations/state"][:], dtype=np.float32)
            images = file["observations/image"]
            actions = np.asarray(file["actions/absolute_target"][:], dtype=np.float32)
            for index in range(episode.frames):
                dataset.add_frame(
                    {
                        "observation.images.base": images[index],
                        "observation.state": states[index],
                        "action": actions[index],
                        "task": episode.task,
                    }
                )
                converted_frames += 1
        dataset.save_episode()
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    output_path.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "schema_version": 1,
        "split_seed": split_seed,
        "train": [_episode_manifest(episode) for episode in train],
        "validation": [_episode_manifest(episode) for episode in validation],
    }
    dataset_contract = {
        "interface_version": contract.INTERFACE_VERSION,
        "robot_type": contract.ROBOT_TYPE,
        "end_effector": contract.END_EFFECTOR,
        "task": contract.TASK_PROMPT,
        "fps": contract.CONTROL_HZ,
        "state_layout": list(contract.STATE_LAYOUT),
        "action_layout": list(contract.ACTION_LAYOUT),
        "action_semantics": "absolute_joint_positions_and_normalized_gripper",
        "training_transform": "first_six_state_relative_deltas_gripper_absolute",
        "camera_roles": {"base": "required", "wrist": "absent"},
        "model_sha256": train[0].model_sha256,
        "converted_train_frames": converted_frames,
    }
    simulation_only = {
        "deployment_scope": "simulation-only",
        "real_robot_deployment_allowed": False,
        "reason": "dataset generated exclusively in MuJoCo",
    }
    for name, value in (
        ("episode_split.json", split_manifest),
        ("dataset_contract.json", dataset_contract),
        ("SIMULATION_ONLY.json", simulation_only),
    ):
        (output_path / name).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    print(f"converted_train_episodes:{len(train)}")
    print(f"held_out_validation_episodes:{len(validation)}")
    print(f"converted_train_frames:{converted_frames}")
    print(f"lerobot_output_path:{output_path}")
    print("PIPER_HDF5_TO_LEROBOT_OK")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--train-episode-limit",
        type=int,
        help="convert only this deterministic prefix of the training split (for example, 10 for overfit-10)",
    )
    parser.add_argument("--image-mode", choices=("image", "video"), default="video")
    parser.add_argument("--image-writer-processes", type=int, default=4)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not args.dry_run and not args.repo_id:
        parser.error("--repo-id is required unless --dry-run is used")
    episodes = discover_episodes(args.input_path)
    train, validation = split_episodes(episodes, train_fraction=args.train_fraction, split_seed=args.split_seed)
    train = limit_training_episodes(train, args.train_episode_limit)
    train_frames = sum(episode.frames for episode in train)
    print(f"source_episode_count:{len(episodes)}")
    print(f"train_episode_count:{len(train)}")
    print(f"validation_episode_count:{len(validation)}")
    print(f"train_frame_count:{train_frames}")
    for passes in (1, 3, 5):
        print(
            f"recommended_train_steps_batch8_{passes}_passes:"
            f"{contract.training_steps_for_passes(train_frames, passes=passes)}"
        )
    if args.dry_run:
        print("PIPER_HDF5_TO_LEROBOT_DRY_RUN_OK")
        return 0
    convert_training_split(
        train,
        validation,
        repo_id=args.repo_id,
        split_seed=args.split_seed,
        image_mode=args.image_mode,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
