"""Shared configuration, dataset-contract, metrics, and safety helpers for ACT."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tomllib
from typing import Any

import numpy as np

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_SEMANTICS = "absolute_joint_position_target_motor_degrees"
DEPLOYMENT_SCOPE = "simulation-only"
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "red_cube_to_box.toml"
CONTRACT_FILENAME = "dataset_contract.json"
SPLIT_FILENAME = "episode_split.json"
SAFETY_FILENAME = "SIMULATION_ONLY.json"
OVERFIT_GATE_FILENAME = "overfit_gate.json"

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
class DatasetLocation:
    dataset_root: Path
    repo_id: str
    dataset_path: Path


@dataclass(frozen=True)
class DatasetContract:
    repo_id: str
    dataset_root: str
    dataset_path: str
    metadata_fingerprint: str
    fps: float
    num_episodes: int
    num_frames: int
    camera_keys: tuple[str, ...]
    state_feature: str
    action_feature: str
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    joint_names: tuple[str, ...]
    action_semantics: str
    deployment_scope: str = DEPLOYMENT_SCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeSplit:
    repo_id: str
    metadata_fingerprint: str
    seed: int
    validation_fraction: float
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path or CONFIG_PATH).expanduser().resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    config["_path"] = str(config_path)
    return config


def config_value(
    config: dict[str, Any],
    section: str,
    key: str,
    *,
    environment: str | None = None,
    fallback: Any = None,
) -> Any:
    if environment and environment in os.environ:
        return os.environ[environment]
    return config.get(section, {}).get(key, fallback)


def resolve_dataset_location(dataset_root: Path | str | None, repo_id: str) -> DatasetLocation:
    if not repo_id or repo_id.strip() != repo_id:
        raise ValueError("repo_id must be a non-empty value without leading/trailing whitespace")
    repo_path = Path(repo_id)
    if repo_path.is_absolute() or ".." in repo_path.parts:
        raise ValueError(f"repo_id must be a relative path below HF_LEROBOT_HOME: {repo_id!r}")
    root_value = dataset_root or os.environ.get("HF_LEROBOT_HOME")
    if not root_value:
        raise ValueError("dataset root is required through --dataset-root or HF_LEROBOT_HOME")
    root = Path(root_value).expanduser().resolve()
    dataset_path = (root / repo_path).resolve()
    try:
        dataset_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"resolved dataset path escapes dataset root: {dataset_path}") from exc
    return DatasetLocation(root, repo_id, dataset_path)


def _feature_shape(feature: Any) -> tuple[int, ...]:
    value = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
    if value is None:
        raise ValueError(f"feature does not declare a shape: {feature!r}")
    return tuple(int(item) for item in value)


def _feature_names(feature: Any) -> tuple[str, ...]:
    value = feature.get("names") if isinstance(feature, dict) else getattr(feature, "names", None)
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _metadata_number(metadata: Any, key: str) -> int | float:
    direct_names = {
        "fps": ("fps",),
        "total_episodes": ("total_episodes", "num_episodes"),
        "total_frames": ("total_frames", "num_frames"),
    }
    for name in direct_names[key]:
        if hasattr(metadata, name):
            return getattr(metadata, name)
    info = getattr(metadata, "info", {})
    for name in direct_names[key]:
        if name in info:
            return info[name]
    raise ValueError(f"LeRobot metadata does not provide {key}")


def metadata_fingerprint(dataset_path: Path) -> str:
    meta_dir = dataset_path / "meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"LeRobot metadata directory not found: {meta_dir}")
    files = sorted(path for path in meta_dir.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"LeRobot metadata directory is empty: {meta_dir}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(meta_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def build_dataset_contract(
    location: DatasetLocation,
    metadata: Any,
    *,
    action_semantics: str,
    expected_camera_keys: Iterable[str] | None = None,
) -> DatasetContract:
    if not location.dataset_path.is_dir():
        raise FileNotFoundError(f"LeRobot dataset not found: {location.dataset_path}")
    if action_semantics != ACTION_SEMANTICS:
        raise ValueError(f"ACT baseline requires action_semantics={ACTION_SEMANTICS!r}, got {action_semantics!r}")
    features = getattr(metadata, "features", None)
    if not isinstance(features, dict):
        raise ValueError("LeRobot metadata features must be a dictionary")
    state_key = "observation.state"
    action_key = "action"
    missing = [key for key in (state_key, action_key) if key not in features]
    if missing:
        raise ValueError(f"dataset is missing required feature(s): {missing}")

    state_shape = _feature_shape(features[state_key])
    action_shape = _feature_shape(features[action_key])
    expected_shape = (len(JOINT_NAMES),)
    if state_shape != expected_shape or action_shape != expected_shape:
        raise ValueError(f"state/action must both have shape {expected_shape}, got {state_shape}/{action_shape}")
    state_names = _feature_names(features[state_key])
    action_names = _feature_names(features[action_key])
    if state_names != JOINT_NAMES or action_names != JOINT_NAMES:
        raise ValueError(
            "state/action joint order does not match the SO-101 S4 contract: "
            f"state={state_names}, action={action_names}, expected={JOINT_NAMES}"
        )

    camera_keys = tuple(sorted(str(key) for key in getattr(metadata, "camera_keys", ())))
    if not camera_keys:
        camera_keys = tuple(sorted(key for key in features if key.startswith("observation.images.")))
    if not camera_keys:
        raise ValueError("ACT dataset must provide at least one camera feature")
    for key in camera_keys:
        shape = _feature_shape(features[key])
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"camera feature {key!r} must be HWC RGB in metadata, got {shape}")
    if expected_camera_keys is not None:
        expected = tuple(sorted(expected_camera_keys))
        if camera_keys != expected:
            raise ValueError(f"camera keys {camera_keys} do not match requested schema {expected}")

    fps = float(_metadata_number(metadata, "fps"))
    num_episodes = int(_metadata_number(metadata, "total_episodes"))
    num_frames = int(_metadata_number(metadata, "total_frames"))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"dataset fps must be positive and finite, got {fps}")
    if num_episodes <= 0 or num_frames <= 0:
        raise ValueError(f"dataset must contain positive episode/frame counts, got {num_episodes}/{num_frames}")
    return DatasetContract(
        repo_id=location.repo_id,
        dataset_root=str(location.dataset_root),
        dataset_path=str(location.dataset_path),
        metadata_fingerprint=metadata_fingerprint(location.dataset_path),
        fps=fps,
        num_episodes=num_episodes,
        num_frames=num_frames,
        camera_keys=camera_keys,
        state_feature=state_key,
        action_feature=action_key,
        state_shape=state_shape,
        action_shape=action_shape,
        joint_names=JOINT_NAMES,
        action_semantics=action_semantics,
    )


def deterministic_episode_split(
    contract: DatasetContract,
    *,
    validation_fraction: float,
    seed: int,
) -> EpisodeSplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if contract.num_episodes < 2:
        raise ValueError("at least two episodes are required for a train/validation split")
    validation_count = max(1, round(contract.num_episodes * validation_fraction))
    validation_count = min(validation_count, contract.num_episodes - 1)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(contract.num_episodes).tolist()
    validation = tuple(sorted(int(item) for item in shuffled[:validation_count]))
    train = tuple(sorted(int(item) for item in shuffled[validation_count:]))
    return EpisodeSplit(
        repo_id=contract.repo_id,
        metadata_fingerprint=contract.metadata_fingerprint,
        seed=seed,
        validation_fraction=validation_fraction,
        train_episodes=train,
        validation_episodes=validation,
    )


def validate_split(split: EpisodeSplit, contract: DatasetContract) -> None:
    if split.repo_id != contract.repo_id or split.metadata_fingerprint != contract.metadata_fingerprint:
        raise ValueError("episode split was generated for a different dataset or metadata revision")
    combined = (*split.train_episodes, *split.validation_episodes)
    if len(combined) != contract.num_episodes or set(combined) != set(range(contract.num_episodes)):
        raise ValueError("episode split does not cover every current dataset episode exactly once")


def load_episode_split(path: Path, contract: DatasetContract) -> EpisodeSplit:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    split = EpisodeSplit(
        repo_id=str(raw["repo_id"]),
        metadata_fingerprint=str(raw["metadata_fingerprint"]),
        seed=int(raw["seed"]),
        validation_fraction=float(raw["validation_fraction"]),
        train_episodes=tuple(int(item) for item in raw["train_episodes"]),
        validation_episodes=tuple(int(item) for item in raw["validation_episodes"]),
    )
    validate_split(split, contract)
    return split


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def write_contract_and_safety(output_dir: Path, contract: DatasetContract) -> None:
    write_json(output_dir / CONTRACT_FILENAME, contract.to_dict())
    write_json(
        output_dir / SAFETY_FILENAME,
        {
            "deployment_scope": DEPLOYMENT_SCOPE,
            "real_robot_deployment_allowed": False,
            "reason": (
                "The source dataset is an S4 simulation dataset and current action-limit audit status "
                "does not authorize real-robot execution."
            ),
            "repo_id": contract.repo_id,
            "metadata_fingerprint": contract.metadata_fingerprint,
            "action_semantics": contract.action_semantics,
        },
    )


def require_simulation_only_marker(path: Path) -> dict[str, Any]:
    marker_path = path if path.name == SAFETY_FILENAME else path / SAFETY_FILENAME
    with marker_path.open("r", encoding="utf-8") as stream:
        marker = json.load(stream)
    if marker.get("deployment_scope") != DEPLOYMENT_SCOPE:
        raise ValueError(f"checkpoint is not marked {DEPLOYMENT_SCOPE!r}: {marker_path}")
    if marker.get("real_robot_deployment_allowed") is not False:
        raise ValueError(f"invalid real-robot safety boundary in {marker_path}")
    return marker


def motor_limit_statistics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"expected final action dimension {len(JOINT_NAMES)}, got {array.shape}")
    flattened = array.reshape(-1, len(JOINT_NAMES))
    finite = np.isfinite(flattened)
    safe = np.where(finite, flattened, 0.0)
    below = safe < MOTOR_LIMITS_DEGREES[:, 0]
    above = safe > MOTOR_LIMITS_DEGREES[:, 1]
    violation = below | above | ~finite
    correction = np.abs(safe - np.clip(safe, MOTOR_LIMITS_DEGREES[:, 0], MOTOR_LIMITS_DEGREES[:, 1]))
    correction[~finite] = np.inf
    return {
        "count_by_joint": np.count_nonzero(violation, axis=0).astype(int).tolist(),
        "total_count": int(np.count_nonzero(violation)),
        "max_violation_by_joint": np.max(correction, axis=0).tolist(),
        "minimum_by_joint": np.min(flattened, axis=0).tolist(),
        "maximum_by_joint": np.max(flattened, axis=0).tolist(),
    }


class ErrorAccumulator:
    """Streaming per-joint absolute and squared-error metrics."""

    def __init__(self, joint_count: int = len(JOINT_NAMES)) -> None:
        self._absolute = np.zeros(joint_count, dtype=np.float64)
        self._squared = np.zeros(joint_count, dtype=np.float64)
        self._count = np.zeros(joint_count, dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> None:
        prediction_array = np.asarray(prediction, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if prediction_array.shape != target_array.shape or prediction_array.shape[-1] != len(self._count):
            raise ValueError(
                f"prediction/target shapes must match with final joint dimension {len(self._count)}, "
                f"got {prediction_array.shape}/{target_array.shape}"
            )
        valid = np.isfinite(prediction_array) & np.isfinite(target_array)
        if mask is not None:
            expanded_mask = np.asarray(mask, dtype=bool)
            if expanded_mask.shape == prediction_array.shape[:-1]:
                expanded_mask = expanded_mask[..., None]
            if expanded_mask.shape != prediction_array.shape:
                expanded_mask = np.broadcast_to(expanded_mask, prediction_array.shape)
            valid &= expanded_mask
        error = prediction_array - target_array
        self._absolute += np.where(valid, np.abs(error), 0.0).reshape(-1, len(self._count)).sum(axis=0)
        self._squared += np.where(valid, np.square(error), 0.0).reshape(-1, len(self._count)).sum(axis=0)
        self._count += valid.reshape(-1, len(self._count)).sum(axis=0)

    def result(self) -> dict[str, Any]:
        if np.any(self._count == 0):
            raise ValueError("cannot report error metrics with an empty joint")
        mae = self._absolute / self._count
        rmse = np.sqrt(self._squared / self._count)
        total_count = int(self._count.sum())
        return {
            "mae": float(self._absolute.sum() / total_count),
            "rmse": float(np.sqrt(self._squared.sum() / total_count)),
            "mae_by_joint": mae.tolist(),
            "rmse_by_joint": rmse.tolist(),
            "sample_count_by_joint": self._count.astype(int).tolist(),
        }


def resolve_pretrained_model_path(checkpoint: Path | str) -> tuple[Path, Path]:
    value = Path(checkpoint).expanduser().resolve()
    if value.name == "pretrained_model" and value.is_dir():
        return value, value.parent.parent.parent
    direct = value / "pretrained_model"
    if direct.is_dir():
        return direct, value.parent.parent
    last = value / "checkpoints" / "last" / "pretrained_model"
    if last.is_dir():
        return last.resolve(), value
    raise FileNotFoundError(
        "ACT checkpoint must be a pretrained_model directory, a numbered checkpoint directory, "
        f"or a run directory with checkpoints/last: {value}"
    )
