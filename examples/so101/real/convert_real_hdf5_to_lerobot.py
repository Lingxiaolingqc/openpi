"""Convert SO-101 real-robot HDF5 episodes to the pinned LeRobot format.

This converter is intentionally separate from the LeIsaac converter. Real
state/action arrays already contain calibrated motor degrees and are copied
without the LeIsaac radians-to-motor remapping. Source HDF5 files are opened
read-only and an existing LeRobot output is never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

REAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = REAL_DIR.parents[2]
DEFAULT_CALIBRATION_LOCK = REAL_DIR / "calibration" / "calibration_lock.json"

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
BOX_ID_CODES = {"A": 0, "B": 1}
TARGET_IDS = (0, 1, 2)
EXPECTED_IMAGE_SHAPE = (480, 640, 3)
EXPECTED_CONTROL_RATE_HZ = 30.0
EXPECTED_ACTION_MODE = "absolute_calibrated_motor_degrees"
MIN_EPISODE_FRAMES = 30
MAX_FRAME_AGE_S = 0.100
MAX_CAMERA_SKEW_S = 0.100
MIN_EFFECTIVE_CONTROL_HZ = 25.0
MAX_EFFECTIVE_CONTROL_HZ = 35.0
MAX_CONTROL_P95_INTERVAL_S = 0.050
CALIBRATION_RANGE_EPSILON_DEG = 0.25
MODEL_RESOLUTION_MAX = 4095


@dataclass(frozen=True)
class CalibrationContract:
    freeze_id: str
    calibration_hashes: dict[str, str]
    joint_low_deg: np.ndarray
    joint_high_deg: np.ndarray


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    num_samples: int
    image_shape: tuple[int, int, int]
    target_id: int
    task: str
    box_id: str
    control_dt_p50_s: float
    control_dt_p95_s: float
    max_frame_age_s: float
    max_camera_skew_s: float
    repeated_front_timestamps: int
    repeated_wrist_timestamps: int
    state_min: tuple[float, ...]
    state_max: tuple[float, ...]
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_repo_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def load_calibration_contract(
    lock_path: Path = DEFAULT_CALIBRATION_LOCK,
    *,
    repo_root: Path = REPO_ROOT,
) -> CalibrationContract:
    lock_path = lock_path.expanduser().resolve()
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read calibration lock {lock_path}: {exc}") from exc

    if int(lock.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported calibration lock schema in {lock_path}")
    if tuple(lock.get("joint_order", ())) != JOINT_ORDER:
        raise ValueError(f"Calibration lock joint order does not match {JOINT_ORDER}")

    arms = lock.get("arms")
    if not isinstance(arms, list):
        raise ValueError("Calibration lock is missing its arms list")
    by_role = {str(arm.get("role")): arm for arm in arms if isinstance(arm, dict)}
    if set(by_role) != {"leader", "follower"}:
        raise ValueError(f"Calibration lock roles must be leader/follower, got {sorted(by_role)}")

    follower = by_role["follower"]
    snapshot_path = _resolve_repo_path(str(follower["snapshot_path"]), repo_root).resolve()
    expected_snapshot_hash = str(follower["snapshot_sha256"]).upper()
    actual_snapshot_hash = _sha256(snapshot_path)
    if actual_snapshot_hash != expected_snapshot_hash:
        raise ValueError(
            f"Follower calibration snapshot hash mismatch: expected {expected_snapshot_hash}, "
            f"got {actual_snapshot_hash}"
        )
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read follower calibration snapshot {snapshot_path}: {exc}") from exc
    if set(snapshot) != set(JOINT_ORDER):
        raise ValueError("Follower calibration snapshot has unexpected joint names")

    low = []
    high = []
    for name in JOINT_ORDER:
        range_min = int(snapshot[name]["range_min"])
        range_max = int(snapshot[name]["range_max"])
        if not 0 <= range_min < range_max <= MODEL_RESOLUTION_MAX:
            raise ValueError(f"Invalid follower calibration range for {name}: {range_min}..{range_max}")
        midpoint = (range_min + range_max) / 2
        low.append((range_min - midpoint) * 360 / MODEL_RESOLUTION_MAX)
        high.append((range_max - midpoint) * 360 / MODEL_RESOLUTION_MAX)

    return CalibrationContract(
        freeze_id=str(lock["freeze_id"]),
        calibration_hashes={role: str(arm["active_sha256"]).upper() for role, arm in by_role.items()},
        joint_low_deg=np.asarray(low, dtype=np.float64),
        joint_high_deg=np.asarray(high, dtype=np.float64),
    )


def _input_files(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".h5" or input_path.name.endswith(".partial.h5"):
            raise ValueError(f"Expected one completed .h5 episode: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    episode_dir = input_path / "episodes"
    search_dir = episode_dir if episode_dir.is_dir() else input_path
    files = sorted(path for path in search_dir.glob("*.h5") if not path.name.endswith(".partial.h5"))
    if not files:
        raise ValueError(f"No completed .h5 episodes found in {search_dir}")
    return files


def _required_attr(file: h5py.File, name: str) -> Any:
    if name not in file.attrs:
        raise ValueError(f"{file.filename} is missing required root attribute {name!r}")
    return file.attrs[name]


def _text(value: Any, *, field: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty UTF-8 string")
    return value.strip()


def _json_attr(file: h5py.File, name: str) -> Any:
    raw = _required_attr(file, name)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file.filename} attribute {name!r} is not valid JSON") from exc


def _check_joint_ranges(
    path: Path,
    label: str,
    values: np.ndarray,
    contract: CalibrationContract,
) -> None:
    below = values < contract.joint_low_deg - CALIBRATION_RANGE_EPSILON_DEG
    above = values > contract.joint_high_deg + CALIBRATION_RANGE_EPSILON_DEG
    if not (below | above).any():
        return
    violations = []
    for index, name in enumerate(JOINT_ORDER):
        count = int(np.count_nonzero(below[:, index] | above[:, index]))
        if count:
            violations.append(
                f"{name}:{count} samples outside "
                f"{contract.joint_low_deg[index]:.2f}..{contract.joint_high_deg[index]:.2f} deg"
            )
    raise ValueError(f"{path} {label} exceeds frozen calibration: {', '.join(violations)}")


def _validate_episode(path: Path, contract: CalibrationContract) -> EpisodeRef:
    required_datasets = (
        "obs/joint_pos",
        "obs/front",
        "obs/wrist",
        "actions",
        "timestamps/control",
        "timestamps/front",
        "timestamps/wrist",
    )
    with h5py.File(path, "r") as file:
        missing = [name for name in required_datasets if name not in file]
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        if int(_required_attr(file, "schema_version")) != 1:
            raise ValueError(f"{path} has an unsupported schema_version")
        if int(_required_attr(file, "success")) != 1:
            raise ValueError(f"{path} is not marked successful")
        if _text(_required_attr(file, "action_mode"), field="action_mode") != EXPECTED_ACTION_MODE:
            raise ValueError(f"{path} does not contain absolute calibrated motor-degree actions")
        if tuple(_json_attr(file, "joint_order")) != JOINT_ORDER:
            raise ValueError(f"{path} joint_order does not match {JOINT_ORDER}")

        control_rate_hz = float(_required_attr(file, "control_rate_hz"))
        if not np.isclose(control_rate_hz, EXPECTED_CONTROL_RATE_HZ, atol=1e-6):
            raise ValueError(f"{path} control_rate_hz={control_rate_hz:g}, expected {EXPECTED_CONTROL_RATE_HZ:g}")
        freeze_id = _text(_required_attr(file, "calibration_freeze_id"), field="calibration_freeze_id")
        if freeze_id != contract.freeze_id:
            raise ValueError(f"{path} freeze_id={freeze_id!r}, expected {contract.freeze_id!r}")
        hashes = {str(key): str(value).upper() for key, value in _json_attr(file, "calibration_hashes").items()}
        if hashes != contract.calibration_hashes:
            raise ValueError(f"{path} calibration hashes do not match the selected frozen lock")

        target_id = int(_required_attr(file, "target_id"))
        if target_id not in TARGET_IDS:
            raise ValueError(f"{path} has invalid target_id={target_id}")
        task = _text(_required_attr(file, "task"), field="task")
        box_id = _text(_required_attr(file, "box_id"), field="box_id").upper()
        if box_id not in BOX_ID_CODES:
            raise ValueError(f"{path} has forbidden box_id={box_id!r}; only Box-A/Box-B may enter conversion")

        state_ds = file["obs/joint_pos"]
        action_ds = file["actions"]
        front_ds = file["obs/front"]
        wrist_ds = file["obs/wrist"]
        num_samples = int(action_ds.shape[0])
        if num_samples < MIN_EPISODE_FRAMES:
            raise ValueError(f"{path} has only {num_samples} frames; minimum is {MIN_EPISODE_FRAMES}")
        if int(_required_attr(file, "frame_count")) != num_samples:
            raise ValueError(f"{path} frame_count does not match actions length {num_samples}")
        if state_ds.shape != (num_samples, len(JOINT_ORDER)):
            raise ValueError(f"{path} has invalid state shape {state_ds.shape}")
        if action_ds.shape != state_ds.shape:
            raise ValueError(f"{path} action shape {action_ds.shape} does not match state {state_ds.shape}")
        if state_ds.dtype != np.dtype("float32") or action_ds.dtype != np.dtype("float32"):
            raise ValueError(f"{path} state/actions must both be float32")
        for role, image_ds in (("front", front_ds), ("wrist", wrist_ds)):
            if image_ds.shape != (num_samples, *EXPECTED_IMAGE_SHAPE):
                raise ValueError(f"{path} {role} image shape is {image_ds.shape}, expected (T,480,640,3)")
            if image_ds.dtype != np.dtype("uint8"):
                raise ValueError(f"{path} {role} images must be uint8")

        state = state_ds[:]
        action = action_ds[:]
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{path} contains NaN or Inf in state/action")
        _check_joint_ranges(path, "state", state, contract)
        _check_joint_ranges(path, "action", action, contract)

        timestamp_arrays = {
            name: np.asarray(file[f"timestamps/{name}"][:], dtype=np.float64) for name in ("control", "front", "wrist")
        }
        for name, values in timestamp_arrays.items():
            if values.shape != (num_samples,):
                raise ValueError(f"{path} timestamps/{name} shape is {values.shape}, expected {(num_samples,)}")
            if not np.isfinite(values).all():
                raise ValueError(f"{path} timestamps/{name} contains NaN or Inf")
        control = timestamp_arrays["control"]
        front_time = timestamp_arrays["front"]
        wrist_time = timestamp_arrays["wrist"]
        control_dt = np.diff(control)
        if (control_dt <= 0).any():
            raise ValueError(f"{path} control timestamps are not strictly increasing")
        if (np.diff(front_time) < 0).any() or (np.diff(wrist_time) < 0).any():
            raise ValueError(f"{path} camera timestamps move backwards")
        control_dt_p50_s = float(np.percentile(control_dt, 50))
        control_dt_p95_s = float(np.percentile(control_dt, 95))
        effective_hz = 1.0 / control_dt_p50_s
        if not MIN_EFFECTIVE_CONTROL_HZ <= effective_hz <= MAX_EFFECTIVE_CONTROL_HZ:
            raise ValueError(f"{path} effective median control rate is {effective_hz:.2f} Hz")
        if control_dt_p95_s > MAX_CONTROL_P95_INTERVAL_S:
            raise ValueError(
                f"{path} control interval p95 is {control_dt_p95_s * 1000:.1f} ms, "
                f"limit is {MAX_CONTROL_P95_INTERVAL_S * 1000:.1f} ms"
            )

        front_age = control - front_time
        wrist_age = control - wrist_time
        if min(float(front_age.min()), float(wrist_age.min())) < -0.001:
            raise ValueError(f"{path} contains a camera timestamp in the future")
        max_frame_age_s = max(float(front_age.max()), float(wrist_age.max()))
        if max_frame_age_s > MAX_FRAME_AGE_S + 1e-6:
            raise ValueError(
                f"{path} maximum camera frame age is {max_frame_age_s * 1000:.1f} ms, "
                f"limit is {MAX_FRAME_AGE_S * 1000:.1f} ms"
            )
        max_camera_skew_s = float(np.max(np.abs(front_time - wrist_time)))
        if max_camera_skew_s > MAX_CAMERA_SKEW_S + 1e-6:
            raise ValueError(
                f"{path} maximum camera skew is {max_camera_skew_s * 1000:.1f} ms, "
                f"limit is {MAX_CAMERA_SKEW_S * 1000:.1f} ms"
            )

        return EpisodeRef(
            path=path,
            num_samples=num_samples,
            image_shape=EXPECTED_IMAGE_SHAPE,
            target_id=target_id,
            task=task,
            box_id=box_id,
            control_dt_p50_s=control_dt_p50_s,
            control_dt_p95_s=control_dt_p95_s,
            max_frame_age_s=max_frame_age_s,
            max_camera_skew_s=max_camera_skew_s,
            repeated_front_timestamps=int(np.count_nonzero(np.diff(front_time) == 0)),
            repeated_wrist_timestamps=int(np.count_nonzero(np.diff(wrist_time) == 0)),
            state_min=tuple(float(value) for value in state.min(axis=0)),
            state_max=tuple(float(value) for value in state.max(axis=0)),
            action_min=tuple(float(value) for value in action.min(axis=0)),
            action_max=tuple(float(value) for value in action.max(axis=0)),
        )


def discover_real_episodes(
    input_path: Path,
    contract: CalibrationContract,
) -> tuple[list[EpisodeRef], int, int]:
    files = _input_files(input_path)
    episodes = []
    skipped_unsuccessful = 0
    for path in files:
        with h5py.File(path, "r") as file:
            if "success" not in file.attrs:
                raise ValueError(f"{path} is missing required root attribute 'success'")
            if int(file.attrs["success"]) != 1:
                skipped_unsuccessful += 1
                continue
        episodes.append(_validate_episode(path, contract))
    if not episodes:
        raise ValueError("No successful real SO-101 episodes were found")
    if len({episode.image_shape for episode in episodes}) != 1:
        raise ValueError("Successful episodes have inconsistent image shapes")
    return episodes, skipped_unsuccessful, len(files)


def _format_joint_values(values: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(value), 3) for value in values)


def print_source_audit(
    episodes: list[EpisodeRef],
    *,
    skipped_unsuccessful: int,
    file_count: int,
) -> None:
    state_min = np.min(np.asarray([episode.state_min for episode in episodes]), axis=0)
    state_max = np.max(np.asarray([episode.state_max for episode in episodes]), axis=0)
    action_min = np.min(np.asarray([episode.action_min for episode in episodes]), axis=0)
    action_max = np.max(np.asarray([episode.action_max for episode in episodes]), axis=0)
    target_counts = Counter(episode.target_id for episode in episodes)
    box_counts = Counter(episode.box_id for episode in episodes)
    total_frames = sum(episode.num_samples for episode in episodes)
    print("source_file_count:", file_count)
    print("successful_episode_count:", len(episodes))
    print("skipped_unsuccessful_episode_count:", skipped_unsuccessful)
    print("total_frames:", total_frames)
    print("duration_seconds:", round(total_frames / EXPECTED_CONTROL_RATE_HZ, 3))
    print("image_shape:", episodes[0].image_shape)
    print("fps:", int(EXPECTED_CONTROL_RATE_HZ))
    print("target_id_episode_counts:", dict(sorted(target_counts.items())))
    print("box_id_episode_counts:", dict(sorted(box_counts.items())))
    print("state_min_deg:", _format_joint_values(state_min))
    print("state_max_deg:", _format_joint_values(state_max))
    print("action_min_deg:", _format_joint_values(action_min))
    print("action_max_deg:", _format_joint_values(action_max))
    print(
        "control_dt_p95_max_ms:",
        round(max(episode.control_dt_p95_s for episode in episodes) * 1000, 3),
    )
    print(
        "camera_frame_age_max_ms:",
        round(max(episode.max_frame_age_s for episode in episodes) * 1000, 3),
    )
    print(
        "camera_skew_max_ms:",
        round(max(episode.max_camera_skew_s for episode in episodes) * 1000, 3),
    )
    print(
        "repeated_camera_timestamp_steps:",
        {
            "front": sum(episode.repeated_front_timestamps for episode in episodes),
            "wrist": sum(episode.repeated_wrist_timestamps for episode in episodes),
        },
    )


def _lerobot_home() -> Path:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

        return Path(HF_LEROBOT_HOME)
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME

        return Path(LEROBOT_HOME)


def _validate_repo_id(repo_id: str) -> None:
    parts = repo_id.split("/")
    if (
        "\\" in repo_id
        or Path(repo_id).is_absolute()
        or ".." in parts
        or len(parts) != 2
        or any(not part for part in parts)
    ):
        raise ValueError("repo_id must have exactly two safe components, for example local/so101_real_v1")


def _features(image_mode: Literal["image", "video"]) -> dict[str, dict[str, Any]]:
    return {
        "observation.images.front": {
            "dtype": image_mode,
            "shape": EXPECTED_IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": image_mode,
            "shape": EXPECTED_IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_ORDER),),
            "names": list(JOINT_ORDER),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_ORDER),),
            "names": list(JOINT_ORDER),
        },
        "target_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["target_id"],
        },
        "box_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["box_id"],
        },
    }


def _conversion_manifest(
    episodes: list[EpisodeRef],
    *,
    repo_id: str,
    image_mode: str,
    contract: CalibrationContract,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_format": "so101_real_hdf5_v1",
        "repo_id": repo_id,
        "fps": EXPECTED_CONTROL_RATE_HZ,
        "image_mode": image_mode,
        "joint_order": list(JOINT_ORDER),
        "action_mode": EXPECTED_ACTION_MODE,
        "target_id_encoding": {"earbud_case": 0, "sponge": 1, "ballpoint_pen": 2},
        "box_id_encoding": BOX_ID_CODES,
        "box_id_is_model_input": False,
        "calibration_freeze_id": contract.freeze_id,
        "calibration_hashes": contract.calibration_hashes,
        "episodes": [
            {
                "source_file": episode.path.name,
                "frames": episode.num_samples,
                "target_id": episode.target_id,
                "box_id": episode.box_id,
                "task": episode.task,
            }
            for episode in episodes
        ],
    }


def convert_dataset(
    episodes: list[EpisodeRef],
    *,
    repo_id: str,
    image_mode: Literal["image", "video"],
    image_writer_processes: int,
    image_writer_threads: int,
    push_to_hub: bool,
    private: bool,
    contract: CalibrationContract,
    dataset_class: Any | None = None,
    output_root: Path | None = None,
) -> Path:
    _validate_repo_id(repo_id)
    if dataset_class is None:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        dataset_class = LeRobotDataset
    root = _lerobot_home() if output_root is None else output_root.expanduser().resolve()
    output_path = root / repo_id
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing LeRobot dataset: {output_path}")

    dataset = dataset_class.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="so101_follower",
        fps=int(EXPECTED_CONTROL_RATE_HZ),
        features=_features(image_mode),
        use_videos=image_mode == "video",
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    converted_frames = 0
    for episode_index, episode in enumerate(episodes):
        with h5py.File(episode.path, "r") as file:
            state = file["obs/joint_pos"]
            action = file["actions"]
            front = file["obs/front"]
            wrist = file["obs/wrist"]
            for frame_index in range(episode.num_samples):
                dataset.add_frame(
                    {
                        "observation.images.front": front[frame_index],
                        "observation.images.wrist": wrist[frame_index],
                        "observation.state": state[frame_index],
                        "action": action[frame_index],
                        "target_id": np.asarray([episode.target_id], dtype=np.int64),
                        "box_id": np.asarray([BOX_ID_CODES[episode.box_id]], dtype=np.int64),
                        "task": episode.task,
                    }
                )
                converted_frames += 1
                if converted_frames % 250 == 0:
                    print("converted_frames:", converted_frames, flush=True)
        dataset.save_episode()
        print(
            f"saved_episode:{episode_index}:source={episode.path.name}:frames={episode.num_samples}",
            flush=True,
        )

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    if not output_path.is_dir():
        raise RuntimeError(f"LeRobot conversion did not create expected output directory: {output_path}")
    manifest = _conversion_manifest(
        episodes,
        repo_id=repo_id,
        image_mode=image_mode,
        contract=contract,
    )
    manifest_path = output_path / "real_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["so101", "real-robot", "openpi"],
            private=private,
            push_videos=image_mode == "video",
            license="apache-2.0",
        )
    print("converted_episode_count:", len(episodes))
    print("converted_frame_count:", converted_frames)
    print("conversion_manifest:", manifest_path)
    print("lerobot_output_path:", output_path)
    print("REAL_HDF5_TO_LEROBOT_OK")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional LeRobot root; output is created below this directory as <repo-id>.",
    )
    parser.add_argument("--calibration-lock", type=Path, default=DEFAULT_CALIBRATION_LOCK)
    parser.add_argument("--image-mode", choices=("image", "video"), default="video")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.image_writer_processes < 0 or args.image_writer_threads <= 0:
        parser.error("Image writer process/thread counts must be non-negative/positive")
    if not args.dry_run and not args.repo_id:
        parser.error("--repo-id is required unless --dry-run is used")
    if args.push_to_hub and args.dry_run:
        parser.error("--push-to-hub cannot be combined with --dry-run")
    contract = load_calibration_contract(args.calibration_lock)
    episodes, skipped_unsuccessful, file_count = discover_real_episodes(args.input_path, contract)
    print_source_audit(
        episodes,
        skipped_unsuccessful=skipped_unsuccessful,
        file_count=file_count,
    )
    if args.dry_run:
        print("REAL_HDF5_TO_LEROBOT_DRY_RUN_OK")
        return 0
    convert_dataset(
        episodes,
        repo_id=args.repo_id,
        image_mode=args.image_mode,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        push_to_hub=args.push_to_hub,
        private=args.private,
        contract=contract,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
