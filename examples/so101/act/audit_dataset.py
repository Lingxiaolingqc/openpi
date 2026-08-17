"""Audit a local LeRobot SO-101 dataset before ACT training.

The audit never clips labels and never modifies the source dataset. It checks the
metadata/schema contract first, then scans numeric columns without decoding every
video frame. One frame per episode is decoded to validate frame-0 camera content.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.so101.act import common


@dataclass
class _NumericAudit:
    frame_count: int
    episode_ids: set[int]
    frame0_indices: dict[int, int]
    state_min: np.ndarray
    state_max: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    non_finite_state_count: np.ndarray
    non_finite_action_count: np.ndarray
    action_violation_count: np.ndarray
    action_max_violation: np.ndarray
    non_monotonic_timestamp_count: int
    off_rate_timestamp_count: int
    invalid_frame0_index_count: int
    invalid_frame0_timestamp_count: int


def _matrix(value: Any, *, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.stack(array.tolist())
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1 and width == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"expected a two-dimensional width-{width} column, got {array.shape}")
    return array


def _json_vector(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values).reshape(-1)]


def _iter_scalar_batches(dataset: Any, *, batch_size: int):
    columns = ["observation.state", "action", "episode_index", "frame_index", "timestamp"]
    missing = [key for key in columns if key not in dataset.hf_dataset.column_names]
    if missing:
        raise ValueError(f"LeRobot parquet data is missing audit column(s): {missing}")
    scalar_dataset = dataset.hf_dataset.select_columns(columns)
    scalar_dataset.set_format(type="numpy", columns=columns, output_all_columns=False)
    for start in range(0, len(scalar_dataset), batch_size):
        yield start, scalar_dataset[start : min(start + batch_size, len(scalar_dataset))]


def _scan_numeric_columns(dataset: Any, *, fps: float, batch_size: int) -> _NumericAudit:
    width = len(common.JOINT_NAMES)
    state_min = np.full(width, np.inf, dtype=np.float64)
    state_max = np.full(width, -np.inf, dtype=np.float64)
    action_min = np.full(width, np.inf, dtype=np.float64)
    action_max = np.full(width, -np.inf, dtype=np.float64)
    non_finite_state_count = np.zeros(width, dtype=np.int64)
    non_finite_action_count = np.zeros(width, dtype=np.int64)
    violation_count = np.zeros(width, dtype=np.int64)
    maximum_violation = np.zeros(width, dtype=np.float64)
    episode_ids: set[int] = set()
    frame0_indices: dict[int, int] = {}
    previous_timestamps: dict[int, float] = {}
    non_monotonic = 0
    off_rate = 0
    invalid_frame0_index = 0
    invalid_frame0_timestamp = 0
    frame_count = 0
    expected_dt = 1.0 / fps
    timestamp_tolerance = max(1e-4, expected_dt * 0.01)

    for global_start, batch in _iter_scalar_batches(dataset, batch_size=batch_size):
        state = _matrix(batch["observation.state"], width=width)
        action = _matrix(batch["action"], width=width)
        episode_index = np.asarray(batch["episode_index"], dtype=np.int64).reshape(-1)
        frame_index = np.asarray(batch["frame_index"], dtype=np.int64).reshape(-1)
        timestamp = np.asarray(batch["timestamp"], dtype=np.float64).reshape(-1)
        lengths = {len(state), len(action), len(episode_index), len(frame_index), len(timestamp)}
        if len(lengths) != 1:
            raise ValueError(f"misaligned scalar columns in batch at global index {global_start}: {lengths}")

        state_finite = np.isfinite(state)
        action_finite = np.isfinite(action)
        non_finite_state_count += np.count_nonzero(~state_finite, axis=0)
        non_finite_action_count += np.count_nonzero(~action_finite, axis=0)
        if state_finite.any():
            state_min = np.minimum(state_min, np.min(np.where(state_finite, state, np.inf), axis=0))
            state_max = np.maximum(state_max, np.max(np.where(state_finite, state, -np.inf), axis=0))
        if action_finite.any():
            action_min = np.minimum(action_min, np.min(np.where(action_finite, action, np.inf), axis=0))
            action_max = np.maximum(action_max, np.max(np.where(action_finite, action, -np.inf), axis=0))
        safe_action = np.where(action_finite, action, 0.0)
        clipped = np.clip(
            safe_action,
            common.MOTOR_LIMITS_DEGREES[:, 0],
            common.MOTOR_LIMITS_DEGREES[:, 1],
        )
        violation = (safe_action != clipped) | ~action_finite
        violation_count += np.count_nonzero(violation, axis=0)
        correction = np.abs(safe_action - clipped)
        correction[~action_finite] = np.inf
        maximum_violation = np.maximum(maximum_violation, np.max(correction, axis=0))

        for offset, (episode, frame, time_s) in enumerate(zip(episode_index, frame_index, timestamp, strict=True)):
            episode_int = int(episode)
            episode_ids.add(episode_int)
            if episode_int not in frame0_indices:
                frame0_indices[episode_int] = global_start + offset
                invalid_frame0_index += int(frame != 0)
                invalid_frame0_timestamp += int(not np.isfinite(time_s) or abs(time_s) > timestamp_tolerance)
            previous = previous_timestamps.get(episode_int)
            if previous is not None:
                delta = float(time_s - previous)
                non_monotonic += int(not np.isfinite(delta) or delta <= 0.0)
                off_rate += int(not np.isfinite(delta) or abs(delta - expected_dt) > timestamp_tolerance)
            previous_timestamps[episode_int] = float(time_s)
        frame_count += len(state)

    return _NumericAudit(
        frame_count=frame_count,
        episode_ids=episode_ids,
        frame0_indices=frame0_indices,
        state_min=state_min,
        state_max=state_max,
        action_min=action_min,
        action_max=action_max,
        non_finite_state_count=non_finite_state_count,
        non_finite_action_count=non_finite_action_count,
        action_violation_count=violation_count,
        action_max_violation=maximum_violation,
        non_monotonic_timestamp_count=non_monotonic,
        off_rate_timestamp_count=off_rate,
        invalid_frame0_index_count=invalid_frame0_index,
        invalid_frame0_timestamp_count=invalid_frame0_timestamp,
    )


def _decode_frame0_samples(
    dataset: Any,
    *,
    camera_keys: tuple[str, ...],
    frame0_indices: dict[int, int],
    maximum_episodes: int,
) -> dict[str, Any]:
    selected = sorted(frame0_indices.items())
    if maximum_episodes > 0:
        selected = selected[:maximum_episodes]
    cameras = {
        key: {
            "decoded_episode_count": 0,
            "non_finite_episode_count": 0,
            "all_black_episode_count": 0,
            "constant_episode_count": 0,
            "observed_shapes": set(),
        }
        for key in camera_keys
    }
    for _, global_index in selected:
        item = dataset[global_index]
        for key in camera_keys:
            if key not in item:
                raise ValueError(f"decoded frame is missing camera feature {key!r}")
            value = item[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            image = np.asarray(value)
            record = cameras[key]
            record["decoded_episode_count"] += 1
            record["observed_shapes"].add(tuple(int(part) for part in image.shape))
            finite = np.isfinite(image)
            record["non_finite_episode_count"] += int(not finite.all())
            if finite.any():
                finite_values = image[finite]
                record["all_black_episode_count"] += int(float(finite_values.max()) <= 0.0)
                record["constant_episode_count"] += int(float(finite_values.max()) == float(finite_values.min()))
    return {
        key: {
            **{name: value for name, value in record.items() if name != "observed_shapes"},
            "observed_shapes": sorted(record["observed_shapes"]),
        }
        for key, record in cameras.items()
    }


def load_metadata(location: common.DatasetLocation) -> Any:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError as exc:
        raise RuntimeError("LeRobot is required; run this script with `uv run`") from exc
    return LeRobotDatasetMetadata(location.repo_id, root=location.dataset_path)


def audit_dataset(
    location: common.DatasetLocation,
    *,
    action_semantics: str,
    expected_camera_keys: tuple[str, ...] | None,
    scan_batch_size: int,
    frame0_sample_episodes: int,
    metadata_only: bool = False,
) -> tuple[common.DatasetContract, dict[str, Any]]:
    metadata = load_metadata(location)
    contract = common.build_dataset_contract(
        location,
        metadata,
        action_semantics=action_semantics,
        expected_camera_keys=expected_camera_keys,
    )
    report: dict[str, Any] = {
        "schema_gate_passed": True,
        "dataset_contract": contract.to_dict(),
        "source_dataset_modified": False,
        "labels_clipped": False,
        "exact_action_semantics_verified_from_metadata": False,
        "action_semantics_source": "S4 dataset contract supplied through configuration",
    }
    if metadata_only:
        report["numeric_scan_performed"] = False
        return contract, report

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot is required; run this script with `uv run`") from exc
    dataset = LeRobotDataset(
        location.repo_id,
        root=location.dataset_path,
        download_videos=False,
    )
    numeric = _scan_numeric_columns(dataset, fps=contract.fps, batch_size=scan_batch_size)
    frame0 = _decode_frame0_samples(
        dataset,
        camera_keys=contract.camera_keys,
        frame0_indices=numeric.frame0_indices,
        maximum_episodes=frame0_sample_episodes,
    )
    finite_gate = not (np.any(numeric.non_finite_state_count) or np.any(numeric.non_finite_action_count))
    timestamp_gate = not (
        numeric.non_monotonic_timestamp_count
        or numeric.off_rate_timestamp_count
        or numeric.invalid_frame0_index_count
        or numeric.invalid_frame0_timestamp_count
    )
    camera_gate = all(
        not item["non_finite_episode_count"]
        and not item["all_black_episode_count"]
        and not item["constant_episode_count"]
        for item in frame0.values()
    )
    count_gate = numeric.frame_count == contract.num_frames and numeric.episode_ids == set(range(contract.num_episodes))
    report.update(
        {
            "numeric_scan_performed": True,
            "numeric_gate_passed": bool(finite_gate and timestamp_gate and camera_gate and count_gate),
            "scanned_frame_count": numeric.frame_count,
            "scanned_episode_count": len(numeric.episode_ids),
            "state_minimum_by_joint": _json_vector(numeric.state_min),
            "state_maximum_by_joint": _json_vector(numeric.state_max),
            "action_minimum_by_joint": _json_vector(numeric.action_min),
            "action_maximum_by_joint": _json_vector(numeric.action_max),
            "non_finite_state_count_by_joint": numeric.non_finite_state_count.astype(int).tolist(),
            "non_finite_action_count_by_joint": numeric.non_finite_action_count.astype(int).tolist(),
            "action_motor_limit_violation_count_by_joint": (numeric.action_violation_count.astype(int).tolist()),
            "action_motor_limit_violation_total": int(numeric.action_violation_count.sum()),
            "action_motor_limit_max_violation_degrees_by_joint": _json_vector(numeric.action_max_violation),
            "non_monotonic_timestamp_count": numeric.non_monotonic_timestamp_count,
            "off_rate_timestamp_count": numeric.off_rate_timestamp_count,
            "invalid_frame0_index_count": numeric.invalid_frame0_index_count,
            "invalid_frame0_timestamp_count": numeric.invalid_frame0_timestamp_count,
            "frame0_camera_audit": frame0,
            "frame0_alignment_claim": (
                "Structural row/timestamp alignment and decoded frame validity passed; semantic causal "
                "alignment cannot be proven from the converted dataset alone."
            ),
        }
    )
    return contract, report


def _config_preparse() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=common.CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


def _build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    dataset_cfg = config.get("dataset", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(config["_path"]))
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("OPENPI_ACT_REPO_ID", dataset_cfg.get("repo_id")),
    )
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("HF_LEROBOT_HOME"))
    parser.add_argument(
        "--action-semantics",
        default=dataset_cfg.get("action_semantics", common.ACTION_SEMANTICS),
    )
    parser.add_argument(
        "--expected-camera-key",
        action="append",
        dest="expected_camera_keys",
        default=None,
        help="Repeat to require an exact camera schema. Omit to accept metadata-discovered cameras.",
    )
    parser.add_argument("--scan-batch-size", type=int, default=4096)
    parser.add_argument(
        "--frame0-sample-episodes",
        type=int,
        default=0,
        help="Number of episode frame-0 images to decode; 0 audits every episode.",
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    config = common.load_config(_config_preparse())
    args = _build_parser(config).parse_args()
    if not args.repo_id:
        raise SystemExit("--repo-id or OPENPI_ACT_REPO_ID is required")
    if args.scan_batch_size <= 0 or args.frame0_sample_episodes < 0:
        raise SystemExit("scan batch size must be positive and frame0 sample count must be non-negative")
    expected_camera_keys = args.expected_camera_keys
    if expected_camera_keys is None:
        expected_camera_keys = config.get("dataset", {}).get("expected_camera_keys")
    location = common.resolve_dataset_location(args.dataset_root, args.repo_id)
    try:
        contract, report = audit_dataset(
            location,
            action_semantics=args.action_semantics,
            expected_camera_keys=(
                None if expected_camera_keys is None else tuple(str(item) for item in expected_camera_keys)
            ),
            scan_batch_size=args.scan_batch_size,
            frame0_sample_episodes=args.frame0_sample_episodes,
            metadata_only=args.metadata_only,
        )
        if args.output:
            common.write_json(args.output.expanduser().resolve(), report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
        print(f"repo_id: {contract.repo_id}")
        print(f"dataset_path: {contract.dataset_path}")
        print(f"episode_count: {contract.num_episodes}")
        print(f"frame_count: {contract.num_frames}")
        print(f"fps: {contract.fps:g}")
        print(f"camera_keys: {contract.camera_keys}")
        if report.get("numeric_scan_performed"):
            print(
                "action_motor_limit_violation_count_by_joint:",
                tuple(report["action_motor_limit_violation_count_by_joint"]),
            )
            if not report["numeric_gate_passed"]:
                print("ACT_DATASET_AUDIT_FAILED: numeric gate failed", file=sys.stderr)
                return 1
        print("ACT_DATASET_AUDIT_OK")
        return 0
    except Exception as exc:
        print(f"ACT_DATASET_AUDIT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
