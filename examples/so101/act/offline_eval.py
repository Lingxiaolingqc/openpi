"""Evaluate an ACT checkpoint on held-out LeRobot episodes without clipping predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.so101.act import audit_dataset
from examples.so101.act import common
from examples.so101.act.policy_utils import load_act_policy
from examples.so101.act.policy_utils import move_batch_to_device
from examples.so101.act.policy_utils import predict_action_chunk


class _LimitAccumulator:
    def __init__(self) -> None:
        width = len(common.JOINT_NAMES)
        self.count = np.zeros(width, dtype=np.int64)
        self.maximum = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.observed_maximum = np.full(width, -np.inf, dtype=np.float64)
        self.non_finite = np.zeros(width, dtype=np.int64)

    def update(self, values: np.ndarray, valid_steps: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        mask = np.asarray(valid_steps, dtype=bool)
        if array.ndim != 3 or array.shape[-1] != len(common.JOINT_NAMES):
            raise ValueError(f"expected action batch (batch,horizon,6), got {array.shape}")
        if mask.shape != array.shape[:-1]:
            raise ValueError(f"action mask {mask.shape} does not match action batch {array.shape}")
        expanded = np.broadcast_to(mask[..., None], array.shape)
        finite = np.isfinite(array)
        valid = expanded & finite
        self.non_finite += np.count_nonzero(expanded & ~finite, axis=(0, 1))
        if valid.any():
            self.minimum = np.minimum(
                self.minimum,
                np.min(np.where(valid, array, np.inf), axis=(0, 1)),
            )
            self.observed_maximum = np.maximum(
                self.observed_maximum,
                np.max(np.where(valid, array, -np.inf), axis=(0, 1)),
            )
        safe = np.where(finite, array, 0.0)
        clipped = np.clip(
            safe,
            common.MOTOR_LIMITS_DEGREES[:, 0],
            common.MOTOR_LIMITS_DEGREES[:, 1],
        )
        violation = expanded & ((safe != clipped) | ~finite)
        self.count += np.count_nonzero(violation, axis=(0, 1))
        correction = np.where(expanded, np.abs(safe - clipped), 0.0)
        correction[expanded & ~finite] = np.inf
        self.maximum = np.maximum(self.maximum, np.max(correction, axis=(0, 1)))

    def result(self) -> dict[str, Any]:
        return {
            "count_by_joint": self.count.astype(int).tolist(),
            "total_count": int(self.count.sum()),
            "maximum_violation_degrees_by_joint": self.maximum.tolist(),
            "minimum_by_joint": self.minimum.tolist(),
            "maximum_by_joint": self.observed_maximum.tolist(),
            "non_finite_count_by_joint": self.non_finite.astype(int).tolist(),
        }


def _config_preparse() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=common.CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


def _build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    dataset_cfg = config.get("dataset", {})
    evaluation_cfg = config.get("evaluation", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(config["_path"]))
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("OPENPI_ACT_CHECKPOINT"))
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("OPENPI_ACT_REPO_ID", dataset_cfg.get("repo_id")),
    )
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("HF_LEROBOT_HOME"))
    parser.add_argument(
        "--action-semantics",
        default=dataset_cfg.get("action_semantics", common.ACTION_SEMANTICS),
    )
    parser.add_argument("--expected-camera-key", action="append", dest="expected_camera_keys")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--device", default=config.get("policy", {}).get("device", "cuda"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(evaluation_cfg.get("batch_size", 8)),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(evaluation_cfg.get("num_workers", 4)),
    )
    parser.add_argument(
        "--maximum-batches",
        type=int,
        default=int(evaluation_cfg.get("maximum_batches", 0)),
        help="0 evaluates the full held-out split.",
    )
    parser.add_argument(
        "--actions-per-inference",
        type=int,
        default=0,
        help="0 uses the checkpoint n_action_steps; otherwise evaluate this chunk prefix.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-only-audit", action="store_true")
    return parser


def _load_validation_dataset(
    *,
    location: common.DatasetLocation,
    episodes: tuple[int, ...],
    policy: Any,
):
    try:
        from lerobot.common.datasets.factory import resolve_delta_timestamps
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError as exc:
        raise RuntimeError("LeRobot ACT is required; run this command with `uv run`") from exc
    metadata = LeRobotDatasetMetadata(location.repo_id, root=location.dataset_path)
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        location.repo_id,
        root=location.dataset_path,
        episodes=list(episodes),
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    if common.ensure_original_episode_index_lookup(dataset, episodes):
        print("act_lerobot_episode_index_compatibility: expanded_original_episode_lookup")
    return dataset


def evaluate(
    *,
    policy: Any,
    dataset: Any,
    device: str,
    batch_size: int,
    num_workers: int,
    maximum_batches: int,
    actions_per_inference: int | None,
) -> dict[str, Any]:
    import torch

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    errors = common.ErrorAccumulator()
    prediction_limits = _LimitAccumulator()
    target_limits = _LimitAccumulator()
    evaluated_batches = 0
    evaluated_steps = 0
    for batch_index, batch in enumerate(dataloader):
        if maximum_batches > 0 and batch_index >= maximum_batches:
            break
        moved = move_batch_to_device(batch, device)
        prediction = predict_action_chunk(
            policy,
            moved,
            actions_per_inference=actions_per_inference,
            apply_temporal_ensemble=False,
        )
        target = moved["action"][:, : prediction.shape[1]]
        pad = moved.get("action_is_pad")
        if pad is None:
            valid = torch.ones(target.shape[:-1], dtype=torch.bool, device=target.device)
        else:
            valid = ~pad[:, : prediction.shape[1]].bool()
        prediction_np = prediction.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        if not np.isfinite(prediction_np[valid_np]).all():
            raise RuntimeError("ACT offline prediction contains NaN or infinity")
        if not np.isfinite(target_np[valid_np]).all():
            raise RuntimeError("ACT offline target contains NaN or infinity")
        errors.update(prediction_np, target_np, valid_np)
        prediction_limits.update(prediction_np, valid_np)
        target_limits.update(target_np, valid_np)
        evaluated_batches += 1
        evaluated_steps += int(np.count_nonzero(valid_np))
    if evaluated_batches == 0:
        raise RuntimeError("offline evaluation selected no batches")
    return {
        "evaluated_batches": evaluated_batches,
        "evaluated_action_steps": evaluated_steps,
        "error": errors.result(),
        "prediction_motor_limits": prediction_limits.result(),
        "target_motor_limits": target_limits.result(),
        "predictions_clipped": False,
        "targets_clipped": False,
    }


def main() -> int:
    config = common.load_config(_config_preparse())
    args = _build_parser(config).parse_args()
    try:
        if args.checkpoint is None:
            raise ValueError("--checkpoint or OPENPI_ACT_CHECKPOINT is required")
        if not args.repo_id:
            raise ValueError("--repo-id or OPENPI_ACT_REPO_ID is required")
        if args.batch_size <= 0 or args.num_workers < 0 or args.maximum_batches < 0:
            raise ValueError("invalid offline evaluation batch/worker limits")
        if args.actions_per_inference < 0:
            raise ValueError("actions_per_inference must be non-negative")
        pretrained_dir, run_root = common.resolve_pretrained_model_path(args.checkpoint)
        marker = common.require_simulation_only_marker(pretrained_dir)
        expected_camera_keys = args.expected_camera_keys
        if expected_camera_keys is None:
            expected_camera_keys = config.get("dataset", {}).get("expected_camera_keys")
        location = common.resolve_dataset_location(args.dataset_root, args.repo_id)
        contract, audit_report = audit_dataset.audit_dataset(
            location,
            action_semantics=args.action_semantics,
            expected_camera_keys=(
                None if expected_camera_keys is None else tuple(str(item) for item in expected_camera_keys)
            ),
            scan_batch_size=4096,
            frame0_sample_episodes=0,
            metadata_only=args.metadata_only_audit,
        )
        if marker.get("repo_id") != contract.repo_id:
            raise ValueError("checkpoint safety marker repo_id does not match the requested dataset")
        if marker.get("metadata_fingerprint") != contract.metadata_fingerprint:
            raise ValueError("checkpoint was trained against a different dataset metadata revision")
        split_path = (
            args.split_file.expanduser().resolve() if args.split_file else pretrained_dir / common.SPLIT_FILENAME
        )
        split = common.load_episode_split(split_path, contract)
        policy = load_act_policy(pretrained_dir, device=args.device)
        checkpoint_cameras = tuple(sorted(policy.config.image_features))
        if checkpoint_cameras != contract.camera_keys:
            raise ValueError(
                f"checkpoint cameras {checkpoint_cameras} do not match dataset cameras {contract.camera_keys}"
            )
        dataset = _load_validation_dataset(
            location=location,
            episodes=split.validation_episodes,
            policy=policy,
        )
        report = {
            "deployment_scope": common.DEPLOYMENT_SCOPE,
            "repo_id": contract.repo_id,
            "metadata_fingerprint": contract.metadata_fingerprint,
            "checkpoint": str(pretrained_dir),
            "validation_episodes": split.validation_episodes,
            "camera_keys": contract.camera_keys,
            "fps": contract.fps,
            "action_semantics": contract.action_semantics,
            "dataset_audit": audit_report,
            **evaluate(
                policy=policy,
                dataset=dataset,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                maximum_batches=args.maximum_batches,
                actions_per_inference=(None if args.actions_per_inference == 0 else args.actions_per_inference),
            ),
        }
        output = args.output or (run_root / "offline_eval.json")
        common.write_json(output.expanduser().resolve(), report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
        print("ACT_OFFLINE_EVAL_OK")
        return 0
    except Exception as exc:
        print(f"ACT_OFFLINE_EVAL_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
