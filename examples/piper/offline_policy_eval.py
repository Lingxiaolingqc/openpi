"""Measure first-action error on the held-out PiPER simulation episodes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from examples.piper import contract
from examples.piper import convert_hdf5_to_lerobot
from examples.piper.policy_rollout import WebsocketPiperPolicy
from examples.piper.policy_rollout import validate_action_chunk


def action_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != contract.ACTION_DIM:
        raise ValueError(
            f"predictions and targets must have matching shape (N, {contract.ACTION_DIM}), "
            f"got {predictions.shape} and {targets.shape}"
        )
    if predictions.shape[0] < 1 or not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("offline metrics require at least one finite prediction and target")
    error = predictions - targets
    return {
        "samples": int(predictions.shape[0]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae_by_action": dict(zip(contract.ACTION_LAYOUT, np.mean(np.abs(error), axis=0).tolist(), strict=True)),
        "rmse_by_action": dict(
            zip(contract.ACTION_LAYOUT, np.sqrt(np.mean(np.square(error), axis=0)).tolist(), strict=True)
        ),
    }


def evaluate(
    *,
    input_path: Path,
    host: str,
    port: int,
    output_path: Path,
    split_seed: int = 2026,
    train_fraction: float = 0.8,
    maximum_frames: int = 400,
) -> dict:
    """Query a live policy on deterministic held-out frames and write metrics."""

    if maximum_frames < 1:
        raise ValueError("maximum_frames must be positive")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing offline report: {output_path}")
    episodes = convert_hdf5_to_lerobot.discover_episodes(input_path)
    _, validation = convert_hdf5_to_lerobot.split_episodes(
        episodes, train_fraction=train_fraction, split_seed=split_seed
    )
    validation_frames = sum(episode.frames for episode in validation)
    stride = max(1, math.ceil(validation_frames / maximum_frames))
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sampled_seeds: list[int] = []
    global_index = 0
    policy = WebsocketPiperPolicy(host=host, port=port)
    metadata = policy.metadata
    try:
        for episode in validation:
            reset = getattr(policy, "reset", None)
            if callable(reset):
                reset()
            episode_sampled = False
            with h5py.File(episode.path, "r") as file:
                states = file["observations/state"]
                images = file["observations/image"]
                actions = file["actions/absolute_target"]
                for frame_index in range(episode.frames):
                    should_sample = global_index % stride == 0 and len(predictions) < maximum_frames
                    global_index += 1
                    if not should_sample:
                        continue
                    response = policy.infer(
                        {
                            "images/base": np.asarray(images[frame_index], dtype=np.uint8),
                            "state": np.asarray(states[frame_index], dtype=np.float32),
                            "prompt": contract.TASK_PROMPT,
                        }
                    )
                    prediction = validate_action_chunk(response, minimum_actions=contract.ACTION_HORIZON)[0]
                    predictions.append(prediction)
                    targets.append(np.asarray(actions[frame_index], dtype=np.float32))
                    episode_sampled = True
            if episode_sampled:
                sampled_seeds.append(episode.seed)
    finally:
        policy.close()
    summary = {
        "metric_scope": "offline_first_action_only",
        "closed_loop_success_claimed": False,
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "validation_episodes": len(validation),
        "validation_frames": validation_frames,
        "sample_stride": stride,
        "sampled_episode_seeds": sampled_seeds,
        "server_metadata": metadata,
        **action_metrics(np.stack(predictions), np.stack(targets)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"offline_report:{output_path}")
    print("PIPER_OFFLINE_ACTION_EVAL_OK")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--maximum-frames", type=int, default=400)
    args = parser.parse_args()
    evaluate(
        input_path=args.input_path,
        host=args.host,
        port=args.port,
        output_path=args.output_path,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        maximum_frames=args.maximum_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
