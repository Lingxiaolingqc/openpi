"""Compare a served SO-101 policy with an exact native HDF5 training sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

try:
    from examples.so101 import convert_leisaac_hdf5_to_lerobot as converter
except ModuleNotFoundError:
    import convert_leisaac_hdf5_to_lerobot as converter


TASK_PROMPT = "Pick up the red cube and place it inside the green box."


def compare_action_chunks(
    predicted: np.ndarray,
    expert: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return per-joint MAE/max error and global RMSE for aligned motor actions."""

    predicted = np.asarray(predicted, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    if predicted.shape != expert.shape or predicted.ndim != 2:
        raise ValueError(
            f"Predicted and expert chunks must have the same 2D shape, got {predicted.shape} and {expert.shape}"
        )
    error = predicted - expert
    return (
        np.mean(np.abs(error), axis=0),
        np.max(np.abs(error), axis=0),
        float(np.sqrt(np.mean(np.square(error)))),
    )


def _rounded(values: np.ndarray, digits: int = 4) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in np.asarray(values).reshape(-1))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=18000)
    parser.add_argument("--prompt", default=TASK_PROMPT)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--source-frame-index",
        type=int,
        default=1,
        help="Native HDF5 frame index; use 1 when conversion used --start-frame 1.",
    )
    parser.add_argument("--horizon", type=int, default=10)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.episode_index < 0 or args.source_frame_index < 0 or args.horizon < 1:
        raise ValueError("episode/frame indices must be non-negative and horizon must be positive")

    episodes, skipped_failures, file_count = converter.discover_successful_episodes(args.input_path)
    if args.episode_index >= len(episodes):
        raise IndexError(f"episode-index {args.episode_index} is outside {len(episodes)} successful episodes")
    episode = episodes[args.episode_index]
    frame_stop = min(episode.num_samples, args.source_frame_index + args.horizon)
    if args.source_frame_index >= frame_stop:
        raise IndexError(
            f"source-frame-index {args.source_frame_index} is outside episode length {episode.num_samples}"
        )

    with h5py.File(episode.path, "r") as h5_file:
        demo = h5_file["data"][episode.name]
        front = np.asarray(demo["obs/front"][args.source_frame_index], dtype=np.uint8)
        state_rad = np.asarray(demo["obs/joint_pos"][args.source_frame_index], dtype=np.float32)
        expert_rad = np.asarray(
            demo["actions"][args.source_frame_index:frame_stop],
            dtype=np.float32,
        )

    state_motor = converter.leisaac_radians_to_motor_degrees(state_rad[None])[0]
    expert_motor = converter.leisaac_radians_to_motor_degrees(expert_rad)

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    try:
        response = client.infer(
            {
                "images/front": front,
                "state": state_motor,
                "prompt": args.prompt,
            }
        )
    finally:
        websocket = getattr(client, "_ws", None)
        if websocket is not None:
            websocket.close()

    if "actions" not in response:
        raise ValueError(f"Policy response does not contain actions: {sorted(response)}")
    predicted_motor = np.asarray(response["actions"], dtype=np.float32)
    if predicted_motor.ndim != 2 or predicted_motor.shape[1] < len(converter.JOINT_NAMES):
        raise ValueError(f"Unexpected policy action shape: {predicted_motor.shape}")
    predicted_motor = predicted_motor[: len(expert_motor), : len(converter.JOINT_NAMES)]
    if len(predicted_motor) != len(expert_motor):
        raise ValueError(
            f"Policy returned {len(predicted_motor)} steps, but {len(expert_motor)} expert steps were requested"
        )

    mae_by_joint, max_error_by_joint, rmse = compare_action_chunks(predicted_motor, expert_motor)
    first_error = predicted_motor[0] - expert_motor[0]
    print("source_file_count:", file_count)
    print("skipped_failed_episode_count:", skipped_failures)
    print("selected_episode:", f"{episode.path}:{episode.name}")
    print("source_frame_index:", args.source_frame_index)
    print("compared_horizon:", len(expert_motor))
    print("joint_names:", converter.JOINT_NAMES)
    print("front_stats:", f"min={int(front.min())}:max={int(front.max())}:mean={float(front.mean()):.3f}")
    print("state_motor:", _rounded(state_motor))
    print("expert_first_action_motor:", _rounded(expert_motor[0]))
    print("policy_first_action_motor:", _rounded(predicted_motor[0]))
    print("first_action_error_motor:", _rounded(first_error))
    print("action_mae_by_joint_motor:", _rounded(mae_by_joint))
    print("action_max_error_by_joint_motor:", _rounded(max_error_by_joint))
    print("action_rmse_motor:", round(rmse, 4))
    print("RED_CUBE_TO_BOX_POLICY_HDF5_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
