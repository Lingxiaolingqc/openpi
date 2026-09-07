"""Replay one audited PiPER HDF5 episode and verify deterministic state alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from examples.piper import convert_hdf5_to_lerobot
from examples.piper import mujoco_env


def replay(
    episode_path: Path,
    *,
    model_dir: Path | None,
    video_path: Path | None,
    state_tolerance: float,
) -> dict:
    reference = convert_hdf5_to_lerobot.validate_episode(episode_path)
    with h5py.File(reference.path, "r") as file:
        expected_states = np.asarray(file["observations/state"][:], dtype=np.float32)
        actions = np.asarray(file["actions/absolute_target"][:], dtype=np.float32)
    frames = []
    maximum_state_error = 0.0
    with mujoco_env.PiperRedCubeToBoxEnv(model_dir=model_dir, render=video_path is not None) as env:
        observation, _ = env.reset(seed=reference.seed)
        for index, action in enumerate(actions):
            error = float(np.max(np.abs(observation.state - expected_states[index])))
            maximum_state_error = max(maximum_state_error, error)
            if error > state_tolerance:
                raise RuntimeError(
                    f"replay diverged before frame {index}: maximum state error {error:.6f} > {state_tolerance:.6f}"
                )
            if video_path is not None:
                frames.append(observation.image)
            result = env.step(action, rate_limit=False)
            observation = result.observation
        success = bool(result.info["success"])
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=20)
    summary = {
        "episode": str(reference.path),
        "frames": reference.frames,
        "maximum_state_error": maximum_state_error,
        "success": success,
        "video": str(video_path) if video_path else None,
    }
    print(summary)
    print("PIPER_EPISODE_REPLAY_OK")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--state-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    replay(
        args.episode,
        model_dir=args.model_dir,
        video_path=args.video,
        state_tolerance=args.state_tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
