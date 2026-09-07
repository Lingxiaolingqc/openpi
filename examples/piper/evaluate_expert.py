"""Evaluate the PiPER scripted expert over deterministic randomized seeds."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from examples.piper import expert
from examples.piper import mujoco_env


def evaluate(*, model_dir: Path | None, episodes: int, start_seed: int) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    reports = []
    with mujoco_env.PiperRedCubeToBoxEnv(model_dir=model_dir, render=False) as env:
        for seed in range(start_seed, start_seed + episodes):
            report = expert.run_expert_episode(env, seed=seed)
            reports.append(report)
            print(json.dumps(report.__dict__, sort_keys=True), flush=True)
    successes = sum(report.success for report in reports)
    failure_reasons = Counter(report.failure_reason for report in reports if not report.success)
    summary = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "failure_reasons": dict(sorted(failure_reasons.items())),
    }
    print(json.dumps(summary, sort_keys=True))
    sentinel = "PIPER_EXPERT_EVAL_OK" if successes / episodes >= 0.95 else "PIPER_EXPERT_EVAL_FAILED"
    print(sentinel)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1000)
    args = parser.parse_args()
    summary = evaluate(model_dir=args.model_dir, episodes=args.episodes, start_seed=args.start_seed)
    return 0 if summary["success_rate"] >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
