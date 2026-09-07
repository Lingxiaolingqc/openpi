"""Collect deterministic PiPER MuJoCo demonstrations with the scripted expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from examples.piper import contract
from examples.piper import expert
from examples.piper import mujoco_env
from examples.piper import record


def _menagerie_revision(model_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(model_dir.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def collect(
    *,
    output_dir: Path,
    episodes: int,
    start_seed: int,
    model_dir: Path | None,
    keep_failures: bool,
) -> dict:
    if episodes < 1 or start_seed < 0:
        raise ValueError("episodes must be positive and start_seed must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    successes = 0
    attempts = 0
    with mujoco_env.PiperRedCubeToBoxEnv(model_dir=model_dir, render=True) as env:
        revision = _menagerie_revision(env.model_dir)
        seed = start_seed
        while successes < episodes:
            if attempts >= episodes * 10:
                raise RuntimeError(
                    f"expert produced only {successes}/{episodes} successful episodes after {attempts} attempts"
                )
            attempts += 1
            destination = output_dir / f"episode_{successes:06d}_seed_{seed:08d}.h5"
            temporary = output_dir / f"attempt_{attempts:06d}_seed_{seed:08d}.h5"
            metadata = record.EpisodeMetadata(seed=seed, model_sha256=env.model_sha256, menagerie_revision=revision)
            recorder = record.PiperEpisodeRecorder(temporary, metadata)
            try:
                report = expert.run_expert_episode(
                    env,
                    seed=seed,
                    record_step=lambda observation, action, phase, recorder=recorder: recorder.append(
                        observation, action, phase
                    ),
                )
                recorder.finalize(
                    success=report.success,
                    final_phase=report.final_phase,
                    failure_reason=report.failure_reason,
                )
            except Exception:
                recorder.abort()
                raise
            report_dict = {
                "seed": seed,
                "success": report.success,
                "steps": report.steps,
                "final_phase": report.final_phase,
                "failure_reason": report.failure_reason,
                "file": str(temporary),
            }
            if report.success:
                temporary.replace(destination)
                report_dict["file"] = str(destination)
                successes += 1
            elif not keep_failures:
                temporary.unlink()
            reports.append(report_dict)
            print(json.dumps(report_dict, sort_keys=True), flush=True)
            seed += 1
    summary = {
        "requested_successes": episodes,
        "successful_episodes": successes,
        "attempts": attempts,
        "start_seed": start_seed,
        "next_seed": seed,
        "task": contract.TASK_PROMPT,
        "reports": reports,
    }
    summary_path = output_dir / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"collection_summary:{summary_path}")
    print("PIPER_EXPERT_COLLECTION_OK")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--keep-failures", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    collect(
        output_dir=args.output_dir,
        episodes=args.episodes,
        start_seed=args.start_seed,
        model_dir=args.model_dir,
        keep_failures=args.keep_failures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
