"""Run simulation-only π0.5 action chunks in the PiPER MuJoCo environment."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import imageio.v2 as imageio
import numpy as np

from examples.piper import contract
from examples.piper import mujoco_env


class PolicyLike(Protocol):
    @property
    def metadata(self) -> dict: ...

    def infer(self, observation: dict) -> dict: ...

    def close(self) -> None: ...


class WebsocketPiperPolicy:
    """Strict simulation-only wrapper around openpi-client."""

    def __init__(self, *, host: str, port: int) -> None:
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port, protocol_version=1)
        self._metadata = dict(self._client.get_server_metadata())
        if self._metadata.get("deployment_scope") != "simulation-only":
            self.close()
            raise ValueError("PiPER rollout requires server deployment_scope='simulation-only'")
        if self._metadata.get("real_robot_deployment_allowed") is not False:
            self.close()
            raise ValueError("PiPER simulation server must advertise real_robot_deployment_allowed=false")
        if self._metadata.get("robot_type") != contract.ROBOT_TYPE:
            self.close()
            raise ValueError(f"server robot_type must be {contract.ROBOT_TYPE!r}")

    @property
    def metadata(self) -> dict:
        return self._metadata

    def infer(self, observation: dict) -> dict:
        return self._client.infer(observation)

    def reset(self) -> None:
        if self._metadata.get("supports_remote_reset", False):
            response = self._client.infer({"__reset__": True})
            if response.get("reset_ack") is not True:
                raise RuntimeError(f"policy server did not acknowledge reset: {response}")

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class RolloutReport:
    seed: int
    success: bool
    steps: int
    failure_reason: str | None
    video: str | None


def validate_action_chunk(response: dict, *, minimum_actions: int) -> np.ndarray:
    if "actions" not in response:
        raise ValueError(f"policy response has no actions key: {sorted(response)}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < minimum_actions or actions.shape[1] != contract.ACTION_DIM:
        raise ValueError(
            f"expected actions shape (at least {minimum_actions}, {contract.ACTION_DIM}), got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("policy action chunk contains NaN or infinity")
    for action in actions[:minimum_actions]:
        contract.validate_action(action)
    return actions[:minimum_actions]


def run_policy_episode(
    env,
    policy: PolicyLike,
    *,
    seed: int,
    actions_per_inference: int = contract.ACTION_HORIZON,
    video_path: Path | None = None,
) -> RolloutReport:
    observation, _ = env.reset(seed=seed)
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    queue: deque[np.ndarray] = deque()
    frames = [observation.image]
    failure_reason = None
    while env.step_count < env.maximum_steps:
        if not queue:
            try:
                response = policy.infer(
                    {
                        "images/base": observation.image,
                        "state": observation.state,
                        "prompt": contract.TASK_PROMPT,
                    }
                )
                queue.extend(validate_action_chunk(response, minimum_actions=actions_per_inference))
            except Exception as exc:
                queue.clear()
                env.hold_measured_pose()
                failure_reason = f"policy_fault:{type(exc).__name__}:{exc}"
                break
        action = queue.popleft()
        try:
            result = env.step(action)
        except Exception as exc:
            queue.clear()
            env.hold_measured_pose()
            failure_reason = f"execution_fault:{type(exc).__name__}:{exc}"
            break
        observation = result.observation
        frames.append(observation.image)
        if result.terminated:
            if video_path is not None:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_path, frames, fps=contract.CONTROL_HZ)
            return RolloutReport(
                seed=seed,
                success=True,
                steps=env.step_count,
                failure_reason=None,
                video=str(video_path) if video_path else None,
            )
        if result.truncated:
            failure_reason = "environment_truncated"
            break
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, frames, fps=contract.CONTROL_HZ)
    return RolloutReport(
        seed=seed,
        success=False,
        steps=env.step_count,
        failure_reason=failure_reason or "task_not_successful",
        video=str(video_path) if video_path else None,
    )


def evaluate(
    *,
    model_dir: Path | None,
    host: str,
    port: int,
    episodes: int,
    start_seed: int,
    output_dir: Path,
    actions_per_inference: int,
    seeds: list[int] | None = None,
    minimum_success_rate: float = 0.60,
) -> dict:
    episode_seeds = list(range(start_seed, start_seed + episodes)) if seeds is None else list(seeds)
    if not episode_seeds:
        raise ValueError("episodes must be positive")
    if len(set(episode_seeds)) != len(episode_seeds) or any(seed < 0 for seed in episode_seeds):
        raise ValueError("rollout seeds must be unique non-negative integers")
    if not 0.0 <= minimum_success_rate <= 1.0:
        raise ValueError("minimum_success_rate must be in [0, 1]")
    if actions_per_inference != contract.ACTION_HORIZON:
        raise ValueError(f"PiPER v1 executes exactly {contract.ACTION_HORIZON} actions per inference")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty rollout directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = WebsocketPiperPolicy(host=host, port=port)
    reports = []
    try:
        with mujoco_env.PiperRedCubeToBoxEnv(model_dir=model_dir, render=True) as env:
            for seed in episode_seeds:
                report = run_policy_episode(
                    env,
                    policy,
                    seed=seed,
                    actions_per_inference=actions_per_inference,
                    video_path=output_dir / f"seed_{seed:08d}.mp4",
                )
                reports.append(report)
                print(json.dumps(asdict(report), sort_keys=True), flush=True)
    finally:
        policy.close()
    successes = sum(report.success for report in reports)
    success_rate = successes / len(episode_seeds)
    summary = {
        "episodes": len(episode_seeds),
        "seeds": episode_seeds,
        "successes": successes,
        "success_rate": success_rate,
        "quality_target": minimum_success_rate,
        "quality_accepted": success_rate >= minimum_success_rate,
        "server_metadata": policy.metadata,
        "reports": [asdict(report) for report in reports],
    }
    summary_path = output_dir / "rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"rollout_summary:{summary_path}")
    print("PIPER_POLICY_ROLLOUT_OK" if summary["quality_accepted"] else "PIPER_POLICY_ROLLOUT_FAILED")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", help="explicit seeds, useful for overfit-set evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actions-per-inference", type=int, default=contract.ACTION_HORIZON)
    parser.add_argument("--minimum-success-rate", type=float, default=0.60)
    args = parser.parse_args()
    summary = evaluate(
        model_dir=args.model_dir,
        host=args.host,
        port=args.port,
        episodes=args.episodes,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
        actions_per_inference=args.actions_per_inference,
        seeds=args.seeds,
        minimum_success_rate=args.minimum_success_rate,
    )
    return 0 if summary["quality_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
