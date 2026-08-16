"""Evaluate an OpenPI SO-101 checkpoint in the RedCubeToBox simulation."""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np


TASK_PROMPT = "Pick up the red cube and place it inside the green box."
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def normalize_action_chunk(value: object, *, num_envs: int = 1, action_dim: int = 6) -> np.ndarray:
    """Return policy actions as a finite ``(horizon, num_envs, action_dim)`` float32 array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim == 2:
        actions = actions[:, None, :]
    expected_tail = (num_envs, action_dim)
    if actions.ndim != 3 or actions.shape[1:] != expected_tail:
        raise ValueError(
            "Expected policy action chunk shape "
            f"(horizon, {num_envs}, {action_dim}), got {tuple(actions.shape)}"
        )
    if actions.shape[0] < 1:
        raise ValueError("Policy returned an empty action chunk")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned a non-finite action chunk")
    return actions


def clip_action_chunk(
    actions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Clip to simulator soft limits while returning auditable violation statistics."""

    lower = np.asarray(lower, dtype=np.float32).reshape(1, 1, -1)
    upper = np.asarray(upper, dtype=np.float32).reshape(1, 1, -1)
    if lower.shape[-1] != actions.shape[-1] or upper.shape[-1] != actions.shape[-1]:
        raise ValueError("Joint-limit dimension does not match policy action dimension")
    clipped = np.clip(actions, lower, upper)
    violation = np.abs(actions - clipped)
    maximum_violation = float(violation.max()) if violation.size else 0.0
    return clipped, int(np.count_nonzero(violation)), maximum_violation


def reset_with_camera_warmup(
    env,
    robot,
    *,
    joint_ids: list[int],
    warmup_steps: int,
    dynamic_gripper_reset=None,
):
    """Reset and advance held simulation steps before exposing a refreshed camera frame."""

    if warmup_steps < 0:
        raise ValueError("camera warmup steps must be non-negative")
    observations, _ = env.reset()
    for warmup_index in range(warmup_steps):
        hold_action = robot.data.joint_pos[:, joint_ids].clone()
        if dynamic_gripper_reset is not None:
            dynamic_gripper_reset(env, "so101leader")
        observations, _, terminated, truncated, _ = env.step(hold_action)
        if bool(terminated.any()) or bool(truncated.any()):
            raise RuntimeError(
                f"Environment terminated/truncated during camera warmup step {warmup_index + 1}"
            )
    print(f"policy_reset_camera_warmup_steps: {warmup_steps}", flush=True)
    return observations


class OpenPISO101Client:
    """Small adapter around openpi-client with LeIsaac's audited motor conversion."""

    def __init__(self, *, host: str, port: int, camera_names: tuple[str, ...], prompt: str) -> None:
        from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot
        from leisaac.utils.robot_utils import convert_lerobot_action_to_leisaac
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._camera_names = camera_names
        self._prompt = prompt
        self._state_to_motor = convert_leisaac_action_to_lerobot
        self._action_to_sim = convert_lerobot_action_to_leisaac

    @property
    def server_metadata(self) -> dict:
        return self._client.get_server_metadata()

    def get_action(self, observation: dict) -> np.ndarray:
        request = {
            f"images/{name}": observation[name][0].detach().cpu().numpy().astype(np.uint8, copy=False)
            for name in self._camera_names
        }
        request["state"] = self._state_to_motor(observation["joint_pos"]).squeeze(0).astype(np.float32)
        request["prompt"] = self._prompt
        response = self._client.infer(request)
        if "actions" not in response:
            raise ValueError(f"Policy response does not contain actions: {sorted(response)}")
        motor_actions = np.asarray(response["actions"], dtype=np.float32)
        simulator_actions = self._action_to_sim(motor_actions)
        return np.asarray(simulator_actions, dtype=np.float32)[:, None, :]

    def close(self) -> None:
        websocket = getattr(self._client, "_ws", None)
        if websocket is not None:
            websocket.close()


def _build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets_root", default=os.environ.get("LEISAAC_ASSETS_ROOT"))
    parser.add_argument("--policy_host", default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=18000)
    parser.add_argument("--prompt", default=TASK_PROMPT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maximum_steps", type=int, default=2400)
    parser.add_argument("--actions_per_inference", type=int, default=10)
    parser.add_argument("--success_stable_steps", type=int, default=30)
    parser.add_argument("--minimum_lift_m", type=float, default=0.03)
    parser.add_argument("--minimum_success_rate", type=float, default=1.0)
    parser.add_argument(
        "--reset_camera_warmup_steps",
        type=int,
        default=0,
        help=(
            "Held simulation steps after reset before the first observation is sent to the policy. "
            "Use 1 only after confirming that reset observation frame 0 is stale."
        ),
    )
    parser.add_argument("--record_dir", type=Path)
    parser.add_argument("--record_every", type=int, default=4)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _numpy_row(tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy().copy()


def _rounded(values, digits: int = 5) -> tuple[float, ...]:
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    return tuple(round(float(value), digits) for value in array.reshape(-1))


class _RolloutRecorder:
    """Write sampled JPEG frames, JSONL diagnostics, and an offline viewer."""

    def __init__(
        self,
        root: Path,
        *,
        seed: int,
        episode_index: int,
        record_every: int,
        jpeg_quality: int,
        image_class,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.run_dir = root / f"policy-seed{seed}-episode{episode_index:03d}-{timestamp}-pid{os.getpid()}"
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.result_path = self.run_dir / "result.json"
        self.viewer_path = self.run_dir / "index.html"
        self._record_every = record_every
        self._jpeg_quality = jpeg_quality
        self._image_class = image_class
        self._frames: list[dict[str, object]] = []

    def capture(
        self,
        *,
        step: int,
        observations: dict,
        env,
        action: np.ndarray | None,
        inference_index: int,
        stable_steps: int,
        force: bool = False,
    ) -> None:
        if not force and step % self._record_every:
            return
        if self._frames and self._frames[-1]["step"] == step:
            return
        front = observations["policy"]["front"]
        if front.ndim != 4 or front.shape[0] != 1 or front.shape[-1] != 3:
            raise RuntimeError(f"Unexpected front image shape: {tuple(front.shape)}")
        image = front[0].detach().cpu().numpy()
        relative_path = Path("frames") / f"frame_{len(self._frames):06d}.jpg"
        self._image_class.fromarray(image).save(
            self.run_dir / relative_path,
            format="JPEG",
            quality=self._jpeg_quality,
        )
        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        robot = env.scene["robot"]
        record = {
            "frame": relative_path.as_posix(),
            "step": step,
            "inference_index": inference_index,
            "stable_steps": stable_steps,
            "joint_pos_rad": _rounded(robot.data.joint_pos[0]),
            "action_rad": None if action is None else _rounded(action),
            "cube_pos_w": _rounded(cube.data.root_pos_w[0]),
            "cube_offset_from_box": _rounded(cube.data.root_pos_w[0] - floor.data.root_pos_w[0]),
            "pick_cube": bool(observations["subtask_terms"]["pick_cube"][0].item()),
        }
        self._frames.append(record)
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def finish(self, result: dict[str, object]) -> None:
        self.result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        payload = json.dumps(self._frames, allow_nan=False)
        self.viewer_path.write_text(
            """<!doctype html><meta charset=\"utf-8\"><title>SO-101 policy rollout</title>
<style>body{font-family:system-ui;background:#111;color:#eee;margin:24px}img{max-width:960px;width:100%}
pre{white-space:pre-wrap}input{width:min(960px,100%)}</style>
<h1>SO-101 RedCubeToBox policy rollout</h1><input id=slider type=range min=0 value=0 step=1>
<p id=label></p><img id=frame><pre id=data></pre><script>
const frames=__FRAMES__;const slider=document.getElementById('slider');slider.max=Math.max(0,frames.length-1);
function draw(){const x=frames[Number(slider.value)];if(!x)return;frame.src=x.frame;label.textContent=`frame ${slider.value}/${frames.length-1}, step ${x.step}`;data.textContent=JSON.stringify(x,null,2)}
slider.oninput=draw;draw();</script>""".replace("__FRAMES__", payload),
            encoding="utf-8",
        )
        print(f"policy_recording_dir: {self.run_dir}", flush=True)


def _validate_camera(observations: dict) -> tuple[str, ...]:
    policy = observations["policy"]
    camera_names = tuple(name for name in ("front", "wrist") if name in policy)
    if "front" not in camera_names:
        raise RuntimeError("RedCubeToBox policy observations do not contain the front camera")
    for name in camera_names:
        image = policy[name]
        image_min = int(image.min().item())
        image_max = int(image.max().item())
        image_mean = float(image.float().mean().item())
        print(
            f"policy_camera:{name}:shape={tuple(image.shape)}:dtype={image.dtype}:"
            f"min={image_min}:max={image_max}:mean={image_mean:.3f}",
            flush=True,
        )
        if image_max == 0:
            raise RuntimeError(f"Camera {name} returned an all-black frame")
    return camera_names


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.enable_cameras and args.rendering_mode is None:
        args.rendering_mode = "performance"
    if not args.headless or not args.enable_cameras:
        parser.error("Policy rollout requires --headless --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")
    if args.episodes < 1 or args.maximum_steps < 1 or args.actions_per_inference < 1:
        parser.error("episode, step, and action-horizon values must be positive")
    if args.success_stable_steps < 1 or args.record_every < 1:
        parser.error("stable-step and recording intervals must be positive")
    if args.reset_camera_warmup_steps < 0:
        parser.error("--reset_camera_warmup_steps must be non-negative")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        parser.error("--minimum_success_rate must be between 0 and 1")
    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    print("RED_CUBE_TO_BOX_POLICY_PHASE=before_launcher", flush=True)
    print(f"policy_endpoint: ws://{args.policy_host}:{args.policy_port}", flush=True)
    print(f"episodes: {args.episodes}", flush=True)
    print(f"maximum_steps: {args.maximum_steps}", flush=True)
    print(f"actions_per_inference: {args.actions_per_inference}", flush=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    status = 1
    env = None
    policy = None
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        import leisaac.tasks  # noqa: F401
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
        from PIL import Image
        import red_cube_to_box_task
        from red_cube_to_box_task import mdp
        import torch

        env_cfg = parse_env_cfg(red_cube_to_box_task.TASK_ID, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        env_cfg.seed = args.seed
        env_cfg.recorders = None
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        env = gym.make(red_cube_to_box_task.TASK_ID, cfg=env_cfg).unwrapped
        robot = env.scene["robot"]
        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        joint_ids = [list(robot.data.joint_names).index(name) for name in JOINT_NAMES]
        soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()

        observations = reset_with_camera_warmup(
            env,
            robot,
            joint_ids=joint_ids,
            warmup_steps=args.reset_camera_warmup_steps,
            dynamic_gripper_reset=(
                dynamic_reset_gripper_effort_limit_sim if env.cfg.dynamic_reset_gripper_effort_limit else None
            ),
        )
        camera_names = _validate_camera(observations)
        policy = OpenPISO101Client(
            host=args.policy_host,
            port=args.policy_port,
            camera_names=camera_names,
            prompt=args.prompt,
        )
        print("RED_CUBE_TO_BOX_POLICY_CONNECTED_OK", flush=True)
        print(f"policy_server_metadata: {policy.server_metadata}", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"joint_names: {JOINT_NAMES}", flush=True)
        print(f"soft_joint_lower_rad: {_rounded(soft_limits[:, 0])}", flush=True)
        print(f"soft_joint_upper_rad: {_rounded(soft_limits[:, 1])}", flush=True)
        print("RED_CUBE_TO_BOX_POLICY_PHASE=evaluating", flush=True)

        successes = 0
        episode_results: list[dict[str, object]] = []
        with torch.inference_mode():
            for episode_index in range(args.episodes):
                if episode_index:
                    observations = reset_with_camera_warmup(
                        env,
                        robot,
                        joint_ids=joint_ids,
                        warmup_steps=args.reset_camera_warmup_steps,
                        dynamic_gripper_reset=(
                            dynamic_reset_gripper_effort_limit_sim
                            if env.cfg.dynamic_reset_gripper_effort_limit
                            else None
                        ),
                    )
                recorder = None
                if args.record_dir is not None:
                    recorder = _RolloutRecorder(
                        args.record_dir.expanduser().resolve(),
                        seed=args.seed,
                        episode_index=episode_index,
                        record_every=args.record_every,
                        jpeg_quality=args.jpeg_quality,
                        image_class=Image,
                    )
                    recorder.capture(
                        step=0,
                        observations=observations,
                        env=env,
                        action=None,
                        inference_index=0,
                        stable_steps=0,
                        force=True,
                    )

                initial_cube_z = float(cube.data.root_pos_w[0, 2].item())
                ever_grasped = False
                ever_lifted = False
                stable_steps = 0
                completed_steps = 0
                inference_count = 0
                clip_count = 0
                maximum_clip_rad = 0.0
                inference_latencies_ms: list[float] = []
                rewards_finite = True
                unexpected_reset = False

                while (
                    completed_steps < args.maximum_steps
                    and stable_steps < args.success_stable_steps
                    and not unexpected_reset
                ):
                    inference_start = time.perf_counter()
                    raw_chunk = policy.get_action(observations["policy"])
                    inference_latency_ms = 1000.0 * (time.perf_counter() - inference_start)
                    inference_latencies_ms.append(inference_latency_ms)
                    action_chunk = normalize_action_chunk(raw_chunk)
                    action_chunk, chunk_clip_count, chunk_max_clip = clip_action_chunk(
                        action_chunk,
                        soft_limits[:, 0],
                        soft_limits[:, 1],
                    )
                    inference_count += 1
                    clip_count += chunk_clip_count
                    maximum_clip_rad = max(maximum_clip_rad, chunk_max_clip)
                    print(
                        "policy_inference:"
                        f"episode={episode_index}:index={inference_count}:step={completed_steps}:"
                        f"latency_ms={inference_latency_ms:.1f}:horizon={action_chunk.shape[0]}:"
                        f"action_min={float(action_chunk.min()):.5f}:action_max={float(action_chunk.max()):.5f}:"
                        f"clip_count={chunk_clip_count}:max_clip_rad={chunk_max_clip:.6f}",
                        flush=True,
                    )

                    steps_from_chunk = min(args.actions_per_inference, action_chunk.shape[0])
                    for action_index in range(steps_from_chunk):
                        if completed_steps >= args.maximum_steps or stable_steps >= args.success_stable_steps:
                            break
                        action_numpy = action_chunk[action_index]
                        action = torch.as_tensor(action_numpy, dtype=torch.float32, device=env.device)
                        if env.cfg.dynamic_reset_gripper_effort_limit:
                            dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
                        observations, rewards, terminated, truncated, _ = env.step(action)
                        completed_steps += 1
                        rewards_finite &= bool(torch.isfinite(rewards).all())
                        step_reset = bool(terminated.any()) or bool(truncated.any())
                        unexpected_reset |= step_reset
                        pick_cube = bool(observations["subtask_terms"]["pick_cube"][0].item())
                        ever_grasped |= pick_cube
                        cube_z = float(cube.data.root_pos_w[0, 2].item())
                        ever_lifted |= cube_z >= initial_cube_z + args.minimum_lift_m
                        inside = bool(mdp.cube_inside_target_box(env).all().item())
                        stable_steps = stable_steps + 1 if inside else 0
                        if recorder is not None:
                            recorder.capture(
                                step=completed_steps,
                                observations=observations,
                                env=env,
                                action=action_numpy[0],
                                inference_index=inference_count,
                                stable_steps=stable_steps,
                            )
                        if step_reset:
                            break

                settled_inside = stable_steps >= args.success_stable_steps
                success = bool(settled_inside and ever_lifted and rewards_finite and not unexpected_reset)
                successes += int(success)
                final_offset = _numpy_row(cube.data.root_pos_w - floor.data.root_pos_w)
                final_speed = float(torch.linalg.vector_norm(cube.data.root_lin_vel_w[0]).item())
                result = {
                    "episode": episode_index,
                    "success": success,
                    "settled_inside": settled_inside,
                    "ever_grasped": ever_grasped,
                    "ever_lifted": ever_lifted,
                    "rewards_finite": rewards_finite,
                    "unexpected_reset": unexpected_reset,
                    "completed_steps": completed_steps,
                    "inference_count": inference_count,
                    "mean_inference_latency_ms": float(np.mean(inference_latencies_ms)),
                    "policy_action_clip_count": clip_count,
                    "policy_action_max_clip_rad": maximum_clip_rad,
                    "cube_final_pos_w": list(_rounded(cube.data.root_pos_w[0])),
                    "cube_offset_from_box": final_offset.tolist(),
                    "cube_final_speed": final_speed,
                }
                episode_results.append(result)
                if recorder is not None:
                    recorder.capture(
                        step=completed_steps,
                        observations=observations,
                        env=env,
                        action=None,
                        inference_index=inference_count,
                        stable_steps=stable_steps,
                        force=True,
                    )
                    recorder.finish(result)
                print(
                    "policy_episode:"
                    f"index={episode_index}:success={success}:steps={completed_steps}:"
                    f"ever_grasped={ever_grasped}:ever_lifted={ever_lifted}:"
                    f"settled_inside={settled_inside}:clip_count={clip_count}:"
                    f"final_offset={tuple(round(float(x), 5) for x in final_offset)}",
                    flush=True,
                )

        success_rate = successes / args.episodes
        print(f"completed_episodes: {args.episodes}", flush=True)
        print(f"successful_episodes: {successes}", flush=True)
        print(f"success_rate: {success_rate:.3f}", flush=True)
        print(f"policy_episode_results: {json.dumps(episode_results, allow_nan=False)}", flush=True)
        if not math.isfinite(success_rate) or success_rate < args.minimum_success_rate:
            raise RuntimeError(
                f"Policy success rate {success_rate:.3f} is below required {args.minimum_success_rate:.3f}"
            )
        print("RED_CUBE_TO_BOX_POLICY_ROLLOUT_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("RED_CUBE_TO_BOX_POLICY_ROLLOUT_FAILED", flush=True)
    finally:
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
        print("RED_CUBE_TO_BOX_POLICY_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
