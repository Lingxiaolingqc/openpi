"""Replay native expert joint targets through the learned-policy action path."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import traceback

import h5py
import numpy as np

try:
    from examples.so101 import convert_leisaac_hdf5_to_lerobot as converter
except ModuleNotFoundError:
    import convert_leisaac_hdf5_to_lerobot as converter


def summarize_tracking_errors(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return per-joint MAE/RMSE/max and global RMSE for replay state errors."""

    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[0] < 1:
        raise ValueError(f"Tracking errors must have shape (steps, joints), got {errors.shape}")
    return (
        np.mean(np.abs(errors), axis=0),
        np.sqrt(np.mean(np.square(errors), axis=0)),
        np.max(np.abs(errors), axis=0),
        float(np.sqrt(np.mean(np.square(errors)))),
    )


def first_divergence_step(errors: np.ndarray, threshold_rad: float) -> int | None:
    """Return the first transition whose largest joint error exceeds the threshold."""

    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2:
        raise ValueError(f"Tracking errors must be two-dimensional, got {errors.shape}")
    if threshold_rad <= 0.0:
        raise ValueError("Divergence threshold must be positive")
    divergent = np.flatnonzero(np.max(np.abs(errors), axis=1) > threshold_rad)
    return int(divergent[0]) if len(divergent) else None


def clip_joint_targets(
    targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip joint targets and report per-joint counts and maximum corrections."""

    targets = np.asarray(targets, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32).reshape(1, -1)
    upper = np.asarray(upper, dtype=np.float32).reshape(1, -1)
    if targets.ndim != 2 or targets.shape[1:] != lower.shape[1:] or lower.shape != upper.shape:
        raise ValueError("Target and joint-limit dimensions do not match")
    clipped = np.clip(targets, lower, upper)
    correction = np.abs(targets - clipped)
    return clipped, np.count_nonzero(correction, axis=0), np.max(correction, axis=0)


def _rounded(values: np.ndarray, digits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in np.asarray(values).reshape(-1))


def _build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets_root", default=os.environ.get("LEISAAC_ASSETS_ROOT"))
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--source_frame_index", type=int, default=0)
    parser.add_argument("--maximum_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--divergence_threshold_rad", type=float, default=0.05)
    parser.add_argument(
        "--joint_damping",
        type=float,
        help="Override all robot joint damping values before reset; use 10.0 to match the polar expert.",
    )
    parser.add_argument(
        "--disable_robot_gravity",
        action="store_true",
        help="Disable robot-link gravity to match the polar state-machine collection environment.",
    )
    parser.add_argument(
        "--no_clip_actions",
        action="store_true",
        help="Send raw expert targets instead of matching rollout soft-limit clipping.",
    )
    parser.add_argument(
        "--require_tracking_match",
        action="store_true",
        help="Exit nonzero when any replay transition exceeds the divergence threshold.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    args = parser.parse_args()
    if args.enable_cameras and args.rendering_mode is None:
        args.rendering_mode = "performance"
    if not args.headless or not args.enable_cameras:
        parser.error("Expert joint replay requires --headless --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")
    if args.episode_index < 0 or args.source_frame_index < 0:
        parser.error("episode and source-frame indices must be non-negative")
    if args.maximum_steps < 1 or args.log_every < 1 or args.divergence_threshold_rad <= 0.0:
        parser.error("step counts and divergence threshold must be positive")
    if args.joint_damping is not None and args.joint_damping < 0.0:
        parser.error("--joint_damping must be non-negative")
    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    episodes, skipped_failures, file_count = converter.discover_successful_episodes(args.input_path)
    if args.episode_index >= len(episodes):
        raise IndexError(f"episode-index {args.episode_index} is outside {len(episodes)} successful episodes")
    episode = episodes[args.episode_index]
    if args.source_frame_index + 1 >= episode.num_samples:
        raise IndexError(
            f"source-frame-index {args.source_frame_index} leaves no transition in {episode.num_samples} frames"
        )
    transition_count = min(args.maximum_steps, episode.num_samples - args.source_frame_index - 1)
    frame_stop = args.source_frame_index + transition_count
    with h5py.File(episode.path, "r") as h5_file:
        demo = h5_file["data"][episode.name]
        teacher_states = np.asarray(
            demo["obs/joint_pos"][args.source_frame_index : frame_stop + 1],
            dtype=np.float32,
        )
        raw_actions = np.asarray(
            demo["actions"][args.source_frame_index:frame_stop],
            dtype=np.float32,
        )
        teacher_cube_start = np.asarray(
            demo["diagnostics/cube_pose"][args.source_frame_index, :3],
            dtype=np.float32,
        )

    print("RED_CUBE_TO_BOX_EXPERT_JOINT_REPLAY_PHASE=before_launcher", flush=True)
    print(f"source_file_count: {file_count}", flush=True)
    print(f"skipped_failed_episode_count: {skipped_failures}", flush=True)
    print(f"selected_episode: {episode.path}:{episode.name}", flush=True)
    print(f"source_frame_index: {args.source_frame_index}", flush=True)
    print(f"requested_transitions: {transition_count}", flush=True)
    print(f"clip_actions: {not args.no_clip_actions}", flush=True)
    print(f"joint_damping_override: {args.joint_damping}", flush=True)
    print(f"disable_robot_gravity: {args.disable_robot_gravity}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    status = 1
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        import leisaac.tasks  # noqa: F401
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
        import red_cube_to_box_task
        import torch

        env_cfg = parse_env_cfg(red_cube_to_box_task.TASK_ID, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        if args.disable_robot_gravity:
            env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        env_cfg.seed = args.seed
        env_cfg.recorders = None
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        env = gym.make(red_cube_to_box_task.TASK_ID, cfg=env_cfg).unwrapped
        robot = env.scene["robot"]
        cube = env.scene["cube"]
        joint_ids = [list(robot.data.joint_names).index(name) for name in converter.JOINT_NAMES]
        soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()
        if args.joint_damping is not None:
            robot.write_joint_damping_to_sim(damping=args.joint_damping)
        if args.no_clip_actions:
            replay_actions = raw_actions.copy()
            clip_count_by_joint = np.zeros(len(converter.JOINT_NAMES), dtype=np.int64)
            max_clip_by_joint = np.zeros(len(converter.JOINT_NAMES), dtype=np.float32)
        else:
            replay_actions, clip_count_by_joint, max_clip_by_joint = clip_joint_targets(
                raw_actions,
                soft_limits[:, 0],
                soft_limits[:, 1],
            )

        env.reset()
        initial_live = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().copy()
        initial_cube = cube.data.root_pos_w[0].detach().cpu().numpy().copy()
        initial_error = initial_live - teacher_states[0]
        cube_initial_error = initial_cube - teacher_cube_start
        print("RED_CUBE_TO_BOX_EXPERT_JOINT_REPLAY_ENV_CREATED_OK", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"joint_names: {converter.JOINT_NAMES}", flush=True)
        print(f"soft_joint_lower_rad: {_rounded(soft_limits[:, 0])}", flush=True)
        print(f"soft_joint_upper_rad: {_rounded(soft_limits[:, 1])}", flush=True)
        print(f"replay_initial_live_joint_rad: {_rounded(initial_live)}", flush=True)
        print(f"replay_initial_teacher_joint_rad: {_rounded(teacher_states[0])}", flush=True)
        print(f"replay_initial_joint_error_rad: {_rounded(initial_error)}", flush=True)
        print(f"replay_initial_cube_error_w: {_rounded(cube_initial_error)}", flush=True)
        print(f"replay_action_clip_count_by_joint: {_rounded(clip_count_by_joint, 0)}", flush=True)
        print(f"replay_action_max_clip_by_joint_rad: {_rounded(max_clip_by_joint)}", flush=True)
        print("RED_CUBE_TO_BOX_EXPERT_JOINT_REPLAY_PHASE=replaying", flush=True)

        errors: list[np.ndarray] = []
        first_divergence: int | None = None
        rewards_finite = True
        unexpected_reset = False
        with torch.inference_mode():
            for transition_index, action_np in enumerate(replay_actions):
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
                action = torch.as_tensor(action_np, dtype=torch.float32, device=env.device).unsqueeze(0)
                step_result = env.step(action)
                live_next = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().copy()
                written_target = robot.data.joint_pos_target[0, joint_ids].detach().cpu().numpy().copy()
                teacher_next = teacher_states[transition_index + 1]
                error = live_next - teacher_next
                errors.append(error)
                max_error = float(np.max(np.abs(error)))
                if first_divergence is None and max_error > args.divergence_threshold_rad:
                    first_divergence = transition_index
                should_log = (
                    transition_index < 5
                    or transition_index % args.log_every == 0
                    or transition_index == first_divergence
                    or transition_index == transition_count - 1
                )
                if should_log:
                    source_index = args.source_frame_index + transition_index
                    print(
                        "expert_joint_replay:"
                        f"transition={transition_index}:source_frame={source_index}:"
                        f"raw_action_rad={_rounded(raw_actions[transition_index])}:"
                        f"replay_action_rad={_rounded(action_np)}:"
                        f"written_target_rad={_rounded(written_target)}:"
                        f"teacher_next_rad={_rounded(teacher_next)}:"
                        f"live_next_rad={_rounded(live_next)}:"
                        f"tracking_error_rad={_rounded(error)}:max_error_rad={max_error:.6f}",
                        flush=True,
                    )
                rewards_finite &= bool(torch.isfinite(step_result[1]).all())
                unexpected_reset |= bool(step_result[2].any()) or bool(step_result[3].any())

        error_array = np.asarray(errors, dtype=np.float64)
        mae, rmse_by_joint, maximum, global_rmse = summarize_tracking_errors(error_array)
        tracking_match = first_divergence is None
        print(f"replayed_transitions: {len(error_array)}", flush=True)
        print(f"replay_tracking_mae_by_joint_rad: {_rounded(mae)}", flush=True)
        print(f"replay_tracking_rmse_by_joint_rad: {_rounded(rmse_by_joint)}", flush=True)
        print(f"replay_tracking_max_by_joint_rad: {_rounded(maximum)}", flush=True)
        print(f"replay_tracking_global_rmse_rad: {global_rmse:.6f}", flush=True)
        print(f"replay_first_divergence_transition: {first_divergence}", flush=True)
        print(f"replay_divergence_threshold_rad: {args.divergence_threshold_rad:.6f}", flush=True)
        print(f"replay_rewards_finite: {rewards_finite}", flush=True)
        print(f"replay_unexpected_reset: {unexpected_reset}", flush=True)
        print(f"replay_tracking_match: {tracking_match}", flush=True)
        if args.require_tracking_match and not tracking_match:
            raise RuntimeError(
                f"Expert joint replay diverged at transition {first_divergence} "
                f"with threshold {args.divergence_threshold_rad:.6f} rad"
            )
        print("RED_CUBE_TO_BOX_EXPERT_JOINT_REPLAY_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("RED_CUBE_TO_BOX_EXPERT_JOINT_REPLAY_FAILED", flush=True)
    finally:
        print("RED_CUBE_TO_BOX_EXPERT_JOINT_REPLAY_PHASE=immediate_close", flush=True)
        if env is not None:
            env.close()
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
