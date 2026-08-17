"""Collect successful polar-expert trajectories into resumable HDF5 shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EXPERT_NAME = "autogen_polar_retreat_transport"
DEFAULT_EXPERT_VERSION = "so101-redcube-polar-s4-v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets_root", default=os.environ.get("LEISAAC_ASSETS_ROOT"))
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--successful_episodes", type=int, default=20)
    parser.add_argument("--maximum_attempts", type=int, default=50)
    parser.add_argument("--shard_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert_version", default=DEFAULT_EXPERT_VERSION)
    parser.add_argument(
        "--camera_refreshes_before_recording",
        type=int,
        default=0,
        help="Render and refresh camera sensors this many times after reset without stepping physics.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _existing_attempts(dataset_dir: Path) -> int:
    manifest = dataset_dir / "collection_manifest.json"
    if not manifest.exists():
        return 0
    return int(json.loads(manifest.read_text(encoding="utf-8")).get("attempts", 0))


def _numpy_row(tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy().copy()


def snapshot_pre_step(observations, robot, cube, *, joint_ids: list[int], gripper_body_index: int) -> dict:
    """Copy obs_t before stepping so mutable simulator buffers cannot shift alignment."""
    policy = observations["policy"]
    sample = {
        "obs/joint_pos": _numpy_row(robot.data.joint_pos[:, joint_ids]).astype(np.float32),
        "obs/front": _numpy_row(policy["front"]).astype(np.uint8),
        "diagnostics/joint_vel": _numpy_row(robot.data.joint_vel[:, joint_ids]).astype(np.float32),
        "diagnostics/cube_pose": np.concatenate(
            (_numpy_row(cube.data.root_pos_w), _numpy_row(cube.data.root_quat_w))
        ).astype(np.float32),
        "diagnostics/gripper_pose": np.concatenate(
            (
                _numpy_row(robot.data.body_pos_w[:, gripper_body_index]),
                _numpy_row(robot.data.body_quat_w[:, gripper_body_index]),
            )
        ).astype(np.float32),
    }
    if "wrist" in policy:
        sample["obs/wrist"] = _numpy_row(policy["wrist"]).astype(np.uint8)
    return sample


def attach_applied_joint_target(sample: dict, robot, *, joint_ids: list[int], timestamp: float) -> dict:
    """Pair pre-step obs_t with the six absolute targets written during that step."""
    result = dict(sample)
    result["actions"] = _numpy_row(robot.data.joint_pos_target[:, joint_ids]).astype(np.float32)
    result["timestamps"] = np.float64(timestamp)
    return result


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.enable_cameras and args.rendering_mode is None:
        args.rendering_mode = "performance"
    if args.successful_episodes < 1 or args.maximum_attempts < 1 or args.shard_size < 1:
        parser.error("episode and shard counts must be positive")
    if args.maximum_attempts < args.successful_episodes:
        parser.error("--maximum_attempts must be at least --successful_episodes")
    if args.camera_refreshes_before_recording < 0:
        parser.error("--camera_refreshes_before_recording must be non-negative")
    if not args.headless or not args.enable_cameras:
        parser.error("Dataset collection requires --headless --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")
    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    resumed_attempts = _existing_attempts(dataset_dir)
    effective_seed = args.seed + resumed_attempts
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    print("RED_CUBE_TO_BOX_DATASET_PHASE=before_launcher", flush=True)
    print(f"dataset_dir: {dataset_dir}", flush=True)
    print(f"target_successful_episodes: {args.successful_episodes}", flush=True)
    print(f"maximum_new_attempts: {args.maximum_attempts}", flush=True)
    print(f"resume_attempt_offset: {resumed_attempts}", flush=True)
    print(f"effective_seed: {effective_seed}", flush=True)
    print(f"camera_refreshes_before_recording: {args.camera_refreshes_before_recording}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    status = 1
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        import leisaac.tasks  # noqa: F401
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
        from examples.so101.utils.red_cube_to_box_camera import refresh_camera_observations_without_control
        from examples.so101.utils.red_cube_to_box_hdf5 import JOINT_NAMES
        from examples.so101.utils.red_cube_to_box_hdf5 import ShardedDatasetWriter
        import red_cube_to_box_task
        from red_cube_to_box_task.autogen_polar_retreat_transport_state_machine import (
            RedCubeToBoxAutogenPolarRetreatTransportStateMachine,
        )
        from red_cube_to_box_task.phase_aware_ik_action import configure_servo_ik_action
        import torch

        writer = ShardedDatasetWriter(
            dataset_dir,
            shard_size=args.shard_size,
            metadata={
                "expert": EXPERT_NAME,
                "expert_version": args.expert_version,
                "base_seed": args.seed,
                "assets_root": str(assets_root),
                "observation_alignment": "pre_step",
                "action_semantics": "absolute_joint_pos_target_written_by_action_terms",
                "camera_refreshes_before_recording": args.camera_refreshes_before_recording,
                "robot_gravity_disabled": True,
                "robot_joint_damping": 10.0,
                "action_target_soft_limit_clipped_before_apply": False,
            },
            resume=True,
        )
        target_total = args.successful_episodes

        env_cfg = parse_env_cfg(red_cube_to_box_task.TASK_ID, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101_state_machine")
        env_cfg.seed = effective_seed
        env_cfg.recorders = None
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        configure_servo_ik_action(env_cfg)
        env = gym.make(red_cube_to_box_task.TASK_ID, cfg=env_cfg).unwrapped
        state_machine = RedCubeToBoxAutogenPolarRetreatTransportStateMachine()
        state_machine.setup(env)
        robot = env.scene["robot"]
        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        joint_ids = [list(robot.data.joint_names).index(name) for name in JOINT_NAMES]
        gripper_body_index = list(robot.data.body_names).index("gripper")
        camera_names = tuple(name for name in ("front", "wrist") if name in env.scene.sensors)
        step_dt = float(env.step_dt)

        print("RED_CUBE_TO_BOX_DATASET_ENV_CREATED_OK", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"camera_names: {camera_names}", flush=True)
        print(f"step_dt: {step_dt}", flush=True)
        print(f"recovered_staging_groups: {writer.recovery.removed_staging_groups}", flush=True)
        print(f"starting_successful_episodes: {writer.successful_episodes}", flush=True)
        print("RED_CUBE_TO_BOX_DATASET_PHASE=collecting", flush=True)

        camera_stats_reported = False
        with torch.inference_mode():
            for new_attempt_index in range(args.maximum_attempts):
                if writer.successful_episodes >= target_total:
                    break
                observations, _ = env.reset()
                observations = refresh_camera_observations_without_control(
                    env,
                    observations,
                    camera_names=camera_names,
                    refreshes=args.camera_refreshes_before_recording,
                )
                state_machine.reset()
                if not camera_stats_reported:
                    for camera_name in camera_names:
                        image = observations["policy"][camera_name]
                        image_min = int(image.min().item())
                        image_max = int(image.max().item())
                        image_mean = float(image.to(torch.float32).mean().item())
                        print(
                            f"camera_frame_stats:{camera_name}:shape={tuple(image.shape)}:dtype={image.dtype}:"
                            f"min={image_min}:max={image_max}:mean={image_mean:.3f}",
                            flush=True,
                        )
                        if image.dtype != torch.uint8 or image_max == 0:
                            raise RuntimeError(f"Camera {camera_name} returned invalid or all-black RGB")
                    camera_stats_reported = True
                initial_cube_pos = _numpy_row(cube.data.root_pos_w)
                episode = writer.begin_attempt(
                    {
                        "new_attempt_index": new_attempt_index,
                        "effective_seed": effective_seed,
                        "initial_cube_pos_w": initial_cube_pos.tolist(),
                    }
                )
                ever_grasped = False
                rewards_finite = True
                unexpected_reset = False
                completed_steps = 0
                try:
                    while not state_machine.is_episode_done:
                        if env.cfg.dynamic_reset_gripper_effort_limit:
                            dynamic_reset_gripper_effort_limit_sim(env, "so101_state_machine")
                        action = state_machine.get_action(env)
                        if state_machine.is_episode_done:
                            break
                        if not bool(torch.isfinite(action).all()):
                            raise RuntimeError("Expert produced a non-finite Cartesian action")
                        pre_step = snapshot_pre_step(
                            observations,
                            robot,
                            cube,
                            joint_ids=joint_ids,
                            gripper_body_index=gripper_body_index,
                        )
                        step_result = env.step(action)
                        episode.append(
                            attach_applied_joint_target(
                                pre_step,
                                robot,
                                joint_ids=joint_ids,
                                timestamp=completed_steps * step_dt,
                            )
                        )
                        observations = step_result[0]
                        rewards_finite &= bool(torch.isfinite(step_result[1]).all())
                        unexpected_reset |= bool(step_result[2].any()) or bool(step_result[3].any())
                        pick_cube = bool(observations["subtask_terms"]["pick_cube"][0].item())
                        observe_pick_cube = getattr(state_machine, "observe_pick_cube", None)
                        if callable(observe_pick_cube):
                            observe_pick_cube(pick_cube, env)
                        ever_grasped |= pick_cube
                        state_machine.advance()
                        completed_steps += 1

                    expert_success = state_machine.check_success(env)
                    abort_reason = getattr(state_machine, "servo_abort_reason", None)
                    timeout_phase = getattr(state_machine, "servo_timeout_phase", None)
                    success = bool(
                        expert_success
                        and ever_grasped
                        and rewards_finite
                        and not unexpected_reset
                        and abort_reason is None
                        and timeout_phase is None
                    )
                    final_offset = _numpy_row(cube.data.root_pos_w - floor.data.root_pos_w)
                    summary = {
                        "success": success,
                        "expert_success": expert_success,
                        "ever_grasped": ever_grasped,
                        "rewards_finite": rewards_finite,
                        "unexpected_reset": unexpected_reset,
                        "servo_abort_reason": abort_reason,
                        "servo_timeout_phase": timeout_phase,
                        "completed_steps": completed_steps,
                        "final_offset": final_offset.tolist(),
                    }
                    demo_name = episode.finish(success=success, summary=summary)
                    print(
                        f"dataset_attempt:{episode.attempt_id}:success={success}:demo={demo_name}:"
                        f"steps={completed_steps}:abort={abort_reason}:timeout={timeout_phase}",
                        flush=True,
                    )
                except Exception as exc:
                    if not episode.is_closed:
                        episode.finish(
                            success=False,
                            summary={"success": False, "exception": f"{type(exc).__name__}: {exc}"},
                        )
                    raise

        print(f"completed_new_attempts: {writer.attempts - resumed_attempts}", flush=True)
        print(f"total_attempts: {writer.attempts}", flush=True)
        print(f"total_successful_episodes: {writer.successful_episodes}", flush=True)
        if writer.successful_episodes < target_total:
            raise RuntimeError(f"Collected {writer.successful_episodes} total successes, below target {target_total}")
        print("RED_CUBE_TO_BOX_EXPERT_DATASET_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("RED_CUBE_TO_BOX_EXPERT_DATASET_FAILED", flush=True)
    finally:
        print("RED_CUBE_TO_BOX_DATASET_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
