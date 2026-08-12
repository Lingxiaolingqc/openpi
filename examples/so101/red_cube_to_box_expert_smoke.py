"""Run one bounded scripted-expert episode in RedCubeToBox."""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

AUTOGEN_REFERENCE_EXPERTS = frozenset(
    {
        "autogen_reference",
        "autogen_reference_slow_grasp",
        "autogen_reference_axis_align_slow_grasp",
    }
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets_root", default=os.environ.get("LEISAAC_ASSETS_ROOT"))
    parser.add_argument(
        "--expert",
        choices=(
            "legacy",
            "legacy_dynamic_grasp_offset",
            "legacy_dynamic_grasp_offset_residual_corrected",
            "autogen_retreat_transport",
            "autogen_independent_retreat_transport",
            "autogen_polar_retreat_transport",
            "autogen_reference",
            "autogen_reference_slow_grasp",
            "autogen_reference_axis_align_slow_grasp",
            "legacy_gripper_anchor",
            "legacy_gripper_anchor_align_then_lower",
            "legacy_gripper_anchor_position_align_then_lower",
            "legacy_gripper_anchor_weighted_position_align_then_lower",
            "legacy_gripper_anchor_safe_planar_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
            "jaw_frame_xyz_tilt",
            "legacy_gripper_anchor_relaxed_ik",
            "legacy_gripper_anchor_planar_ik",
            "adaptive",
            "servo",
            "weighted_servo",
            "legacy_weighted_servo",
            "legacy_position_servo",
            "legacy_pd_position_servo",
            "legacy_trajectory_pd_servo",
        ),
        default="legacy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--record_dir",
        default=os.environ.get("RED_CUBE_TO_BOX_RECORD_DIR"),
        help="Optional root directory for sampled front-camera JPEGs, trace JSONL, and an offline HTML viewer.",
    )
    parser.add_argument("--record_every", type=int, default=4, help="Record one frame every N control steps.")
    parser.add_argument("--record_fps", type=float, default=15.0, help="Playback rate used by the HTML viewer.")
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument(
        "--autogen_ray_axis",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        default="-z",
        help="Gripper-frame local axis used as the active Autogen green ray; all six axes are visualized.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _rounded_row(values, digits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(value), digits) for value in values.detach().cpu().tolist())


def _print_fields(prefix: str, /, **fields: object) -> None:
    """Print one diagnostic field per line while preserving its grep-friendly prefix."""

    lines = [prefix, *(f"  {name}={value}" for name, value in fields.items())]
    print("\n" + "\n".join(lines), flush=True)


def _finite_or_none(value):
    """Keep diagnostic JSON standards-compliant before the first safety update."""

    if value is None:
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


class _DiagnosticRecorder:
    """Stream sampled camera frames and state into a copyable run directory."""

    def __init__(
        self,
        root: Path,
        expert: str,
        seed: int,
        record_every: int,
        playback_fps: float,
        jpeg_quality: int,
        image_class,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.run_dir = root / f"{expert}-seed{seed}-{timestamp}-pid{os.getpid()}"
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.result_path = self.run_dir / "result.json"
        self.viewer_path = self.run_dir / "index.html"
        self._record_every = record_every
        self._playback_fps = playback_fps
        self._jpeg_quality = jpeg_quality
        self._image_class = image_class
        self._frames: list[dict[str, object]] = []
        self._last_recorded_step: int | None = None
        self._finished = False

    def capture(
        self,
        step: int,
        phase: str,
        observations: dict,
        env,
        state_machine=None,
        *,
        force: bool = False,
    ) -> None:
        if not force and step % self._record_every != 0:
            return
        if self._last_recorded_step == step:
            return

        front = observations["policy"]["front"]
        if front.ndim != 4 or front.shape[0] != 1 or front.shape[-1] != 3:
            raise RuntimeError(f"Unexpected recorder front camera shape: {tuple(front.shape)}")

        relative_path = Path("frames") / f"frame_{len(self._frames):06d}.jpg"
        image = front[0].detach().cpu().numpy()
        self._image_class.fromarray(image).save(
            self.run_dir / relative_path,
            format="JPEG",
            quality=self._jpeg_quality,
        )

        robot = env.scene["robot"]
        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        ee_frame = env.scene["ee_frame"]
        record = {
            "frame": relative_path.as_posix(),
            "step": step,
            "phase": phase,
            "joint_pos_rad": _rounded_row(robot.data.joint_pos[0]),
            "gripper_pos_w": _rounded_row(ee_frame.data.target_pos_w[0, 0]),
            "jaw_detection_pos_w": _rounded_row(ee_frame.data.target_pos_w[0, 1]),
            "cube_pos_w": _rounded_row(cube.data.root_pos_w[0]),
            "cube_offset_from_box": _rounded_row(cube.data.root_pos_w[0] - floor.data.root_pos_w[0]),
            "pick_cube": bool(observations["subtask_terms"]["pick_cube"][0].item()),
        }
        if state_machine is not None:
            ee_point_w = getattr(state_machine, "ee_point_w", None)
            ee_table_projection_w = getattr(state_machine, "ee_table_projection_w", None)
            descent_xy_correction_w = getattr(state_machine, "descent_xy_correction_w", None)
            axis_alignment_closing_axis_w = getattr(state_machine, "axis_alignment_closing_axis_w", None)
            axis_alignment_cube_x_axis_w = getattr(state_machine, "axis_alignment_cube_x_axis_w", None)
            axis_alignment_cube_y_axis_w = getattr(state_machine, "axis_alignment_cube_y_axis_w", None)
            axis_alignment_desired_axis_w = getattr(state_machine, "axis_alignment_desired_axis_w", None)
            axis_alignment_direct_joint_target = getattr(state_machine, "axis_alignment_direct_joint_target", None)
            record.update(
                {
                    "ik_runtime_mode": getattr(state_machine, "ik_runtime_mode", "pose"),
                    "safety_mode": getattr(state_machine, "safety_mode", None),
                    "cube_clearance": _finite_or_none(getattr(state_machine, "cube_clearance", None)),
                    "minimum_robot_clearance": _finite_or_none(getattr(state_machine, "minimum_robot_clearance", None)),
                    "maximum_box_contact_force": _finite_or_none(
                        getattr(state_machine, "maximum_box_contact_force", None)
                    ),
                    "align_cube_z_reference": _finite_or_none(getattr(state_machine, "align_cube_z_reference", None)),
                    "align_cube_z_error": _finite_or_none(getattr(state_machine, "align_cube_z_error", None)),
                    "ee_marker_pos_w": None if ee_point_w is None else _rounded_row(ee_point_w[0]),
                    "ee_table_projection_marker_pos_w": (
                        None if ee_table_projection_w is None else _rounded_row(ee_table_projection_w[0])
                    ),
                    "lower_target_error": _finite_or_none(getattr(state_machine, "lower_target_error", None)),
                    "lower_target_stable_streak": getattr(state_machine, "lower_target_stable_streak", None),
                    "lower_hold_steps": getattr(state_machine, "lower_hold_steps", None),
                    "release_authorized": getattr(state_machine, "release_authorized", None),
                    "green_ray_hit": getattr(state_machine, "green_ray_hit", None),
                    "green_ray_obb_hit": getattr(state_machine, "green_ray_obb_hit", None),
                    "green_ray_within_grasp_reach": getattr(state_machine, "green_ray_within_grasp_reach", None),
                    "green_ray_hit_distance": _finite_or_none(getattr(state_machine, "green_ray_hit_distance", None)),
                    "wrist_to_gripper_length": _finite_or_none(getattr(state_machine, "wrist_to_gripper_length", None)),
                    "wrist_to_jaw_length": _finite_or_none(getattr(state_machine, "wrist_to_jaw_length", None)),
                    "approach_tracking_error": _finite_or_none(getattr(state_machine, "approach_tracking_error", None)),
                    "descent_ray_xy_error": _finite_or_none(getattr(state_machine, "descent_ray_xy_error", None)),
                    "descent_xy_correction_w": (
                        None if descent_xy_correction_w is None else _rounded_row(descent_xy_correction_w[0])
                    ),
                    "ray_alignment_streak": getattr(state_machine, "ray_alignment_streak", None),
                    "axis_alignment_complete": getattr(state_machine, "axis_alignment_complete", None),
                    "axis_alignment_selected_cube_axis": getattr(
                        state_machine, "axis_alignment_selected_cube_axis", None
                    ),
                    "axis_alignment_closing_axis_w": (
                        None
                        if axis_alignment_closing_axis_w is None
                        else _rounded_row(axis_alignment_closing_axis_w[0])
                    ),
                    "axis_alignment_desired_axis_w": (
                        None
                        if axis_alignment_desired_axis_w is None
                        else _rounded_row(axis_alignment_desired_axis_w[0])
                    ),
                    "axis_alignment_cube_x_axis_w": (
                        None if axis_alignment_cube_x_axis_w is None else _rounded_row(axis_alignment_cube_x_axis_w[0])
                    ),
                    "axis_alignment_cube_y_axis_w": (
                        None if axis_alignment_cube_y_axis_w is None else _rounded_row(axis_alignment_cube_y_axis_w[0])
                    ),
                    "axis_alignment_error_rad": _finite_or_none(getattr(state_machine, "axis_alignment_error", None)),
                    "axis_alignment_signed_error_rad": _finite_or_none(
                        getattr(state_machine, "axis_alignment_signed_error", None)
                    ),
                    "axis_alignment_streak": getattr(state_machine, "axis_alignment_streak", None),
                    "axis_alignment_final_gate_streak": getattr(
                        state_machine, "axis_alignment_final_gate_streak", None
                    ),
                    "axis_alignment_wrist_roll_target_rad": _finite_or_none(
                        getattr(state_machine, "axis_alignment_wrist_roll_target", None)
                    ),
                    "axis_alignment_wrist_roll_position_rad": _finite_or_none(
                        getattr(state_machine, "axis_alignment_wrist_roll_position", None)
                    ),
                    "axis_alignment_wrist_roll_velocity_rad_s": _finite_or_none(
                        getattr(state_machine, "axis_alignment_wrist_roll_velocity", None)
                    ),
                    "axis_alignment_max_arm_joint_velocity_rad_s": _finite_or_none(
                        getattr(state_machine, "axis_alignment_max_arm_joint_velocity", None)
                    ),
                    "axis_alignment_wrist_position_error_m": _finite_or_none(
                        getattr(state_machine, "axis_alignment_wrist_position_error", None)
                    ),
                    "axis_alignment_direct_target_error_rad": _finite_or_none(
                        getattr(state_machine, "axis_alignment_direct_target_error", None)
                    ),
                    "axis_alignment_frozen_joint_drift_rad": _finite_or_none(
                        getattr(state_machine, "axis_alignment_frozen_joint_drift", None)
                    ),
                    "axis_alignment_direct_joint_target_rad": (
                        None
                        if axis_alignment_direct_joint_target is None
                        else _rounded_row(axis_alignment_direct_joint_target[0])
                    ),
                    "axis_alignment_direct_hold_active": getattr(
                        state_machine, "axis_alignment_direct_hold_active", None
                    ),
                    "axis_alignment_direct_hold_release_reason": getattr(
                        state_machine, "axis_alignment_direct_hold_release_reason", None
                    ),
                    "ik_handoff_streak": getattr(state_machine, "ik_handoff_streak", None),
                    "ik_handoff_wrist_position_error_m": _finite_or_none(
                        getattr(state_machine, "ik_handoff_wrist_position_error", None)
                    ),
                    "ik_handoff_max_arm_joint_velocity_rad_s": _finite_or_none(
                        getattr(state_machine, "ik_handoff_max_arm_joint_velocity", None)
                    ),
                    "measured_gripper_above_jaw_z": (
                        None
                        if getattr(state_machine, "measured_gripper_above_jaw_z", None) is None
                        else _rounded_row(state_machine.measured_gripper_above_jaw_z[0:1])
                    ),
                    "safe_jaw_target_z": (
                        None
                        if getattr(state_machine, "safe_jaw_target_z", None) is None
                        else _rounded_row(state_machine.safe_jaw_target_z[0:1])
                    ),
                    "safe_release_gripper_target_z": (
                        None
                        if getattr(state_machine, "safe_release_gripper_target_z", None) is None
                        else _rounded_row(state_machine.safe_release_gripper_target_z[0:1])
                    ),
                }
            )
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, allow_nan=False) + "\n")
        self._frames.append(record)
        self._last_recorded_step = step
        self._write_viewer()

    def finish(self, result: dict[str, object]) -> None:
        if self._finished:
            return
        self.result_path.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._write_viewer(result)
        self._finished = True

    def _write_viewer(self, result: dict[str, object] | None = None) -> None:
        frames_json = json.dumps(self._frames, allow_nan=False)
        result_json = json.dumps(result or {"status": "running"}, allow_nan=False)
        interval_ms = max(round(1000.0 / self._playback_fps), 1)
        self.viewer_path.write_text(
            f"""<!doctype html>
<meta charset="utf-8">
<title>RedCubeToBox expert diagnostic</title>
<style>
body {{ background:#111; color:#eee; font:16px system-ui; margin:24px; }}
img {{ display:block; max-width:100%; border:1px solid #555; margin:12px 0; }}
button,input {{ margin-right:8px; }} pre {{ white-space:pre-wrap; }}
</style>
<h1>RedCubeToBox expert diagnostic</h1>
<pre id="result"></pre>
<img id="frame" alt="recorded front camera frame">
<button id="play">Play</button><button id="pause">Pause</button>
<input id="slider" type="range" min="0" max="0" value="0">
<pre id="state"></pre>
<script>
const frames = {frames_json};
const result = {result_json};
const intervalMs = {interval_ms};
let index = 0;
let timer = null;
const image = document.getElementById('frame');
const slider = document.getElementById('slider');
const state = document.getElementById('state');
document.getElementById('result').textContent = JSON.stringify(result, null, 2);
slider.max = Math.max(frames.length - 1, 0);
function show(next) {{
  if (!frames.length) return;
  index = Math.max(0, Math.min(next, frames.length - 1));
  slider.value = index;
  image.src = frames[index].frame;
  state.textContent = JSON.stringify(frames[index], null, 2);
}}
function play() {{
  if (timer || !frames.length) return;
  timer = setInterval(() => show((index + 1) % frames.length), intervalMs);
}}
function pause() {{ clearInterval(timer); timer = null; }}
document.getElementById('play').onclick = play;
document.getElementById('pause').onclick = pause;
slider.oninput = () => show(Number(slider.value));
show(0);
</script>
""",
            encoding="utf-8",
        )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.headless:
        parser.error("This expert smoke requires --headless")
    if not args.enable_cameras:
        parser.error("The environment requires --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")
    if args.record_every <= 0:
        parser.error("--record_every must be positive")
    if args.record_fps <= 0:
        parser.error("--record_fps must be positive")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg_quality must be between 1 and 95")

    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)
    record_root = Path(args.record_dir).expanduser().resolve() if args.record_dir else None

    print("RED_CUBE_TO_BOX_EXPERT_PHASE=before_launcher", flush=True)
    print(f"assets_root: {assets_root}", flush=True)
    print(f"requested_device: {args.device}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac Sim must be launched before importing the remaining simulation modules.
    # isort: off
    import gymnasium as gym
    from PIL import Image
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    import leisaac.tasks  # noqa: F401
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
    import red_cube_to_box_task
    from red_cube_to_box_task.adaptive_state_machine import RedCubeToBoxAdaptiveStateMachine
    from red_cube_to_box_task.autogen_retreat_transport_state_machine import (
        RedCubeToBoxAutogenRetreatTransportStateMachine,
    )
    from red_cube_to_box_task.autogen_independent_retreat_transport_state_machine import (
        RedCubeToBoxAutogenIndependentRetreatTransportStateMachine,
    )
    from red_cube_to_box_task.autogen_polar_retreat_transport_state_machine import (
        RedCubeToBoxAutogenPolarRetreatTransportStateMachine,
    )
    from red_cube_to_box_task.autogen_reference_state_machine import (
        RedCubeToBoxAutogenReferenceStateMachine,
        configure_autogen_reference_action,
    )
    from red_cube_to_box_task.autogen_reference_slow_grasp_state_machine import (
        RedCubeToBoxAutogenReferenceSlowGraspStateMachine,
    )
    from red_cube_to_box_task.autogen_reference_axis_align_slow_grasp_state_machine import (
        RedCubeToBoxAutogenReferenceAxisAlignSlowGraspStateMachine,
    )
    from red_cube_to_box_task.env_cfg import configure_planar_safety_sensors
    from red_cube_to_box_task.legacy_gripper_anchor_state_machine import (
        RedCubeToBoxLegacyGripperAnchorStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_relaxed_ik_state_machine import (
        RedCubeToBoxLegacyGripperAnchorRelaxedIkStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_planar_ik_state_machine import (
        RedCubeToBoxLegacyGripperAnchorPlanarIkStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_position_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorPositionAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_weighted_position_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorWeightedPositionAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_planar_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafePlanarAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeJawTrajectoryDirectXyzPanNullspaceAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeXyzPanNullspaceAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeXyzPitchPanAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.legacy_gripper_anchor_safe_xyz_tilt_align_then_lower_state_machine import (
        RedCubeToBoxLegacyGripperAnchorSafeXyzTiltAlignThenLowerStateMachine,
    )
    from red_cube_to_box_task.jaw_frame_xyz_tilt_state_machine import RedCubeToBoxJawFrameXyzTiltStateMachine
    from red_cube_to_box_task.legacy_dynamic_grasp_offset_state_machine import (
        RedCubeToBoxLegacyDynamicGraspOffsetStateMachine,
    )
    from red_cube_to_box_task.legacy_dynamic_grasp_offset_residual_corrected_state_machine import (
        RedCubeToBoxLegacyDynamicGraspOffsetResidualCorrectedStateMachine,
    )
    from red_cube_to_box_task.legacy_weighted_servo_state_machine import (
        RedCubeToBoxLegacyWeightedServoStateMachine,
    )
    from red_cube_to_box_task.legacy_position_servo_state_machine import (
        RedCubeToBoxLegacyPositionServoStateMachine,
    )
    from red_cube_to_box_task.legacy_pd_position_servo_state_machine import (
        RedCubeToBoxLegacyPdPositionServoStateMachine,
    )
    from red_cube_to_box_task.legacy_trajectory_pd_servo_state_machine import (
        RedCubeToBoxLegacyTrajectoryPdServoStateMachine,
    )
    from red_cube_to_box_task.phase_aware_ik_action import (
        configure_dynamic_control_frame_offset,
        configure_servo_ik_action,
        resolve_action_term,
    )
    from red_cube_to_box_task.servo_state_machine import RedCubeToBoxServoStateMachine
    from red_cube_to_box_task.state_machine import RedCubeToBoxStateMachine
    from red_cube_to_box_task.weighted_servo_state_machine import RedCubeToBoxWeightedServoStateMachine
    # isort: on

    status = 1
    completed_steps = 0
    recorder = None
    try:
        task_id = red_cube_to_box_task.TASK_ID
        print("RED_CUBE_TO_BOX_EXPERT_PHASE=app_ready", flush=True)
        print(f"app_launcher_device_id: {app_launcher.device_id}", flush=True)
        print(f"task_id: {task_id}", flush=True)
        print(f"expert_variant: {args.expert}", flush=True)

        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101_state_machine")
        env_cfg.seed = args.seed
        env_cfg.recorders = None
        env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        if args.expert in {
            "legacy_gripper_anchor_planar_ik",
            "legacy_gripper_anchor_safe_planar_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
            "jaw_frame_xyz_tilt",
        }:
            configure_planar_safety_sensors(env_cfg)
        if args.expert in {
            "legacy_gripper_anchor_relaxed_ik",
            "legacy_gripper_anchor_planar_ik",
            "legacy_gripper_anchor_position_align_then_lower",
            "legacy_gripper_anchor_weighted_position_align_then_lower",
            "legacy_gripper_anchor_safe_planar_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
            "jaw_frame_xyz_tilt",
            "servo",
            "weighted_servo",
            "legacy_weighted_servo",
            "legacy_position_servo",
            "legacy_pd_position_servo",
            "legacy_trajectory_pd_servo",
            "autogen_retreat_transport",
            "autogen_independent_retreat_transport",
            "autogen_polar_retreat_transport",
        }:
            configure_servo_ik_action(env_cfg)
        if args.expert in AUTOGEN_REFERENCE_EXPERTS:
            configure_autogen_reference_action(env_cfg)
        if args.expert in {
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
            "jaw_frame_xyz_tilt",
        }:
            configure_dynamic_control_frame_offset(env_cfg)
        print(
            "state_machine_gripper_close_expr: "
            f"{getattr(env_cfg.actions.gripper_action, 'close_command_expr', 'continuous_joint_position')}",
            flush=True,
        )
        print(f"expert_ik_command_type: {env_cfg.actions.arm_action.controller.command_type}", flush=True)
        orientation_policy = {
            "legacy": "fixed_world",
            "legacy_dynamic_grasp_offset": "legacy_fixed_world,dynamic_per-grasp_placement_xy",
            "legacy_dynamic_grasp_offset_residual_corrected": (
                "legacy_fixed_world,dynamic_per-grasp_placement_xy,single_post-transfer_residual_correction"
            ),
            "autogen_retreat_transport": (
                "legacy_pickup,live_gripper_retreat,live_gripper_transport,full_6d_pose,"
                "target_box_floor_center_xy,no_grasp_offset,no_residual_correction"
            ),
            "autogen_independent_retreat_transport": (
                "independent_pickup,retreat_and_lift_until_converged,"
                "transport_until_converged,full_6d_pose,target_box_floor_center_xy,no_offset"
            ),
            "autogen_polar_retreat_transport": (
                "independent_pickup,root_relative_5_over_7_retreat,root-centered_arc_with_yaw,"
                "radial_box_approach,full_6d_pose,actual_xyz_and_bearing_completion"
            ),
            "autogen_reference": (
                "bundled_autogen_state_flow,robot-base_coordinates,original_green_ray_obb,"
                "wrist_xyz_ik_plus_wrist_flex_posture_correction,continuous_gripper"
            ),
            "autogen_reference_slow_grasp": (
                "autogen_reference,grasp_close_duration_80_to_240_steps,no_other_behavior_change"
            ),
            "autogen_reference_axis_align_slow_grasp": (
                "autogen_reference_slow_grasp,post_descend_gripper_local_x_to_nearest_cube_local_x_or_y,"
                "wrist_xyz_plus_wrist_roll,recenter_before_grasp"
            ),
            "legacy_gripper_anchor": "legacy_fixed_world,jaw_anchored_placement",
            "legacy_gripper_anchor_align_then_lower": "legacy_fixed_world,align_high_then_descend",
            "legacy_gripper_anchor_position_align_then_lower": "position_only_high_align,legacy_pose_descent",
            "legacy_gripper_anchor_weighted_position_align_then_lower": (
                "shoulder_pan_priority_xyz_align,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_planar_align_then_lower": (
                "xy_plus_orientation_high_align,physical_z_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower": (
                "xyz_plus_world_tilt_high_align,free_yaw,physical_safety_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower": (
                "xyz_plus_pitch_plus_explicit_shoulder_pan_target,physical_safety_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower": (
                "xyz_plus_explicit_shoulder_pan_target,nullspace_joint_limit_avoidance,"
                "physical_safety_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower": (
                "direct_closed_jaw_xyz_plus_explicit_shoulder_pan_target,nullspace_joint_limit_avoidance,"
                "physical_safety_gates,legacy_pose_descent"
            ),
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower": (
                "smooth_jaw_transfer_to_floor_center,direct_closed_jaw_xyz_plus_explicit_shoulder_pan_target,"
                "persistent_joint_target_slew,nullspace_joint_limit_avoidance,physical_safety_gates"
            ),
            "jaw_frame_xyz_tilt": "live_jaw_frame_xyz_plus_world_tilt_all_phases,free_yaw",
            "legacy_gripper_anchor_relaxed_ik": "legacy_fixed_world,jaw_anchor_then_staged_relaxation",
            "legacy_gripper_anchor_planar_ik": "legacy_fixed_world,collision_gated_xy_plus_orientation",
            "adaptive": "fixed_during_grasp,current_after_grasp",
            "servo": "fixed_world_through_lift,position_only_ik_after_lift",
            "weighted_servo": "fixed_world_through_lift,translation_priority_ik_after_lift",
            "legacy_weighted_servo": "legacy_exact_through_lift,translation_priority_ik_after_lift",
            "legacy_position_servo": "legacy_exact_through_lift,position_only_ik_after_lift",
            "legacy_pd_position_servo": "legacy_exact_through_lift,velocity_damped_position_only_ik_after_lift",
            "legacy_trajectory_pd_servo": "legacy_exact_through_lift,smooth_reference_pd_position_ik_after_lift",
        }[args.expert]
        print(f"expert_orientation_policy: {orientation_policy}", flush=True)

        print("RED_CUBE_TO_BOX_EXPERT_PHASE=creating_env", flush=True)
        env = gym.make(task_id, cfg=env_cfg).unwrapped
        observations, _ = env.reset()

        state_machine_class = {
            "legacy": RedCubeToBoxStateMachine,
            "legacy_dynamic_grasp_offset": RedCubeToBoxLegacyDynamicGraspOffsetStateMachine,
            "legacy_dynamic_grasp_offset_residual_corrected": (
                RedCubeToBoxLegacyDynamicGraspOffsetResidualCorrectedStateMachine
            ),
            "autogen_retreat_transport": RedCubeToBoxAutogenRetreatTransportStateMachine,
            "autogen_independent_retreat_transport": (RedCubeToBoxAutogenIndependentRetreatTransportStateMachine),
            "autogen_polar_retreat_transport": RedCubeToBoxAutogenPolarRetreatTransportStateMachine,
            "autogen_reference": RedCubeToBoxAutogenReferenceStateMachine,
            "autogen_reference_slow_grasp": RedCubeToBoxAutogenReferenceSlowGraspStateMachine,
            "autogen_reference_axis_align_slow_grasp": (RedCubeToBoxAutogenReferenceAxisAlignSlowGraspStateMachine),
            "legacy_gripper_anchor": RedCubeToBoxLegacyGripperAnchorStateMachine,
            "legacy_gripper_anchor_align_then_lower": RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine,
            "legacy_gripper_anchor_position_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorPositionAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_weighted_position_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorWeightedPositionAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_planar_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafePlanarAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeXyzTiltAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeXyzPitchPanAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeXyzPanNullspaceAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine
            ),
            "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower": (
                RedCubeToBoxLegacyGripperAnchorSafeJawTrajectoryDirectXyzPanNullspaceAlignThenLowerStateMachine
            ),
            "jaw_frame_xyz_tilt": RedCubeToBoxJawFrameXyzTiltStateMachine,
            "legacy_gripper_anchor_relaxed_ik": RedCubeToBoxLegacyGripperAnchorRelaxedIkStateMachine,
            "legacy_gripper_anchor_planar_ik": RedCubeToBoxLegacyGripperAnchorPlanarIkStateMachine,
            "adaptive": RedCubeToBoxAdaptiveStateMachine,
            "servo": RedCubeToBoxServoStateMachine,
            "weighted_servo": RedCubeToBoxWeightedServoStateMachine,
            "legacy_weighted_servo": RedCubeToBoxLegacyWeightedServoStateMachine,
            "legacy_position_servo": RedCubeToBoxLegacyPositionServoStateMachine,
            "legacy_pd_position_servo": RedCubeToBoxLegacyPdPositionServoStateMachine,
            "legacy_trajectory_pd_servo": RedCubeToBoxLegacyTrajectoryPdServoStateMachine,
        }[args.expert]
        if args.expert in AUTOGEN_REFERENCE_EXPERTS:
            state_machine = state_machine_class(green_ray_axis=args.autogen_ray_axis)
        else:
            state_machine = state_machine_class()
        state_machine.setup(env)
        state_machine.reset()
        servo_parameters = getattr(state_machine, "servo_parameters", None)
        if isinstance(servo_parameters, dict):
            _print_fields("servo_parameters", **servo_parameters)
        else:
            _print_fields("servo_parameters", value="not_applicable")

        if record_root is not None:
            recorder = _DiagnosticRecorder(
                root=record_root,
                expert=args.expert,
                seed=args.seed,
                record_every=args.record_every,
                playback_fps=args.record_fps,
                jpeg_quality=args.jpeg_quality,
                image_class=Image,
            )
            print(f"diagnostic_record_dir: {recorder.run_dir}", flush=True)

        cube = env.scene["cube"]
        floor = env.scene["target_box_floor"]
        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        print("RED_CUBE_TO_BOX_EXPERT_ENV_CREATED_OK", flush=True)
        print(f"simulation_device: {env.device}", flush=True)
        print(f"action_space: {env.action_space}", flush=True)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        print(f"expert_ik_action_class: {type(arm_action_term).__name__}", flush=True)
        print(f"cube_initial_pos_w: {_rounded_row(cube.data.root_pos_w[0])}", flush=True)
        print(f"target_box_floor_pos_w: {_rounded_row(floor.data.root_pos_w[0])}", flush=True)

        all_rewards_finite = True
        unexpected_reset = False
        previous_phase = None
        previous_ik_runtime_mode = None
        previous_pick_cube = bool(observations["subtask_terms"]["pick_cube"][0].item())
        if recorder is not None:
            recorder.capture(0, state_machine.phase_name, observations, env, state_machine, force=True)

        with torch.inference_mode():
            while not state_machine.is_episode_done:
                phase = state_machine.phase_name
                phase_changed = phase != previous_phase
                if phase_changed:
                    _print_fields(f"expert_phase:{phase}", step=state_machine.step_count)
                    gripper_pos = ee_frame.data.target_pos_w[0, 0]
                    jaw_pos = ee_frame.data.target_pos_w[0, 1]
                    jaw_cube_distance = torch.linalg.vector_norm(jaw_pos - cube.data.root_pos_w[0])
                    pick_cube = observations["subtask_terms"]["pick_cube"][0]
                    _print_fields(
                        f"expert_state:{phase}",
                        gripper_pos_w=_rounded_row(gripper_pos),
                        jaw_pos_w=_rounded_row(jaw_pos),
                        cube_pos_w=_rounded_row(cube.data.root_pos_w[0]),
                        jaw_cube_distance=f"{jaw_cube_distance.item():.5f}",
                        gripper_joint=f"{robot.data.joint_pos[0, -1].item():.5f}",
                        pick_cube=bool(pick_cube.item()),
                    )
                    previous_phase = phase

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, "so101_state_machine")

                action = state_machine.get_action(env)
                expected_action_shape = (env.num_envs, env.action_manager.total_action_dim)
                if action.shape != expected_action_shape:
                    raise RuntimeError(f"Unexpected expert action shape: {tuple(action.shape)}")
                if not bool(torch.isfinite(action).all()):
                    raise RuntimeError("Expert produced a non-finite action")
                if phase_changed:
                    _print_fields(f"expert_action:{phase}", action=_rounded_row(action[0]))
                    desired_cube = getattr(state_machine, "last_desired_cube_w", None)
                    cube_error = getattr(state_machine, "last_cube_error_w", None)
                    gripper_target = getattr(state_machine, "last_gripper_target_w", None)
                    if desired_cube is not None and cube_error is not None and gripper_target is not None:
                        print(
                            f"expert_feedback:{phase}:"
                            f"desired_cube_w={_rounded_row(desired_cube[0])}:"
                            f"cube_error_w={_rounded_row(cube_error[0])}:"
                            f"gripper_target_w={_rounded_row(gripper_target[0])}",
                            flush=True,
                        )
                if args.expert in AUTOGEN_REFERENCE_EXPERTS and (phase_changed or completed_steps % 30 == 0):
                    print(
                        f"expert_autogen_reference:{phase}:"
                        f"phase_step={state_machine.phase_step}:"
                        f"command_position_b={_rounded_row(state_machine.command_position_b[0], digits=7)}:"
                        f"gripper_command={state_machine.gripper_command:.7f}:"
                        f"green_ray_hit={state_machine.green_ray_hit}:"
                        f"green_ray_obb_hit={state_machine.green_ray_obb_hit}:"
                        f"green_ray_within_grasp_reach={state_machine.green_ray_within_grasp_reach}:"
                        f"green_ray_origin_w="
                        f"{None if state_machine.green_ray_origin_w is None else _rounded_row(state_machine.green_ray_origin_w[0], digits=7)}:"
                        f"green_ray_direction_w="
                        f"{None if state_machine.green_ray_direction_w is None else _rounded_row(state_machine.green_ray_direction_w[0], digits=7)}:"
                        f"green_ray_hit_distance="
                        f"{None if state_machine.green_ray_hit_distance is None else _rounded_row(state_machine.green_ray_hit_distance, digits=7)}:"
                        f"wrist_to_gripper_length="
                        f"{None if state_machine.wrist_to_gripper_length is None else _rounded_row(state_machine.wrist_to_gripper_length, digits=7)}:"
                        f"wrist_to_jaw_length="
                        f"{None if state_machine.wrist_to_jaw_length is None else _rounded_row(state_machine.wrist_to_jaw_length, digits=7)}:"
                        f"cube_projection_on_green_ray="
                        f"{None if state_machine.cube_projection_on_green_ray is None else _rounded_row(state_machine.cube_projection_on_green_ray, digits=7)}:"
                        f"cube_distance_to_green_ray="
                        f"{None if state_machine.cube_distance_to_green_ray is None else _rounded_row(state_machine.cube_distance_to_green_ray, digits=7)}:"
                        f"gripper_frame_position_w="
                        f"{None if state_machine.gripper_frame_position_w is None else _rounded_row(state_machine.gripper_frame_position_w[0], digits=7)}:"
                        f"jaw_detection_position_w="
                        f"{None if state_machine.jaw_detection_position_w is None else _rounded_row(state_machine.jaw_detection_position_w[0], digits=7)}:"
                        f"wrist_height_above_cube="
                        f"{None if state_machine.wrist_height_above_cube is None else _rounded_row(state_machine.wrist_height_above_cube, digits=7)}:"
                        f"gripper_frame_height_above_cube="
                        f"{None if state_machine.gripper_frame_height_above_cube is None else _rounded_row(state_machine.gripper_frame_height_above_cube, digits=7)}:"
                        f"jaw_height_above_cube="
                        f"{None if state_machine.jaw_height_above_cube is None else _rounded_row(state_machine.jaw_height_above_cube, digits=7)}:"
                        f"posture_target_rad="
                        f"{None if state_machine.posture_target is None else _rounded_row(state_machine.posture_target, digits=7)}:"
                        f"temp_jaw_angle_rad={state_machine.held_gripper_angle}:"
                        f"temp_jaw_capture_step={state_machine.held_gripper_angle_capture_step}:"
                        f"wrist_position_w="
                        f"{None if state_machine.wrist_position_w is None else _rounded_row(state_machine.wrist_position_w[0], digits=7)}:"
                        f"descent_wrist_xy_error="
                        f"{None if state_machine.descent_wrist_xy_error is None else _rounded_row(state_machine.descent_wrist_xy_error, digits=7)}:"
                        f"approach_tracking_error="
                        f"{None if state_machine.approach_tracking_error is None else _rounded_row(state_machine.approach_tracking_error, digits=7)}:"
                        f"descent_ray_xy_error="
                        f"{None if state_machine.descent_ray_xy_error is None else _rounded_row(state_machine.descent_ray_xy_error, digits=7)}:"
                        f"descent_xy_correction_w="
                        f"{None if state_machine.descent_xy_correction_w is None else _rounded_row(state_machine.descent_xy_correction_w[0], digits=7)}:"
                        f"ray_alignment_streak={state_machine.ray_alignment_streak}:"
                        f"axis_alignment_complete={getattr(state_machine, 'axis_alignment_complete', None)}:"
                        f"axis_alignment_selected_cube_axis="
                        f"{getattr(state_machine, 'axis_alignment_selected_cube_axis', None)}:"
                        f"axis_alignment_closing_axis_w="
                        f"{None if getattr(state_machine, 'axis_alignment_closing_axis_w', None) is None else _rounded_row(state_machine.axis_alignment_closing_axis_w[0], digits=7)}:"
                        f"axis_alignment_cube_x_axis_w="
                        f"{None if getattr(state_machine, 'axis_alignment_cube_x_axis_w', None) is None else _rounded_row(state_machine.axis_alignment_cube_x_axis_w[0], digits=7)}:"
                        f"axis_alignment_cube_y_axis_w="
                        f"{None if getattr(state_machine, 'axis_alignment_cube_y_axis_w', None) is None else _rounded_row(state_machine.axis_alignment_cube_y_axis_w[0], digits=7)}:"
                        f"axis_alignment_desired_axis_w="
                        f"{None if getattr(state_machine, 'axis_alignment_desired_axis_w', None) is None else _rounded_row(state_machine.axis_alignment_desired_axis_w[0], digits=7)}:"
                        f"axis_alignment_signed_error_rad="
                        f"{None if getattr(state_machine, 'axis_alignment_signed_error', None) is None else _rounded_row(state_machine.axis_alignment_signed_error, digits=7)}:"
                        f"axis_alignment_error_rad="
                        f"{None if getattr(state_machine, 'axis_alignment_error', None) is None else _rounded_row(state_machine.axis_alignment_error, digits=7)}:"
                        f"axis_alignment_streak={getattr(state_machine, 'axis_alignment_streak', None)}:"
                        f"axis_alignment_final_gate_streak="
                        f"{getattr(state_machine, 'axis_alignment_final_gate_streak', None)}:"
                        f"axis_alignment_wrist_roll_target_rad="
                        f"{None if getattr(state_machine, 'axis_alignment_wrist_roll_target', None) is None else _rounded_row(state_machine.axis_alignment_wrist_roll_target, digits=7)}:"
                        f"axis_alignment_wrist_roll_position_rad="
                        f"{None if getattr(state_machine, 'axis_alignment_wrist_roll_position', None) is None else _rounded_row(state_machine.axis_alignment_wrist_roll_position, digits=7)}:"
                        f"axis_alignment_wrist_roll_velocity_rad_s="
                        f"{None if getattr(state_machine, 'axis_alignment_wrist_roll_velocity', None) is None else _rounded_row(state_machine.axis_alignment_wrist_roll_velocity, digits=7)}:"
                        f"axis_alignment_max_arm_joint_velocity_rad_s="
                        f"{None if getattr(state_machine, 'axis_alignment_max_arm_joint_velocity', None) is None else _rounded_row(state_machine.axis_alignment_max_arm_joint_velocity, digits=7)}:"
                        f"axis_alignment_wrist_position_error_m="
                        f"{None if getattr(state_machine, 'axis_alignment_wrist_position_error', None) is None else _rounded_row(state_machine.axis_alignment_wrist_position_error, digits=7)}:"
                        f"axis_alignment_direct_target_error_rad="
                        f"{None if getattr(state_machine, 'axis_alignment_direct_target_error', None) is None else _rounded_row(state_machine.axis_alignment_direct_target_error, digits=7)}:"
                        f"axis_alignment_frozen_joint_drift_rad="
                        f"{None if getattr(state_machine, 'axis_alignment_frozen_joint_drift', None) is None else _rounded_row(state_machine.axis_alignment_frozen_joint_drift, digits=7)}:"
                        f"axis_alignment_direct_joint_target_rad="
                        f"{None if getattr(state_machine, 'axis_alignment_direct_joint_target', None) is None else _rounded_row(state_machine.axis_alignment_direct_joint_target[0], digits=7)}:"
                        f"axis_alignment_direct_hold_active="
                        f"{getattr(state_machine, 'axis_alignment_direct_hold_active', None)}:"
                        f"axis_alignment_direct_hold_release_reason="
                        f"{getattr(state_machine, 'axis_alignment_direct_hold_release_reason', None)}:"
                        f"ik_handoff_streak={getattr(state_machine, 'ik_handoff_streak', None)}:"
                        f"ik_handoff_wrist_position_error_m="
                        f"{None if getattr(state_machine, 'ik_handoff_wrist_position_error', None) is None else _rounded_row(state_machine.ik_handoff_wrist_position_error, digits=7)}:"
                        f"ik_handoff_max_arm_joint_velocity_rad_s="
                        f"{None if getattr(state_machine, 'ik_handoff_max_arm_joint_velocity', None) is None else _rounded_row(state_machine.ik_handoff_max_arm_joint_velocity, digits=7)}:"
                        f"gripper_target_error="
                        f"{None if state_machine.gripper_target_error is None else _rounded_row(state_machine.gripper_target_error, digits=7)}:"
                        f"gripper_joint_velocity="
                        f"{None if state_machine.gripper_joint_velocity is None else _rounded_row(state_machine.gripper_joint_velocity, digits=7)}:"
                        f"gripper_settle_streak={state_machine.gripper_settle_streak}:"
                        f"gripper_settle_reason={state_machine.gripper_settle_reason}:"
                        f"retreat_target_b="
                        f"{None if state_machine.retreat_target_b is None else _rounded_row(state_machine.retreat_target_b[0], digits=7)}:"
                        f"transport_target_b="
                        f"{None if state_machine.transport_target_b is None else _rounded_row(state_machine.transport_target_b[0], digits=7)}",
                        flush=True,
                    )
                if (
                    args.expert
                    in {
                        "legacy_dynamic_grasp_offset",
                        "legacy_dynamic_grasp_offset_residual_corrected",
                    }
                    and phase in {"lift_cube", "retreat_to_safe", "transfer_to_box", "lower_into_box", "release_cube"}
                    and (phase_changed or completed_steps % 25 == 0)
                ):
                    print(
                        f"expert_dynamic_grasp_offset:{phase}:"
                        f"sample_count={state_machine.grasp_offset_sample_count}:"
                        f"gripper_to_cube_xy="
                        f"{None if state_machine.gripper_to_cube_xy is None else _rounded_row(state_machine.gripper_to_cube_xy[0], digits=7)}:"
                        f"floor_center_xy={_rounded_row(floor.data.root_pos_w[0, :2], digits=7)}:"
                        f"desired_cube_xy="
                        f"{None if state_machine.desired_cube_xy is None else _rounded_row(state_machine.desired_cube_xy[0], digits=7)}:"
                        f"gripper_target_xy="
                        f"{None if state_machine.dynamic_gripper_target_xy is None else _rounded_row(state_machine.dynamic_gripper_target_xy[0], digits=7)}:"
                        f"dynamic_place_offset_xy="
                        f"{None if state_machine.dynamic_place_offset_xy is None else _rounded_row(state_machine.dynamic_place_offset_xy[0], digits=7)}",
                        flush=True,
                    )
                if (
                    args.expert
                    in {
                        "legacy_dynamic_grasp_offset_residual_corrected",
                    }
                    and phase in {"lower_into_box", "release_cube", "retract_gripper", "settle"}
                    and (phase_changed or completed_steps % 25 == 0)
                ):
                    print(
                        f"expert_transfer_residual_correction:{phase}:"
                        f"transfer_sample_count={state_machine.transfer_grasp_offset_sample_count}:"
                        f"transfer_gripper_to_cube_xy="
                        f"{None if state_machine.transfer_gripper_to_cube_xy is None else _rounded_row(state_machine.transfer_gripper_to_cube_xy[0], digits=7)}:"
                        f"raw_target_correction_xy="
                        f"{None if state_machine.raw_target_correction_xy is None else _rounded_row(state_machine.raw_target_correction_xy[0], digits=7)}:"
                        f"applied_target_correction_xy="
                        f"{None if state_machine.applied_target_correction_xy is None else _rounded_row(state_machine.applied_target_correction_xy[0], digits=7)}:"
                        f"corrected_gripper_target_xy="
                        f"{None if state_machine.corrected_gripper_target_xy is None else _rounded_row(state_machine.corrected_gripper_target_xy[0], digits=7)}:"
                        f"actual_cube_xy={_rounded_row(cube.data.root_pos_w[0, :2], digits=7)}:"
                        f"desired_cube_xy={_rounded_row(state_machine.desired_cube_xy[0], digits=7)}:"
                        f"measured_gripper_above_jaw_z="
                        f"{None if state_machine.measured_gripper_above_jaw_z is None else _rounded_row(state_machine.measured_gripper_above_jaw_z[0:1], digits=7)}:"
                        f"safe_jaw_target_z="
                        f"{None if state_machine.safe_jaw_target_z is None else _rounded_row(state_machine.safe_jaw_target_z[0:1], digits=7)}:"
                        f"safe_release_gripper_target_z="
                        f"{None if state_machine.safe_release_gripper_target_z is None else _rounded_row(state_machine.safe_release_gripper_target_z[0:1], digits=7)}",
                        flush=True,
                    )
                    if phase == "lower_into_box":
                        print(
                            f"expert_lower_release_gate:{phase}:"
                            f"target_error={state_machine.lower_target_error}:"
                            f"tolerance={state_machine.servo_parameters['lower_target_tolerance']}:"
                            f"stable_streak={state_machine.lower_target_stable_streak}:"
                            f"hold_steps={state_machine.lower_hold_steps}:"
                            f"release_authorized={state_machine.release_authorized}",
                            flush=True,
                        )
                if (
                    args.expert == "autogen_retreat_transport"
                    and phase in {"retreat_to_safe", "transfer_to_box"}
                    and (phase_changed or completed_steps % 25 == 0)
                ):
                    print(
                        f"expert_autogen_path:{phase}:"
                        f"retreat_start_gripper_w="
                        f"{None if state_machine.retreat_start_gripper_w is None else _rounded_row(state_machine.retreat_start_gripper_w[0], digits=7)}:"
                        f"retreat_safe_target_w="
                        f"{None if state_machine.retreat_safe_target_w is None else _rounded_row(state_machine.retreat_safe_target_w[0], digits=7)}:"
                        f"transport_start_gripper_w="
                        f"{None if state_machine.transport_start_gripper_w is None else _rounded_row(state_machine.transport_start_gripper_w[0], digits=7)}:"
                        f"transport_target_w="
                        f"{None if state_machine.transport_target_w is None else _rounded_row(state_machine.transport_target_w[0], digits=7)}:"
                        f"placement_xy_policy=target_box_floor_center_without_offset",
                        flush=True,
                    )
                if (
                    args.expert == "autogen_independent_retreat_transport"
                    and phase in {"retreat_to_safe", "transfer_to_box", "lower_into_box", "retract_gripper"}
                    and (phase_changed or completed_steps % 25 == 0)
                ):
                    print(
                        f"expert_independent_path:{phase}:"
                        f"phase_step={state_machine.phase_step}:"
                        f"retreat_subphase={state_machine.retreat_subphase}:"
                        f"motion_start_w="
                        f"{None if state_machine.motion_start_w is None else _rounded_row(state_machine.motion_start_w[0], digits=7)}:"
                        f"motion_target_w="
                        f"{None if state_machine.motion_target_w is None else _rounded_row(state_machine.motion_target_w[0], digits=7)}:"
                        f"current_target_w="
                        f"{None if state_machine.current_target_w is None else _rounded_row(state_machine.current_target_w[0], digits=7)}:"
                        f"target_error={state_machine.target_error}:"
                        f"stable_streak={state_machine.target_stable_streak}:"
                        f"jaw_cube_distance={state_machine.jaw_cube_distance}:"
                        f"grasp_confirmed={state_machine.grasp_confirmed}",
                        flush=True,
                    )
                if (
                    args.expert == "autogen_polar_retreat_transport"
                    and phase == "close_gripper"
                    and (phase_changed or completed_steps % 25 == 0)
                ):
                    _print_fields(
                        f"expert_polar_close:{phase}",
                        phase_step=state_machine.phase_step,
                        gripper_target_error=state_machine.gripper_target_error,
                        gripper_joint_velocity=state_machine.gripper_joint_velocity,
                        gripper_settle_streak=state_machine.gripper_settle_streak,
                        gripper_settle_reason=state_machine.gripper_settle_reason,
                        jaw_cube_distance=state_machine.jaw_cube_distance,
                        pick_cube=bool(observations["subtask_terms"]["pick_cube"][0].item()),
                    )
                if (
                    args.expert == "autogen_polar_retreat_transport"
                    and phase
                    in {
                        "retreat_to_safe",
                        "arc_transfer",
                        "radial_transfer",
                        "lower_into_box",
                        "retract_gripper",
                    }
                    and (phase_changed or completed_steps % 25 == 0)
                ):
                    _print_fields(
                        f"expert_polar_path:{phase}",
                        phase_step=state_machine.phase_step,
                        path_segment=state_machine.retreat_subphase,
                        control_body=state_machine.retreat_control_body,
                        motion_start_w=(
                            None
                            if state_machine.motion_start_w is None
                            else _rounded_row(state_machine.motion_start_w[0], digits=7)
                        ),
                        motion_target_w=(
                            None
                            if state_machine.motion_target_w is None
                            else _rounded_row(state_machine.motion_target_w[0], digits=7)
                        ),
                        current_target_w=(
                            None
                            if state_machine.current_target_w is None
                            else _rounded_row(state_machine.current_target_w[0], digits=7)
                        ),
                        current_target_quat_w=(
                            None
                            if state_machine.current_target_quat_w is None
                            else _rounded_row(state_machine.current_target_quat_w[0], digits=7)
                        ),
                        actual_gripper_w=_rounded_row(ee_frame.data.target_pos_w[0, 0], digits=7),
                        actual_retreat_control_w=(
                            None
                            if state_machine.retreat_actual_w is None
                            else _rounded_row(state_machine.retreat_actual_w[0], digits=7)
                        ),
                        retreat_z_error=state_machine.retreat_z_error,
                        retreat_xz_error=state_machine.retreat_xz_error,
                        retreat_radial_error=state_machine.retreat_radial_error,
                        target_error=state_machine.target_error,
                        bearing_error=state_machine.bearing_error,
                        stable_streak=state_machine.target_stable_streak,
                        retreat_worsening_streak=state_machine.retreat_worsening_streak,
                        retreat_safety_reason=state_machine.retreat_safety_reason,
                        wrist_flex_target=(
                            None
                            if state_machine.retreat_wrist_flex_target is None
                            else _rounded_row(state_machine.retreat_wrist_flex_target, digits=7)
                        ),
                        wrist_flex_position=round(
                            robot.data.joint_pos[0, robot.data.joint_names.index("wrist_flex")].item(), 7
                        ),
                        ik_task_error=(
                            None
                            if arm_action_term.last_task_error is None
                            else _rounded_row(arm_action_term.last_task_error[0], digits=7)
                        ),
                        ik_task_singular_values=(
                            None
                            if arm_action_term.last_task_singular_values is None
                            else _rounded_row(arm_action_term.last_task_singular_values[0], digits=7)
                        ),
                        jaw_cube_distance=state_machine.jaw_cube_distance,
                        grasp_confirmed=state_machine.grasp_confirmed,
                    )
                ik_runtime_mode = getattr(state_machine, "ik_runtime_mode", "pose")
                if phase_changed or ik_runtime_mode != previous_ik_runtime_mode:
                    _print_fields(f"expert_ik_runtime_mode:{phase}", mode=ik_runtime_mode)
                    previous_ik_runtime_mode = ik_runtime_mode
                if (
                    args.expert
                    in {
                        "legacy_gripper_anchor",
                        "legacy_gripper_anchor_align_then_lower",
                        "legacy_gripper_anchor_position_align_then_lower",
                        "legacy_gripper_anchor_weighted_position_align_then_lower",
                        "legacy_gripper_anchor_safe_planar_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
                        "legacy_gripper_anchor_relaxed_ik",
                        "legacy_gripper_anchor_planar_ik",
                    }
                    and phase in {"lower_into_box", "align_over_box"}
                    and (phase_changed or state_machine.step_count % 25 == 0)
                ):
                    desired_jaw = state_machine.last_desired_jaw_w
                    jaw_error = state_machine.last_jaw_error_w
                    gripper_target = state_machine.last_gripper_target_w
                    if desired_jaw is not None and jaw_error is not None and gripper_target is not None:
                        print(
                            f"expert_jaw_anchor:{phase}:"
                            f"desired_jaw_w={_rounded_row(desired_jaw[0])}:"
                            f"jaw_error_w={_rounded_row(jaw_error[0])}:"
                            f"gripper_target_w={_rounded_row(gripper_target[0])}:"
                            f"alignment_streak={state_machine.alignment_streak}",
                            flush=True,
                        )
                if (
                    args.expert
                    in {
                        "legacy_gripper_anchor_planar_ik",
                        "legacy_gripper_anchor_safe_planar_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
                        "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
                        "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
                    }
                    and phase in {"lower_into_box", "align_over_box", "release_cube"}
                    and (phase_changed or state_machine.step_count % 25 == 0)
                ):
                    print(
                        f"expert_safety:{phase}:"
                        f"mode={state_machine.safety_mode}:"
                        f"cube_clearance={state_machine.cube_clearance:.6f}:"
                        f"minimum_robot_clearance={state_machine.minimum_robot_clearance:.6f}:"
                        f"maximum_box_contact_force={state_machine.maximum_box_contact_force:.6f}:"
                        f"align_cube_z_reference={getattr(state_machine, 'align_cube_z_reference', None)}:"
                        f"align_cube_z_error={getattr(state_machine, 'align_cube_z_error', None)}",
                        flush=True,
                    )
                if (
                    args.expert == "legacy_gripper_anchor_weighted_position_align_then_lower"
                    and phase == "align_over_box"
                    and (phase_changed or state_machine.step_count % 25 == 0)
                ):
                    weighted_delta = state_machine.last_weighted_delta_joint_pos
                    if weighted_delta is not None:
                        weighted_joint_names = state_machine.weighted_joint_names
                        weighted_joint_ids = [
                            robot.data.joint_names.index(joint_name) for joint_name in weighted_joint_names
                        ]
                        arm_joint_pos = robot.data.joint_pos[0, weighted_joint_ids]
                        print(
                            f"expert_weighted_ik:{phase}:"
                            f"joint_names={weighted_joint_names}:"
                            f"joint_pos={_rounded_row(arm_joint_pos)}:"
                            f"delta_joint_pos={_rounded_row(weighted_delta[0], digits=7)}",
                            flush=True,
                        )
                if (
                    args.expert
                    in {
                        "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
                        "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
                        "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
                    }
                    and phase == "align_over_box"
                    and (phase_changed or state_machine.step_count % 25 == 0)
                ):
                    pan_index = robot.data.joint_names.index("shoulder_pan")
                    pan_target = state_machine.shoulder_pan_target
                    pan_entry = state_machine.shoulder_pan_entry
                    bearing_error = state_machine.bearing_error
                    delta_joint_pos = arm_action_term.last_delta_joint_pos
                    task_error = arm_action_term.last_task_error
                    singular_values = arm_action_term.last_task_singular_values
                    primary_delta = arm_action_term.last_primary_delta_joint_pos
                    nullspace_delta = arm_action_term.last_nullspace_delta_joint_pos
                    unlimited_delta = arm_action_term.last_unlimited_delta_joint_pos
                    joint_position_target = arm_action_term.last_joint_position_target
                    joint_target_slew_step = arm_action_term.last_joint_target_slew_step
                    pan_entry_value = None if pan_entry is None else round(pan_entry[0].item(), 7)
                    pan_target_value = None if pan_target is None else round(pan_target[0].item(), 7)
                    bearing_error_value = None if bearing_error is None else round(bearing_error[0].item(), 7)
                    print(
                        f"expert_pan_objective:{phase}:"
                        f"entry={pan_entry_value}:"
                        f"target={pan_target_value}:"
                        f"actual={robot.data.joint_pos[0, pan_index].item():.7f}:"
                        f"bearing_error={bearing_error_value}:"
                        f"joint_names={arm_action_term.controlled_joint_names}:"
                        f"delta_joint_pos={None if delta_joint_pos is None else _rounded_row(delta_joint_pos[0], digits=7)}:"
                        f"primary_delta={None if primary_delta is None else _rounded_row(primary_delta[0], digits=7)}:"
                        f"nullspace_delta={None if nullspace_delta is None else _rounded_row(nullspace_delta[0], digits=7)}:"
                        f"unlimited_delta={None if unlimited_delta is None else _rounded_row(unlimited_delta[0], digits=7)}:"
                        f"joint_position_target={None if joint_position_target is None else _rounded_row(joint_position_target[0], digits=7)}:"
                        f"joint_target_slew_step={None if joint_target_slew_step is None else _rounded_row(joint_target_slew_step[0], digits=7)}:"
                        f"task_error={None if task_error is None else _rounded_row(task_error[0], digits=7)}:"
                        f"singular_values={None if singular_values is None else _rounded_row(singular_values[0], digits=7)}",
                        flush=True,
                    )
                if args.expert in {
                    "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
                    "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
                    "jaw_frame_xyz_tilt",
                } and (phase_changed or state_machine.step_count % 25 == 0):
                    print(
                        f"expert_direct_jaw_control:{phase}:"
                        f"desired_jaw_w={None if state_machine.last_desired_jaw_w is None else _rounded_row(state_machine.last_desired_jaw_w[0], digits=7)}:"
                        f"offset_pos={None if state_machine.direct_jaw_offset_pos is None else _rounded_row(state_machine.direct_jaw_offset_pos[0], digits=7)}:"
                        f"offset_quat={None if state_machine.direct_jaw_offset_quat is None else _rounded_row(state_machine.direct_jaw_offset_quat[0], digits=7)}:"
                        f"controlled_point_w={None if state_machine.controlled_jaw_point_w is None else _rounded_row(state_machine.controlled_jaw_point_w[0], digits=7)}:"
                        f"table_projection_w={None if state_machine.table_projection_point_w is None else _rounded_row(state_machine.table_projection_point_w[0], digits=7)}",
                        flush=True,
                    )
                    if args.expert == "jaw_frame_xyz_tilt":
                        five_dimensional_error = state_machine.last_five_dimensional_error
                        print(
                            f"expert_jaw_five_dimensional_task:{phase}:"
                            f"error_xyz_roll_pitch="
                            f"{None if five_dimensional_error is None else _rounded_row(five_dimensional_error[0], digits=7)}:"
                            f"alignment_streak={state_machine.alignment_streak}",
                            flush=True,
                        )
                if (
                    args.expert == "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower"
                    and phase == "transfer_to_box"
                    and (phase_changed or state_machine.step_count % 25 == 0)
                ):
                    print(
                        f"expert_transfer_target:{phase}:"
                        f"floor_center_w={_rounded_row(floor.data.root_pos_w[0], digits=7)}:"
                        f"start_jaw_w={None if state_machine.transfer_start_jaw_w is None else _rounded_row(state_machine.transfer_start_jaw_w[0], digits=7)}:"
                        f"target_jaw_w={None if state_machine.transfer_target_jaw_w is None else _rounded_row(state_machine.transfer_target_jaw_w[0], digits=7)}:"
                        f"actual_jaw_w={_rounded_row(env.scene['ee_frame'].data.target_pos_w[0, 1], digits=7)}",
                        flush=True,
                    )
                if (
                    args.expert
                    in {
                        "legacy_gripper_anchor",
                        "legacy_gripper_anchor_align_then_lower",
                        "legacy_gripper_anchor_position_align_then_lower",
                        "legacy_gripper_anchor_weighted_position_align_then_lower",
                        "legacy_gripper_anchor_safe_planar_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_tilt_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_pitch_pan_align_then_lower",
                        "legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower",
                        "legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower",
                        "legacy_gripper_anchor_safe_jaw_trajectory_direct_xyz_pan_nullspace_align_then_lower",
                        "legacy_gripper_anchor_relaxed_ik",
                        "legacy_gripper_anchor_planar_ik",
                        "servo",
                        "weighted_servo",
                        "legacy_weighted_servo",
                        "legacy_position_servo",
                        "legacy_pd_position_servo",
                        "legacy_trajectory_pd_servo",
                        "autogen_retreat_transport",
                        "autogen_independent_retreat_transport",
                        "autogen_polar_retreat_transport",
                    }
                    and phase
                    in {
                        "lift_cube",
                        "retreat_to_safe",
                        "transfer_to_box",
                        "arc_transfer",
                        "radial_transfer",
                        "lower_into_box",
                        "align_over_box",
                    }
                    and (phase_changed or state_machine.step_count % 25 == 0)
                ):
                    gripper_pos = ee_frame.data.target_pos_w[0, 0]
                    jaw_pos = ee_frame.data.target_pos_w[0, 1]
                    cube_pos = cube.data.root_pos_w[0]
                    _print_fields(
                        f"expert_tracking:{phase}",
                        step=state_machine.step_count,
                        gripper_pos_w=_rounded_row(gripper_pos),
                        gripper_quat_w=_rounded_row(ee_frame.data.target_quat_w[0, 0]),
                        jaw_pos_w=_rounded_row(jaw_pos),
                        cube_pos_w=_rounded_row(cube_pos),
                        jaw_cube_distance=f"{torch.linalg.vector_norm(jaw_pos - cube_pos).item():.6f}",
                        joint_pos=_rounded_row(robot.data.joint_pos[0]),
                        pick_cube=bool(observations["subtask_terms"]["pick_cube"][0].item()),
                    )
                if (
                    args.expert
                    in {
                        "servo",
                        "weighted_servo",
                        "legacy_weighted_servo",
                        "legacy_position_servo",
                        "legacy_pd_position_servo",
                        "legacy_trajectory_pd_servo",
                    }
                    and phase
                    in {
                        "transfer_to_box",
                        "lower_into_box",
                        "align_over_box",
                    }
                    and (phase_changed or state_machine.step_count % 50 == 0)
                ):
                    servo_delta = state_machine.last_servo_delta_w
                    servo_error_norm = state_machine.last_servo_error_norm
                    servo_velocity_norm = state_machine.last_servo_velocity_norm
                    if servo_delta is not None and servo_error_norm is not None:
                        print(
                            f"expert_servo:{phase}:"
                            f"error_norm={servo_error_norm[0, 0].item():.6f}:"
                            f"velocity_norm={servo_velocity_norm[0, 0].item():.6f}:"
                            f"delta_w={_rounded_row(servo_delta[0])}:"
                            f"stable_streak={state_machine.servo_stable_streak}",
                            flush=True,
                        )

                if state_machine.is_episode_done:
                    print(
                        f"expert_abort_before_step:{getattr(state_machine, 'servo_abort_reason', None)}",
                        flush=True,
                    )
                    break

                step_result = env.step(action)
                observations = step_result[0]
                all_rewards_finite = all_rewards_finite and bool(torch.isfinite(step_result[1]).all())
                unexpected_reset = unexpected_reset or bool(step_result[2].any()) or bool(step_result[3].any())
                pick_cube_after = bool(observations["subtask_terms"]["pick_cube"][0].item())
                observe_pick_cube = getattr(state_machine, "observe_pick_cube", None)
                if callable(observe_pick_cube) and observe_pick_cube(pick_cube_after, env):
                    print(
                        f"expert_temp_jaw_angle_captured:phase={state_machine.phase_name}:"
                        f"control_step={completed_steps + 1}:"
                        f"angle_rad={state_machine.held_gripper_angle:.7f}:"
                        f"gripper_velocity_rad_s={abs(robot.data.joint_vel[0, -1].item()):.7f}",
                        flush=True,
                    )
                if previous_pick_cube and not pick_cube_after:
                    print(
                        f"expert_grasp_event:lost:phase={phase}:"
                        f"state_step={state_machine.step_count}:control_step={completed_steps + 1}",
                        flush=True,
                    )
                previous_pick_cube = pick_cube_after
                state_machine.advance()
                completed_steps += 1
                if recorder is not None:
                    recorder.capture(completed_steps, phase, observations, env, state_machine)

        success = state_machine.check_success(env)
        cube_offset = cube.data.root_pos_w - floor.data.root_pos_w
        cube_speed = torch.linalg.vector_norm(cube.data.root_lin_vel_w, dim=-1)

        print(f"completed_steps: {completed_steps}", flush=True)
        print(f"all_rewards_finite: {all_rewards_finite}", flush=True)
        print(f"unexpected_reset: {unexpected_reset}", flush=True)
        print(f"cube_final_pos_w: {_rounded_row(cube.data.root_pos_w[0])}", flush=True)
        print(f"cube_offset_from_box: {_rounded_row(cube_offset[0])}", flush=True)
        print(f"cube_final_speed: {cube_speed[0].item():.6f}", flush=True)
        print(f"pick_cube_final: {bool(observations['subtask_terms']['pick_cube'][0].item())}", flush=True)
        print(f"grasp_confirmed: {getattr(state_machine, 'grasp_confirmed', 'not_tracked')}", flush=True)
        print(
            f"grasp_lost_before_release: {getattr(state_machine, 'grasp_lost_before_release', 'not_tracked')}",
            flush=True,
        )
        print(f"retry_used: {getattr(state_machine, 'retry_used', 'not_tracked')}", flush=True)
        print(
            f"box_aligned_before_release: {getattr(state_machine, 'box_aligned_before_release', 'not_tracked')}",
            flush=True,
        )
        servo_timeout_phase = getattr(state_machine, "servo_timeout_phase", None)
        servo_abort_reason = getattr(state_machine, "servo_abort_reason", None)
        release_block_reason = getattr(state_machine, "release_block_reason", None)
        print(f"servo_timeout_phase: {servo_timeout_phase}", flush=True)
        print(f"servo_abort_reason: {servo_abort_reason}", flush=True)
        print(f"release_block_reason: {release_block_reason}", flush=True)
        print(f"expert_success: {success}", flush=True)

        failure_message = None
        if not all_rewards_finite:
            failure_message = "A non-finite reward was observed"
        elif unexpected_reset:
            failure_message = "The environment reset before the expert episode completed"
        elif servo_abort_reason is not None:
            failure_message = f"The expert aborted: {servo_abort_reason}"
        elif servo_timeout_phase is not None:
            failure_message = f"The expert timed out in phase: {servo_timeout_phase}"
        elif release_block_reason is not None:
            failure_message = f"Release was safely blocked: {release_block_reason}"
        elif not success:
            failure_message = "The scripted expert did not place a settled cube inside the target box"

        if recorder is not None:
            recorder.capture(
                completed_steps,
                state_machine.phase_name,
                observations,
                env,
                state_machine,
                force=True,
            )
            recorder.finish(
                {
                    "status": "failed" if failure_message else "passed",
                    "expert": args.expert,
                    "seed": args.seed,
                    "completed_steps": completed_steps,
                    "expert_success": success,
                    "failure_message": failure_message,
                    "box_aligned_before_release": getattr(
                        state_machine,
                        "box_aligned_before_release",
                        None,
                    ),
                    "timeout_phase": servo_timeout_phase,
                    "cube_final_pos_w": _rounded_row(cube.data.root_pos_w[0]),
                    "cube_offset_from_box": _rounded_row(cube_offset[0]),
                    "gripper_final_pos_w": _rounded_row(ee_frame.data.target_pos_w[0, 0]),
                    "jaw_final_pos_w": _rounded_row(ee_frame.data.target_pos_w[0, 1]),
                    "safety_mode": getattr(state_machine, "safety_mode", None),
                    "cube_clearance": _finite_or_none(getattr(state_machine, "cube_clearance", None)),
                    "minimum_robot_clearance": _finite_or_none(getattr(state_machine, "minimum_robot_clearance", None)),
                    "maximum_box_contact_force": _finite_or_none(
                        getattr(state_machine, "maximum_box_contact_force", None)
                    ),
                }
            )

        if failure_message is not None:
            raise RuntimeError(failure_message)

        print("RED_CUBE_TO_BOX_EXPERT_SMOKE_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        print("RED_CUBE_TO_BOX_EXPERT_SMOKE_FAILED", flush=True)
    finally:
        if recorder is not None:
            try:
                recorder.finish(
                    {
                        "status": "passed" if status == 0 else "failed",
                        "expert": args.expert,
                        "seed": args.seed,
                        "completed_steps": completed_steps,
                    }
                )
                print(f"diagnostic_record_saved: {recorder.run_dir}", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"diagnostic_record_finalize_failed: {recorder.run_dir}", flush=True)
        print("RED_CUBE_TO_BOX_EXPERT_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
