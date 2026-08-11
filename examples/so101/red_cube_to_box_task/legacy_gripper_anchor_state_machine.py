"""Legacy SO-101 pickup with jaw-anchored placement."""

from __future__ import annotations

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
import torch

from .state_machine import RedCubeToBoxStateMachine

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_JAW_RELEASE_HEIGHT = 0.09
_JAW_RETRACT_HEIGHT = 0.12
_JAW_XY_THRESHOLD = 0.006
_JAW_Z_THRESHOLD = 0.008
_MIN_ALIGNMENT_STREAK = 10
_PLACEMENT_PHASES = {
    "lower_into_box",
    "align_over_box",
    "release_cube",
    "retract_gripper",
    "settle",
}


class RedCubeToBoxLegacyGripperAnchorStateMachine(RedCubeToBoxStateMachine):
    """Keep legacy pickup and transport, then place through the measured jaw frame.

    Isaac Lab's IK action commands the gripper frame, while the grasped cube follows
    the offset jaw detection frame. During placement this expert measures that full
    3D offset on every step and converts a desired jaw pose into a gripper target.
    """

    _PHASES = (
        ("approach_cube", 120),
        ("descend_to_cube", 120),
        ("close_gripper", 80),
        ("lift_cube", 120),
        ("transfer_to_box", 160),
        ("lower_into_box", 120),
        ("align_over_box", 240),
        ("release_cube", 100),
        ("retract_gripper", 100),
        ("settle", 180),
    )
    MAX_STEPS = sum(duration for _, duration in _PHASES)

    def __init__(self) -> None:
        super().__init__()
        self._last_phase: str | None = None
        self._lower_start_jaw_w: torch.Tensor | None = None
        self._alignment_streak = 0
        self._box_aligned_before_release = False
        self._servo_timeout_phase: str | None = None
        self._last_desired_jaw_w: torch.Tensor | None = None
        self._last_jaw_error_w: torch.Tensor | None = None
        self._last_gripper_target_w: torch.Tensor | None = None

    def get_action(self, env) -> torch.Tensor:
        """Return the unmodified legacy action until placement begins."""

        phase_name, phase_step, phase_duration = self._phase_state()
        if phase_name not in _PLACEMENT_PHASES:
            self._clear_placement_diagnostics()
            return super().get_action(env)

        self._initialize_anchors(env)
        assert self._floor_anchor is not None

        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        gripper_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]
        self._on_phase_entry(phase_name, jaw_pos_w)

        release_jaw_w = self._release_jaw_target()
        if phase_name == "lower_into_box":
            assert self._lower_start_jaw_w is not None
            desired_jaw_w = self._interpolate(
                self._lower_start_jaw_w,
                release_jaw_w,
                phase_step,
                phase_duration,
            )
            gripper = _GRIPPER_CLOSE
        elif phase_name == "align_over_box":
            desired_jaw_w = release_jaw_w
            gripper = _GRIPPER_CLOSE
        elif phase_name == "release_cube":
            desired_jaw_w = release_jaw_w
            gripper = _GRIPPER_OPEN
        elif phase_name == "retract_gripper":
            retract_jaw_w = release_jaw_w.clone()
            retract_jaw_w[:, 2] += _JAW_RETRACT_HEIGHT
            desired_jaw_w = self._interpolate(
                release_jaw_w,
                retract_jaw_w,
                phase_step,
                phase_duration,
            )
            gripper = _GRIPPER_OPEN
        else:
            desired_jaw_w = release_jaw_w.clone()
            desired_jaw_w[:, 2] += _JAW_RETRACT_HEIGHT
            gripper = _GRIPPER_OPEN

        # Algebraically this is desired_jaw_w - (jaw_pos_w - gripper_pos_w).
        # Expressing it as feedback makes the controlled reference explicit.
        jaw_error_w = desired_jaw_w - jaw_pos_w
        target_pos_w = gripper_pos_w + jaw_error_w
        self._last_desired_jaw_w = desired_jaw_w.clone()
        self._last_jaw_error_w = jaw_error_w.clone()
        self._last_gripper_target_w = target_pos_w.clone()
        self._update_alignment(phase_name, jaw_error_w)

        return self._compose_pose_action(env, robot, target_pos_w, gripper)

    def advance(self) -> None:
        phase_name, phase_step, phase_duration = self._phase_state()
        if phase_name == "align_over_box":
            if self._box_aligned_before_release:
                self._step_count = self._phase_start("release_cube")
                return
            if phase_step + 1 >= phase_duration:
                self._servo_timeout_phase = phase_name
                self._episode_done = True
                return
        super().advance()

    def reset(self) -> None:
        super().reset()
        self._last_phase = None
        self._lower_start_jaw_w = None
        self._alignment_streak = 0
        self._box_aligned_before_release = False
        self._servo_timeout_phase = None
        self._clear_placement_diagnostics()

    def _on_phase_entry(self, phase_name: str, jaw_pos_w: torch.Tensor) -> None:
        if phase_name == self._last_phase:
            return
        self._last_phase = phase_name
        if phase_name == "lower_into_box":
            self._lower_start_jaw_w = jaw_pos_w.clone()

    def _release_jaw_target(self) -> torch.Tensor:
        assert self._floor_anchor is not None
        target = self._floor_anchor.clone()
        target[:, 2] += _JAW_RELEASE_HEIGHT
        return target

    def _update_alignment(self, phase_name: str, jaw_error_w: torch.Tensor) -> None:
        if phase_name != "align_over_box":
            return
        xy_error = torch.linalg.vector_norm(jaw_error_w[:, :2], dim=-1)
        z_error = torch.abs(jaw_error_w[:, 2])
        aligned = torch.logical_and(xy_error < _JAW_XY_THRESHOLD, z_error < _JAW_Z_THRESHOLD)
        if bool(aligned.all().item()):
            self._alignment_streak += 1
        else:
            self._alignment_streak = 0
        self._box_aligned_before_release = self._alignment_streak >= _MIN_ALIGNMENT_STREAK

    def _clear_placement_diagnostics(self) -> None:
        self._last_desired_jaw_w = None
        self._last_jaw_error_w = None
        self._last_gripper_target_w = None

    @staticmethod
    def _compose_pose_action(
        env,
        robot,
        target_pos_w: torch.Tensor,
        gripper: float,
        target_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        robot_base_pos_w = robot.data.root_pos_w
        robot_base_quat_w = robot.data.root_quat_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), target_pos_w - robot_base_pos_w)
        if target_quat_w is None:
            zero = torch.zeros((), device=env.device)
            target_quat_w = quat_from_euler_xyz(zero, zero, zero).repeat(env.num_envs, 1)
        target_quat_local = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)
        gripper_command = torch.full((env.num_envs, 1), gripper, device=env.device)
        return torch.cat([target_pos_local, target_quat_local, gripper_command], dim=-1)

    def _phase_start(self, target_phase: str) -> int:
        phase_start = 0
        for name, duration in self._PHASES:
            if name == target_phase:
                return phase_start
            phase_start += duration
        raise ValueError(f"Unknown phase: {target_phase}")

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        return {
            "pickup_controller": "legacy_exact",
            "transport_controller": "legacy_exact",
            "placement_reference": "jaw_detection_frame",
            "jaw_release_height": _JAW_RELEASE_HEIGHT,
            "jaw_xy_threshold": _JAW_XY_THRESHOLD,
            "jaw_z_threshold": _JAW_Z_THRESHOLD,
            "stable_steps": _MIN_ALIGNMENT_STREAK,
        }

    @property
    def last_desired_jaw_w(self) -> torch.Tensor | None:
        return self._last_desired_jaw_w

    @property
    def last_jaw_error_w(self) -> torch.Tensor | None:
        return self._last_jaw_error_w

    @property
    def last_gripper_target_w(self) -> torch.Tensor | None:
        return self._last_gripper_target_w

    @property
    def alignment_streak(self) -> int:
        return self._alignment_streak

    @property
    def box_aligned_before_release(self) -> bool:
        return self._box_aligned_before_release

    @property
    def servo_timeout_phase(self) -> str | None:
        return self._servo_timeout_phase

    @property
    def servo_abort_reason(self) -> str | None:
        return None
