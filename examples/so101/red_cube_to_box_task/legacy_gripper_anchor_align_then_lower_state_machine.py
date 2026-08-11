"""Legacy jaw-anchor expert that aligns above the box before descending."""

from __future__ import annotations

import torch

from .legacy_gripper_anchor_state_machine import RedCubeToBoxLegacyGripperAnchorStateMachine
from .state_machine import RedCubeToBoxStateMachine

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_JAW_XY_THRESHOLD = 0.006
_JAW_Z_THRESHOLD = 0.008
_MIN_ALIGNMENT_STREAK = 10
_DESCENT_STEPS = 120
_DESCENT_CONFIRMATION_STEPS = 60


class RedCubeToBoxLegacyGripperAnchorAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorStateMachine
):
    """Preserve legacy pose IK while swapping horizontal alignment and descent."""

    _PHASES = (
        ("approach_cube", 120),
        ("descend_to_cube", 120),
        ("close_gripper", 80),
        ("lift_cube", 120),
        ("transfer_to_box", 160),
        ("align_over_box", 240),
        ("lower_into_box", _DESCENT_STEPS + _DESCENT_CONFIRMATION_STEPS),
        ("release_cube", 100),
        ("retract_gripper", 100),
        ("settle", 180),
    )
    MAX_STEPS = sum(duration for _, duration in _PHASES)

    def __init__(self) -> None:
        super().__init__()
        self._align_start_jaw_w: torch.Tensor | None = None
        self._horizontal_alignment_streak = 0
        self._final_alignment_streak = 0

    def get_action(self, env) -> torch.Tensor:
        phase_name, phase_step, phase_duration = self._phase_state()
        if phase_name not in {
            "align_over_box",
            "lower_into_box",
            "release_cube",
            "retract_gripper",
            "settle",
        }:
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
        if phase_name == "align_over_box":
            assert self._align_start_jaw_w is not None
            desired_jaw_w = self._align_start_jaw_w.clone()
            desired_jaw_w[:, :2] = self._floor_anchor[:, :2]
            gripper = _GRIPPER_CLOSE
        elif phase_name == "lower_into_box":
            assert self._lower_start_jaw_w is not None
            lower_start_w = self._lower_start_jaw_w.clone()
            lower_start_w[:, :2] = self._floor_anchor[:, :2]
            desired_jaw_w = self._interpolate(
                lower_start_w,
                release_jaw_w,
                phase_step,
                _DESCENT_STEPS,
            )
            gripper = _GRIPPER_CLOSE
        elif phase_name == "release_cube":
            desired_jaw_w = release_jaw_w
            gripper = _GRIPPER_OPEN
        elif phase_name == "retract_gripper":
            retract_jaw_w = release_jaw_w.clone()
            retract_jaw_w[:, 2] += 0.12
            desired_jaw_w = self._interpolate(
                release_jaw_w,
                retract_jaw_w,
                phase_step,
                phase_duration,
            )
            gripper = _GRIPPER_OPEN
        else:
            desired_jaw_w = release_jaw_w.clone()
            desired_jaw_w[:, 2] += 0.12
            gripper = _GRIPPER_OPEN

        jaw_error_w = desired_jaw_w - jaw_pos_w
        target_pos_w = gripper_pos_w + jaw_error_w
        self._last_desired_jaw_w = desired_jaw_w.clone()
        self._last_jaw_error_w = jaw_error_w.clone()
        self._last_gripper_target_w = target_pos_w.clone()
        self._update_phase_alignment(phase_name, jaw_pos_w, release_jaw_w)
        return self._compose_pose_action(env, robot, target_pos_w, gripper)

    def advance(self) -> None:
        phase_name, phase_step, phase_duration = self._phase_state()
        if phase_name == "align_over_box":
            if self._horizontal_alignment_streak >= _MIN_ALIGNMENT_STREAK:
                self._step_count = self._phase_start("lower_into_box")
                return
            if phase_step + 1 >= phase_duration:
                self._servo_timeout_phase = phase_name
                self._episode_done = True
                return
        elif phase_name == "lower_into_box":
            if self._box_aligned_before_release:
                self._step_count = self._phase_start("release_cube")
                return
            if phase_step + 1 >= phase_duration:
                self._servo_timeout_phase = phase_name
                self._episode_done = True
                return
        RedCubeToBoxStateMachine.advance(self)

    def reset(self) -> None:
        super().reset()
        self._align_start_jaw_w = None
        self._horizontal_alignment_streak = 0
        self._final_alignment_streak = 0

    def _on_phase_entry(self, phase_name: str, jaw_pos_w: torch.Tensor) -> None:
        if phase_name == self._last_phase:
            return
        self._last_phase = phase_name
        if phase_name == "align_over_box":
            self._align_start_jaw_w = jaw_pos_w.clone()
        elif phase_name == "lower_into_box":
            self._lower_start_jaw_w = jaw_pos_w.clone()

    def _update_phase_alignment(
        self,
        phase_name: str,
        jaw_pos_w: torch.Tensor,
        release_jaw_w: torch.Tensor,
    ) -> None:
        assert self._floor_anchor is not None
        xy_error = torch.linalg.vector_norm(jaw_pos_w[:, :2] - self._floor_anchor[:, :2], dim=-1)
        xy_aligned = bool((xy_error < _JAW_XY_THRESHOLD).all().item())

        if phase_name == "align_over_box":
            self._horizontal_alignment_streak = self._horizontal_alignment_streak + 1 if xy_aligned else 0
            self._alignment_streak = self._horizontal_alignment_streak
            return

        if phase_name != "lower_into_box":
            return
        z_error = torch.abs(jaw_pos_w[:, 2] - release_jaw_w[:, 2])
        final_aligned = xy_aligned and bool((z_error < _JAW_Z_THRESHOLD).all().item())
        self._final_alignment_streak = self._final_alignment_streak + 1 if final_aligned else 0
        self._alignment_streak = self._final_alignment_streak
        self._box_aligned_before_release = self._final_alignment_streak >= _MIN_ALIGNMENT_STREAK

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "placement_order": "align_xy_high_then_descend_z",
                "horizontal_alignment_height": "measured_at_phase_entry",
                "descent_xy_reference": "target_box_floor_center",
                "descent_steps": _DESCENT_STEPS,
                "descent_confirmation_budget": _DESCENT_CONFIRMATION_STEPS,
            }
        )
        return parameters
