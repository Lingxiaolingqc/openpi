"""Smoothly transfer the jaw to the box center before direct-jaw high alignment."""

from __future__ import annotations

import torch

from .legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine,
)

_GRIPPER_CLOSE = -1.0
_TRANSFER_MOVE_STEPS = 120
_TRANSFER_HOLD_STEPS = 40
_TRANSFER_JAW_HEIGHT_ABOVE_FLOOR = 0.188
_JOINT_TARGET_SLEW_MAX_STEP = 0.01


class RedCubeToBoxLegacyGripperAnchorSafeJawTrajectoryDirectXyzPanNullspaceAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine
):
    """Replace the empirical transfer offset and current-relative align clamp."""

    def __init__(self) -> None:
        super().__init__()
        self._transfer_start_jaw_w: torch.Tensor | None = None
        self._transfer_target_jaw_w: torch.Tensor | None = None

    def setup(self, env) -> None:
        super().setup(env)
        assert self._arm_action_term is not None
        self._arm_action_term.set_maximum_joint_target_step(maximum_step=None)
        self._arm_action_term.set_joint_target_slew_limit(maximum_step=_JOINT_TARGET_SLEW_MAX_STEP)

    def get_action(self, env) -> torch.Tensor:
        phase_name, phase_step, _ = self._phase_state()
        if phase_name != "transfer_to_box":
            return super().get_action(env)

        assert self._arm_action_term is not None
        self._arm_action_term.set_identity_control_frame_offset()
        self._arm_action_term.reset_joint_target_slew_reference()
        self._update_point_visualization(env)
        self._initialize_anchors(env)
        assert self._floor_anchor is not None

        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        gripper_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]
        if self._transfer_start_jaw_w is None:
            self._transfer_start_jaw_w = jaw_pos_w.clone()
            self._transfer_target_jaw_w = self._floor_anchor.clone()
            self._transfer_target_jaw_w[:, 2] += _TRANSFER_JAW_HEIGHT_ABOVE_FLOOR

        assert self._transfer_target_jaw_w is not None
        progress = min(phase_step / max(_TRANSFER_MOVE_STEPS - 1, 1), 1.0)
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        desired_jaw_w = self._transfer_start_jaw_w + smooth_progress * (
            self._transfer_target_jaw_w - self._transfer_start_jaw_w
        )
        jaw_error_w = desired_jaw_w - jaw_pos_w
        target_pos_w = gripper_pos_w + jaw_error_w
        self._last_desired_jaw_w = desired_jaw_w.clone()
        self._last_jaw_error_w = jaw_error_w.clone()
        self._last_gripper_target_w = target_pos_w.clone()
        return self._compose_pose_action(env, robot, target_pos_w, _GRIPPER_CLOSE)

    def reset(self) -> None:
        super().reset()
        self._transfer_start_jaw_w = None
        self._transfer_target_jaw_w = None

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "transfer_target_reference": "target_box_floor_center_jaw_frame",
                "transfer_trajectory": "cubic_smoothstep_then_hold",
                "transfer_move_steps": _TRANSFER_MOVE_STEPS,
                "transfer_hold_steps": _TRANSFER_HOLD_STEPS,
                "transfer_jaw_height_above_floor": _TRANSFER_JAW_HEIGHT_ABOVE_FLOOR,
                "align_joint_limit_mode": "persistent_joint_target_slew",
                "align_joint_target_slew_max_step": _JOINT_TARGET_SLEW_MAX_STEP,
            }
        )
        return parameters

    @property
    def transfer_start_jaw_w(self) -> torch.Tensor | None:
        return self._transfer_start_jaw_w

    @property
    def transfer_target_jaw_w(self) -> torch.Tensor | None:
        return self._transfer_target_jaw_w
