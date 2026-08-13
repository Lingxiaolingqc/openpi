"""Control and evaluate every RedCubeToBox phase at the jaw frame."""

from __future__ import annotations

from isaaclab.utils.math import compute_pose_error
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
from isaaclab.utils.math import subtract_frame_transforms
import torch

from ..state_machine import RedCubeToBoxStateMachine
from .legacy_gripper_anchor_safe_direct_jaw_xyz_pan_nullspace_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine,
)

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_PICK_HOVER_HEIGHT = 0.20
_PICK_LIFT_HEIGHT = 0.18
_BOX_HOVER_HEIGHT = 0.188
_JAW_RELEASE_HEIGHT = 0.09
_JAW_RETRACT_HEIGHT = 0.12
_ALIGN_POSITION_THRESHOLD = 0.006
_ALIGN_TILT_THRESHOLD = 0.05
_MIN_ALIGNMENT_STREAK = 10
_JOINT_TARGET_SLEW_MAX_STEP = 0.01


class RedCubeToBoxJawFrameXyzTiltStateMachine(
    RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine
):
    """Use the live jaw frame and a five-row XYZ-plus-tilt task throughout.

    The virtual IK body offset is recomputed every control step. This matters
    while the gripper opens or closes: the jaw-to-gripper transform is not a
    constant rigid offset over the complete episode.
    """

    _PHASES = (
        ("approach_cube", 120),
        ("descend_to_cube", 120),
        ("close_gripper", 80),
        ("lift_cube", 120),
        ("transfer_to_box", 160),
        ("align_over_box", 240),
        ("lower_into_box", 120),
        ("release_cube", 100),
        ("retract_gripper", 100),
        ("settle", 180),
    )
    MAX_STEPS = sum(duration for _, duration in _PHASES)

    def __init__(self) -> None:
        super().__init__()
        self._initial_jaw_pos_w: torch.Tensor | None = None
        self._initial_jaw_quat_w: torch.Tensor | None = None
        self._last_five_dimensional_error: torch.Tensor | None = None

    def setup(self, env) -> None:
        super().setup(env)
        assert self._arm_action_term is not None
        self._arm_action_term.set_maximum_joint_target_step(maximum_step=None)
        self._arm_action_term.set_joint_target_slew_limit(maximum_step=_JOINT_TARGET_SLEW_MAX_STEP)

    def get_action(self, env) -> torch.Tensor:
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a jaw-frame action")

        self._initialize_jaw_anchors(env)
        assert self._initial_jaw_pos_w is not None
        assert self._initial_jaw_quat_w is not None
        assert self._cube_anchor is not None
        assert self._floor_anchor is not None

        robot = env.scene["robot"]
        ee_frame = env.scene["ee_frame"]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]
        jaw_quat_w = ee_frame.data.target_quat_w[:, 1, :]

        # The gripper joint changes the measured jaw frame. Updating this
        # transform every step keeps both the error and Jacobian jaw-relative.
        offset_pos, offset_quat = subtract_frame_transforms(
            ee_frame.data.target_pos_w[:, 0, :],
            ee_frame.data.target_quat_w[:, 0, :],
            jaw_pos_w,
            jaw_quat_w,
        )
        self._direct_jaw_offset_pos = offset_pos.detach().clone()
        self._direct_jaw_offset_quat = offset_quat.detach().clone()
        self._arm_action_term.set_control_frame_offset(position=offset_pos, orientation=offset_quat)
        self._arm_action_term.set_xyz_tilt(enabled=True)
        self._safety_mode = "live_jaw_frame_xyz_plus_world_tilt_free_yaw"
        self._update_point_visualization(env)

        phase_name, phase_step, phase_duration = self._phase_state()
        pick_hover = self._cube_anchor.clone()
        pick_hover[:, 2] += _PICK_HOVER_HEIGHT
        pick_grasp = self._cube_anchor.clone()
        pick_lift = self._cube_anchor.clone()
        pick_lift[:, 2] += _PICK_LIFT_HEIGHT
        box_hover = self._floor_anchor.clone()
        box_hover[:, 2] += _BOX_HOVER_HEIGHT
        box_release = self._floor_anchor.clone()
        box_release[:, 2] += _JAW_RELEASE_HEIGHT
        box_retract = box_release.clone()
        box_retract[:, 2] += _JAW_RETRACT_HEIGHT

        if phase_name == "approach_cube":
            desired_jaw_w = self._interpolate(
                self._initial_jaw_pos_w,
                pick_hover,
                phase_step,
                phase_duration,
            )
            gripper = _GRIPPER_OPEN
        elif phase_name == "descend_to_cube":
            desired_jaw_w = self._interpolate(pick_hover, pick_grasp, phase_step, phase_duration)
            gripper = _GRIPPER_OPEN
        elif phase_name == "close_gripper":
            desired_jaw_w = pick_grasp
            gripper = _GRIPPER_CLOSE
        elif phase_name == "lift_cube":
            desired_jaw_w = self._interpolate(pick_grasp, pick_lift, phase_step, phase_duration)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "transfer_to_box":
            desired_jaw_w = self._interpolate(pick_lift, box_hover, phase_step, phase_duration)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "align_over_box":
            desired_jaw_w = box_hover
            gripper = _GRIPPER_CLOSE
        elif phase_name == "lower_into_box":
            desired_jaw_w = self._interpolate(box_hover, box_release, phase_step, phase_duration)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "release_cube":
            desired_jaw_w = box_release
            gripper = _GRIPPER_OPEN
        elif phase_name == "retract_gripper":
            desired_jaw_w = self._interpolate(box_release, box_retract, phase_step, phase_duration)
            gripper = _GRIPPER_OPEN
        else:
            desired_jaw_w = box_retract
            gripper = _GRIPPER_OPEN

        position_error, orientation_error = compute_pose_error(
            jaw_pos_w,
            jaw_quat_w,
            desired_jaw_w,
            self._initial_jaw_quat_w,
            rot_error_type="axis_angle",
        )
        self._last_desired_jaw_w = desired_jaw_w.detach().clone()
        self._last_jaw_error_w = position_error.detach().clone()
        self._last_gripper_target_w = None
        self._last_five_dimensional_error = (
            torch.cat(
                (position_error, orientation_error[:, :2]),
                dim=-1,
            )
            .detach()
            .clone()
        )
        self._update_jaw_alignment(phase_name, position_error, orientation_error)

        target_pos_local = quat_apply(quat_inv(robot.data.root_quat_w), desired_jaw_w - robot.data.root_pos_w)
        target_quat_local = quat_mul(quat_inv(robot.data.root_quat_w), self._initial_jaw_quat_w)
        gripper_command = torch.full((env.num_envs, 1), gripper, device=env.device)
        return torch.cat((target_pos_local, target_quat_local, gripper_command), dim=-1)

    def advance(self) -> None:
        phase_name, phase_step, phase_duration = self._phase_state()
        if phase_name == "align_over_box":
            if self._box_aligned_before_release:
                self._step_count = self._phase_start("lower_into_box")
                return
            if phase_step + 1 >= phase_duration:
                self._servo_timeout_phase = phase_name
                self._episode_done = True
                return
        RedCubeToBoxStateMachine.advance(self)

    def reset(self) -> None:
        super().reset()
        self._initial_jaw_pos_w = None
        self._initial_jaw_quat_w = None
        self._last_five_dimensional_error = None
        if self._arm_action_term is not None:
            self._arm_action_term.reset_joint_target_slew_reference()

    def _initialize_jaw_anchors(self, env) -> None:
        self._initialize_anchors(env)
        if self._initial_jaw_pos_w is not None:
            return
        ee_frame = env.scene["ee_frame"]
        self._initial_jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :].clone()
        self._initial_jaw_quat_w = ee_frame.data.target_quat_w[:, 1, :].clone()

    def _update_jaw_alignment(
        self,
        phase_name: str,
        position_error: torch.Tensor,
        orientation_error: torch.Tensor,
    ) -> None:
        if phase_name != "align_over_box":
            self._alignment_streak = 0
            self._box_aligned_before_release = False
            return
        position_aligned = torch.linalg.vector_norm(position_error, dim=-1) < _ALIGN_POSITION_THRESHOLD
        tilt_aligned = torch.linalg.vector_norm(orientation_error[:, :2], dim=-1) < _ALIGN_TILT_THRESHOLD
        if bool(torch.logical_and(position_aligned, tilt_aligned).all().item()):
            self._alignment_streak += 1
        else:
            self._alignment_streak = 0
        self._box_aligned_before_release = self._alignment_streak >= _MIN_ALIGNMENT_STREAK

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        return {
            "control_frame": "live_jaw_detection_frame_every_step",
            "task_constraints": "jaw_xyz_plus_world_roll_pitch",
            "constraint_dimension": 5,
            "free_orientation_component": "world_yaw",
            "phase_reference": "jaw_detection_frame",
            "pickup_xy_offset": "none",
            "placement_xy_offset": "none",
            "target_tilt_reference": "initial_jaw_world_tilt",
            "joint_target_slew_max_step": _JOINT_TARGET_SLEW_MAX_STEP,
            "align_position_threshold": _ALIGN_POSITION_THRESHOLD,
            "align_tilt_threshold": _ALIGN_TILT_THRESHOLD,
            "stable_steps": _MIN_ALIGNMENT_STREAK,
        }

    @property
    def last_five_dimensional_error(self) -> torch.Tensor | None:
        return self._last_five_dimensional_error
