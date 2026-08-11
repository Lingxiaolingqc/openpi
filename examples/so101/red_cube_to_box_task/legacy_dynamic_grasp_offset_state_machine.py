"""Legacy expert with a per-grasp dynamic placement offset."""

from __future__ import annotations

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
import torch

from .env_cfg import TARGET_BOX_FLOOR_THICKNESS
from .state_machine import RedCubeToBoxStateMachine

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
# These reproduce the original legacy lift keyframe exactly. They are not
# placement compensation and remain fixed to keep this a single-variable test.
_LEGACY_PICK_XY_OFFSET = (-0.02, 0.0)
_LEGACY_PICK_LIFT_HEIGHT = 0.26
_GRASP_OFFSET_SAMPLE_START_STEP = 80
_MINIMUM_CAPTURED_LIFT = 0.05
_BOX_TO_ROOT_SAFETY_DISTANCE = 0.005


class RedCubeToBoxLegacyDynamicGraspOffsetStateMachine(RedCubeToBoxStateMachine):
    """Change only legacy placement XY using the measured grasp transform.

    Pickup, lift, fixed-world orientation, phase durations, IK and gripper
    commands delegate to the original legacy expert. During the final third of
    lift, the expert samples ``cube_xy - gripper_xy``. The median sample is
    frozen for the episode and used to place the cube, rather than the gripper,
    at the box target.
    """

    def __init__(self) -> None:
        super().__init__()
        self._gripper_to_cube_xy_samples: list[torch.Tensor] = []
        self._gripper_to_cube_xy: torch.Tensor | None = None
        self._desired_cube_xy: torch.Tensor | None = None
        self._dynamic_gripper_target_xy: torch.Tensor | None = None
        self._dynamic_place_offset_xy: torch.Tensor | None = None
        self._last_gripper_target_w: torch.Tensor | None = None

    def get_action(self, env) -> torch.Tensor:
        phase_name, phase_step, phase_duration = self._phase_state()
        if phase_name in {"approach_cube", "descend_to_cube", "close_gripper", "lift_cube"}:
            action = super().get_action(env)
            if phase_name == "lift_cube":
                self._sample_grasp_offset(env, phase_step)
            self._last_gripper_target_w = None
            return action

        self._initialize_anchors(env)
        assert self._cube_anchor is not None
        assert self._floor_anchor is not None
        self._finalize_grasp_offset(env)
        assert self._dynamic_gripper_target_xy is not None
        placement_gripper_target_xy = self._placement_gripper_target_xy(env, phase_name)

        pick_lift = self._cube_anchor.clone()
        pick_lift[:, 0] += _LEGACY_PICK_XY_OFFSET[0]
        pick_lift[:, 1] += _LEGACY_PICK_XY_OFFSET[1]
        pick_lift[:, 2] += _LEGACY_PICK_LIFT_HEIGHT

        box_hover = self._floor_anchor.clone()
        box_hover[:, :2] = placement_gripper_target_xy
        box_hover[:, 2] += 0.25

        box_release = self._floor_anchor.clone()
        box_release[:, :2] = placement_gripper_target_xy
        box_release[:, 2] = self._placement_release_z(env, box_release[:, 2], phase_name)

        if phase_name == "transfer_to_box":
            target_pos_w = self._interpolate(pick_lift, box_hover, phase_step, phase_duration)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "lower_into_box":
            target_pos_w = self._interpolate(box_hover, box_release, phase_step, phase_duration)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "release_cube":
            target_pos_w = box_release
            gripper = _GRIPPER_OPEN
        elif phase_name == "retract_gripper":
            target_pos_w = self._interpolate(box_release, box_hover, phase_step, phase_duration)
            gripper = _GRIPPER_OPEN
        else:
            target_pos_w = box_hover
            gripper = _GRIPPER_OPEN

        self._last_gripper_target_w = target_pos_w.detach().clone()
        return self._compose_legacy_pose_action(env, target_pos_w, gripper)

    def _placement_gripper_target_xy(self, env, phase_name: str) -> torch.Tensor:
        """Return the frozen placement target, with an override point for comparisons."""
        del env, phase_name
        assert self._dynamic_gripper_target_xy is not None
        return self._dynamic_gripper_target_xy

    def _placement_release_z(self, env, floor_center_z: torch.Tensor, phase_name: str) -> torch.Tensor:
        """Return the original legacy release height, with an override point for comparisons."""
        del env, phase_name
        return floor_center_z + TARGET_BOX_FLOOR_THICKNESS / 2.0 + 0.13

    def reset(self) -> None:
        super().reset()
        self._gripper_to_cube_xy_samples = []
        self._gripper_to_cube_xy = None
        self._desired_cube_xy = None
        self._dynamic_gripper_target_xy = None
        self._dynamic_place_offset_xy = None
        self._last_gripper_target_w = None

    def _sample_grasp_offset(self, env, phase_step: int) -> None:
        if phase_step < _GRASP_OFFSET_SAMPLE_START_STEP:
            return
        assert self._cube_anchor is not None
        cube_pos_w = env.scene["cube"].data.root_pos_w
        lifted = cube_pos_w[:, 2] - self._cube_anchor[:, 2] > _MINIMUM_CAPTURED_LIFT
        if not bool(lifted.all().item()):
            return
        gripper_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        self._gripper_to_cube_xy_samples.append((cube_pos_w[:, :2] - gripper_pos_w[:, :2]).detach().clone())

    def _finalize_grasp_offset(self, env) -> None:
        if self._gripper_to_cube_xy is not None:
            return
        if not self._gripper_to_cube_xy_samples:
            raise RuntimeError("No lifted-cube samples were available for the dynamic grasp offset")

        assert self._floor_anchor is not None
        self._gripper_to_cube_xy = torch.median(
            torch.stack(self._gripper_to_cube_xy_samples, dim=0),
            dim=0,
        ).values
        robot_root_xy = env.scene["robot"].data.root_pos_w[:, :2]
        box_to_root = robot_root_xy - self._floor_anchor[:, :2]
        box_to_root_norm = torch.linalg.vector_norm(box_to_root, dim=-1, keepdim=True)
        if bool((box_to_root_norm < 1e-8).any().item()):
            raise RuntimeError("Robot root and target-box center have indistinguishable XY positions")
        box_to_root_unit = box_to_root / box_to_root_norm
        self._desired_cube_xy = self._floor_anchor[:, :2] + _BOX_TO_ROOT_SAFETY_DISTANCE * box_to_root_unit
        self._dynamic_gripper_target_xy = self._desired_cube_xy - self._gripper_to_cube_xy
        self._dynamic_place_offset_xy = self._dynamic_gripper_target_xy - self._floor_anchor[:, :2]

    @staticmethod
    def _compose_legacy_pose_action(env, target_pos_w: torch.Tensor, gripper: float) -> torch.Tensor:
        robot = env.scene["robot"]
        target_pos_local = quat_apply(
            quat_inv(robot.data.root_quat_w),
            target_pos_w - robot.data.root_pos_w,
        )
        zero = torch.zeros((), device=env.device)
        target_quat_w = quat_from_euler_xyz(zero, zero, zero).repeat(env.num_envs, 1)
        target_quat_local = quat_mul(quat_inv(robot.data.root_quat_w), target_quat_w)
        gripper_command = torch.full((env.num_envs, 1), gripper, device=env.device)
        return torch.cat((target_pos_local, target_quat_local, gripper_command), dim=-1)

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        return {
            "base_expert": "legacy_exact",
            "changed_component": "placement_xy_target_only",
            "grasp_offset_measurement": "median_cube_xy_minus_gripper_xy_during_late_lift",
            "grasp_offset_sample_start_step": _GRASP_OFFSET_SAMPLE_START_STEP,
            "minimum_captured_lift": _MINIMUM_CAPTURED_LIFT,
            "box_to_root_safety_distance": _BOX_TO_ROOT_SAFETY_DISTANCE,
        }

    @property
    def gripper_to_cube_xy(self) -> torch.Tensor | None:
        return self._gripper_to_cube_xy

    @property
    def grasp_offset_sample_count(self) -> int:
        return len(self._gripper_to_cube_xy_samples)

    @property
    def desired_cube_xy(self) -> torch.Tensor | None:
        return self._desired_cube_xy

    @property
    def dynamic_gripper_target_xy(self) -> torch.Tensor | None:
        return self._dynamic_gripper_target_xy

    @property
    def dynamic_place_offset_xy(self) -> torch.Tensor | None:
        return self._dynamic_place_offset_xy

    @property
    def last_gripper_target_w(self) -> torch.Tensor | None:
        return self._last_gripper_target_w
