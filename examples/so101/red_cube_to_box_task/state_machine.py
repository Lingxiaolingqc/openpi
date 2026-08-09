"""Scripted SO-101 expert for the RedCubeToBox task."""

from __future__ import annotations

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
from leisaac.datagen.state_machine.base import StateMachineBase
import torch

from . import mdp
from .env_cfg import TARGET_BOX_FLOOR_THICKNESS

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_PICK_XY_OFFSET = (-0.02, 0.0)
_PICK_GRASP_HEIGHT = 0.08
_PLACE_XY_OFFSET = (-0.084, 0.003)


class RedCubeToBoxStateMachine(StateMachineBase):
    """Absolute-pose IK expert with smooth Cartesian keyframe transitions."""

    _PHASES = (
        ("approach_cube", 120),
        ("descend_to_cube", 120),
        ("close_gripper", 80),
        ("lift_cube", 120),
        ("transfer_to_box", 160),
        ("lower_into_box", 120),
        ("release_cube", 100),
        ("retract_gripper", 100),
        ("settle", 180),
    )
    MAX_STEPS = sum(duration for _, duration in _PHASES)

    def __init__(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._cube_anchor: torch.Tensor | None = None
        self._floor_anchor: torch.Tensor | None = None

    def setup(self, env) -> None:
        """Apply the damping used by LeIsaac's existing state-machine expert."""

        env.scene["robot"].write_joint_damping_to_sim(damping=10.0)

    def check_success(self, env) -> bool:
        """Evaluate the same geometric and settled-speed predicate as the task."""

        return bool(mdp.cube_inside_target_box(env).all().item())

    def get_action(self, env) -> torch.Tensor:
        """Return an 8D absolute IK pose plus binary gripper command."""

        robot = env.scene["robot"]
        self._initialize_anchors(env)
        assert self._initial_ee_pos is not None
        assert self._cube_anchor is not None
        assert self._floor_anchor is not None

        pick_hover = self._cube_anchor.clone()
        pick_hover[:, 0] += _PICK_XY_OFFSET[0]
        pick_hover[:, 1] += _PICK_XY_OFFSET[1]
        pick_hover[:, 2] += 0.20

        pick_grasp = self._cube_anchor.clone()
        pick_grasp[:, 0] += _PICK_XY_OFFSET[0]
        pick_grasp[:, 1] += _PICK_XY_OFFSET[1]
        pick_grasp[:, 2] += _PICK_GRASP_HEIGHT

        pick_lift = self._cube_anchor.clone()
        pick_lift[:, 0] += _PICK_XY_OFFSET[0]
        pick_lift[:, 1] += _PICK_XY_OFFSET[1]
        pick_lift[:, 2] += 0.26

        box_hover = self._floor_anchor.clone()
        box_hover[:, 0] += _PLACE_XY_OFFSET[0]
        box_hover[:, 1] += _PLACE_XY_OFFSET[1]
        box_hover[:, 2] += 0.25

        box_release = self._floor_anchor.clone()
        box_release[:, 0] += _PLACE_XY_OFFSET[0]
        box_release[:, 1] += _PLACE_XY_OFFSET[1]
        box_release[:, 2] += TARGET_BOX_FLOOR_THICKNESS / 2.0 + 0.13

        phase_name, phase_step, phase_duration = self._phase_state()
        if phase_name == "approach_cube":
            target_pos_w = self._interpolate(self._initial_ee_pos, pick_hover, phase_step, phase_duration)
            gripper = _GRIPPER_OPEN
        elif phase_name == "descend_to_cube":
            target_pos_w = self._interpolate(pick_hover, pick_grasp, phase_step, phase_duration)
            gripper = _GRIPPER_OPEN
        elif phase_name == "close_gripper":
            target_pos_w = pick_grasp
            gripper = _GRIPPER_CLOSE
        elif phase_name == "lift_cube":
            target_pos_w = self._interpolate(pick_grasp, pick_lift, phase_step, phase_duration)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "transfer_to_box":
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

        robot_base_pos_w = robot.data.root_pos_w
        robot_base_quat_w = robot.data.root_quat_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), target_pos_w - robot_base_pos_w)

        zero = torch.zeros((), device=env.device)
        target_quat_w = quat_from_euler_xyz(zero, zero, zero).repeat(env.num_envs, 1)
        target_quat_local = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)
        gripper_command = torch.full((env.num_envs, 1), gripper, device=env.device)
        return torch.cat([target_pos_local, target_quat_local, gripper_command], dim=-1)

    def advance(self) -> None:
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS:
            self._episode_done = True

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._initial_ee_pos = None
        self._cube_anchor = None
        self._floor_anchor = None

    def _initialize_anchors(self, env) -> None:
        if self._initial_ee_pos is not None:
            return
        self._initial_ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :].clone()
        self._cube_anchor = env.scene["cube"].data.root_pos_w.clone()
        self._floor_anchor = env.scene["target_box_floor"].data.root_pos_w.clone()

    def _phase_state(self) -> tuple[str, int, int]:
        phase_start = 0
        for name, duration in self._PHASES:
            phase_end = phase_start + duration
            if self._step_count < phase_end:
                return name, self._step_count - phase_start, duration
            phase_start = phase_end
        name, duration = self._PHASES[-1]
        return name, duration - 1, duration

    @staticmethod
    def _interpolate(start: torch.Tensor, end: torch.Tensor, step: int, duration: int) -> torch.Tensor:
        alpha = min((step + 1) / duration, 1.0)
        return torch.lerp(start, end, alpha)

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def phase_name(self) -> str:
        return self._phase_state()[0]

    @property
    def step_count(self) -> int:
        return self._step_count
