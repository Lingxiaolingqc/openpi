"""Closed-loop scripted SO-101 expert for the RedCubeToBox task."""

from __future__ import annotations

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
from leisaac.datagen.state_machine.base import StateMachineBase
import torch

from . import mdp

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_APPROACH_CLEARANCE = 0.14
_ALIGN_CLEARANCE = 0.003
_LIFT_HEIGHT = 0.18
_BOX_HOVER_HEIGHT = 0.20
_BOX_RELEASE_HEIGHT = 0.09
_RETRACT_HEIGHT = 0.12
_GRASP_DISTANCE_THRESHOLD = 0.02
_GRIPPER_POSITION_THRESHOLD = 0.26
_MIN_CLOSE_STEPS = 60


class RedCubeToBoxAdaptiveStateMachine(StateMachineBase):
    """Jaw-feedback expert kept separate from the fixed-offset legacy expert."""

    _PHASES = (
        ("approach_cube", 120),
        ("descend_to_cube", 160),
        ("close_gripper", 240),
        ("lift_cube", 140),
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
        self._initial_jaw_pos: torch.Tensor | None = None
        self._cube_anchor: torch.Tensor | None = None
        self._floor_anchor: torch.Tensor | None = None
        self._lift_start_cube: torch.Tensor | None = None
        self._cube_to_jaw: torch.Tensor | None = None
        self._release_gripper_target: torch.Tensor | None = None
        self._last_phase: str | None = None
        self._grasp_confirmed = False
        self._grasp_lost_before_release = False

    def setup(self, env) -> None:
        """Apply the damping used by LeIsaac's existing state-machine expert."""

        env.scene["robot"].write_joint_damping_to_sim(damping=10.0)

    def check_success(self, env) -> bool:
        """Evaluate the task's geometric and settled-speed predicate."""

        return bool(mdp.cube_inside_target_box(env).all().item())

    def get_action(self, env) -> torch.Tensor:
        """Return an 8D pose action using live jaw-to-target feedback."""

        self._initialize_anchors(env)
        assert self._initial_jaw_pos is not None
        assert self._cube_anchor is not None
        assert self._floor_anchor is not None

        robot = env.scene["robot"]
        cube = env.scene["cube"]
        ee_frame = env.scene["ee_frame"]
        gripper_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1, :]
        cube_pos_w = cube.data.root_pos_w

        phase_name, phase_step, phase_duration = self._phase_state()
        self._on_phase_entry(phase_name, cube_pos_w, gripper_pos_w, jaw_pos_w)
        self._update_grasp_status(phase_name, robot, cube_pos_w, jaw_pos_w)

        hover_jaw = self._cube_anchor.clone()
        hover_jaw[:, 2] += _APPROACH_CLEARANCE
        aligned_jaw = self._cube_anchor.clone()
        aligned_jaw[:, 2] += _ALIGN_CLEARANCE

        if phase_name == "approach_cube":
            desired_jaw_w = self._interpolate(
                self._initial_jaw_pos,
                hover_jaw,
                phase_step,
                phase_duration,
            )
            target_pos_w = self._gripper_target_for_jaw(gripper_pos_w, jaw_pos_w, desired_jaw_w)
            gripper = _GRIPPER_OPEN
        elif phase_name == "descend_to_cube":
            desired_jaw_w = self._interpolate(hover_jaw, aligned_jaw, phase_step, phase_duration)
            target_pos_w = self._gripper_target_for_jaw(gripper_pos_w, jaw_pos_w, desired_jaw_w)
            gripper = _GRIPPER_OPEN
        elif phase_name == "close_gripper":
            desired_jaw_w = cube_pos_w.clone()
            target_pos_w = self._gripper_target_for_jaw(gripper_pos_w, jaw_pos_w, desired_jaw_w)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "lift_cube":
            assert self._lift_start_cube is not None
            desired_cube_w = self._lift_start_cube.clone()
            desired_cube_w[:, 2] += _LIFT_HEIGHT * min((phase_step + 1) / phase_duration, 1.0)
            desired_jaw_w = self._jaw_target_for_cube(desired_cube_w)
            target_pos_w = self._gripper_target_for_jaw(gripper_pos_w, jaw_pos_w, desired_jaw_w)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "transfer_to_box":
            desired_cube_w = self._interpolate(
                self._lift_target_cube(),
                self._box_hover_cube(),
                phase_step,
                phase_duration,
            )
            desired_jaw_w = self._jaw_target_for_cube(desired_cube_w)
            target_pos_w = self._gripper_target_for_jaw(gripper_pos_w, jaw_pos_w, desired_jaw_w)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "lower_into_box":
            desired_cube_w = self._interpolate(
                self._box_hover_cube(),
                self._box_release_cube(),
                phase_step,
                phase_duration,
            )
            desired_jaw_w = self._jaw_target_for_cube(desired_cube_w)
            target_pos_w = self._gripper_target_for_jaw(gripper_pos_w, jaw_pos_w, desired_jaw_w)
            gripper = _GRIPPER_CLOSE
        elif phase_name == "release_cube":
            assert self._release_gripper_target is not None
            target_pos_w = self._release_gripper_target
            gripper = _GRIPPER_OPEN
        elif phase_name == "retract_gripper":
            assert self._release_gripper_target is not None
            retract_target = self._release_gripper_target.clone()
            retract_target[:, 2] += _RETRACT_HEIGHT
            target_pos_w = self._interpolate(
                self._release_gripper_target,
                retract_target,
                phase_step,
                phase_duration,
            )
            gripper = _GRIPPER_OPEN
        else:
            assert self._release_gripper_target is not None
            target_pos_w = self._release_gripper_target.clone()
            target_pos_w[:, 2] += _RETRACT_HEIGHT
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
        phase_name, phase_step, phase_duration = self._phase_state()
        if phase_name == "close_gripper" and self._grasp_confirmed and phase_step >= _MIN_CLOSE_STEPS:
            self._step_count += phase_duration - phase_step
        else:
            self._step_count += 1
        if self._step_count >= self.MAX_STEPS:
            self._episode_done = True

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._initial_jaw_pos = None
        self._cube_anchor = None
        self._floor_anchor = None
        self._lift_start_cube = None
        self._cube_to_jaw = None
        self._release_gripper_target = None
        self._last_phase = None
        self._grasp_confirmed = False
        self._grasp_lost_before_release = False

    def _initialize_anchors(self, env) -> None:
        if self._initial_jaw_pos is not None:
            return
        ee_frame = env.scene["ee_frame"]
        self._initial_jaw_pos = ee_frame.data.target_pos_w[:, 1, :].clone()
        self._cube_anchor = env.scene["cube"].data.root_pos_w.clone()
        self._floor_anchor = env.scene["target_box_floor"].data.root_pos_w.clone()

    def _on_phase_entry(
        self,
        phase_name: str,
        cube_pos_w: torch.Tensor,
        gripper_pos_w: torch.Tensor,
        jaw_pos_w: torch.Tensor,
    ) -> None:
        if phase_name == self._last_phase:
            return
        self._last_phase = phase_name
        if phase_name == "lift_cube":
            self._lift_start_cube = cube_pos_w.clone()
            self._cube_to_jaw = cube_pos_w.clone() - jaw_pos_w.clone()
        elif phase_name == "release_cube":
            self._release_gripper_target = gripper_pos_w.clone()

    def _update_grasp_status(
        self,
        phase_name: str,
        robot,
        cube_pos_w: torch.Tensor,
        jaw_pos_w: torch.Tensor,
    ) -> None:
        jaw_distance = torch.linalg.vector_norm(jaw_pos_w - cube_pos_w, dim=-1)
        gripper_closed = robot.data.joint_pos[:, -1] < _GRIPPER_POSITION_THRESHOLD
        grasped = bool(torch.logical_and(jaw_distance < _GRASP_DISTANCE_THRESHOLD, gripper_closed).all().item())
        if grasped:
            self._grasp_confirmed = True
        elif self._grasp_confirmed and phase_name in {
            "lift_cube",
            "transfer_to_box",
            "lower_into_box",
        }:
            self._grasp_lost_before_release = True

    @staticmethod
    def _gripper_target_for_jaw(
        gripper_pos_w: torch.Tensor,
        jaw_pos_w: torch.Tensor,
        desired_jaw_w: torch.Tensor,
    ) -> torch.Tensor:
        return gripper_pos_w + (desired_jaw_w - jaw_pos_w)

    def _jaw_target_for_cube(self, desired_cube_w: torch.Tensor) -> torch.Tensor:
        if self._cube_to_jaw is None:
            return desired_cube_w
        return desired_cube_w - self._cube_to_jaw

    def _lift_target_cube(self) -> torch.Tensor:
        assert self._lift_start_cube is not None
        target = self._lift_start_cube.clone()
        target[:, 2] += _LIFT_HEIGHT
        return target

    def _box_hover_cube(self) -> torch.Tensor:
        assert self._floor_anchor is not None
        target = self._floor_anchor.clone()
        target[:, 2] += _BOX_HOVER_HEIGHT
        return target

    def _box_release_cube(self) -> torch.Tensor:
        assert self._floor_anchor is not None
        target = self._floor_anchor.clone()
        target[:, 2] += _BOX_RELEASE_HEIGHT
        return target

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

    @property
    def grasp_confirmed(self) -> bool:
        return self._grasp_confirmed

    @property
    def grasp_lost_before_release(self) -> bool:
        return self._grasp_lost_before_release
