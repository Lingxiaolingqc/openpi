"""Independent retreat-then-transport expert for RedCubeToBox."""

from __future__ import annotations

from typing import ClassVar

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
from leisaac.datagen.state_machine.base import StateMachineBase
import torch

from . import mdp
from .env_cfg import TARGET_BOX_FLOOR_THICKNESS
from .env_cfg import TARGET_BOX_WALL_TOP_Z
from .phase_aware_ik_action import PhaseAwareDifferentialInverseKinematicsAction
from .phase_aware_ik_action import resolve_action_term

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_PICK_XY_OFFSET = (-0.02, 0.0)
_PICK_GRASP_HEIGHT = 0.08
_PICK_HOVER_HEIGHT = 0.20
_RETREAT_RADIAL_SCALE = 5.0 / 7.0
_RETREAT_HEIGHT_ABOVE_FLOOR_CENTER = 0.25
_BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER = 0.25
_MINIMUM_JAW_CLEARANCE_ABOVE_WALL = 0.03
_CARTESIAN_REFERENCE_STEP = 0.0025
_RETREAT_REFERENCE_STEP = 0.0008
_RETREAT_RADIAL_ALLOWANCE = 0.035
_GRASP_CONFIRM_DISTANCE = 0.015
_GRASP_LOSS_DISTANCE = 0.025


class RedCubeToBoxAutogenIndependentRetreatTransportStateMachine(StateMachineBase):
    """Pick, retreat-and-lift, transport, lower, and release.

    This class deliberately does not inherit any legacy RedCubeToBox expert.
    The known-good pickup keyframes are repeated locally, then one slow
    ``retreat_to_safe`` Cartesian segment raises the grasped cube while moving
    the arm inward. Motion phases transition only after measured physical
    completion conditions remain true for several consecutive control steps.
    All arm commands retain the complete legacy 6D pose target.
    """

    _FIXED_PHASE_STEPS: ClassVar[dict[str, int]] = {
        "approach_cube": 120,
        "descend_to_cube": 120,
        "close_gripper": 80,
        "release_cube": 100,
        "settle": 180,
    }
    _MOTION_PHASE_LIMITS: ClassVar[dict[str, tuple[int, int, float, int]]] = {
        "retreat_to_safe": (180, 900, 0.015, 15),
        "transfer_to_box": (120, 600, 0.020, 15),
        "lower_into_box": (120, 600, 0.015, 10),
        "retract_gripper": (100, 400, 0.020, 10),
    }
    _PHASES = (
        "approach_cube",
        "descend_to_cube",
        "close_gripper",
        "retreat_to_safe",
        "transfer_to_box",
        "lower_into_box",
        "release_cube",
        "retract_gripper",
        "settle",
    )
    MAX_STEPS = sum(_FIXED_PHASE_STEPS.values()) + sum(
        maximum_steps for _, maximum_steps, _, _ in _MOTION_PHASE_LIMITS.values()
    )

    def __init__(self) -> None:
        self._phase_index = 0
        self._phase_step = 0
        self._step_count = 0
        self._episode_done = False
        self._arm_action_term: PhaseAwareDifferentialInverseKinematicsAction | None = None
        self._initial_gripper_w: torch.Tensor | None = None
        self._cube_anchor_w: torch.Tensor | None = None
        self._floor_anchor_w: torch.Tensor | None = None
        self._motion_start_w: torch.Tensor | None = None
        self._motion_target_w: torch.Tensor | None = None
        self._current_target_w: torch.Tensor | None = None
        self._target_error: float | None = None
        self._target_stable_streak = 0
        self._motion_ready = False
        self._retreat_subphase: str | None = None
        self._retreat_safe_z: torch.Tensor | None = None
        self._retreat_target_radius: torch.Tensor | None = None
        self._jaw_cube_distance: float | None = None
        self._grasp_confirmed = False
        self._grasp_lost_before_release = False
        self._safe_release_gripper_z: torch.Tensor | None = None
        self._servo_timeout_phase: str | None = None
        self._servo_abort_reason: str | None = None
        self._release_block_reason: str | None = None

    def setup(self, env) -> None:
        env.scene["robot"].write_joint_damping_to_sim(damping=10.0)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise TypeError(
                "autogen_independent_retreat_transport requires PhaseAwareDifferentialInverseKinematicsAction"
            )
        self._arm_action_term = arm_action_term
        self._arm_action_term.set_orientation_weight(weight=1.0)

    def get_action(self, env) -> torch.Tensor:
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting an independent AutoGen expert action")
        self._arm_action_term.set_orientation_weight(weight=1.0)
        self._initialize_anchors(env)
        assert self._initial_gripper_w is not None
        assert self._cube_anchor_w is not None
        assert self._floor_anchor_w is not None

        phase = self.phase_name
        pick_hover_w, pick_grasp_w = self._pickup_targets()
        if phase == "approach_cube":
            duration = self._FIXED_PHASE_STEPS[phase]
            target_w = self._interpolate(self._initial_gripper_w, pick_hover_w, self._phase_step, duration)
            gripper = _GRIPPER_OPEN
        elif phase == "descend_to_cube":
            duration = self._FIXED_PHASE_STEPS[phase]
            target_w = self._interpolate(pick_hover_w, pick_grasp_w, self._phase_step, duration)
            gripper = _GRIPPER_OPEN
        elif phase == "close_gripper":
            target_w = pick_grasp_w
            gripper = _GRIPPER_CLOSE
        elif phase == "retreat_to_safe":
            self._initialize_retreat(env)
            if self._detect_retreat_grasp_loss(env):
                target_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
            else:
                target_w = self._retreat_reference()
            gripper = _GRIPPER_CLOSE
        elif phase == "transfer_to_box":
            self._initialize_transport(env)
            target_w = self._motion_reference(phase)
            gripper = _GRIPPER_CLOSE
        elif phase == "lower_into_box":
            self._initialize_lower(env)
            target_w = self._motion_reference(phase)
            gripper = _GRIPPER_CLOSE
        elif phase == "release_cube":
            self._initialize_release(env)
            assert self._motion_target_w is not None
            target_w = self._motion_target_w
            gripper = _GRIPPER_OPEN
        elif phase == "retract_gripper":
            self._initialize_retract(env)
            target_w = self._motion_reference(phase)
            gripper = _GRIPPER_OPEN
        else:
            self._initialize_settle(env)
            assert self._motion_target_w is not None
            target_w = self._motion_target_w
            gripper = _GRIPPER_OPEN

        self._current_target_w = target_w.detach().clone()
        if phase == "retreat_to_safe" and not self._episode_done:
            self._update_retreat_progress(env)
        elif phase in self._MOTION_PHASE_LIMITS:
            self._update_motion_convergence(env, phase)
        return self._compose_pose_action(env, target_w, gripper)

    def advance(self) -> None:
        if self._episode_done:
            return
        phase = self.phase_name
        self._step_count += 1
        self._phase_step += 1

        if phase in self._FIXED_PHASE_STEPS:
            if self._phase_step >= self._FIXED_PHASE_STEPS[phase]:
                self._advance_phase()
            return

        minimum_steps, maximum_steps, _, _ = self._MOTION_PHASE_LIMITS[phase]
        if self._phase_step >= minimum_steps and self._motion_ready:
            self._advance_phase()
        elif self._phase_step >= maximum_steps:
            self._servo_timeout_phase = phase
            self._release_block_reason = (
                f"{phase}_target_not_reached_before_timeout:"
                f"error={self._target_error}:stable_streak={self._target_stable_streak}"
            )
            self._episode_done = True

    def reset(self) -> None:
        arm_action_term = self._arm_action_term
        self.__init__()
        self._arm_action_term = arm_action_term
        if self._arm_action_term is not None:
            self._arm_action_term.set_orientation_weight(weight=1.0)

    def check_success(self, env) -> bool:
        return bool(mdp.cube_inside_target_box(env).all().item())

    def _advance_phase(self) -> None:
        if self._phase_index == len(self._PHASES) - 1:
            self._episode_done = True
            return
        self._phase_index += 1
        self._phase_step = 0
        self._motion_start_w = None
        self._motion_target_w = None
        self._current_target_w = None
        self._target_error = None
        self._target_stable_streak = 0
        self._motion_ready = False

    def _initialize_anchors(self, env) -> None:
        if self._initial_gripper_w is not None:
            return
        self._initial_gripper_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        self._cube_anchor_w = env.scene["cube"].data.root_pos_w.detach().clone()
        self._floor_anchor_w = env.scene["target_box_floor"].data.root_pos_w.detach().clone()

    def _pickup_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._cube_anchor_w is not None
        pick_hover_w = self._cube_anchor_w.clone()
        pick_hover_w[:, 0] += _PICK_XY_OFFSET[0]
        pick_hover_w[:, 1] += _PICK_XY_OFFSET[1]
        pick_hover_w[:, 2] += _PICK_HOVER_HEIGHT
        pick_grasp_w = self._cube_anchor_w.clone()
        pick_grasp_w[:, 0] += _PICK_XY_OFFSET[0]
        pick_grasp_w[:, 1] += _PICK_XY_OFFSET[1]
        pick_grasp_w[:, 2] += _PICK_GRASP_HEIGHT
        return pick_hover_w, pick_grasp_w

    def _initialize_retreat(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        robot_root_w = env.scene["robot"].data.root_pos_w
        target_w = start_w.clone()
        start_radius = torch.linalg.vector_norm(start_w[:, :2] - robot_root_w[:, :2], dim=-1)
        target_w[:, :2] = robot_root_w[:, :2] + _RETREAT_RADIAL_SCALE * (start_w[:, :2] - robot_root_w[:, :2])
        target_w[:, 2] = torch.maximum(
            start_w[:, 2],
            self._floor_anchor_w[:, 2] + _RETREAT_HEIGHT_ABOVE_FLOOR_CENTER,
        )
        jaw_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        cube_w = env.scene["cube"].data.root_pos_w
        jaw_cube_distance = torch.linalg.vector_norm(jaw_w - cube_w, dim=-1)
        self._jaw_cube_distance = float(jaw_cube_distance.max().item())
        self._grasp_confirmed = self._jaw_cube_distance <= _GRASP_CONFIRM_DISTANCE
        self._retreat_subphase = "combined_slow_retreat"
        self._retreat_safe_z = target_w[:, 2].detach().clone()
        self._retreat_target_radius = (_RETREAT_RADIAL_SCALE * start_radius).detach().clone()
        self._set_motion(start_w, target_w)

    def _retreat_reference(self) -> torch.Tensor:
        assert self._motion_start_w is not None
        assert self._motion_target_w is not None
        return self._bounded_reference(
            self._motion_start_w,
            self._motion_target_w,
            self._phase_step,
            _RETREAT_REFERENCE_STEP,
        )

    def _detect_retreat_grasp_loss(self, env) -> bool:
        jaw_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :]
        cube_w = env.scene["cube"].data.root_pos_w
        jaw_cube_distance = torch.linalg.vector_norm(jaw_w - cube_w, dim=-1)
        self._jaw_cube_distance = float(jaw_cube_distance.max().item())
        if not self._grasp_confirmed and self._jaw_cube_distance > _GRASP_LOSS_DISTANCE:
            self._servo_abort_reason = (
                "grasp_not_confirmed_at_retreat_start:"
                f"jaw_cube_distance={self._jaw_cube_distance}:"
                f"threshold={_GRASP_LOSS_DISTANCE}"
            )
            self._episode_done = True
            return True
        if self._grasp_confirmed and self._jaw_cube_distance > _GRASP_LOSS_DISTANCE:
            self._grasp_lost_before_release = True
            self._servo_abort_reason = (
                "grasp_lost_during_retreat:"
                f"subphase={self._retreat_subphase}:"
                f"jaw_cube_distance={self._jaw_cube_distance}:"
                f"threshold={_GRASP_LOSS_DISTANCE}"
            )
            self._episode_done = True
            return True
        return False

    def _update_retreat_progress(self, env) -> None:
        assert self._retreat_safe_z is not None
        assert self._retreat_target_radius is not None
        assert self._motion_target_w is not None
        actual_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        height_deficit = torch.clamp(self._retreat_safe_z - actual_w[:, 2], min=0.0)
        robot_root_xy = env.scene["robot"].data.root_pos_w[:, :2]
        actual_radius = torch.linalg.vector_norm(actual_w[:, :2] - robot_root_xy, dim=-1)
        radial_excess = torch.clamp(
            actual_radius - (self._retreat_target_radius + _RETREAT_RADIAL_ALLOWANCE),
            min=0.0,
        )
        physical_error = torch.maximum(height_deficit, radial_excess)
        self._target_error = float(physical_error.max().item())
        _, _, _, stable_steps = self._MOTION_PHASE_LIMITS["retreat_to_safe"]
        if self._target_error <= 1e-6:
            self._target_stable_streak += 1
        else:
            self._target_stable_streak = 0
        self._motion_ready = self._target_stable_streak >= stable_steps

    def _initialize_transport(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        target_w = self._floor_anchor_w.clone()
        target_w[:, 2] += _BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER
        self._set_motion(start_w, target_w)

    def _initialize_lower(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        ee_frame = env.scene["ee_frame"]
        gripper_above_jaw_z = torch.clamp(
            ee_frame.data.target_pos_w[:, 0, 2] - ee_frame.data.target_pos_w[:, 1, 2],
            min=0.0,
        )
        legacy_release_z = self._floor_anchor_w[:, 2] + TARGET_BOX_FLOOR_THICKNESS / 2.0 + 0.13
        safe_release_z = torch.maximum(
            legacy_release_z,
            TARGET_BOX_WALL_TOP_Z + _MINIMUM_JAW_CLEARANCE_ABOVE_WALL + gripper_above_jaw_z,
        )
        target_w = self._floor_anchor_w.clone()
        target_w[:, 2] = safe_release_z
        self._safe_release_gripper_z = safe_release_z.detach().clone()
        self._set_motion(start_w, target_w)

    def _initialize_retract(self, env) -> None:
        if self._motion_start_w is not None:
            return
        assert self._floor_anchor_w is not None
        start_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        target_w = self._floor_anchor_w.clone()
        target_w[:, 2] += _BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER
        self._set_motion(start_w, target_w)

    def _initialize_release(self, env) -> None:
        if self._motion_target_w is not None:
            return
        assert self._floor_anchor_w is not None
        if self._safe_release_gripper_z is None:
            raise RuntimeError("Release height was not initialized by lower_into_box")
        target_w = self._floor_anchor_w.clone()
        target_w[:, 2] = self._safe_release_gripper_z
        self._set_motion(
            env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone(),
            target_w,
        )

    def _initialize_settle(self, env) -> None:
        if self._motion_target_w is not None:
            return
        assert self._floor_anchor_w is not None
        target_w = self._floor_anchor_w.clone()
        target_w[:, 2] += _BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER
        self._set_motion(
            env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone(),
            target_w,
        )

    def _set_motion(self, start_w: torch.Tensor, target_w: torch.Tensor) -> None:
        self._motion_start_w = start_w
        self._motion_target_w = target_w.detach().clone()

    def _motion_reference(self, phase: str) -> torch.Tensor:
        assert self._motion_start_w is not None
        assert self._motion_target_w is not None
        if phase not in self._MOTION_PHASE_LIMITS:
            raise ValueError(f"No motion limits configured for phase: {phase}")
        return self._bounded_reference(
            self._motion_start_w,
            self._motion_target_w,
            self._phase_step,
            _CARTESIAN_REFERENCE_STEP,
        )

    @staticmethod
    def _bounded_reference(
        start_w: torch.Tensor,
        target_w: torch.Tensor,
        elapsed_steps: int,
        maximum_step: float,
    ) -> torch.Tensor:
        displacement = target_w - start_w
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        traveled = torch.full_like(distance, (elapsed_steps + 1) * maximum_step)
        progress = torch.clamp(traveled / torch.clamp(distance, min=1e-8), max=1.0)
        return start_w + progress * displacement

    def _update_motion_convergence(self, env, phase: str) -> None:
        assert self._motion_target_w is not None
        actual_gripper_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        error = torch.linalg.vector_norm(self._motion_target_w - actual_gripper_w, dim=-1)
        self._target_error = float(error.max().item())
        _, _, tolerance, stable_steps = self._MOTION_PHASE_LIMITS[phase]
        if self._target_error <= tolerance:
            self._target_stable_streak += 1
        else:
            self._target_stable_streak = 0
        self._motion_ready = self._target_stable_streak >= stable_steps

    @staticmethod
    def _interpolate(start: torch.Tensor, end: torch.Tensor, step: int, duration: int) -> torch.Tensor:
        alpha = min((step + 1) / duration, 1.0)
        return torch.lerp(start, end, alpha)

    @staticmethod
    def _compose_pose_action(env, target_pos_w: torch.Tensor, gripper: float) -> torch.Tensor:
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
            "implementation": "independent_no_legacy_inheritance",
            "phase_sequence": "approach,descend,close,retreat_and_lift,transport,lower,release,retract,settle",
            "placement_xy_policy": "target_box_floor_center_without_offset",
            "ik_mode": "full_6d_pose",
            "transition_policy": "actual_gripper_position_stable_before_phase_advance",
            "cartesian_reference_step": _CARTESIAN_REFERENCE_STEP,
            "retreat_radial_scale": _RETREAT_RADIAL_SCALE,
            "retreat_reference_step": _RETREAT_REFERENCE_STEP,
            "retreat_height_above_floor_center": _RETREAT_HEIGHT_ABOVE_FLOOR_CENTER,
            "retreat_path": "single_combined_xyz_segment",
            "retreat_completion": "safe_height_and_reduced_root_radius",
            "grasp_loss_distance": _GRASP_LOSS_DISTANCE,
        }

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def phase_name(self) -> str:
        return self._PHASES[self._phase_index]

    @property
    def phase_step(self) -> int:
        return self._phase_step

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "uninitialized"
        return self._arm_action_term.runtime_mode

    @property
    def motion_start_w(self) -> torch.Tensor | None:
        return self._motion_start_w

    @property
    def motion_target_w(self) -> torch.Tensor | None:
        return self._motion_target_w

    @property
    def current_target_w(self) -> torch.Tensor | None:
        return self._current_target_w

    @property
    def target_error(self) -> float | None:
        return self._target_error

    @property
    def target_stable_streak(self) -> int:
        return self._target_stable_streak

    @property
    def retreat_subphase(self) -> str | None:
        return self._retreat_subphase

    @property
    def jaw_cube_distance(self) -> float | None:
        return self._jaw_cube_distance

    @property
    def grasp_confirmed(self) -> bool:
        return self._grasp_confirmed

    @property
    def grasp_lost_before_release(self) -> bool:
        return self._grasp_lost_before_release

    @property
    def servo_timeout_phase(self) -> str | None:
        return self._servo_timeout_phase

    @property
    def servo_abort_reason(self) -> str | None:
        return self._servo_abort_reason

    @property
    def release_block_reason(self) -> str | None:
        return self._release_block_reason

    @property
    def safe_release_gripper_z(self) -> torch.Tensor | None:
        return self._safe_release_gripper_z
