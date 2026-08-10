"""Rate-limited Cartesian P-servo SO-101 expert for RedCubeToBox."""

from __future__ import annotations

from isaaclab.utils.math import quat_from_euler_xyz
import torch

from .adaptive_state_machine import RedCubeToBoxAdaptiveStateMachine
from .phase_aware_ik_action import resolve_action_term

_SERVO_KP = 0.25
_SERVO_MAX_STEP = 0.003
_SERVO_DEADBAND = 0.0015
_SERVO_STABLE_STEPS = 10
_SERVO_PHASES = (
    "transfer_to_box",
    "lower_into_box",
    "align_over_box",
)
TRANSPORT_IK_PHASES = (
    "transfer_to_box",
    "lower_into_box",
    "align_over_box",
    "release_cube",
    "retract_gripper",
    "settle",
)
_NEXT_PHASE = {
    "transfer_to_box": "lower_into_box",
    "lower_into_box": "align_over_box",
    "align_over_box": "release_cube",
}


class RedCubeToBoxServoStateMachine(RedCubeToBoxAdaptiveStateMachine):
    """Adaptive grasp plus a bounded Cartesian P-servo transport controller."""

    _PHASES = (
        ("approach_cube", 120),
        ("descend_to_cube", 120),
        ("close_gripper", 200),
        ("retry_retract", 100),
        ("retry_descend", 120),
        ("retry_close", 200),
        ("lift_cube", 140),
        ("transfer_to_box", 300),
        ("lower_into_box", 300),
        ("align_over_box", 300),
        ("release_cube", 100),
        ("retract_gripper", 100),
        ("settle", 180),
    )
    MAX_STEPS = sum(duration for _, duration in _PHASES)

    def __init__(self) -> None:
        super().__init__()
        self._arm_action_term = None
        self._servo_stable_phase: str | None = None
        self._servo_stable_streak = 0
        self._servo_arrived_phases: set[str] = set()
        self._servo_timeout_phase: str | None = None
        self._servo_abort_reason: str | None = None
        self._last_servo_delta_w: torch.Tensor | None = None
        self._last_servo_error_norm: torch.Tensor | None = None

    def setup(self, env) -> None:
        super().setup(env)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not hasattr(arm_action_term, "set_position_only"):
            raise RuntimeError(
                "The servo expert requires PhaseAwareDifferentialInverseKinematicsAction; "
                f"received {type(arm_action_term).__name__}"
            )
        self._arm_action_term = arm_action_term
        self._arm_action_term.set_position_only(enabled=False)

    def reset(self) -> None:
        super().reset()
        self._servo_stable_phase = None
        self._servo_stable_streak = 0
        self._servo_arrived_phases = set()
        self._servo_timeout_phase = None
        self._servo_abort_reason = None
        self._last_servo_delta_w = None
        self._last_servo_error_norm = None
        if self._arm_action_term is not None:
            self._arm_action_term.set_position_only(enabled=False)

    def get_action(self, env) -> torch.Tensor:
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting a servo-expert action")
        orientation_weight = self.transport_orientation_weight if self.phase_name in TRANSPORT_IK_PHASES else 1.0
        self._arm_action_term.set_orientation_weight(weight=orientation_weight)
        action = super().get_action(env)
        if self.abort_on_grasp_loss and self.grasp_lost_before_release:
            self._servo_abort_reason = f"grasp_lost:{self.phase_name}"
            self._episode_done = True
        return action

    def advance(self) -> None:
        phase_name, phase_step, phase_duration = self._phase_state()
        if self.abort_on_grasp_loss and self.grasp_lost_before_release:
            self._servo_abort_reason = f"grasp_lost:{phase_name}"
            self._episode_done = True
            return
        if phase_name not in _SERVO_PHASES:
            super().advance()
            return

        if self._servo_stable_phase != phase_name:
            self._servo_stable_phase = phase_name
            self._servo_stable_streak = 0

        error = self.last_cube_error_w
        arrived = False
        if error is not None:
            error_norm = torch.linalg.vector_norm(error, dim=-1)
            arrived = bool((error_norm <= _SERVO_DEADBAND).all().item())

        self._servo_stable_streak = self._servo_stable_streak + 1 if arrived else 0
        if self._servo_stable_streak >= _SERVO_STABLE_STEPS:
            self._servo_arrived_phases.add(phase_name)
            self._servo_stable_streak = 0
            self._step_count = self._phase_start(_NEXT_PHASE[phase_name])
        elif phase_step + 1 >= phase_duration:
            self._servo_timeout_phase = phase_name
            self._episode_done = True
        else:
            self._step_count += 1

    def _desired_lift_cube(self, phase_step: int, phase_duration: int) -> torch.Tensor:
        return super()._desired_lift_cube(phase_step, phase_duration)

    def _desired_transfer_cube(self, phase_step: int, phase_duration: int) -> torch.Tensor:
        del phase_step, phase_duration
        return self._box_hover_cube()

    def _desired_lower_cube(self, phase_step: int, phase_duration: int) -> torch.Tensor:
        del phase_step, phase_duration
        return self._box_release_cube()

    def _gripper_target_from_cube_feedback(
        self,
        gripper_pos_w: torch.Tensor,
        cube_pos_w: torch.Tensor,
        desired_cube_w: torch.Tensor,
    ) -> torch.Tensor:
        if self.phase_name == "lift_cube":
            self._last_servo_delta_w = None
            self._last_servo_error_norm = None
            return super()._gripper_target_from_cube_feedback(gripper_pos_w, cube_pos_w, desired_cube_w)

        error = desired_cube_w - cube_pos_w
        error_norm = torch.linalg.vector_norm(error, dim=-1, keepdim=True)
        delta = _SERVO_KP * error
        delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(
            _SERVO_MAX_STEP / torch.clamp(delta_norm, min=1e-8),
            max=1.0,
        )
        delta = delta * scale
        delta = torch.where(error_norm <= _SERVO_DEADBAND, torch.zeros_like(delta), delta)
        target = gripper_pos_w + delta

        self._last_desired_cube_w = desired_cube_w.clone()
        self._last_cube_error_w = error.clone()
        self._last_gripper_target_w = target.clone()
        self._last_servo_delta_w = delta.clone()
        self._last_servo_error_norm = error_norm.clone()
        return target

    def _target_orientation_w(self, env, phase_name: str, ee_frame) -> torch.Tensor:
        del phase_name, ee_frame
        zero = torch.zeros((), device=env.device)
        return quat_from_euler_xyz(zero, zero, zero).repeat(env.num_envs, 1)

    @property
    def box_aligned_before_release(self) -> bool:
        return "align_over_box" in self._servo_arrived_phases

    @property
    def servo_timeout_phase(self) -> str | None:
        return self._servo_timeout_phase

    @property
    def servo_abort_reason(self) -> str | None:
        return self._servo_abort_reason

    @property
    def servo_stable_streak(self) -> int:
        return self._servo_stable_streak

    @property
    def last_servo_delta_w(self) -> torch.Tensor | None:
        return self._last_servo_delta_w

    @property
    def last_servo_error_norm(self) -> torch.Tensor | None:
        return self._last_servo_error_norm

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "unconfigured"
        return self._arm_action_term.runtime_mode

    @property
    def transport_orientation_weight(self) -> float:
        return 0.0

    @property
    def abort_on_grasp_loss(self) -> bool:
        """Whether one failed grasp observation immediately aborts the episode."""

        return True

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        return {
            "kp": _SERVO_KP,
            "max_step": _SERVO_MAX_STEP,
            "deadband": _SERVO_DEADBAND,
            "stable_steps": _SERVO_STABLE_STEPS,
            "activation_phase": "transfer_to_box",
            "ik_transport_mode": "position_only",
            "transport_orientation_weight": self.transport_orientation_weight,
        }
