"""Phase-based IK expert for the PiPER RedCubeToBox task."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from examples.piper import contract
from examples.piper import ik


class Phase(str, Enum):
    PREGRASP = "pregrasp"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    TRANSPORT = "transport"
    LOWER = "lower"
    RELEASE = "release"
    RETREAT = "retreat"
    DONE = "done"


MOVE_PHASES = {Phase.PREGRASP, Phase.DESCEND, Phase.LIFT, Phase.TRANSPORT, Phase.LOWER, Phase.RETREAT}
DOWNWARD_ROTATION = np.diag([-1.0, 1.0, -1.0])


@dataclass(frozen=True)
class ExpertReport:
    seed: int
    success: bool
    steps: int
    final_phase: str
    failure_reason: str | None


class ScriptedExpert:
    """Closed-loop waypoint controller with explicit, auditable task phases."""

    _ORDER = (
        Phase.PREGRASP,
        Phase.DESCEND,
        Phase.CLOSE,
        Phase.LIFT,
        Phase.TRANSPORT,
        Phase.LOWER,
        Phase.RELEASE,
        Phase.RETREAT,
        Phase.DONE,
    )

    def __init__(self, env, *, phase_timeout_steps: int = 200, hold_steps: int = 10) -> None:
        self.env = env
        self.phase_timeout_steps = phase_timeout_steps
        self.hold_steps = hold_steps
        self.phase = Phase.PREGRASP
        self.action_phase = self.phase
        self.phase_step = 0
        self._target_action: np.ndarray | None = None
        self.failure_reason: str | None = None
        self._cube_start = env.cube_position.copy()
        self._target = env.target_position.copy()

    def _position_for_phase(self, phase: Phase) -> np.ndarray:
        if phase is Phase.PREGRASP:
            return self._cube_start + np.array([0.0, 0.0, 0.075])
        if phase is Phase.DESCEND:
            return self._cube_start + np.array([0.0, 0.0, 0.025])
        if phase in {Phase.LIFT, Phase.TRANSPORT}:
            xy = self._cube_start[:2] if phase is Phase.LIFT else self._target[:2]
            return np.array([xy[0], xy[1], 0.105])
        if phase is Phase.LOWER:
            return self._target + np.array([0.0, 0.0, 0.028])
        if phase is Phase.RETREAT:
            return self._target + np.array([0.0, 0.0, 0.075])
        raise ValueError(f"phase {phase.value} has no Cartesian target")

    def _plan_move(self) -> None:
        relaxed_orientation = self.phase in {Phase.LIFT, Phase.TRANSPORT, Phase.RETREAT}
        result = ik.solve_site_ik(
            self.env,
            self._position_for_phase(self.phase),
            target_rotation=DOWNWARD_ROTATION,
            initial_q_rad=self.env.state()[:6],
            orientation_weight=0.08 if relaxed_orientation else 0.25,
            max_iterations=180,
            position_tolerance_m=0.010 if relaxed_orientation else 0.004,
            orientation_tolerance_rad=0.45 if relaxed_orientation else 0.12,
        )
        approximate_ok = result.position_error_m <= (
            0.025 if relaxed_orientation else 0.010
        ) and result.orientation_error_rad <= (0.50 if relaxed_orientation else 0.15)
        if not result.converged and not approximate_ok:
            self.failure_reason = (
                f"ik_{self.phase.value}:position={result.position_error_m:.4f},"
                f"orientation={result.orientation_error_rad:.4f}"
            )
            return
        gripper = 1.0 if self.phase in {Phase.PREGRASP, Phase.DESCEND, Phase.RETREAT} else 0.0
        self._target_action = np.concatenate([result.q_rad, [gripper]]).astype(np.float32)

    def _advance(self) -> None:
        index = self._ORDER.index(self.phase)
        self.phase = self._ORDER[index + 1]
        self.phase_step = 0
        self._target_action = None

    def next_action(self) -> np.ndarray:
        self.action_phase = self.phase
        if self.phase is Phase.DONE:
            return self.env.state()
        if self.failure_reason is not None:
            return self.env.state()
        self.phase_step += 1

        if self.phase in MOVE_PHASES:
            if self._target_action is None:
                self._plan_move()
                if self.failure_reason is not None:
                    return self.env.state()
            assert self._target_action is not None
            action = contract.limit_action_step(self.env.state(), self._target_action)
            cartesian_error = float(np.linalg.norm(self.env.gripper_position - self._position_for_phase(self.phase)))
            tolerance = 0.020 if self.phase in {Phase.LIFT, Phase.TRANSPORT, Phase.LOWER, Phase.RETREAT} else 0.012
            if cartesian_error < tolerance and self.phase_step >= 3:
                self._advance()
            elif self.phase_step >= self.phase_timeout_steps:
                self.failure_reason = f"phase_timeout:{self.phase.value}"
            return action

        action = self.env.state().copy()
        completed = False
        if self.phase is Phase.CLOSE:
            action[6] = 0.0
            completed = self.phase_step >= self.hold_steps and self.env.has_two_finger_contact()
        elif self.phase is Phase.RELEASE:
            action[6] = 1.0
            completed = self.phase_step >= self.hold_steps and (
                self.env.state()[6] >= 0.92 or not self.env.has_two_finger_contact()
            )
            completed = completed or self.phase_step >= 40
        if completed:
            self._advance()
        elif self.phase is not Phase.RELEASE and self.phase_step >= self.phase_timeout_steps:
            self.failure_reason = f"gripper_timeout:{self.action_phase.value}"
        return contract.limit_action_step(self.env.state(), action)


def run_expert_episode(
    env,
    *,
    seed: int,
    record_step: Callable[[contract.Observation, np.ndarray, Phase], None] | None = None,
) -> ExpertReport:
    """Reset and execute one expert rollout, optionally recording pre-step pairs."""

    observation, _ = env.reset(seed=seed)
    expert = ScriptedExpert(env)
    while env.step_count < env.maximum_steps:
        action = expert.next_action()
        if expert.failure_reason is not None:
            return ExpertReport(
                seed=seed,
                success=False,
                steps=env.step_count,
                final_phase=expert.phase.value,
                failure_reason=expert.failure_reason,
            )
        result = env.step(action)
        if record_step is not None:
            record_step(observation, result.info["executed_action"], expert.action_phase)
        observation = result.observation
        if result.terminated:
            return ExpertReport(
                seed=seed,
                success=True,
                steps=env.step_count,
                final_phase=expert.phase.value,
                failure_reason=None,
            )
        if result.truncated:
            return ExpertReport(
                seed=seed,
                success=False,
                steps=env.step_count,
                final_phase=expert.phase.value,
                failure_reason="environment_truncated",
            )
        if expert.phase is Phase.DONE:
            break
    return ExpertReport(
        seed=seed,
        success=False,
        steps=env.step_count,
        final_phase=expert.phase.value,
        failure_reason="task_not_successful",
    )
