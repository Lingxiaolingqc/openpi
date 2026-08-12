"""Debounce LeIsaac's pick signal and freeze a confirmed gripper angle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GripperPickLatchUpdate:
    """Result of one pick-feedback update."""

    command_angle: float
    captured: bool
    tightened: bool


class GripperPickLatch:
    """Hold the confirmation-window angle unchanged until an explicit release."""

    def __init__(
        self,
        *,
        confirmation_steps: int,
        velocity_tolerance: float,
        loss_clear_steps: int,
        minimum_angle: float,
        safety_closure: float = 0.0,
    ) -> None:
        if confirmation_steps <= 0:
            raise ValueError("confirmation_steps must be positive")
        if velocity_tolerance < 0.0:
            raise ValueError("velocity_tolerance must be non-negative")
        if loss_clear_steps <= 0:
            raise ValueError("loss_clear_steps must be positive")
        if safety_closure < 0.0:
            raise ValueError("safety_closure must be non-negative")

        self.confirmation_steps = confirmation_steps
        self.velocity_tolerance = velocity_tolerance
        self.loss_clear_steps = loss_clear_steps
        self.minimum_angle = minimum_angle
        self.safety_closure = safety_closure
        self.reset()

    def reset(self) -> None:
        self.held_angle: float | None = None
        self.candidate_min_angle: float | None = None
        self.minimum_pick_angle: float | None = None
        self.confirmation_streak = 0
        self.loss_streak = 0

    def release(self) -> bool:
        """Clear the frozen hold only for an explicit release transition."""

        had_hold = self.held_angle is not None
        self.reset()
        return had_hold

    def _safe_angle(self, measured_angle: float) -> float:
        return max(self.minimum_angle, measured_angle - self.safety_closure)

    def update(
        self,
        *,
        picked: bool,
        measured_angle: float,
        measured_velocity: float,
        nominal_angle: float,
        allow_capture: bool,
        track_loss: bool,
    ) -> GripperPickLatchUpdate:
        """Update debounce state and return the angle that should be commanded."""

        captured = False
        tightened = False

        if self.held_angle is None:
            self.loss_streak = 0
            candidate_pick = allow_capture and picked
            if candidate_pick:
                self.confirmation_streak += 1
                self.candidate_min_angle = (
                    measured_angle
                    if self.candidate_min_angle is None
                    else min(self.candidate_min_angle, measured_angle)
                )
            else:
                self.confirmation_streak = 0
                self.candidate_min_angle = None
            stable_pick = candidate_pick and abs(measured_velocity) <= self.velocity_tolerance
            if self.confirmation_streak >= self.confirmation_steps and stable_pick:
                assert self.candidate_min_angle is not None
                self.minimum_pick_angle = self.candidate_min_angle
                self.held_angle = self._safe_angle(self.minimum_pick_angle)
                self.candidate_min_angle = None
                self.confirmation_streak = 0
                captured = True
        else:
            self.confirmation_streak = 0
            if track_loss:
                self.loss_streak = 0 if picked else min(self.loss_streak + 1, self.loss_clear_steps)
            else:
                self.loss_streak = 0

        command_angle = nominal_angle if self.held_angle is None else self.held_angle
        return GripperPickLatchUpdate(
            command_angle=command_angle,
            captured=captured,
            tightened=tightened,
        )
