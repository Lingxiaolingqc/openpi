"""Debounce LeIsaac's geometric pick signal before holding a gripper angle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GripperPickLatchUpdate:
    """Result of one pick-feedback update."""

    command_angle: float
    captured: bool
    released: bool


class GripperPickLatch:
    """Latch a measured gripper angle only after stable, low-speed pick feedback."""

    def __init__(self, *, confirmation_steps: int, velocity_tolerance: float, loss_clear_steps: int) -> None:
        if confirmation_steps <= 0:
            raise ValueError("confirmation_steps must be positive")
        if velocity_tolerance < 0.0:
            raise ValueError("velocity_tolerance must be non-negative")
        if loss_clear_steps <= 0:
            raise ValueError("loss_clear_steps must be positive")

        self.confirmation_steps = confirmation_steps
        self.velocity_tolerance = velocity_tolerance
        self.loss_clear_steps = loss_clear_steps
        self.reset()

    def reset(self) -> None:
        self.held_angle: float | None = None
        self.confirmation_streak = 0
        self.loss_streak = 0

    def update(
        self,
        *,
        picked: bool,
        measured_angle: float,
        measured_velocity: float,
        nominal_angle: float,
        allow_capture: bool,
        allow_release: bool,
    ) -> GripperPickLatchUpdate:
        """Update debounce state and return the angle that should be commanded."""

        captured = False
        released = False

        if self.held_angle is None:
            self.loss_streak = 0
            candidate_pick = allow_capture and picked
            self.confirmation_streak = self.confirmation_streak + 1 if candidate_pick else 0
            stable_pick = candidate_pick and abs(measured_velocity) <= self.velocity_tolerance
            if self.confirmation_streak >= self.confirmation_steps and stable_pick:
                self.held_angle = measured_angle
                self.confirmation_streak = 0
                captured = True
        else:
            self.confirmation_streak = 0
            if allow_release:
                self.loss_streak = 0 if picked else self.loss_streak + 1
                if self.loss_streak >= self.loss_clear_steps:
                    self.held_angle = None
                    self.loss_streak = 0
                    released = True
            else:
                self.loss_streak = 0

        command_angle = nominal_angle if self.held_angle is None else self.held_angle
        return GripperPickLatchUpdate(
            command_angle=command_angle,
            captured=captured,
            released=released,
        )
