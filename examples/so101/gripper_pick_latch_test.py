"""Behavioral regression tests for monotonic gripper-angle holding."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

MODULE_PATH = Path(__file__).resolve().parent / "red_cube_to_box_task" / "gripper_pick_latch.py"
MODULE_SPEC = importlib.util.spec_from_file_location("gripper_pick_latch", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load gripper pick latch from {MODULE_PATH}")
latch_module = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = latch_module
MODULE_SPEC.loader.exec_module(latch_module)

GripperPickLatch = latch_module.GripperPickLatch

NOMINAL_ANGLE = 0.238561
CONFIRMATION_STEPS = 20
LOSS_CLEAR_STEPS = 3
MINIMUM_ANGLE = -0.174533


@pytest.fixture
def latch():
    return GripperPickLatch(
        confirmation_steps=CONFIRMATION_STEPS,
        velocity_tolerance=0.01,
        loss_clear_steps=LOSS_CLEAR_STEPS,
        minimum_angle=MINIMUM_ANGLE,
    )


def _update(
    latch,
    *,
    picked: bool,
    measured_angle: float = 0.259811,
    velocity: float = 0.0,
    nominal_angle: float = NOMINAL_ANGLE,
):
    return latch.update(
        picked=picked,
        measured_angle=measured_angle,
        measured_velocity=velocity,
        nominal_angle=nominal_angle,
        allow_capture=True,
        track_loss=True,
    )


def _capture(latch, *, measured_angle: float = 0.259811, nominal_angle: float = NOMINAL_ANGLE) -> None:
    for _ in range(CONFIRMATION_STEPS):
        update = _update(latch, picked=True, measured_angle=measured_angle, nominal_angle=nominal_angle)
    assert update.captured


def test_single_frame_pick_does_not_latch(latch) -> None:
    first = _update(latch, picked=True)
    lost = _update(latch, picked=False)

    assert first.command_angle == NOMINAL_ANGLE
    assert not first.captured
    assert latch.held_angle is None
    assert latch.confirmation_streak == 0
    assert latch.candidate_min_angle is None
    assert lost.command_angle == NOMINAL_ANGLE


def test_fast_pick_signal_does_not_latch_until_confirmation_frame_is_slow(latch) -> None:
    for _ in range(CONFIRMATION_STEPS + 2):
        update = _update(latch, picked=True, velocity=0.290102)

    assert not update.captured
    assert update.command_angle == NOMINAL_ANGLE
    assert latch.held_angle is None

    captured = _update(latch, picked=True, measured_angle=0.2439685)
    assert captured.captured
    assert captured.command_angle == NOMINAL_ANGLE


def test_confirmation_uses_streak_minimum_instead_of_rebound_angle(latch) -> None:
    measurements = [0.259811] * (CONFIRMATION_STEPS - 2) + [0.232, 0.2439685]
    for measured_angle in measurements:
        update = _update(latch, picked=True, measured_angle=measured_angle, nominal_angle=0.30)

    assert update.captured
    assert latch.minimum_pick_angle == 0.232
    assert update.command_angle == 0.232


def test_capture_never_opens_beyond_nominal_target(latch) -> None:
    _capture(latch, measured_angle=0.2439685)

    assert latch.minimum_pick_angle == 0.2439685
    assert latch.held_angle == NOMINAL_ANGLE


def test_confirmed_hold_can_only_tighten(latch) -> None:
    _capture(latch, measured_angle=0.24, nominal_angle=0.30)
    commands = [latch.held_angle]
    for measured_angle in (0.27, 0.22, 0.28, 0.21, 0.40):
        update = _update(latch, picked=True, measured_angle=measured_angle, nominal_angle=0.30)
        commands.append(update.command_angle)

    assert commands == sorted(commands, reverse=True)
    assert commands[-1] == 0.21


def test_pick_loss_never_reopens_confirmed_hold(latch) -> None:
    _capture(latch, measured_angle=0.22)

    for _ in range(LOSS_CLEAR_STEPS + 2):
        update = _update(latch, picked=False, measured_angle=0.45)

    assert update.command_angle == 0.22
    assert latch.held_angle == 0.22
    assert latch.loss_streak >= LOSS_CLEAR_STEPS


def test_only_explicit_release_clears_monotonic_hold(latch) -> None:
    _capture(latch, measured_angle=0.22)

    assert latch.release()
    assert latch.held_angle is None
    assert latch.minimum_pick_angle is None
    assert not latch.release()


def test_safety_closure_and_joint_limit_are_applied() -> None:
    latch = GripperPickLatch(
        confirmation_steps=1,
        velocity_tolerance=0.01,
        loss_clear_steps=LOSS_CLEAR_STEPS,
        minimum_angle=MINIMUM_ANGLE,
        safety_closure=0.01,
    )

    first = _update(latch, picked=True, measured_angle=0.25, nominal_angle=0.30)
    assert first.command_angle == pytest.approx(0.24)

    tighter = _update(latch, picked=True, measured_angle=-0.20, nominal_angle=0.30)
    assert tighter.command_angle == MINIMUM_ANGLE
