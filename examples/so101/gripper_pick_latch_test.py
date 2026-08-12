"""Behavioral regression tests for debounced gripper-angle holding."""

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
MEASURED_ANGLE = 0.259811
CONFIRMATION_STEPS = 8
LOSS_CLEAR_STEPS = 3


@pytest.fixture
def latch():
    return GripperPickLatch(
        confirmation_steps=CONFIRMATION_STEPS,
        velocity_tolerance=0.01,
        loss_clear_steps=LOSS_CLEAR_STEPS,
    )


def _update(latch, *, picked: bool, velocity: float = 0.0):
    return latch.update(
        picked=picked,
        measured_angle=MEASURED_ANGLE,
        measured_velocity=velocity,
        nominal_angle=NOMINAL_ANGLE,
        allow_capture=True,
        allow_release=True,
    )


def _capture(latch) -> None:
    for _ in range(CONFIRMATION_STEPS):
        update = _update(latch, picked=True)
    assert update.captured


def test_single_frame_pick_does_not_latch(latch) -> None:
    first = _update(latch, picked=True)
    lost = _update(latch, picked=False)

    assert first.command_angle == NOMINAL_ANGLE
    assert not first.captured
    assert latch.held_angle is None
    assert latch.confirmation_streak == 0
    assert lost.command_angle == NOMINAL_ANGLE


def test_fast_pick_signal_does_not_latch(latch) -> None:
    for _ in range(CONFIRMATION_STEPS + 2):
        update = _update(latch, picked=True, velocity=0.290102)

    assert not update.captured
    assert update.command_angle == NOMINAL_ANGLE
    assert latch.held_angle is None
    assert latch.confirmation_streak == CONFIRMATION_STEPS + 2

    captured_after_settling = _update(latch, picked=True)
    assert captured_after_settling.captured
    assert captured_after_settling.command_angle == MEASURED_ANGLE


def test_consecutive_low_speed_pick_latches_on_final_frame(latch) -> None:
    for _ in range(CONFIRMATION_STEPS - 1):
        update = _update(latch, picked=True)
        assert not update.captured
        assert update.command_angle == NOMINAL_ANGLE

    captured = _update(latch, picked=True)
    repeated = _update(latch, picked=True)

    assert captured.captured
    assert captured.command_angle == MEASURED_ANGLE
    assert latch.held_angle == MEASURED_ANGLE
    assert not repeated.captured


def test_pick_loss_before_confirmation_restores_nominal_target(latch) -> None:
    for _ in range(CONFIRMATION_STEPS - 1):
        _update(latch, picked=True)

    lost = _update(latch, picked=False)

    assert latch.confirmation_streak == 0
    assert latch.held_angle is None
    assert not lost.released
    assert lost.command_angle == NOMINAL_ANGLE


def test_pick_loss_after_latch_is_debounced_then_restores_nominal_target(latch) -> None:
    _capture(latch)

    for expected_loss_streak in range(1, LOSS_CLEAR_STEPS):
        update = _update(latch, picked=False)
        assert not update.released
        assert update.command_angle == MEASURED_ANGLE
        assert latch.loss_streak == expected_loss_streak

    released = _update(latch, picked=False)

    assert released.released
    assert released.command_angle == NOMINAL_ANGLE
    assert latch.held_angle is None
    assert latch.loss_streak == 0


def test_latch_does_not_clear_after_capture_when_release_is_not_allowed(latch) -> None:
    _capture(latch)

    for _ in range(LOSS_CLEAR_STEPS + 2):
        update = latch.update(
            picked=False,
            measured_angle=MEASURED_ANGLE,
            measured_velocity=0.0,
            nominal_angle=NOMINAL_ANGLE,
            allow_capture=False,
            allow_release=False,
        )

    assert not update.released
    assert update.command_angle == MEASURED_ANGLE
    assert latch.held_angle == MEASURED_ANGLE
