from __future__ import annotations

import types

import pytest

from examples.so101.red_cube_to_box_camera import refresh_camera_observations_without_control


def test_refresh_camera_observations_renders_and_updates_without_control_step() -> None:
    events: list[object] = []

    class FakeCamera:
        def update(self, dt, *, force_recompute):
            events.append(("camera", dt, force_recompute))

    class FakeObservationManager:
        def compute(self, *, update_history):
            events.append(("observations", update_history))
            return {"refresh": len(events)}

    env = types.SimpleNamespace(
        sim=types.SimpleNamespace(render=lambda: events.append("render")),
        scene=types.SimpleNamespace(sensors={"front": FakeCamera()}),
        observation_manager=FakeObservationManager(),
        physics_dt=1.0 / 120.0,
    )

    result = refresh_camera_observations_without_control(
        env,
        {"refresh": 0},
        camera_names=("front",),
        refreshes=1,
    )

    assert result == {"refresh": 3}
    assert events == ["render", ("camera", 1.0 / 120.0, True), ("observations", False)]


def test_refresh_camera_observations_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        refresh_camera_observations_without_control(
            object(),
            {},
            camera_names=("front",),
            refreshes=-1,
        )
