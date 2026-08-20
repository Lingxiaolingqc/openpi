from __future__ import annotations

import types

import numpy as np
import pytest

from examples.so101.s6 import camera_safety


def _image(value: int) -> np.ndarray:
    return np.full((1, 8, 10, 3), value, dtype=np.uint8)


def _token(value: float) -> camera_safety.CameraFrameToken:
    return camera_safety.CameraFrameToken(source="_timestamp_last_update", values=(value,))


def test_reads_audited_isaaclab_camera_update_timestamp() -> None:
    sensors = {"front": types.SimpleNamespace(_timestamp_last_update=np.array([0.25], dtype=np.float32))}

    tokens = camera_safety.read_camera_frame_tokens(sensors, ("front",))

    assert tokens["front"].source == "_timestamp_last_update"
    assert tokens["front"].values == pytest.approx((0.25,))


def test_missing_camera_frame_token_is_typed_fault() -> None:
    with pytest.raises(camera_safety.CameraSafetyError) as caught:
        camera_safety.read_camera_frame_tokens({"front": types.SimpleNamespace()}, ("front",))

    assert caught.value.fault_type == "camera_frame_token_unavailable"
    assert caught.value.details["max_undetected_old_action_steps"] == 0


def test_camera_fingerprint_is_deterministic_and_detects_sampled_rgb_change() -> None:
    image = _image(10)
    changed = image.copy()
    changed[0, 0, 0, 0] = 11

    first = camera_safety.camera_observation_fingerprint(image, grid_size=4)

    assert first == camera_safety.camera_observation_fingerprint(image.copy(), grid_size=4)
    assert first != camera_safety.camera_observation_fingerprint(changed, grid_size=4)


def test_tracker_allows_one_30fps_repeat_then_rejects_freeze() -> None:
    tracker = camera_safety.CameraFreshnessTracker(("front",), max_stale_steps=1)
    observation = {"front": _image(10)}
    tokens = {"front": _token(0.0)}
    tracker.reset(observation, tokens)

    first_repeat = tracker.observe(observation, tokens, environment_step_after=1, observed_monotonic_ns=100)

    assert first_repeat["stale_steps_by_camera"] == {"front": 1}
    with pytest.raises(camera_safety.CameraSafetyError) as caught:
        tracker.observe(
            observation,
            tokens,
            environment_step_after=2,
            injection_active=True,
            observed_monotonic_ns=200,
        )
    assert caught.value.fault_type == "camera_freeze"
    assert caught.value.details["camera_name"] == "front"
    assert caught.value.details["environment_step_after"] == 2
    assert caught.value.details["stale_observation_steps"] == 2
    assert caught.value.details["max_camera_stale_steps"] == 1
    assert caught.value.details["max_undetected_old_action_steps"] == 2
    assert caught.value.details["frame_token"] == {
        "source": "_timestamp_last_update",
        "values": [0.0],
    }
    assert caught.value.details["detection_source"] == "sensor_update_token_and_rgb_fingerprint"
    assert caught.value.details["freeze_injected"] is True
    assert caught.value.detection_started_monotonic_ns == 100


def test_tracker_requires_both_token_and_rgb_to_remain_unchanged() -> None:
    tracker = camera_safety.CameraFreshnessTracker(("front",), max_stale_steps=1)
    tracker.reset({"front": _image(10)}, {"front": _token(0.0)})

    token_advanced = tracker.observe({"front": _image(10)}, {"front": _token(0.1)}, environment_step_after=1)
    rgb_changed = tracker.observe({"front": _image(11)}, {"front": _token(0.1)}, environment_step_after=2)

    assert token_advanced["maximum_stale_steps_observed"] == 0
    assert rgb_changed["maximum_stale_steps_observed"] == 0


def test_tracker_accepts_normal_30fps_pattern_during_60fps_control() -> None:
    tracker = camera_safety.CameraFreshnessTracker(("front",), max_stale_steps=1)
    tracker.reset({"front": _image(0)}, {"front": _token(0.0)})

    maximum_stale = 0
    for environment_step in range(1, 21):
        camera_frame = environment_step // 2
        report = tracker.observe(
            {"front": _image(camera_frame)},
            {"front": _token(float(camera_frame))},
            environment_step_after=environment_step,
        )
        maximum_stale = max(maximum_stale, report["maximum_stale_steps_observed"])

    assert maximum_stale == 1


def test_freeze_injector_only_replaces_policy_cameras_and_monitor_tokens() -> None:
    injector = camera_safety.CameraFreezeInjector(("front",), freeze_after_step=3)
    injector.reset({"front": _image(0), "joint_pos": "reset-state"}, {"front": _token(0.0)})

    before, before_tokens, active, started = injector.apply(
        {"front": _image(1), "joint_pos": "state-1"},
        {"front": _token(0.1)},
        environment_step_after=2,
    )
    frozen, frozen_tokens, active, started = injector.apply(
        {"front": _image(2), "joint_pos": "state-2"},
        {"front": _token(0.2)},
        environment_step_after=3,
    )

    assert before["joint_pos"] == "state-1"
    assert before_tokens["front"] == _token(0.1)
    assert frozen["joint_pos"] == "state-2"
    np.testing.assert_array_equal(frozen["front"], _image(1))
    assert frozen_tokens["front"] == _token(0.1)
    assert active is True
    assert started is True

    _, _, active, started = injector.apply(
        {"front": _image(3), "joint_pos": "state-3"},
        {"front": _token(0.3)},
        environment_step_after=4,
    )
    assert active is True
    assert started is False


def test_injected_freeze_reaches_typed_fault_after_bounded_steps() -> None:
    tracker = camera_safety.CameraFreshnessTracker(("front",), max_stale_steps=1)
    injector = camera_safety.CameraFreezeInjector(("front",), freeze_after_step=1)
    baseline_observation = {"front": _image(0)}
    baseline_tokens = {"front": _token(0.0)}
    tracker.reset(baseline_observation, baseline_tokens)
    injector.reset(baseline_observation, baseline_tokens)

    first_observation, first_tokens, active, started = injector.apply(
        {"front": _image(1)},
        {"front": _token(0.1)},
        environment_step_after=1,
    )
    first_check = tracker.observe(
        first_observation,
        first_tokens,
        environment_step_after=1,
        injection_active=active,
    )

    assert started is True
    assert first_check["maximum_stale_steps_observed"] == 1
    second_observation, second_tokens, active, _ = injector.apply(
        {"front": _image(2)},
        {"front": _token(0.2)},
        environment_step_after=2,
    )
    with pytest.raises(camera_safety.CameraSafetyError, match="unchanged for 2") as caught:
        tracker.observe(
            second_observation,
            second_tokens,
            environment_step_after=2,
            injection_active=active,
        )
    assert caught.value.fault_type == "camera_freeze"
    assert caught.value.details["max_undetected_old_action_steps"] == 2
