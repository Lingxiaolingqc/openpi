from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from examples.so101 import red_cube_to_box_policy_rollout as rollout
from examples.so101.s6 import action_safety
from examples.so101.s6 import safety_log


def test_normalize_action_chunk_accepts_server_and_compact_shapes() -> None:
    compact = np.arange(60, dtype=np.float32).reshape(10, 6)
    expanded = rollout.normalize_action_chunk(compact)

    assert expanded.shape == (10, 1, 6)
    np.testing.assert_array_equal(expanded[:, 0], compact)
    np.testing.assert_array_equal(rollout.normalize_action_chunk(expanded), expanded)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros((10, 7), dtype=np.float32), "action chunk shape"),
        (np.zeros((0, 6), dtype=np.float32), "empty action chunk"),
        (np.full((10, 6), np.nan, dtype=np.float32), "non-finite action chunk"),
    ],
)
def test_normalize_action_chunk_rejects_invalid_outputs(value: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rollout.normalize_action_chunk(value)


def test_clip_action_chunk_reports_every_clipped_scalar() -> None:
    actions = np.array(
        [
            [
                [-2.0, -0.5, 0.0, 0.5, 2.0, 0.25],
            ]
        ],
        dtype=np.float32,
    )
    lower = np.full(6, -1.0, dtype=np.float32)
    upper = np.full(6, 1.0, dtype=np.float32)

    clipped, count, maximum = rollout.clip_action_chunk(actions, lower, upper)

    assert count == 2
    assert maximum == pytest.approx(1.0)
    np.testing.assert_allclose(clipped[0, 0], [-1.0, -0.5, 0.0, 0.5, 1.0, 0.25])


def test_clip_action_chunk_rejects_limit_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="Joint-limit dimension"):
        rollout.clip_action_chunk(np.zeros((10, 1, 6), dtype=np.float32), np.zeros(5), np.ones(5))


def test_action_chunk_clip_by_joint_reports_axis_specific_corrections() -> None:
    raw = np.zeros((2, 1, 3), dtype=np.float32)
    raw[0, 0] = [-2.0, 0.0, 2.0]
    raw[1, 0] = [-3.0, 0.5, 1.5]
    clipped = np.clip(raw, -1.0, 1.0)

    counts, maximum = rollout.action_chunk_clip_by_joint(raw, clipped)

    np.testing.assert_array_equal(counts, [2, 0, 2])
    np.testing.assert_allclose(maximum, [2.0, 0.0, 1.0])


def test_action_chunk_clip_by_joint_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        rollout.action_chunk_clip_by_joint(
            np.zeros((2, 1, 6), dtype=np.float32),
            np.zeros((1, 1, 6), dtype=np.float32),
        )


def test_s6_soft_limit_violation_is_rejected_without_clipping() -> None:
    with pytest.raises(action_safety.ActionSafetyError) as caught:
        rollout.reject_s6_soft_limit_violations(
            violation_count=2,
            maximum_violation_rad=0.25,
            violation_count_by_joint=np.array([0, 1, 0, 1, 0, 0]),
            maximum_violation_by_joint_rad=np.array([0.0, 0.1, 0.0, 0.25, 0.0, 0.0]),
        )

    assert caught.value.fault_type == "invalid_action_out_of_range"
    assert caught.value.details["soft_limit_violation_count"] == 2
    assert caught.value.details["maximum_soft_limit_violation_rad"] == pytest.approx(0.25)


def test_s6_and_match_expert_dynamics_preserve_valid_raw_targets() -> None:
    raw = np.array([[[0.125, -0.25, 0.5, -0.75, 1.0, 0.25]]], dtype=np.float32)
    clipped = raw.copy()

    selected = rollout.select_action_chunk_for_execution(
        raw,
        clipped,
        s6_safety=True,
        match_expert_dynamics=True,
    )

    assert selected is raw
    np.testing.assert_array_equal(selected, raw)
    assert (
        rollout.policy_action_out_of_range_behavior(
            s6_safety=True,
            match_expert_dynamics=True,
        )
        == "reject"
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((True, False), "reject"),
        ((True, True), "reject"),
        ((False, True), "execute_raw"),
        ((False, False), "clip"),
    ],
)
def test_policy_action_out_of_range_behavior_is_auditable(
    flags: tuple[bool, bool],
    expected: str,
) -> None:
    s6_safety, match_expert_dynamics = flags
    assert (
        rollout.policy_action_out_of_range_behavior(
            s6_safety=s6_safety,
            match_expert_dynamics=match_expert_dynamics,
        )
        == expected
    )


def _guarded_chunk(horizon: int = 10) -> action_safety.GuardedActionChunk:
    request = {
        "request_id": "request-1",
        "client_session_id": "session-1",
        "connection_epoch": 0,
        "observation_id": "observation-1",
        "request_created_unix_ns": 100,
    }
    response = {
        **request,
        "response_id": "response-1",
        "server_received_unix_ns": 110,
        "server_completed_unix_ns": 120,
    }
    return action_safety.GuardedActionChunk.create(
        np.zeros((horizon, 1, 6), dtype=np.float32),
        request_metadata=request,
        response_metadata=response,
        ttl_s=1.0,
        received_unix_ns=1_000,
        received_monotonic_ns=2_000,
    )


def test_s6_fault_path_clears_queue_holds_pose_and_requires_recovery(capsys) -> None:
    queue = action_safety.S6ActionQueue()
    queue.load(_guarded_chunk())
    for _ in range(3):
        queue.peek_next(current_connection_epoch=0, now_monotonic_ns=2_100)
        queue.mark_executed()

    class FakeEnv:
        def __init__(self) -> None:
            self.actions: list[np.ndarray] = []

        def step(self, action):
            self.actions.append(action)
            return ({}, np.zeros(1), _WarmupBool(value=False), _WarmupBool(value=False), {})

    env = FakeEnv()
    measured_pose = np.arange(6, dtype=np.float32)[None]
    robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=_WarmupJointPositions(measured_pose)))
    fake_torch = types.SimpleNamespace(isfinite=lambda value: types.SimpleNamespace(all=lambda: True))
    context = rollout._enter_s6_safe_state(  # noqa: SLF001
        exc=action_safety.ActionSafetyError("injected_disconnect", "test fault"),
        queue=queue,
        policy=None,
        env=env,
        robot=robot,
        joint_ids=list(range(6)),
        dynamic_gripper_reset=None,
        torch_module=fake_torch,
        detection_started_ns=1,
    )
    rollout._emit_s6_terminated(context)  # noqa: SLF001

    records = [
        json.loads(line.removeprefix(safety_log.LOG_PREFIX))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(safety_log.LOG_PREFIX)
    ]
    assert [record["event"] for record in records] == [
        "rollout_fault_detected",
        "action_queue_cancelled",
        "safe_hold_applied",
        "simulation_terminated",
        "recovery_required",
    ]
    assert records[1]["queued_actions_before"] == 7
    assert records[1]["queued_actions_after"] == 0
    assert records[1]["details"]["post_fault_old_action_steps"] == 0
    assert records[-1]["state_to"] == "recovery_required"
    assert queue.remaining_actions == 0
    assert queue.post_fault_old_action_steps == 0
    np.testing.assert_array_equal(env.actions, [measured_pose])


class _WarmupBool:
    def __init__(self, *, value: bool) -> None:
        self._value = value

    def any(self) -> bool:
        return self._value


class _WarmupJointPositions:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def __getitem__(self, item):
        return self

    def clone(self) -> np.ndarray:
        return self._values.copy()


def test_reset_camera_warmup_is_explicit_and_holds_current_joints() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.actions: list[np.ndarray] = []

        def reset(self):
            return {"frame": 0}, {}

        def step(self, action):
            self.actions.append(action)
            return (
                {"frame": len(self.actions)},
                None,
                _WarmupBool(value=False),
                _WarmupBool(value=False),
                {},
            )

    env = FakeEnv()
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(joint_pos=_WarmupJointPositions(np.arange(6, dtype=np.float32)[None]))
    )
    reset_calls: list[tuple[object, str]] = []

    observation = rollout.reset_with_camera_warmup(
        env,
        robot,
        joint_ids=list(range(6)),
        warmup_steps=1,
        dynamic_gripper_reset=lambda current_env, robot_name: reset_calls.append((current_env, robot_name)),
    )

    assert observation == {"frame": 1}
    assert len(env.actions) == 1
    np.testing.assert_array_equal(env.actions[0], np.arange(6, dtype=np.float32)[None])
    assert reset_calls == [(env, "so101leader")]


def test_reset_camera_warmup_zero_keeps_reset_observation() -> None:
    class FakeEnv:
        def reset(self):
            return {"frame": 0}, {}

        def step(self, action):
            raise AssertionError("warmup step must remain opt-in")

    robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=None))

    assert rollout.reset_with_camera_warmup(FakeEnv(), robot, joint_ids=list(range(6)), warmup_steps=0) == {"frame": 0}


def test_reset_camera_refresh_does_not_step_environment(monkeypatch) -> None:
    refresh_calls: list[tuple[object, object, tuple[str, ...], int]] = []

    module_name = "examples.so101.utils.red_cube_to_box_camera"
    camera_module = types.ModuleType(module_name)

    def fake_refresh(env, observations, *, camera_names, refreshes):
        refresh_calls.append((env, observations, camera_names, refreshes))
        return {"frame": "refreshed"}

    camera_module.refresh_camera_observations_without_control = fake_refresh
    monkeypatch.setitem(sys.modules, module_name, camera_module)

    class FakeEnv:
        def reset(self):
            return {"frame": 0}, {}

        def step(self, action):
            raise AssertionError("sensor-only refresh must not advance the environment")

    env = FakeEnv()
    robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=None))

    observation = rollout.reset_with_camera_warmup(
        env,
        robot,
        joint_ids=list(range(6)),
        warmup_steps=0,
        camera_names=("front",),
        camera_refreshes=1,
    )

    assert observation == {"frame": "refreshed"}
    assert refresh_calls == [(env, {"frame": 0}, ("front",), 1)]


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __getitem__(self, item) -> _FakeTensor:
        return _FakeTensor(self.value[item])

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


def test_openpi_adapter_builds_so101_request_and_restores_simulator_actions(monkeypatch) -> None:
    requests: list[dict] = []
    closed: list[bool] = []

    class FakeWebsocketClientPolicy:
        def __init__(self, *, host: str, port: int) -> None:
            assert (host, port) == ("policy-host", 18000)
            self._ws = types.SimpleNamespace(close=lambda: closed.append(True))

        def get_server_metadata(self) -> dict:
            return {
                "model": "fake",
                "deployment_scope": "simulation-only",
                "supports_remote_reset": True,
            }

        def infer(self, request: dict) -> dict:
            requests.append(request)
            if request.get("__reset__") is True:
                return {"reset_ack": True}
            return {
                "actions": np.full((10, 6), 2.0, dtype=np.float32),
                "policy_diagnostics": {"actions_clipped": False},
            }

    robot_utils = types.ModuleType("leisaac.utils.robot_utils")
    robot_utils.convert_leisaac_action_to_lerobot = lambda value: value.detach().cpu().numpy() + 10.0
    robot_utils.convert_lerobot_action_to_leisaac = lambda value: value - 1.0
    websocket_module = types.SimpleNamespace(WebsocketClientPolicy=FakeWebsocketClientPolicy)
    openpi_client = types.ModuleType("openpi_client")
    openpi_client.websocket_client_policy = websocket_module
    monkeypatch.setitem(sys.modules, "leisaac", types.ModuleType("leisaac"))
    monkeypatch.setitem(sys.modules, "leisaac.utils", types.ModuleType("leisaac.utils"))
    monkeypatch.setitem(sys.modules, "leisaac.utils.robot_utils", robot_utils)
    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client)

    client = rollout.OpenPISO101Client(
        host="policy-host",
        port=18000,
        camera_names=("front",),
        prompt=rollout.TASK_PROMPT,
        required_deployment_scope="simulation-only",
    )
    client.reset()
    actions = client.get_action(
        {
            "front": _FakeTensor(np.zeros((1, 4, 5, 3), dtype=np.uint8)),
            "joint_pos": _FakeTensor(np.arange(6, dtype=np.float32)[None]),
        }
    )

    assert client.server_metadata["model"] == "fake"
    assert client.last_policy_diagnostics == {"actions_clipped": False}
    assert actions.shape == (10, 1, 6)
    np.testing.assert_array_equal(actions, np.ones((10, 1, 6), dtype=np.float32))
    assert requests[0] == {"__reset__": True}
    assert requests[1]["images/front"].shape == (4, 5, 3)
    np.testing.assert_array_equal(requests[1]["state"], np.arange(6, dtype=np.float32) + 10.0)
    assert requests[1]["prompt"] == rollout.TASK_PROMPT
    client.close()
    assert closed == [True]


def test_openpi_adapter_s6_mode_exposes_correlation_and_bounded_watchdog(monkeypatch) -> None:
    health_checks: list[float] = []
    closed: list[bool] = []

    class FakeWebsocketClientPolicy:
        def __init__(self, **kwargs) -> None:
            assert kwargs == {
                "host": "policy-host",
                "port": 18000,
                "protocol_version": 1,
                "connect_timeout_s": 3.0,
                "inference_timeout_s": 4.0,
            }
            self.connection_epoch = 2
            self.last_request_metadata = None
            self.last_response_metadata = None

        def get_server_metadata(self) -> dict:
            return {"deployment_scope": "simulation-only", "openpi_protocol_versions": [1]}

        def infer(self, request: dict) -> dict:
            self.last_request_metadata = {
                "request_id": "request-1",
                "client_session_id": "session-1",
                "connection_epoch": 2,
                "observation_id": "observation-1",
                "request_created_unix_ns": 100,
            }
            self.last_response_metadata = {
                **self.last_request_metadata,
                "response_id": "response-1",
                "server_received_unix_ns": 110,
                "server_completed_unix_ns": 120,
            }
            return {"actions": np.zeros((10, 6), dtype=np.float32)}

        def health_check(self, *, timeout_s: float) -> None:
            health_checks.append(timeout_s)

        def close(self) -> None:
            closed.append(True)

    robot_utils = types.ModuleType("leisaac.utils.robot_utils")
    robot_utils.convert_leisaac_action_to_lerobot = lambda value: value.detach().cpu().numpy()
    robot_utils.convert_lerobot_action_to_leisaac = lambda value: value
    openpi_client = types.ModuleType("openpi_client")
    openpi_client.websocket_client_policy = types.SimpleNamespace(WebsocketClientPolicy=FakeWebsocketClientPolicy)
    monkeypatch.setitem(sys.modules, "leisaac", types.ModuleType("leisaac"))
    monkeypatch.setitem(sys.modules, "leisaac.utils", types.ModuleType("leisaac.utils"))
    monkeypatch.setitem(sys.modules, "leisaac.utils.robot_utils", robot_utils)
    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client)

    client = rollout.OpenPISO101Client(
        host="policy-host",
        port=18000,
        camera_names=("front",),
        prompt=rollout.TASK_PROMPT,
        required_deployment_scope="simulation-only",
        s6_client_options={"connect_timeout_s": 3.0, "inference_timeout_s": 4.0},
    )
    client.get_action(
        {
            "front": _FakeTensor(np.zeros((1, 4, 5, 3), dtype=np.uint8)),
            "joint_pos": _FakeTensor(np.arange(6, dtype=np.float32)[None]),
        }
    )
    client.health_check(timeout_s=0.25)

    assert client.connection_epoch == 2
    assert client.last_request_metadata["request_id"] == "request-1"
    assert client.last_response_metadata["response_id"] == "response-1"
    assert client.last_response_received_unix_ns is not None
    assert client.last_response_received_monotonic_ns is not None
    assert health_checks == [0.25]
    client.close()
    assert closed == [True]


def test_openpi_adapter_rejects_wrong_policy_scope(monkeypatch) -> None:
    class FakeWebsocketClientPolicy:
        def __init__(self, *, host: str, port: int) -> None:
            pass

        def get_server_metadata(self) -> dict:
            return {"deployment_scope": "real"}

    robot_utils = types.ModuleType("leisaac.utils.robot_utils")
    robot_utils.convert_leisaac_action_to_lerobot = lambda value: value
    robot_utils.convert_lerobot_action_to_leisaac = lambda value: value
    openpi_client = types.ModuleType("openpi_client")
    openpi_client.websocket_client_policy = types.SimpleNamespace(WebsocketClientPolicy=FakeWebsocketClientPolicy)
    monkeypatch.setitem(sys.modules, "leisaac", types.ModuleType("leisaac"))
    monkeypatch.setitem(sys.modules, "leisaac.utils", types.ModuleType("leisaac.utils"))
    monkeypatch.setitem(sys.modules, "leisaac.utils.robot_utils", robot_utils)
    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client)

    with pytest.raises(ValueError, match="deployment scope"):
        rollout.OpenPISO101Client(
            host="policy-host",
            port=18000,
            camera_names=("front",),
            prompt=rollout.TASK_PROMPT,
            required_deployment_scope="simulation-only",
        )
