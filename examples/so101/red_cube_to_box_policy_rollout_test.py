from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from examples.so101 import red_cube_to_box_policy_rollout as rollout


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

    camera_module = types.ModuleType("red_cube_to_box_camera")

    def fake_refresh(env, observations, *, camera_names, refreshes):
        refresh_calls.append((env, observations, camera_names, refreshes))
        return {"frame": "refreshed"}

    camera_module.refresh_camera_observations_without_control = fake_refresh
    monkeypatch.setitem(sys.modules, "red_cube_to_box_camera", camera_module)

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
