import time
import uuid

import numpy as np
import pytest

from examples.franka import backend
from examples.franka import contract
from examples.franka import runtime


class FakeClient:
    def __init__(self, *, fault: str | None = None) -> None:
        self._fault = fault
        self._request = None
        self._response = None
        self.infer_count = 0

    @property
    def connection_epoch(self) -> int:
        return 0

    @property
    def last_request_metadata(self):
        return self._request

    @property
    def last_response_metadata(self):
        return self._response

    def get_server_metadata(self):
        metadata = contract.expected_server_metadata()
        metadata["openpi_protocol_versions"] = [1]
        metadata["real_robot_deployment_allowed"] = False
        return metadata

    def infer(self, observation):
        self.infer_count += 1
        created = time.time_ns()
        self._request = {
            "request_id": str(uuid.uuid4()),
            "client_session_id": "fake-client",
            "connection_epoch": 0,
            "observation_id": str(uuid.uuid4()),
            "request_created_unix_ns": created,
        }
        self._response = {**self._request, "response_id": str(uuid.uuid4())}
        actions = np.repeat(np.asarray(observation["state"])[None], contract.ACTION_HORIZON, axis=0)
        if self._fault == "shape":
            actions = actions[:, :-1]
        return {"actions": actions}


def test_observe_and_shadow_never_command_backend() -> None:
    robot = backend.MockFrankaBackend()
    observe = runtime.FrankaRuntime(backend=robot)
    assert observe.step()["mode"] == "observe"
    assert robot.applied_targets == []

    shadow = runtime.FrankaRuntime(backend=robot, mode=contract.RuntimeMode.SHADOW, policy=FakeClient())
    result = shadow.step()
    assert result["validated_actions"].shape == (16, 8)
    assert robot.applied_targets == []


def test_mock_execute_consumes_one_correlated_action_per_step() -> None:
    robot = backend.MockFrankaBackend()
    client = FakeClient()
    policy_runtime = runtime.FrankaRuntime(
        backend=robot,
        mode=contract.RuntimeMode.EXECUTE,
        policy=client,
    )

    assert policy_runtime.step()["remaining_actions"] == 15
    assert policy_runtime.step()["remaining_actions"] == 14
    assert client.infer_count == 1
    assert len(robot.applied_targets) == 2


def test_fault_cancels_queue_holds_once_and_requires_explicit_recovery() -> None:
    robot = backend.MockFrankaBackend()
    policy_runtime = runtime.FrankaRuntime(
        backend=robot,
        mode=contract.RuntimeMode.EXECUTE,
        policy=FakeClient(fault="shape"),
    )

    with pytest.raises(runtime.FrankaRuntimeError):
        policy_runtime.step()
    assert policy_runtime.remaining_actions == 0
    assert policy_runtime.post_fault_old_action_steps == 0
    assert robot.hold_count == 1
    with pytest.raises(runtime.FrankaRuntimeError):
        policy_runtime.step()
    assert robot.hold_count == 1
    assert robot.applied_targets == []

    policy_runtime.recover()
    assert policy_runtime.fault is None


def test_stale_sensor_mid_chunk_cancels_old_actions() -> None:
    robot = backend.MockFrankaBackend()
    policy_runtime = runtime.FrankaRuntime(
        backend=robot,
        mode=contract.RuntimeMode.EXECUTE,
        policy=FakeClient(),
        safety_config=contract.SafetyConfig(max_sensor_age_s=0.01),
    )
    policy_runtime.step()
    assert policy_runtime.remaining_actions == 15
    original_read_snapshot = robot.read_snapshot

    def stale_snapshot() -> contract.FrankaSnapshot:
        snapshot = original_read_snapshot()
        stale = time.monotonic_ns() - 20_000_000
        return contract.FrankaSnapshot(
            **{
                **snapshot.__dict__,
                "captured_monotonic_ns": stale,
                "base_image_monotonic_ns": stale,
                "wrist_image_monotonic_ns": stale,
            }
        )

    robot.read_snapshot = stale_snapshot  # type: ignore[method-assign]
    with pytest.raises(runtime.FrankaRuntimeError, match="stale"):
        policy_runtime.step()
    assert policy_runtime.remaining_actions == 0
    assert policy_runtime.post_fault_old_action_steps == 0
    assert len(robot.applied_targets) == 1
    assert robot.hold_count == 1


def test_close_cancels_chunk_holds_once_and_is_idempotent() -> None:
    robot = backend.MockFrankaBackend()
    policy_runtime = runtime.FrankaRuntime(
        backend=robot,
        mode=contract.RuntimeMode.EXECUTE,
        policy=FakeClient(),
    )
    policy_runtime.step()

    policy_runtime.close()
    policy_runtime.close()

    assert policy_runtime.remaining_actions == 0
    assert policy_runtime.post_fault_old_action_steps == 0
    assert robot.hold_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        policy_runtime.step()
