import time
import uuid

import numpy as np
import pytest

from examples.franka import action_safety
from examples.franka import backend
from examples.franka import contract


def _metadata() -> tuple[dict, dict]:
    created = time.time_ns()
    request = {
        "request_id": str(uuid.uuid4()),
        "client_session_id": str(uuid.uuid4()),
        "connection_epoch": 0,
        "observation_id": str(uuid.uuid4()),
        "request_created_unix_ns": created,
    }
    response = {**request, "response_id": str(uuid.uuid4())}
    return request, response


def _safe_actions(snapshot: contract.FrankaSnapshot) -> np.ndarray:
    state = np.concatenate([snapshot.q_rad, [snapshot.gripper_width_m]])
    return np.repeat(state[None], contract.ACTION_HORIZON, axis=0)


def test_safe_chunk_and_cancelled_queue_never_emit_old_action() -> None:
    snapshot = backend.MockFrankaBackend().read_snapshot()
    request, response = _metadata()
    chunk = action_safety.GuardedFrankaChunk.create(
        _safe_actions(snapshot),
        snapshot=snapshot,
        config=contract.SafetyConfig(),
        request_metadata=request,
        response_metadata=response,
    )
    queue = action_safety.FrankaActionQueue()
    queue.load(chunk)
    target = queue.peek(connection_epoch=0)
    np.testing.assert_array_equal(target.q_target_rad, snapshot.q_rad)

    assert queue.cancel() == (contract.ACTION_HORIZON, 0)
    with pytest.raises(action_safety.FrankaSafetyError, match="after a fault"):
        queue.peek(connection_epoch=0)
    assert queue.post_fault_old_action_steps == 0


@pytest.mark.parametrize("mutation", ["shape", "nan", "joint", "velocity", "gripper"])
def test_unsafe_chunks_are_rejected(mutation: str) -> None:
    snapshot = backend.MockFrankaBackend().read_snapshot()
    actions = _safe_actions(snapshot)
    if mutation == "shape":
        actions = actions[:, :-1]
    elif mutation == "nan":
        actions[0, 0] = np.nan
    elif mutation == "joint":
        actions[0, 0] = contract.Q_MAX_RAD[0] + 1.0
    elif mutation == "velocity":
        actions[0, 0] += 0.1
    elif mutation == "gripper":
        actions[0, 7] = 1.0
    request, response = _metadata()

    with pytest.raises(action_safety.FrankaSafetyError):
        action_safety.GuardedFrankaChunk.create(
            actions,
            snapshot=snapshot,
            config=contract.SafetyConfig(),
            request_metadata=request,
            response_metadata=response,
        )


def test_response_correlation_is_mandatory() -> None:
    snapshot = backend.MockFrankaBackend().read_snapshot()
    request, response = _metadata()
    response["observation_id"] = "wrong"
    with pytest.raises(action_safety.FrankaSafetyError, match="does not match"):
        action_safety.GuardedFrankaChunk.create(
            _safe_actions(snapshot),
            snapshot=snapshot,
            config=contract.SafetyConfig(),
            request_metadata=request,
            response_metadata=response,
        )
