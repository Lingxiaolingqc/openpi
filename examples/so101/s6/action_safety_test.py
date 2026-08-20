from __future__ import annotations

import numpy as np
import pytest

from examples.so101.s6 import action_safety


def _metadata(*, request_id: str = "request-1", response_id: str = "response-1", epoch: int = 2):
    request = {
        "request_id": request_id,
        "client_session_id": "session-1",
        "connection_epoch": epoch,
        "observation_id": "observation-1",
        "request_created_unix_ns": 100,
    }
    response = {
        **request,
        "response_id": response_id,
        "server_received_unix_ns": 110,
        "server_completed_unix_ns": 120,
    }
    return request, response


def _chunk(*, horizon: int = 10, ttl_s: float = 1.0):
    request, response = _metadata()
    return action_safety.GuardedActionChunk.create(
        np.arange(horizon * 6, dtype=np.float32).reshape(horizon, 1, 6),
        request_metadata=request,
        response_metadata=response,
        ttl_s=ttl_s,
        received_unix_ns=1_000,
        received_monotonic_ns=2_000,
    )


def test_fault_cancellation_clears_remaining_actions_and_prevents_reuse() -> None:
    queue = action_safety.S6ActionQueue()
    queue.load(_chunk())

    for _ in range(3):
        queue.peek_next(current_connection_epoch=2, now_monotonic_ns=2_100)
        queue.mark_executed()

    assert queue.cancel() == (7, 0)
    assert queue.post_fault_old_action_steps == 0
    with pytest.raises(action_safety.ActionSafetyError, match="cancelled"):
        queue.peek_next(current_connection_epoch=2, now_monotonic_ns=2_100)
    assert queue.post_fault_old_action_steps == 0


def test_marking_action_after_fault_is_detected_as_invariant_violation() -> None:
    queue = action_safety.S6ActionQueue()
    queue.load(_chunk())
    queue.cancel()

    with pytest.raises(action_safety.ActionSafetyError) as caught:
        queue.mark_executed()

    assert caught.value.fault_type == "action_executed_after_fault"
    assert queue.post_fault_old_action_steps == 1


def test_chunk_rejects_expiry_before_next_step() -> None:
    queue = action_safety.S6ActionQueue()
    queue.load(_chunk(ttl_s=0.001))

    with pytest.raises(action_safety.ActionSafetyError) as caught:
        queue.peek_next(current_connection_epoch=2, now_monotonic_ns=1_002_001)

    assert caught.value.fault_type == "action_chunk_expired"


def test_chunk_rejects_connection_epoch_change() -> None:
    queue = action_safety.S6ActionQueue()
    queue.load(_chunk())

    with pytest.raises(action_safety.ActionSafetyError) as caught:
        queue.peek_next(current_connection_epoch=3, now_monotonic_ns=2_100)

    assert caught.value.fault_type == "action_chunk_epoch_mismatch"


def test_chunk_rejects_mismatched_response_correlation() -> None:
    request, response = _metadata()
    response["observation_id"] = "stale-observation"

    with pytest.raises(action_safety.ActionSafetyError) as caught:
        action_safety.GuardedActionChunk.create(
            np.zeros((10, 1, 6), dtype=np.float32),
            request_metadata=request,
            response_metadata=response,
            ttl_s=1.0,
        )

    assert caught.value.fault_type == "stale_action_correlation"


def test_queue_refuses_to_replace_unexecuted_old_chunk() -> None:
    queue = action_safety.S6ActionQueue()
    queue.load(_chunk())

    with pytest.raises(action_safety.ActionSafetyError) as caught:
        queue.load(_chunk())

    assert caught.value.fault_type == "action_queue_not_empty"


def test_completed_chunk_can_be_replaced_without_reconnect() -> None:
    queue = action_safety.S6ActionQueue()
    first = _chunk(horizon=1)
    queue.load(first)
    queue.peek_next(current_connection_epoch=2, now_monotonic_ns=2_100)
    queue.mark_executed()

    request, response = _metadata(request_id="request-2", response_id="response-2")
    second = action_safety.GuardedActionChunk.create(
        np.ones((1, 1, 6), dtype=np.float32),
        request_metadata=request,
        response_metadata=response,
        ttl_s=1.0,
        received_unix_ns=2_000,
        received_monotonic_ns=3_000,
    )
    queue.load(second)

    np.testing.assert_array_equal(
        queue.peek_next(current_connection_epoch=2, now_monotonic_ns=3_100),
        second.actions[0],
    )
