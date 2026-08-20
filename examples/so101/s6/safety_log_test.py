from __future__ import annotations

import json

import pytest

from examples.so101.s6 import safety_log


def test_s6_event_is_json_safe_and_correlated() -> None:
    record = safety_log.build_s6_event(
        "action_queue_cancelled",
        state_from=safety_log.SafeState.FAULT_DETECTED,
        state_to=safety_log.SafeState.ACTION_QUEUE_CANCELLED,
        fault_type="inference_timeout",
        request_id="request-1",
        action_chunk_id="chunk-1",
        connection_epoch=2,
        queued_actions_before=9,
        queued_actions_after=0,
        detection_latency_ms=12.5,
        safe_action="measured_pose_hold",
        event_time_unix_ns=100,
        event_monotonic_ns=200,
    )

    line = safety_log.format_s6_event(record)
    decoded = json.loads(line.removeprefix(safety_log.LOG_PREFIX))

    assert decoded["request_id"] == "request-1"
    assert decoded["queued_actions_before"] == 9
    assert decoded["queued_actions_after"] == 0
    assert decoded["recovery_condition"] == safety_log.MANUAL_RECOVERY


@pytest.mark.parametrize("latency", [-1.0, float("nan"), float("inf")])
def test_s6_event_rejects_invalid_detection_latency(latency: float) -> None:
    with pytest.raises(ValueError, match="detection_latency_ms"):
        safety_log.build_s6_event(
            "fault",
            state_from=safety_log.SafeState.NORMAL,
            state_to=safety_log.SafeState.FAULT_DETECTED,
            detection_latency_ms=latency,
        )


def test_s6_event_does_not_replace_explicit_invalid_timestamp() -> None:
    with pytest.raises(ValueError, match="event_time_unix_ns"):
        safety_log.build_s6_event(
            "fault",
            state_from=safety_log.SafeState.NORMAL,
            state_to=safety_log.SafeState.FAULT_DETECTED,
            event_time_unix_ns=0,
        )
