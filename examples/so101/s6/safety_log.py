"""Machine-readable S6 fault and safe-state event schema."""

from __future__ import annotations

import enum
import json
import math
import time
from typing import Any

SCHEMA_VERSION = 1
LOG_PREFIX = "S6_EVENT "
MANUAL_RECOVERY = "explicit_manual_confirmation_required"


class SafeState(enum.Enum):
    NORMAL = "normal"
    FAULT_DETECTED = "fault_detected"
    RESPONSE_REJECTED = "response_rejected"
    ACTION_QUEUE_CANCELLED = "action_queue_cancelled"
    SAFE_HOLD = "safe_hold"
    SIMULATION_TERMINATED = "simulation_terminated"
    RECOVERY_REQUIRED = "recovery_required"


def build_s6_event(
    event: str,
    *,
    state_from: SafeState,
    state_to: SafeState,
    fault_type: str | None = None,
    request_id: str | None = None,
    response_id: str | None = None,
    observation_id: str | None = None,
    action_chunk_id: str | None = None,
    connection_epoch: int | None = None,
    queued_actions_before: int | None = None,
    queued_actions_after: int | None = None,
    detection_latency_ms: float | None = None,
    safe_action: str | None = None,
    recovery_condition: str = MANUAL_RECOVERY,
    details: dict[str, Any] | None = None,
    event_time_unix_ns: int | None = None,
    event_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe S6 event with stable correlation fields."""

    if not event:
        raise ValueError("event must be non-empty")
    if connection_epoch is not None and (
        isinstance(connection_epoch, bool) or not isinstance(connection_epoch, int) or connection_epoch < 0
    ):
        raise ValueError("connection_epoch must be non-negative")
    for name, value in (
        ("queued_actions_before", queued_actions_before),
        ("queued_actions_after", queued_actions_after),
    ):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be non-negative")
    if detection_latency_ms is not None and (not math.isfinite(detection_latency_ms) or detection_latency_ms < 0.0):
        raise ValueError("detection_latency_ms must be finite and non-negative")
    event_time_unix_ns = time.time_ns() if event_time_unix_ns is None else event_time_unix_ns
    event_monotonic_ns = time.monotonic_ns() if event_monotonic_ns is None else event_monotonic_ns
    for name, value in (
        ("event_time_unix_ns", event_time_unix_ns),
        ("event_monotonic_ns", event_monotonic_ns),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not recovery_condition:
        raise ValueError("recovery_condition must be non-empty")
    record = {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "event_time_unix_ns": event_time_unix_ns,
        "event_monotonic_ns": event_monotonic_ns,
        "state_from": state_from.value,
        "state_to": state_to.value,
        "fault_type": fault_type,
        "request_id": request_id,
        "response_id": response_id,
        "observation_id": observation_id,
        "action_chunk_id": action_chunk_id,
        "connection_epoch": connection_epoch,
        "queued_actions_before": queued_actions_before,
        "queued_actions_after": queued_actions_after,
        "detection_latency_ms": detection_latency_ms,
        "safe_action": safe_action,
        "recovery_condition": recovery_condition,
        "details": details or {},
    }
    json.dumps(record, allow_nan=False)
    return record


def format_s6_event(record: dict[str, Any]) -> str:
    """Serialize an S6 event for direct terminal logging."""

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported S6 log schema version: {record.get('schema_version')!r}")
    return LOG_PREFIX + json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
