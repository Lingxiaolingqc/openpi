"""Correlation, expiry, and cancellation invariants for S6 action chunks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

CORRELATION_FIELDS = (
    "request_id",
    "client_session_id",
    "connection_epoch",
    "observation_id",
    "request_created_unix_ns",
)


class ActionSafetyError(RuntimeError):
    """Raised when an action chunk is unsafe to enqueue or execute."""

    def __init__(self, fault_type: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.fault_type = fault_type
        self.details = details or {}


@dataclass(frozen=True)
class ActionChunkCorrelation:
    request_id: str
    response_id: str
    client_session_id: str
    connection_epoch: int
    observation_id: str
    request_created_unix_ns: int
    server_received_unix_ns: int
    server_completed_unix_ns: int

    @classmethod
    def from_metadata(
        cls,
        request_metadata: dict[str, Any] | None,
        response_metadata: dict[str, Any] | None,
    ) -> ActionChunkCorrelation:
        if not isinstance(request_metadata, dict) or not isinstance(response_metadata, dict):
            raise ActionSafetyError(
                "missing_action_correlation",
                "S6 action chunks require protocol v1 request and response metadata",
            )
        mismatches = {
            name: (request_metadata.get(name), response_metadata.get(name))
            for name in CORRELATION_FIELDS
            if request_metadata.get(name) != response_metadata.get(name)
        }
        if mismatches:
            raise ActionSafetyError(
                "stale_action_correlation",
                f"Action response metadata does not match its request: {mismatches}",
            )
        string_fields = ("request_id", "response_id", "client_session_id", "observation_id")
        for name in string_fields:
            value = response_metadata.get(name)
            if not isinstance(value, str) or not value:
                raise ActionSafetyError("missing_action_correlation", f"Action metadata {name!r} is missing")
        integer_fields = (
            "connection_epoch",
            "request_created_unix_ns",
            "server_received_unix_ns",
            "server_completed_unix_ns",
        )
        for name in integer_fields:
            value = response_metadata.get(name)
            minimum = 0 if name == "connection_epoch" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ActionSafetyError("missing_action_correlation", f"Action metadata {name!r} is invalid")
        return cls(
            request_id=response_metadata["request_id"],
            response_id=response_metadata["response_id"],
            client_session_id=response_metadata["client_session_id"],
            connection_epoch=response_metadata["connection_epoch"],
            observation_id=response_metadata["observation_id"],
            request_created_unix_ns=response_metadata["request_created_unix_ns"],
            server_received_unix_ns=response_metadata["server_received_unix_ns"],
            server_completed_unix_ns=response_metadata["server_completed_unix_ns"],
        )


@dataclass
class GuardedActionChunk:
    actions: np.ndarray
    correlation: ActionChunkCorrelation
    received_unix_ns: int
    received_monotonic_ns: int
    expires_monotonic_ns: int
    next_action_index: int = 0

    @classmethod
    def create(
        cls,
        actions: np.ndarray,
        *,
        request_metadata: dict[str, Any] | None,
        response_metadata: dict[str, Any] | None,
        ttl_s: float,
        received_unix_ns: int | None = None,
        received_monotonic_ns: int | None = None,
    ) -> GuardedActionChunk:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] < 1:
            raise ActionSafetyError("invalid_action_shape", f"Unsafe action chunk shape: {tuple(actions.shape)}")
        if not np.isfinite(actions).all():
            raise ActionSafetyError("invalid_action_nonfinite", "Unsafe action chunk contains NaN or infinity")
        if not math.isfinite(ttl_s) or ttl_s <= 0.0:
            raise ActionSafetyError("invalid_action_ttl", "Action chunk TTL must be finite and positive")
        received_unix_ns = time.time_ns() if received_unix_ns is None else received_unix_ns
        received_monotonic_ns = time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
        for name, value in (
            ("received_unix_ns", received_unix_ns),
            ("received_monotonic_ns", received_monotonic_ns),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ActionSafetyError("invalid_action_timestamp", f"{name} must be a positive integer")
        ttl_ns = max(1, int(ttl_s * 1_000_000_000))
        return cls(
            actions=actions.copy(),
            correlation=ActionChunkCorrelation.from_metadata(request_metadata, response_metadata),
            received_unix_ns=received_unix_ns,
            received_monotonic_ns=received_monotonic_ns,
            expires_monotonic_ns=received_monotonic_ns + ttl_ns,
        )

    @property
    def action_chunk_id(self) -> str:
        return self.correlation.response_id

    @property
    def remaining_actions(self) -> int:
        return self.actions.shape[0] - self.next_action_index

    def validate_before_step(self, *, current_connection_epoch: int, now_monotonic_ns: int | None = None) -> None:
        now_monotonic_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        if current_connection_epoch != self.correlation.connection_epoch:
            raise ActionSafetyError(
                "action_chunk_epoch_mismatch",
                "Action chunk connection epoch changed from "
                f"{self.correlation.connection_epoch} to {current_connection_epoch}",
            )
        if now_monotonic_ns > self.expires_monotonic_ns:
            expired_ms = (now_monotonic_ns - self.expires_monotonic_ns) / 1_000_000.0
            raise ActionSafetyError(
                "action_chunk_expired",
                f"Action chunk exceeded its TTL by {expired_ms:.3f} ms",
            )


class S6ActionQueue:
    """Single-chunk queue that cannot emit actions after fault cancellation."""

    def __init__(self) -> None:
        self._chunk: GuardedActionChunk | None = None
        self._fault_detected = False
        self._post_fault_old_action_steps = 0

    @property
    def chunk(self) -> GuardedActionChunk | None:
        return self._chunk

    @property
    def remaining_actions(self) -> int:
        return 0 if self._chunk is None else self._chunk.remaining_actions

    @property
    def post_fault_old_action_steps(self) -> int:
        return self._post_fault_old_action_steps

    def load(self, chunk: GuardedActionChunk) -> None:
        if self._fault_detected:
            raise ActionSafetyError("action_queue_cancelled", "Cannot load an action chunk after a fault")
        if self.remaining_actions:
            raise ActionSafetyError(
                "action_queue_not_empty",
                f"Cannot replace {self.remaining_actions} queued actions with a newer chunk",
            )
        self._chunk = chunk

    def peek_next(self, *, current_connection_epoch: int, now_monotonic_ns: int | None = None) -> np.ndarray:
        if self._fault_detected:
            raise ActionSafetyError("action_queue_cancelled", "Action queue was cancelled after a fault")
        if self._chunk is None or self._chunk.remaining_actions < 1:
            raise ActionSafetyError("action_queue_empty", "No action is queued")
        self._chunk.validate_before_step(
            current_connection_epoch=current_connection_epoch,
            now_monotonic_ns=now_monotonic_ns,
        )
        return self._chunk.actions[self._chunk.next_action_index]

    def mark_executed(self) -> None:
        if self._fault_detected:
            self._post_fault_old_action_steps += 1
            raise ActionSafetyError(
                "action_executed_after_fault",
                "An old policy action was marked executed after fault detection",
            )
        if self._chunk is None or self._chunk.remaining_actions < 1:
            raise ActionSafetyError("action_queue_empty", "Cannot mark an empty queue action as executed")
        self._chunk.next_action_index += 1

    def cancel(self) -> tuple[int, int]:
        self._fault_detected = True
        queued_before = self.remaining_actions
        if self._chunk is not None:
            self._chunk.next_action_index = self._chunk.actions.shape[0]
        return queued_before, self.remaining_actions

    def discard(self) -> tuple[int, int]:
        """Clear a non-faulted queue at a normal episode boundary."""

        if self._fault_detected:
            raise ActionSafetyError("action_queue_cancelled", "Cancelled action queue cannot be normally discarded")
        queued_before = self.remaining_actions
        self._chunk = None
        return queued_before, self.remaining_actions
