"""Fail-closed validation and queueing for absolute FR3 action chunks."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from examples.franka import contract

CORRELATION_FIELDS = (
    "request_id",
    "client_session_id",
    "connection_epoch",
    "observation_id",
    "request_created_unix_ns",
)


class FrankaSafetyError(RuntimeError):
    def __init__(self, fault_type: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.fault_type = fault_type
        self.details = details or {}


@dataclass(frozen=True)
class Correlation:
    request_id: str
    response_id: str
    client_session_id: str
    connection_epoch: int
    observation_id: str
    request_created_unix_ns: int

    @classmethod
    def create(cls, request: dict[str, Any] | None, response: dict[str, Any] | None) -> Correlation:
        if not isinstance(request, dict) or not isinstance(response, dict):
            raise FrankaSafetyError("missing_correlation", "Franka chunks require protocol-v1 metadata")
        mismatches = {
            name: (request.get(name), response.get(name))
            for name in CORRELATION_FIELDS
            if request.get(name) != response.get(name)
        }
        if mismatches:
            raise FrankaSafetyError("stale_correlation", f"response does not match request: {mismatches}")
        for name in ("request_id", "response_id", "client_session_id", "observation_id"):
            if not isinstance(response.get(name), str) or not response[name]:
                raise FrankaSafetyError("missing_correlation", f"metadata {name!r} is missing")
        epoch = response.get("connection_epoch")
        request_created = response.get("request_created_unix_ns")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise FrankaSafetyError("missing_correlation", "connection_epoch is invalid")
        if isinstance(request_created, bool) or not isinstance(request_created, int) or request_created <= 0:
            raise FrankaSafetyError("missing_correlation", "request_created_unix_ns is invalid")
        return cls(
            request_id=response["request_id"],
            response_id=response["response_id"],
            client_session_id=response["client_session_id"],
            connection_epoch=epoch,
            observation_id=response["observation_id"],
            request_created_unix_ns=request_created,
        )


@dataclass
class GuardedFrankaChunk:
    actions: np.ndarray
    correlation: Correlation
    expires_monotonic_ns: int
    next_action_index: int = 0

    @classmethod
    def create(
        cls,
        actions: object,
        *,
        snapshot: contract.FrankaSnapshot,
        config: contract.SafetyConfig,
        request_metadata: dict[str, Any] | None,
        response_metadata: dict[str, Any] | None,
        received_unix_ns: int | None = None,
        received_monotonic_ns: int | None = None,
    ) -> GuardedFrankaChunk:
        action_array = np.asarray(actions, dtype=np.float64)
        expected_shape = (config.action_horizon, contract.ACTION_DIM)
        if action_array.shape != expected_shape:
            raise FrankaSafetyError(
                "invalid_action_shape", f"expected Franka action chunk {expected_shape}, got {action_array.shape}"
            )
        if not np.isfinite(action_array).all():
            raise FrankaSafetyError("invalid_action_nonfinite", "Franka action chunk contains NaN or infinity")

        q_targets = action_array[:, :7]
        if np.any(q_targets < contract.Q_MIN_RAD) or np.any(q_targets > contract.Q_MAX_RAD):
            raise FrankaSafetyError("joint_position_limit", "Franka action chunk exceeds the application joint limits")
        gripper = action_array[:, 7]
        if np.any(gripper < 0.0) or np.any(gripper > config.max_gripper_width_m):
            raise FrankaSafetyError("gripper_width_limit", "Franka action chunk exceeds the gripper width range")

        q_sequence = np.concatenate([np.asarray(snapshot.q_rad, dtype=np.float64)[None], q_targets], axis=0)
        q_velocity = np.abs(np.diff(q_sequence, axis=0)) * config.control_hz
        allowed_q_velocity = contract.QDOT_RECT_MAX_RAD_S * config.speed_scale
        if np.any(q_velocity > allowed_q_velocity + 1e-9):
            joint_index = int(np.argwhere(q_velocity > allowed_q_velocity + 1e-9)[0, 1])
            raise FrankaSafetyError(
                "joint_velocity_limit",
                f"Franka action chunk exceeds the scaled rectangular velocity limit at joint {joint_index + 1}",
            )
        gripper_sequence = np.concatenate([[snapshot.gripper_width_m], gripper])
        if np.any(np.abs(np.diff(gripper_sequence)) * config.control_hz > config.max_gripper_speed_m_s + 1e-9):
            raise FrankaSafetyError("gripper_speed_limit", "Franka action chunk exceeds the gripper speed limit")

        correlation = Correlation.create(request_metadata, response_metadata)
        received_unix_ns = time.time_ns() if received_unix_ns is None else received_unix_ns
        received_monotonic_ns = time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
        response_age_ns = received_unix_ns - correlation.request_created_unix_ns
        if response_age_ns < 0 or response_age_ns > int(config.max_response_age_s * 1_000_000_000):
            raise FrankaSafetyError("stale_policy_response", "Franka policy response is stale or future-dated")
        return cls(
            actions=action_array.astype(np.float32),
            correlation=correlation,
            expires_monotonic_ns=received_monotonic_ns + int(config.action_chunk_ttl_s * 1_000_000_000),
        )

    @property
    def remaining_actions(self) -> int:
        return self.actions.shape[0] - self.next_action_index

    def target(self, *, connection_epoch: int, now_monotonic_ns: int | None = None) -> contract.FrankaTarget:
        now_monotonic_ns = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        if connection_epoch != self.correlation.connection_epoch:
            raise FrankaSafetyError("connection_epoch_changed", "action chunk belongs to an old connection epoch")
        if now_monotonic_ns > self.expires_monotonic_ns:
            raise FrankaSafetyError("action_chunk_expired", "Franka action chunk expired before execution")
        if self.remaining_actions < 1:
            raise FrankaSafetyError("action_queue_empty", "Franka action chunk is exhausted")
        action = self.actions[self.next_action_index]
        return contract.FrankaTarget(q_target_rad=action[:7].copy(), gripper_width_m=float(action[7]))


class FrankaActionQueue:
    """A single-chunk queue that permanently closes after a fault."""

    def __init__(self) -> None:
        self._chunk: GuardedFrankaChunk | None = None
        self._faulted = False
        self._post_fault_old_action_steps = 0

    @property
    def remaining_actions(self) -> int:
        return 0 if self._chunk is None else self._chunk.remaining_actions

    @property
    def post_fault_old_action_steps(self) -> int:
        return self._post_fault_old_action_steps

    def load(self, chunk: GuardedFrankaChunk) -> None:
        if self._faulted:
            raise FrankaSafetyError("action_queue_cancelled", "cannot load actions after a fault")
        if self.remaining_actions:
            raise FrankaSafetyError("action_queue_not_empty", "cannot replace an active Franka action chunk")
        self._chunk = chunk

    def peek(self, *, connection_epoch: int, now_monotonic_ns: int | None = None) -> contract.FrankaTarget:
        if self._faulted:
            raise FrankaSafetyError("action_queue_cancelled", "cannot execute actions after a fault")
        if self._chunk is None:
            raise FrankaSafetyError("action_queue_empty", "no Franka action chunk is loaded")
        return self._chunk.target(connection_epoch=connection_epoch, now_monotonic_ns=now_monotonic_ns)

    def mark_executed(self) -> None:
        if self._faulted:
            self._post_fault_old_action_steps += 1
            raise FrankaSafetyError("action_after_fault", "an old action was executed after a fault")
        if self._chunk is None or self._chunk.remaining_actions < 1:
            raise FrankaSafetyError("action_queue_empty", "cannot mark an empty queue action executed")
        self._chunk.next_action_index += 1

    def cancel(self) -> tuple[int, int]:
        queued_before = self.remaining_actions
        self._faulted = True
        if self._chunk is not None:
            self._chunk.next_action_index = self._chunk.actions.shape[0]
        return queued_before, self.remaining_actions

    def discard(self) -> None:
        if self._faulted:
            raise FrankaSafetyError("action_queue_cancelled", "cannot discard a faulted queue")
        self._chunk = None
