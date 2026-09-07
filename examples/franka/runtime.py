"""20 Hz task-agnostic Franka policy runtime with fail-closed execution."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Protocol

from examples.franka import action_safety
from examples.franka import backend as backend_module
from examples.franka import contract

logger = logging.getLogger(__name__)


class PolicyClient(Protocol):
    @property
    def connection_epoch(self) -> int: ...

    @property
    def last_request_metadata(self) -> dict[str, Any] | None: ...

    @property
    def last_response_metadata(self) -> dict[str, Any] | None: ...

    def get_server_metadata(self) -> dict[str, Any]: ...

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...


class Recorder(Protocol):
    def append(
        self,
        snapshot: contract.FrankaSnapshot,
        target: contract.FrankaTarget,
    ) -> None: ...


class FrankaRuntimeError(RuntimeError):
    def __init__(self, fault_type: str, message: str, *, hold_error: str | None = None) -> None:
        super().__init__(message)
        self.fault_type = fault_type
        self.hold_error = hold_error


class FrankaRuntime:
    def __init__(
        self,
        *,
        backend: backend_module.FrankaBackend,
        mode: contract.RuntimeMode = contract.RuntimeMode.OBSERVE,
        policy: PolicyClient | None = None,
        safety_config: contract.SafetyConfig | None = None,
        approval_path: Path | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        self._backend = backend
        self._mode = mode
        self._policy = policy
        self._config = safety_config or contract.SafetyConfig()
        self._recorder = recorder
        self._queue = action_safety.FrankaActionQueue()
        self._fault: FrankaRuntimeError | None = None
        self._server_metadata: dict[str, Any] | None = None
        self._closed = False

        if mode is not contract.RuntimeMode.OBSERVE:
            if policy is None:
                raise contract.ContractError(f"{mode.value} mode requires a policy client")
            self._server_metadata = policy.get_server_metadata()
            contract.validate_server_metadata(self._server_metadata)
        if mode is contract.RuntimeMode.EXECUTE:
            capabilities = backend.capabilities
            if capabilities.control_update_hz != 1000:
                raise contract.ContractError("execute mode requires a 1 kHz local controller")
            if not capabilities.position_dependent_velocity_limits or not capabilities.local_watchdog:
                raise contract.ContractError("execute mode requires dynamic limits and a local watchdog")
            if capabilities.real_hardware:
                if (
                    self._server_metadata is None
                    or self._server_metadata.get("real_robot_deployment_allowed") is not True
                ):
                    raise contract.ContractError("policy server metadata forbids real-robot deployment")
                if approval_path is None:
                    raise contract.ContractError("real FR3 execute mode requires an approval JSON")
                contract.validate_real_robot_approval(
                    approval_path,
                    self._server_metadata,
                    requested_speed_scale=self._config.speed_scale,
                )

    @property
    def mode(self) -> contract.RuntimeMode:
        return self._mode

    @property
    def fault(self) -> FrankaRuntimeError | None:
        return self._fault

    @property
    def remaining_actions(self) -> int:
        return self._queue.remaining_actions

    @property
    def post_fault_old_action_steps(self) -> int:
        return self._queue.post_fault_old_action_steps

    def _trip(self, exc: BaseException) -> FrankaRuntimeError:
        self._queue.cancel()
        hold_error = None
        if self._mode is contract.RuntimeMode.EXECUTE:
            try:
                self._backend.hold_position()
            except Exception as hold_exc:  # The original fault remains primary, but the hold failure is never hidden.
                hold_error = f"{type(hold_exc).__name__}: {hold_exc}"
        fault_type = getattr(exc, "fault_type", type(exc).__name__)
        fault = FrankaRuntimeError(str(fault_type), str(exc), hold_error=hold_error)
        self._fault = fault
        return fault

    def _request_chunk(self, snapshot: contract.FrankaSnapshot, prompt: str) -> action_safety.GuardedFrankaChunk:
        assert self._policy is not None
        response = self._policy.infer(snapshot.policy_observation(prompt))
        if not isinstance(response, dict) or "actions" not in response:
            raise action_safety.FrankaSafetyError("invalid_policy_response", "policy response is missing actions")
        return action_safety.GuardedFrankaChunk.create(
            response["actions"],
            snapshot=snapshot,
            config=self._config,
            request_metadata=self._policy.last_request_metadata,
            response_metadata=self._policy.last_response_metadata,
        )

    def step(self, prompt: str = "Perform the instructed manipulation task.") -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Franka runtime is closed")
        if self._fault is not None:
            raise self._fault
        try:
            snapshot = self._backend.read_snapshot()
            contract.validate_snapshot(snapshot, self._config)
            if self._mode is contract.RuntimeMode.OBSERVE:
                return {"mode": self._mode.value, "observation": snapshot.policy_observation(prompt)}

            if self._mode is contract.RuntimeMode.SHADOW:
                chunk = self._request_chunk(snapshot, prompt)
                return {
                    "mode": self._mode.value,
                    "validated_actions": chunk.actions,
                    "action_chunk_id": chunk.correlation.response_id,
                }

            assert self._policy is not None
            if self._queue.remaining_actions == 0:
                self._queue.load(self._request_chunk(snapshot, prompt))
            target = self._queue.peek(connection_epoch=self._policy.connection_epoch)
            contract.validate_target(target, self._config)
            if self._recorder is not None:
                self._recorder.append(snapshot, target)
            self._backend.apply_target(target)
            self._queue.mark_executed()
            return {
                "mode": self._mode.value,
                "q_target_rad": target.q_target_rad.copy(),
                "gripper_width_m": target.gripper_width_m,
                "remaining_actions": self._queue.remaining_actions,
            }
        except FrankaRuntimeError:
            raise
        except Exception as exc:
            raise self._trip(exc) from exc

    def run(self, *, prompt: str, steps: int) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        period_s = 1.0 / self._config.control_hz
        for step_index in range(steps):
            started = time.monotonic()
            result = self.step(prompt)
            logger.info("franka_step=%d result=%s", step_index, result["mode"])
            remaining = period_s - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)

    def recover(self) -> None:
        if self._closed:
            raise RuntimeError("cannot recover a closed Franka runtime")
        if self._fault is None:
            raise RuntimeError("runtime is not faulted")
        self._backend.recover()
        self._queue = action_safety.FrankaActionQueue()
        self._fault = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.cancel()
        close_error: BaseException | None = None
        if self._mode is contract.RuntimeMode.EXECUTE and self._fault is None:
            try:
                self._backend.hold_position()
            except Exception as exc:
                close_error = exc
        try:
            policy_close = getattr(self._policy, "close", None)
            if callable(policy_close):
                policy_close()
        except Exception as exc:
            close_error = close_error or exc
        try:
            self._backend.close()
        except Exception as exc:
            close_error = close_error or exc
        if close_error is not None:
            raise RuntimeError(f"failed to close Franka runtime safely: {close_error}") from close_error
