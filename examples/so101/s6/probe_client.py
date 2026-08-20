"""Exercise the S6 WebSocket client against the local fake policy server."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from openpi_client import websocket_client_policy
from openpi_client import websocket_policy_protocol as protocol

from examples.so101.s6 import safety_log

NO_FAULT = "none"
INVALID_ACTION_SHAPE = "invalid_action_shape"
INVALID_ACTION_NONFINITE = "invalid_action_nonfinite"
INVALID_ACTION_OUT_OF_RANGE = "invalid_action_out_of_range"
APPLICATION_FAULTS = {
    INVALID_ACTION_SHAPE,
    INVALID_ACTION_NONFINITE,
    INVALID_ACTION_OUT_OF_RANGE,
}
EXPECTED_FAULTS = {NO_FAULT, *APPLICATION_FAULTS, *(fault.value for fault in websocket_client_policy.PolicyFaultType)}


class ProbeFaultError(RuntimeError):
    def __init__(self, fault_type: str, message: str) -> None:
        super().__init__(message)
        self.fault_type = fault_type


def validate_action_chunk(
    value: object,
    *,
    expected_horizon: int,
    action_dim: int,
    max_abs_action: float | None,
) -> np.ndarray:
    """Reject invalid fake actions without clipping or repairing them."""

    try:
        actions = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ProbeFaultError(INVALID_ACTION_SHAPE, f"action chunk cannot be converted to float32: {exc}") from exc
    expected_shape = (expected_horizon, action_dim)
    if actions.shape != expected_shape:
        raise ProbeFaultError(INVALID_ACTION_SHAPE, f"expected action shape {expected_shape}, got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ProbeFaultError(INVALID_ACTION_NONFINITE, "action chunk contains NaN or infinity")
    if max_abs_action is not None and bool((np.abs(actions) > max_abs_action).any()):
        maximum = float(np.abs(actions).max())
        raise ProbeFaultError(
            INVALID_ACTION_OUT_OF_RANGE,
            f"maximum absolute action {maximum:.6f} exceeds {max_abs_action:.6f}",
        )
    return actions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--expected-horizon", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--max-abs-action", type=float)
    parser.add_argument("--connect-timeout-s", type=float, default=3.0)
    parser.add_argument("--connect-attempt-timeout-s", type=float, default=1.0)
    parser.add_argument("--metadata-timeout-s", type=float, default=1.0)
    parser.add_argument("--send-timeout-s", type=float, default=1.0)
    parser.add_argument("--inference-timeout-s", type=float, default=1.0)
    parser.add_argument("--heartbeat-timeout-s", type=float, default=0.5)
    parser.add_argument("--expect-fault", choices=sorted(EXPECTED_FAULTS), default=NO_FAULT)
    return parser


def _correlation(client: websocket_client_policy.WebsocketClientPolicy | None) -> dict:
    if client is None:
        return {}
    request_metadata = client.last_request_metadata or {}
    response_metadata = client.last_response_metadata or {}
    return {
        "request_id": request_metadata.get("request_id"),
        "response_id": response_metadata.get("response_id"),
        "observation_id": request_metadata.get("observation_id"),
        "connection_epoch": client.connection_epoch,
    }


def _print_event(event: str, *, client, state_from, state_to, **kwargs) -> None:
    print(
        safety_log.format_s6_event(
            safety_log.build_s6_event(
                event,
                state_from=state_from,
                state_to=state_to,
                **_correlation(client),
                **kwargs,
            )
        ),
        flush=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.requests < 1 or args.expected_horizon < 1 or args.action_dim < 1:
        raise ValueError("requests, expected-horizon, and action-dim must be positive")
    if args.max_abs_action is not None and (not math.isfinite(args.max_abs_action) or args.max_abs_action <= 0.0):
        raise ValueError("max-abs-action must be finite and positive")

    client = None
    observed_fault = NO_FAULT
    operation_started_ns = time.monotonic_ns()
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            protocol_version=protocol.PROTOCOL_VERSION,
            connect_timeout_s=args.connect_timeout_s,
            connect_attempt_timeout_s=args.connect_attempt_timeout_s,
            metadata_timeout_s=args.metadata_timeout_s,
            send_timeout_s=args.send_timeout_s,
            inference_timeout_s=args.inference_timeout_s,
            heartbeat_timeout_s=args.heartbeat_timeout_s,
        )
        for request_index in range(args.requests):
            operation_started_ns = time.monotonic_ns()
            response = client.infer(
                {
                    "state": np.zeros(args.action_dim, dtype=np.float32),
                    "probe_request_index": request_index,
                }
            )
            if "actions" not in response:
                raise ProbeFaultError(INVALID_ACTION_SHAPE, "response does not contain actions")
            actions = validate_action_chunk(
                response["actions"],
                expected_horizon=args.expected_horizon,
                action_dim=args.action_dim,
                max_abs_action=args.max_abs_action,
            )
            latency_ms = (time.monotonic_ns() - operation_started_ns) / 1_000_000.0
            _print_event(
                "inference_accepted",
                client=client,
                state_from=safety_log.SafeState.NORMAL,
                state_to=safety_log.SafeState.NORMAL,
                detection_latency_ms=latency_ms,
                details={
                    "request_index": request_index,
                    "action_shape": list(actions.shape),
                    "action_min": float(actions.min()),
                    "action_max": float(actions.max()),
                },
            )
        operation_started_ns = time.monotonic_ns()
        client.health_check()
    except websocket_client_policy.PolicyClientError as exc:
        observed_fault = exc.fault_type.value
        detection_latency_ms = (time.monotonic_ns() - operation_started_ns) / 1_000_000.0
        _print_event(
            "policy_client_fault",
            client=client,
            state_from=safety_log.SafeState.NORMAL,
            state_to=safety_log.SafeState.FAULT_DETECTED,
            fault_type=observed_fault,
            detection_latency_ms=detection_latency_ms,
            safe_action="no_action_emitted",
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )
    except ProbeFaultError as exc:
        observed_fault = exc.fault_type
        detection_latency_ms = (time.monotonic_ns() - operation_started_ns) / 1_000_000.0
        _print_event(
            "policy_action_rejected",
            client=client,
            state_from=safety_log.SafeState.NORMAL,
            state_to=safety_log.SafeState.RESPONSE_REJECTED,
            fault_type=observed_fault,
            detection_latency_ms=detection_latency_ms,
            safe_action="no_action_emitted",
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )
    finally:
        if client is not None:
            client.close()

    if observed_fault != args.expect_fault:
        print(
            f"S6_PROBE_FAILED: expected_fault={args.expect_fault} observed_fault={observed_fault}",
            flush=True,
        )
        return 1
    print(f"S6_PROBE_OK: expected_fault={args.expect_fault} observed_fault={observed_fault}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
