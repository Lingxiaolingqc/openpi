"""Local fake policy server for SO-101 S6 protocol and fault injection."""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import math
import random
import sys
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import websocket_policy_protocol as protocol
import websockets

try:
    from websockets.asyncio.server import serve
except ImportError:  # websockets 11-12 used by the Windows LeIsaac environment.
    from websockets.server import serve


READY_MARKER = "S6_FAKE_POLICY_SERVER_READY"
INJECTION_PREFIX = "S6_FAULT_INJECTION "


class FaultMode(enum.Enum):
    NORMAL = "normal"
    FIXED_DELAY = "fixed-delay"
    JITTER = "jitter"
    INFERENCE_TIMEOUT = "inference-timeout"
    DROP_RESPONSE = "drop-response"
    DISCONNECT = "disconnect"
    DISCONNECT_AFTER_RESPONSE = "disconnect-after-response"
    SERVER_EXIT = "server-exit"
    BAD_SHAPE = "bad-shape"
    NAN_ACTION = "nan-action"
    INF_ACTION = "inf-action"
    OUT_OF_RANGE_ACTION = "out-of-range-action"
    DUPLICATE_RESPONSE = "duplicate-response"
    STALE_RESPONSE = "stale-response"


def make_action_chunk(mode: FaultMode, *, horizon: int, action_dim: int, action_value: float) -> np.ndarray:
    """Create deterministic normal or invalid fake policy actions."""

    shape = (horizon, action_dim + 1) if mode is FaultMode.BAD_SHAPE else (horizon, action_dim)
    actions = np.full(shape, action_value, dtype=np.float32)
    if mode is FaultMode.NAN_ACTION:
        actions[0, 0] = np.nan
    elif mode is FaultMode.INF_ACTION:
        actions[0, 0] = np.inf
    elif mode is FaultMode.OUT_OF_RANGE_ACTION:
        actions.fill(999.0)
    return actions


class FakePolicyServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._mode = FaultMode(args.fault)
        self._rng = random.Random(args.seed)
        self._request_count = 0
        self._stop_event = asyncio.Event()
        self._packer = msgpack_numpy.Packer()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": "so101-s6-fake",
            "deployment_scope": "simulation-only",
            "real_robot_deployment_allowed": False,
            "supports_remote_reset": False,
            protocol.PROTOCOL_METADATA_KEY: [protocol.PROTOCOL_VERSION],
            "fault_mode": self._mode.value,
        }

    def _log_injection(self, request_metadata: dict[str, Any], **details: Any) -> None:
        record = {
            "fault_mode": self._mode.value,
            "request_count": self._request_count,
            "request_id": request_metadata["request_id"],
            "connection_epoch": request_metadata["connection_epoch"],
            **details,
        }
        print(INJECTION_PREFIX + json.dumps(record, allow_nan=False, sort_keys=True), flush=True)

    async def _send(self, websocket: Any, message: Any) -> None:
        await asyncio.wait_for(websocket.send(self._packer.pack(message)), timeout=self._args.send_timeout_s)

    async def handler(self, websocket: Any) -> None:
        await self._send(websocket, self.metadata)
        pending_duplicate_response: dict[str, Any] | None = None
        try:
            async for packed_request in websocket:
                self._request_count += 1
                request_message = msgpack_numpy.unpackb(packed_request)
                payload, request_metadata = protocol.unpack_request(request_message)
                if request_metadata is None:
                    await websocket.send("S6 fake server requires OpenPI protocol version 1")
                    await websocket.close(code=1002, reason="protocol v1 required")
                    return
                if pending_duplicate_response is not None:
                    duplicate_metadata = pending_duplicate_response[protocol.ENVELOPE_KEY]
                    self._log_injection(
                        request_metadata,
                        response_id=duplicate_metadata["response_id"],
                        duplicated_request_id=duplicate_metadata["request_id"],
                        injection_stage="sent_on_next_request",
                    )
                    await self._send(websocket, pending_duplicate_response)
                    pending_duplicate_response = None
                    continue
                persistent_faults = {
                    FaultMode.FIXED_DELAY,
                    FaultMode.JITTER,
                    FaultMode.INFERENCE_TIMEOUT,
                    FaultMode.DISCONNECT,
                }
                fault_index_matches = (
                    self._request_count >= self._args.fault_at_request
                    if self._mode in persistent_faults
                    else self._request_count == self._args.fault_at_request
                )
                inject = fault_index_matches and self._mode is not FaultMode.NORMAL
                if inject and self._mode is FaultMode.FIXED_DELAY:
                    self._log_injection(request_metadata, delay_s=self._args.delay_s)
                    await asyncio.sleep(self._args.delay_s)
                elif inject and self._mode is FaultMode.JITTER:
                    delay_s = self._rng.uniform(self._args.jitter_min_s, self._args.jitter_max_s)
                    self._log_injection(request_metadata, delay_s=delay_s)
                    await asyncio.sleep(delay_s)
                elif inject and self._mode is FaultMode.INFERENCE_TIMEOUT:
                    self._log_injection(request_metadata, stall_s=self._args.stall_s)
                    await asyncio.sleep(self._args.stall_s)
                    continue
                elif inject and self._mode is FaultMode.DROP_RESPONSE:
                    self._log_injection(request_metadata)
                    continue
                elif inject and self._mode is FaultMode.DISCONNECT:
                    self._log_injection(request_metadata)
                    await websocket.close(code=1011, reason="injected disconnect")
                    return
                elif inject and self._mode is FaultMode.SERVER_EXIT:
                    self._log_injection(request_metadata)
                    self._stop_event.set()
                    await websocket.close(code=1011, reason="injected server exit")
                    return

                action_mode = self._mode if inject else FaultMode.NORMAL
                actions = make_action_chunk(
                    action_mode,
                    horizon=self._args.action_horizon,
                    action_dim=self._args.action_dim,
                    action_value=self._args.action_value,
                )
                response = protocol.new_infer_response(
                    {
                        "actions": actions,
                        "fake_policy_diagnostics": {
                            "fault_mode": action_mode.value,
                            "request_count": self._request_count,
                            "request_keys": sorted(payload),
                        },
                    },
                    request_metadata,
                )
                if inject and self._mode is FaultMode.STALE_RESPONSE:
                    self._log_injection(request_metadata)
                    response[protocol.ENVELOPE_KEY]["request_id"] = f"stale-{request_metadata['request_id']}"
                await self._send(websocket, response)
                if inject and self._mode is FaultMode.DISCONNECT_AFTER_RESPONSE:
                    self._log_injection(
                        request_metadata,
                        response_id=response[protocol.ENVELOPE_KEY]["response_id"],
                        disconnect_after_response_s=self._args.disconnect_after_response_s,
                    )
                    await asyncio.sleep(self._args.disconnect_after_response_s)
                    await websocket.close(code=1011, reason="injected mid-chunk disconnect")
                    return
                if inject and self._mode is FaultMode.DUPLICATE_RESPONSE:
                    pending_duplicate_response = response
        except websockets.exceptions.ConnectionClosed:
            return

    async def run(self) -> None:
        async with serve(
            self.handler,
            self._args.host,
            self._args.port,
            compression=None,
            max_size=None,
        ):
            print(
                f"{READY_MARKER}: ws://{self._args.host}:{self._args.port} fault={self._mode.value}",
                flush=True,
            )
            await self._stop_event.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--fault", choices=[mode.value for mode in FaultMode], default=FaultMode.NORMAL.value)
    parser.add_argument("--fault-at-request", type=int, default=1)
    parser.add_argument("--delay-s", type=float, default=0.25)
    parser.add_argument("--jitter-min-s", type=float, default=0.05)
    parser.add_argument("--jitter-max-s", type=float, default=0.50)
    parser.add_argument("--stall-s", type=float, default=60.0)
    parser.add_argument("--disconnect-after-response-s", type=float, default=0.05)
    parser.add_argument("--send-timeout-s", type=float, default=5.0)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--action-value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.fault_at_request < 1 or args.action_horizon < 1 or args.action_dim < 1:
        raise ValueError("fault-at-request, action-horizon, and action-dim must be positive")
    numeric_values = (
        args.delay_s,
        args.jitter_min_s,
        args.jitter_max_s,
        args.stall_s,
        args.disconnect_after_response_s,
        args.send_timeout_s,
        args.action_value,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("delay, jitter, stall, send timeout, and action value must be finite")
    if args.delay_s < 0.0 or args.jitter_min_s < 0.0 or args.jitter_max_s < args.jitter_min_s:
        raise ValueError("delay and jitter bounds are invalid")
    if args.stall_s <= 0.0 or args.disconnect_after_response_s <= 0.0 or args.send_timeout_s <= 0.0:
        raise ValueError("stall-s, disconnect-after-response-s, and send-timeout-s must be positive")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        asyncio.run(FakePolicyServer(args).run())
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"S6_FAKE_POLICY_SERVER_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
