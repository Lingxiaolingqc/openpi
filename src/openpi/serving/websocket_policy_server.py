import asyncio
import contextlib
import http
import logging
import math
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from openpi_client import websocket_policy_protocol as _protocol
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        send_timeout_s: float = 5.0,
    ) -> None:
        if not math.isfinite(send_timeout_s) or send_timeout_s <= 0.0:
            raise ValueError("send_timeout_s must be finite and positive")
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        self._metadata[_protocol.PROTOCOL_METADATA_KEY] = [_protocol.PROTOCOL_VERSION]
        self._send_timeout_s = send_timeout_s
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await asyncio.wait_for(websocket.send(packer.pack(self._metadata)), timeout=self._send_timeout_s)

        prev_total_time = None
        while True:
            request_metadata = None
            server_received_unix_ns = None
            try:
                start_time = time.monotonic()
                packed_request = await websocket.recv()
                server_received_unix_ns = time.time_ns()
                request_message = msgpack_numpy.unpackb(packed_request)
                obs, request_metadata = _protocol.unpack_request(request_message)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                if not isinstance(action, dict):
                    raise TypeError(f"policy output must be a mapping, got {type(action).__name__}")

                response_payload = dict(action)
                response_payload["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    response_payload["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                if request_metadata is None:
                    response_message = response_payload
                else:
                    response_message = _protocol.new_infer_response(
                        response_payload,
                        request_metadata,
                        server_received_unix_ns=server_received_unix_ns,
                        server_completed_unix_ns=time.time_ns(),
                    )

                await asyncio.wait_for(
                    websocket.send(packer.pack(response_message)),
                    timeout=self._send_timeout_s,
                )
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as exc:
                if request_metadata is None:
                    error_message = traceback.format_exc()
                else:
                    error_message = packer.pack(
                        _protocol.new_error_response(
                            request_metadata,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            server_received_unix_ns=server_received_unix_ns,
                            server_completed_unix_ns=time.time_ns(),
                        )
                    )
                with contextlib.suppress(TimeoutError, websockets.ConnectionClosed):
                    await asyncio.wait_for(websocket.send(error_message), timeout=self._send_timeout_s)
                with contextlib.suppress(TimeoutError, websockets.ConnectionClosed):
                    await asyncio.wait_for(
                        websocket.close(
                            code=websockets.frames.CloseCode.INTERNAL_ERROR,
                            reason="Internal server error. Error details included in previous frame.",
                        ),
                        timeout=self._send_timeout_s,
                    )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
