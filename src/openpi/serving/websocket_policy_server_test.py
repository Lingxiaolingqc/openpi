from __future__ import annotations

import asyncio

from openpi_client import msgpack_numpy
from openpi_client import websocket_policy_protocol as protocol
import pytest

from openpi.serving import websocket_policy_server


class _FakeWebsocket:
    remote_address = ("127.0.0.1", 12345)

    def __init__(self, messages: list[bytes]) -> None:
        self._messages = messages
        self.sent: list[bytes | str] = []
        self.closed = False

    async def recv(self) -> bytes:
        if self._messages:
            return self._messages.pop(0)
        raise asyncio.CancelledError

    async def send(self, message: bytes | str) -> None:
        self.sent.append(message)

    async def close(self, *, code: int, reason: str) -> None:
        del code, reason
        self.closed = True


class _Policy:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[dict] = []

    def infer(self, request: dict) -> dict:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return {"actions": [[0.0] * 6]}


def _request() -> dict:
    return protocol.new_infer_request(
        {"state": [1.0]},
        client_session_id="session",
        connection_epoch=4,
        request_id="request",
        observation_id="observation",
        request_created_unix_ns=100,
    )


def test_server_advertises_and_correlates_protocol_v1() -> None:
    policy = _Policy()
    websocket = _FakeWebsocket([msgpack_numpy.packb(_request())])
    server = websocket_policy_server.WebsocketPolicyServer(policy, metadata={"model": "fake"})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server._handler(websocket))  # noqa: SLF001

    metadata = msgpack_numpy.unpackb(websocket.sent[0])
    response_payload, response_metadata = protocol.unpack_response(msgpack_numpy.unpackb(websocket.sent[1]))
    assert metadata[protocol.PROTOCOL_METADATA_KEY] == [protocol.PROTOCOL_VERSION]
    assert policy.requests == [{"state": [1.0]}]
    assert response_payload["actions"] == [[0.0] * 6]
    assert response_metadata["request_id"] == "request"
    assert response_metadata["connection_epoch"] == 4


def test_server_returns_correlated_error_without_traceback() -> None:
    policy = _Policy(error=ValueError("injected invalid action"))
    websocket = _FakeWebsocket([msgpack_numpy.packb(_request())])
    server = websocket_policy_server.WebsocketPolicyServer(policy)

    with pytest.raises(ValueError, match="injected invalid action"):
        asyncio.run(server._handler(websocket))  # noqa: SLF001

    payload, metadata = protocol.unpack_response(msgpack_numpy.unpackb(websocket.sent[1]))
    assert metadata["message_type"] == protocol.ERROR_RESPONSE
    assert payload == {"error_type": "ValueError", "error_message": "injected invalid action"}
    assert "Traceback" not in payload["error_message"]
    assert websocket.closed


def test_server_preserves_legacy_payload_and_response_shape() -> None:
    policy = _Policy()
    websocket = _FakeWebsocket([msgpack_numpy.packb({"state": [2.0]})])
    server = websocket_policy_server.WebsocketPolicyServer(policy)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server._handler(websocket))  # noqa: SLF001

    response = msgpack_numpy.unpackb(websocket.sent[1])
    assert protocol.ENVELOPE_KEY not in response
    assert response["actions"] == [[0.0] * 6]
    assert "server_timing" in response
    assert policy.requests == [{"state": [2.0]}]
