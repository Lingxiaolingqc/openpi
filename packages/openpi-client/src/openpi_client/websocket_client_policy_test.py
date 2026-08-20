from __future__ import annotations

import threading
from typing import Callable

import pytest

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as client_module
from openpi_client import websocket_policy_protocol as protocol


class _FakeSocket:
    def __init__(self, release: threading.Event | None = None) -> None:
        self.closed = False
        self._release = release

    def shutdown(self, how: int) -> None:
        del how
        self.closed = True
        if self._release is not None:
            self._release.set()

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(
        self,
        *,
        protocol_enabled: bool = True,
        response_builder: Callable[[dict], dict] | None = None,
        metadata_timeout: bool = False,
        inference_timeout: bool = False,
        block_send: bool = False,
        heartbeat_ack: bool = True,
    ) -> None:
        versions = [protocol.PROTOCOL_VERSION] if protocol_enabled else []
        self._responses = [msgpack_numpy.packb({protocol.PROTOCOL_METADATA_KEY: versions})]
        self._response_builder = response_builder
        self._metadata_timeout = metadata_timeout
        self._inference_timeout = inference_timeout
        self._send_release = threading.Event()
        self._block_send = block_send
        self._heartbeat_ack = heartbeat_ack
        self.socket = _FakeSocket(self._send_release)
        self.closed = False
        self.sent_messages: list[dict] = []

    def send(self, packed_message: bytes) -> None:
        if self._block_send:
            self._send_release.wait(1.0)
            return
        message = msgpack_numpy.unpackb(packed_message)
        self.sent_messages.append(message)
        if self._response_builder is not None:
            self._responses.append(msgpack_numpy.packb(self._response_builder(message)))

    def recv(self, timeout: float | None = None) -> bytes:
        del timeout
        if self._metadata_timeout and self._responses:
            self._responses.clear()
            raise TimeoutError("metadata timeout")
        if self._responses:
            return self._responses.pop(0)
        if self._inference_timeout:
            raise TimeoutError("inference timeout")
        raise AssertionError("fake connection has no queued response")

    def ping(self) -> threading.Event:
        pong = threading.Event()
        if self._heartbeat_ack:
            pong.set()
        return pong

    def close(self) -> None:
        self.closed = True


def _response_builder(*, response_id: str = "response-1", stale: bool = False) -> Callable[[dict], dict]:
    def build(request: dict) -> dict:
        request_metadata = dict(request[protocol.ENVELOPE_KEY])
        response = protocol.new_infer_response(
            {"actions": [[0.0] * 6]},
            request_metadata,
            response_id=response_id,
        )
        if stale:
            response[protocol.ENVELOPE_KEY]["request_id"] = "old-request"
        return response

    return build


def _make_client(
    monkeypatch: pytest.MonkeyPatch, connection: _FakeConnection, **kwargs
) -> client_module.WebsocketClientPolicy:
    monkeypatch.setattr(client_module.websockets.sync.client, "connect", lambda *args, **options: connection)
    defaults = {
        "connect_timeout_s": 0.2,
        "connect_attempt_timeout_s": 0.1,
        "metadata_timeout_s": 0.05,
        "send_timeout_s": 0.05,
        "inference_timeout_s": 0.05,
        "heartbeat_timeout_s": 0.01,
        "close_timeout_s": 0.05,
        "retry_interval_s": 0.01,
        "protocol_version": protocol.PROTOCOL_VERSION,
    }
    defaults.update(kwargs)
    return client_module.WebsocketClientPolicy(host="fake", port=1234, **defaults)


def test_protocol_client_accepts_correlated_response(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(response_builder=_response_builder())
    client = _make_client(monkeypatch, connection)

    response = client.infer({"state": [1.0]})

    assert response == {"actions": [[0.0] * 6]}
    assert client.connection_state is client_module.PolicyClientState.READY
    assert client.last_request_metadata["request_id"] == client.last_response_metadata["request_id"]
    client.close()


def test_legacy_client_request_and_response_remain_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(
        protocol_enabled=False,
        response_builder=lambda request: {"actions": [[request["state"][0]] * 6]},
    )
    monkeypatch.setattr(client_module.websockets.sync.client, "connect", lambda *args, **options: connection)
    client = client_module.WebsocketClientPolicy(
        host="fake",
        connect_timeout_s=0.2,
        metadata_timeout_s=0.05,
        send_timeout_s=0.05,
        inference_timeout_s=0.05,
    )

    assert client.infer({"state": [2.0]}) == {"actions": [[2.0] * 6]}
    assert client.last_request_metadata is None
    assert client.connection_state is client_module.PolicyClientState.READY
    client.close()


def test_metadata_timeout_is_typed_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(metadata_timeout=True)
    monkeypatch.setattr(client_module.websockets.sync.client, "connect", lambda *args, **options: connection)

    with pytest.raises(client_module.PolicyMetadataTimeoutError):
        client_module.WebsocketClientPolicy(
            host="fake",
            connect_timeout_s=0.2,
            metadata_timeout_s=0.01,
        )
    assert connection.socket.closed


def test_invalid_protocol_version_metadata_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection()
    connection._responses[0] = msgpack_numpy.packb({protocol.PROTOCOL_METADATA_KEY: "1"})  # noqa: SLF001
    monkeypatch.setattr(client_module.websockets.sync.client, "connect", lambda *args, **options: connection)

    with pytest.raises(client_module.PolicyInvalidResponseError, match="integer list"):
        client_module.WebsocketClientPolicy(host="fake", protocol_version=protocol.PROTOCOL_VERSION)
    assert connection.socket.closed


def test_inference_timeout_faults_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(inference_timeout=True)
    client = _make_client(monkeypatch, connection)

    with pytest.raises(client_module.PolicyInferenceTimeoutError):
        client.infer({"state": [1.0]})

    assert client.connection_state is client_module.PolicyClientState.FAULTED
    assert client.last_fault.fault_type is client_module.PolicyFaultType.INFERENCE_TIMEOUT
    assert connection.socket.closed
    with pytest.raises(client_module.PolicyClientStateError, match="not ready"):
        client.ensure_ready()


def test_send_timeout_faults_connection_and_releases_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(block_send=True)
    client = _make_client(monkeypatch, connection, send_timeout_s=0.01)

    with pytest.raises(client_module.PolicySendTimeoutError):
        client.infer({"state": [1.0]})

    assert client.connection_state is client_module.PolicyClientState.FAULTED
    assert connection.socket.closed


def test_stale_response_is_rejected_and_faults_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(response_builder=_response_builder(stale=True))
    client = _make_client(monkeypatch, connection)

    with pytest.raises(client_module.PolicyStaleResponseError):
        client.infer({"state": [1.0]})

    assert client.connection_state is client_module.PolicyClientState.FAULTED


def test_duplicate_response_id_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(response_builder=_response_builder(response_id="duplicate-id"))
    client = _make_client(monkeypatch, connection)
    assert client.infer({"state": [1.0]})["actions"]

    with pytest.raises(client_module.PolicyDuplicateResponseError):
        client.infer({"state": [2.0]})

    assert client.connection_state is client_module.PolicyClientState.FAULTED


def test_heartbeat_timeout_faults_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _FakeConnection(heartbeat_ack=False)
    client = _make_client(monkeypatch, connection)

    with pytest.raises(client_module.PolicyHeartbeatTimeoutError):
        client.health_check()

    assert client.connection_state is client_module.PolicyClientState.FAULTED


def test_explicit_reconnect_advances_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    connections = [
        _FakeConnection(response_builder=_response_builder(response_id="first")),
        _FakeConnection(response_builder=_response_builder(response_id="second")),
    ]
    monkeypatch.setattr(
        client_module.websockets.sync.client,
        "connect",
        lambda *args, **options: connections.pop(0),
    )
    client = client_module.WebsocketClientPolicy(
        host="fake",
        protocol_version=protocol.PROTOCOL_VERSION,
        retry_interval_s=0.01,
    )
    first_session = client.client_session_id

    client.reconnect()

    assert client.connection_epoch == 1
    assert client.client_session_id == first_session
    assert client.connection_state is client_module.PolicyClientState.READY


def test_connect_retry_has_finite_total_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_module.websockets.sync.client,
        "connect",
        lambda *args, **options: (_ for _ in ()).throw(ConnectionRefusedError("not ready")),
    )
    with pytest.raises(client_module.PolicyConnectionTimeoutError):
        client_module.WebsocketClientPolicy(
            host="fake",
            connect_timeout_s=0.03,
            connect_attempt_timeout_s=0.01,
            retry_interval_s=0.005,
        )
