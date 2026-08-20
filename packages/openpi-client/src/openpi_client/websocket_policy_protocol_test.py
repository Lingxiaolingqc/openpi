from __future__ import annotations

import pytest

from openpi_client import websocket_policy_protocol as protocol


def _request() -> dict:
    return protocol.new_infer_request(
        {"state": [1, 2, 3]},
        client_session_id="session-1",
        connection_epoch=2,
        request_id="request-1",
        observation_id="observation-1",
        request_created_unix_ns=100,
    )


def test_protocol_round_trip_preserves_correlation() -> None:
    request = _request()
    payload, request_metadata = protocol.unpack_request(request)
    assert request_metadata is not None
    response = protocol.new_infer_response(
        {"actions": [[0.0] * 6]},
        request_metadata,
        response_id="response-1",
        server_received_unix_ns=200,
        server_completed_unix_ns=300,
    )

    response_payload, response_metadata = protocol.unpack_response(response)

    assert payload == {"state": [1, 2, 3]}
    assert response_payload == {"actions": [[0.0] * 6]}
    for field in (
        "request_id",
        "client_session_id",
        "connection_epoch",
        "observation_id",
        "request_created_unix_ns",
    ):
        assert response_metadata[field] == request_metadata[field]
    assert response_metadata["response_id"] == "response-1"


def test_unpack_request_accepts_legacy_payload() -> None:
    payload = {"state": [1.0]}
    assert protocol.unpack_request(payload) == (payload, None)


def test_protocol_rejects_reversed_server_timestamps() -> None:
    request_metadata = _request()[protocol.ENVELOPE_KEY]
    with pytest.raises(protocol.ProtocolError, match="precedes"):
        protocol.new_infer_response(
            {"actions": []},
            request_metadata,
            server_received_unix_ns=300,
            server_completed_unix_ns=200,
        )


def test_protocol_does_not_replace_explicit_invalid_timestamp() -> None:
    with pytest.raises(protocol.ProtocolError, match="must be positive"):
        protocol.new_infer_request(
            {},
            client_session_id="session",
            connection_epoch=0,
            request_created_unix_ns=0,
        )


def test_protocol_rejects_boolean_version() -> None:
    request = _request()
    request[protocol.ENVELOPE_KEY]["protocol_version"] = True
    with pytest.raises(protocol.ProtocolError, match="non-negative integer"):
        protocol.unpack_request(request)


def test_protocol_error_response_does_not_require_traceback() -> None:
    request_metadata = _request()[protocol.ENVELOPE_KEY]
    response = protocol.new_error_response(
        request_metadata,
        error_type="ValueError",
        error_message="bad request",
        response_id="response-error",
        server_received_unix_ns=200,
        server_completed_unix_ns=201,
    )

    payload, metadata = protocol.unpack_response(response)

    assert metadata["message_type"] == protocol.ERROR_RESPONSE
    assert payload == {"error_type": "ValueError", "error_message": "bad request"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", ""),
        ("connection_epoch", -1),
        ("request_created_unix_ns", 0),
    ],
)
def test_protocol_rejects_invalid_request_metadata(field: str, value: object) -> None:
    request = _request()
    request[protocol.ENVELOPE_KEY][field] = value
    with pytest.raises(protocol.ProtocolError):
        protocol.unpack_request(request)
