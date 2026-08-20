"""Versioned request and response envelopes for OpenPI WebSocket policies."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple
import uuid


PROTOCOL_VERSION = 1
PROTOCOL_METADATA_KEY = "openpi_protocol_versions"
ENVELOPE_KEY = "__openpi_protocol__"
PAYLOAD_KEY = "payload"

INFER_REQUEST = "infer_request"
INFER_RESPONSE = "infer_response"
ERROR_RESPONSE = "error_response"


class ProtocolError(ValueError):
    """Raised when a WebSocket policy envelope violates the wire contract."""


def _require_mapping(value: Any, name: str) -> Dict:
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} must be a mapping")
    return value


def _require_nonempty_string(metadata: Dict, name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"protocol metadata {name!r} must be a non-empty string")
    return value


def _require_nonnegative_integer(metadata: Dict, name: str) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"protocol metadata {name!r} must be a non-negative integer")
    return value


def _require_positive_integer(metadata: Dict, name: str) -> int:
    value = _require_nonnegative_integer(metadata, name)
    if value == 0:
        raise ProtocolError(f"protocol metadata {name!r} must be positive")
    return value


def _validate_common_metadata(metadata: Dict, expected_message_type: str) -> None:
    protocol_version = _require_nonnegative_integer(metadata, "protocol_version")
    if protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {metadata.get('protocol_version')!r}; expected {PROTOCOL_VERSION}"
        )
    if metadata.get("message_type") != expected_message_type:
        raise ProtocolError(
            f"unexpected protocol message type {metadata.get('message_type')!r}; expected {expected_message_type!r}"
        )
    _require_nonempty_string(metadata, "request_id")
    _require_nonempty_string(metadata, "client_session_id")
    _require_nonnegative_integer(metadata, "connection_epoch")


def new_infer_request(
    payload: Dict,
    *,
    client_session_id: str,
    connection_epoch: int,
    request_id: Optional[str] = None,
    observation_id: Optional[str] = None,
    request_created_unix_ns: Optional[int] = None,
) -> Dict:
    """Wrap an observation in a versioned inference request."""

    _require_mapping(payload, PAYLOAD_KEY)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": INFER_REQUEST,
        "request_id": str(uuid.uuid4()) if request_id is None else request_id,
        "client_session_id": client_session_id,
        "connection_epoch": connection_epoch,
        "observation_id": str(uuid.uuid4()) if observation_id is None else observation_id,
        "request_created_unix_ns": time.time_ns() if request_created_unix_ns is None else request_created_unix_ns,
    }
    validate_request_metadata(metadata)
    return {ENVELOPE_KEY: metadata, PAYLOAD_KEY: payload}


def validate_request_metadata(metadata: Dict) -> None:
    """Validate inference request metadata without inspecting its payload."""

    metadata = _require_mapping(metadata, ENVELOPE_KEY)
    _validate_common_metadata(metadata, INFER_REQUEST)
    _require_nonempty_string(metadata, "observation_id")
    _require_positive_integer(metadata, "request_created_unix_ns")


def unpack_request(message: Dict) -> Tuple[Dict, Optional[Dict]]:
    """Return the policy payload and request metadata, accepting legacy messages."""

    message = _require_mapping(message, "request")
    if ENVELOPE_KEY not in message:
        return message, None
    metadata = _require_mapping(message.get(ENVELOPE_KEY), ENVELOPE_KEY)
    validate_request_metadata(metadata)
    payload = _require_mapping(message.get(PAYLOAD_KEY), PAYLOAD_KEY)
    return payload, dict(metadata)


def new_infer_response(
    payload: Dict,
    request_metadata: Dict,
    *,
    response_id: Optional[str] = None,
    server_received_unix_ns: Optional[int] = None,
    server_completed_unix_ns: Optional[int] = None,
) -> Dict:
    """Wrap policy output while preserving request correlation fields."""

    _require_mapping(payload, PAYLOAD_KEY)
    validate_request_metadata(request_metadata)
    received_ns = time.time_ns() if server_received_unix_ns is None else server_received_unix_ns
    completed_ns = time.time_ns() if server_completed_unix_ns is None else server_completed_unix_ns
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": INFER_RESPONSE,
        "request_id": request_metadata["request_id"],
        "response_id": str(uuid.uuid4()) if response_id is None else response_id,
        "client_session_id": request_metadata["client_session_id"],
        "connection_epoch": request_metadata["connection_epoch"],
        "observation_id": request_metadata["observation_id"],
        "request_created_unix_ns": request_metadata["request_created_unix_ns"],
        "server_received_unix_ns": received_ns,
        "server_completed_unix_ns": completed_ns,
    }
    validate_response_metadata(metadata, expected_message_type=INFER_RESPONSE)
    return {ENVELOPE_KEY: metadata, PAYLOAD_KEY: payload}


def new_error_response(
    request_metadata: Dict,
    *,
    error_type: str,
    error_message: str,
    response_id: Optional[str] = None,
    server_received_unix_ns: Optional[int] = None,
    server_completed_unix_ns: Optional[int] = None,
) -> Dict:
    """Build a correlated error response without exposing a server traceback."""

    validate_request_metadata(request_metadata)
    if not isinstance(error_type, str) or not isinstance(error_message, str) or not error_type or not error_message:
        raise ProtocolError("error_type and error_message must be non-empty strings")
    received_ns = time.time_ns() if server_received_unix_ns is None else server_received_unix_ns
    completed_ns = time.time_ns() if server_completed_unix_ns is None else server_completed_unix_ns
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": ERROR_RESPONSE,
        "request_id": request_metadata["request_id"],
        "response_id": str(uuid.uuid4()) if response_id is None else response_id,
        "client_session_id": request_metadata["client_session_id"],
        "connection_epoch": request_metadata["connection_epoch"],
        "observation_id": request_metadata["observation_id"],
        "request_created_unix_ns": request_metadata["request_created_unix_ns"],
        "server_received_unix_ns": received_ns,
        "server_completed_unix_ns": completed_ns,
    }
    validate_response_metadata(metadata, expected_message_type=ERROR_RESPONSE)
    return {
        ENVELOPE_KEY: metadata,
        PAYLOAD_KEY: {
            "error_type": error_type,
            "error_message": error_message,
        },
    }


def validate_response_metadata(metadata: Dict, *, expected_message_type: Optional[str] = None) -> None:
    """Validate response metadata and its server-side timestamp ordering."""

    metadata = _require_mapping(metadata, ENVELOPE_KEY)
    message_type = metadata.get("message_type")
    if expected_message_type is None:
        if message_type not in {INFER_RESPONSE, ERROR_RESPONSE}:
            raise ProtocolError(f"unexpected protocol response message type {message_type!r}")
        expected_message_type = message_type
    _validate_common_metadata(metadata, expected_message_type)
    _require_nonempty_string(metadata, "response_id")
    _require_nonempty_string(metadata, "observation_id")
    _require_positive_integer(metadata, "request_created_unix_ns")
    received_ns = _require_positive_integer(metadata, "server_received_unix_ns")
    completed_ns = _require_positive_integer(metadata, "server_completed_unix_ns")
    if completed_ns < received_ns:
        raise ProtocolError("server_completed_unix_ns precedes server_received_unix_ns")


def unpack_response(message: Dict) -> Tuple[Dict, Dict]:
    """Validate and return a protocol response payload and metadata."""

    message = _require_mapping(message, "response")
    metadata = _require_mapping(message.get(ENVELOPE_KEY), ENVELOPE_KEY)
    validate_response_metadata(metadata)
    payload = _require_mapping(message.get(PAYLOAD_KEY), PAYLOAD_KEY)
    return payload, dict(metadata)
