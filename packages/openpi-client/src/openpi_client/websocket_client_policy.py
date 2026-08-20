from __future__ import annotations

import enum
import logging
import math
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import uuid

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from openpi_client import websocket_policy_protocol as _protocol


DEFAULT_CONNECT_TIMEOUT_S = 60.0
DEFAULT_CONNECT_ATTEMPT_TIMEOUT_S = 5.0
DEFAULT_METADATA_TIMEOUT_S = 5.0
DEFAULT_SEND_TIMEOUT_S = 5.0
DEFAULT_INFERENCE_TIMEOUT_S = 120.0
DEFAULT_HEARTBEAT_TIMEOUT_S = 2.0
DEFAULT_CLOSE_TIMEOUT_S = 2.0
DEFAULT_RETRY_INTERVAL_S = 1.0
MAX_TRACKED_RESPONSE_IDS = 4096


class PolicyClientState(enum.Enum):
    CONNECTING = "connecting"
    READY = "ready"
    IN_FLIGHT = "in_flight"
    FAULTED = "faulted"
    CLOSED = "closed"


class PolicyFaultType(enum.Enum):
    CONNECT_TIMEOUT = "connect_timeout"
    METADATA_TIMEOUT = "metadata_timeout"
    SEND_TIMEOUT = "send_timeout"
    INFERENCE_TIMEOUT = "inference_timeout"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    DISCONNECTED = "disconnected"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    STALE_RESPONSE = "stale_response"
    DUPLICATE_RESPONSE = "duplicate_response"
    SERVER_ERROR = "server_error"
    CLIENT_STATE = "client_state"


class PolicyClientError(RuntimeError):
    """Base class for typed policy transport and protocol failures."""

    fault_type = PolicyFaultType.CLIENT_STATE

    def __init__(self, message: str, *, request_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class PolicyConnectionTimeoutError(PolicyClientError):
    fault_type = PolicyFaultType.CONNECT_TIMEOUT


class PolicyMetadataTimeoutError(PolicyClientError):
    fault_type = PolicyFaultType.METADATA_TIMEOUT


class PolicySendTimeoutError(PolicyClientError):
    fault_type = PolicyFaultType.SEND_TIMEOUT


class PolicyInferenceTimeoutError(PolicyClientError):
    fault_type = PolicyFaultType.INFERENCE_TIMEOUT


class PolicyHeartbeatTimeoutError(PolicyClientError):
    fault_type = PolicyFaultType.HEARTBEAT_TIMEOUT


class PolicyDisconnectedError(PolicyClientError):
    fault_type = PolicyFaultType.DISCONNECTED


class PolicyInvalidRequestError(PolicyClientError):
    fault_type = PolicyFaultType.INVALID_REQUEST


class PolicyInvalidResponseError(PolicyClientError):
    fault_type = PolicyFaultType.INVALID_RESPONSE


class PolicyStaleResponseError(PolicyClientError):
    fault_type = PolicyFaultType.STALE_RESPONSE


class PolicyDuplicateResponseError(PolicyClientError):
    fault_type = PolicyFaultType.DUPLICATE_RESPONSE


class PolicyServerError(PolicyClientError):
    fault_type = PolicyFaultType.SERVER_ERROR


class PolicyClientStateError(PolicyClientError):
    fault_type = PolicyFaultType.CLIENT_STATE


def _validate_timeout(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Communicate with a policy server with bounded, fail-closed operations.

    Protocol version 1 is opt-in for compatibility with legacy policy servers.
    Regardless of protocol mode, connection, metadata, send, and inference waits
    have finite upper bounds. A transport or response fault permanently faults
    the current connection; only an explicit :meth:`reconnect` creates a new
    connection epoch.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        connect_attempt_timeout_s: float = DEFAULT_CONNECT_ATTEMPT_TIMEOUT_S,
        metadata_timeout_s: float = DEFAULT_METADATA_TIMEOUT_S,
        send_timeout_s: float = DEFAULT_SEND_TIMEOUT_S,
        inference_timeout_s: float = DEFAULT_INFERENCE_TIMEOUT_S,
        heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
        close_timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S,
        retry_interval_s: float = DEFAULT_RETRY_INTERVAL_S,
        protocol_version: Optional[int] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        if protocol_version not in {None, _protocol.PROTOCOL_VERSION}:
            raise ValueError(f"protocol_version must be None or {_protocol.PROTOCOL_VERSION}")

        self._connect_timeout_s = _validate_timeout(connect_timeout_s, "connect_timeout_s")
        self._connect_attempt_timeout_s = _validate_timeout(connect_attempt_timeout_s, "connect_attempt_timeout_s")
        self._metadata_timeout_s = _validate_timeout(metadata_timeout_s, "metadata_timeout_s")
        self._send_timeout_s = _validate_timeout(send_timeout_s, "send_timeout_s")
        self._inference_timeout_s = _validate_timeout(inference_timeout_s, "inference_timeout_s")
        self._heartbeat_timeout_s = _validate_timeout(heartbeat_timeout_s, "heartbeat_timeout_s")
        self._close_timeout_s = _validate_timeout(close_timeout_s, "close_timeout_s")
        self._retry_interval_s = _validate_timeout(retry_interval_s, "retry_interval_s")
        self._protocol_version = protocol_version

        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._client_session_id = str(uuid.uuid4())
        self._connection_epoch = 0
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = PolicyClientState.CONNECTING
        self._last_fault: Optional[PolicyClientError] = None
        self._last_request_metadata: Optional[Dict] = None
        self._last_response_metadata: Optional[Dict] = None
        self._seen_response_ids = set()
        self._response_id_order = []
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._server_metadata: Dict = {}

        try:
            self._ws, self._server_metadata = self._wait_for_server()
        except PolicyClientError as exc:
            self._set_fault(exc)
            raise
        with self._state_lock:
            self._state = PolicyClientState.READY

    @property
    def connection_state(self) -> PolicyClientState:
        with self._state_lock:
            return self._state

    @property
    def connection_epoch(self) -> int:
        return self._connection_epoch

    @property
    def client_session_id(self) -> str:
        return self._client_session_id

    @property
    def last_fault(self) -> Optional[PolicyClientError]:
        with self._state_lock:
            return self._last_fault

    @property
    def last_request_metadata(self) -> Optional[Dict]:
        return None if self._last_request_metadata is None else dict(self._last_request_metadata)

    @property
    def last_response_metadata(self) -> Optional[Dict]:
        return None if self._last_response_metadata is None else dict(self._last_response_metadata)

    def get_server_metadata(self) -> Dict:
        return dict(self._server_metadata)

    def ensure_ready(self) -> None:
        """Raise immediately when the connection isn't safe for new work."""

        with self._state_lock:
            state = self._state
            fault = self._last_fault
        if state is PolicyClientState.READY:
            return
        detail = "" if fault is None else f"; last fault: {fault.fault_type.value}: {fault}"
        raise PolicyClientStateError(f"policy client is {state.value}, not ready{detail}")

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting up to %.1f s for server at %s...", self._connect_timeout_s, self._uri)
        deadline = time.monotonic() + self._connect_timeout_s
        last_error: Optional[BaseException] = None
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                message = f"Timed out after {self._connect_timeout_s:.1f} s waiting for policy server at {self._uri}"
                raise PolicyConnectionTimeoutError(message) from last_error
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=min(self._connect_attempt_timeout_s, remaining_s),
                    close_timeout=self._close_timeout_s,
                )
            except (ConnectionRefusedError, OSError, TimeoutError) as exc:
                last_error = exc
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    continue
                logging.info("Policy server connection not ready: %s", exc)
                time.sleep(min(self._retry_interval_s, remaining_s))
                continue
            except Exception as exc:
                raise PolicyDisconnectedError(
                    f"Could not connect to policy server at {self._uri}: {type(exc).__name__}: {exc}"
                ) from exc

            metadata_timeout_s = min(self._metadata_timeout_s, max(0.001, deadline - time.monotonic()))
            try:
                packed_metadata = conn.recv(timeout=metadata_timeout_s)
            except TimeoutError as exc:
                self._abort_connection(conn)
                raise PolicyMetadataTimeoutError(
                    f"Policy server did not send metadata within {metadata_timeout_s:.3f} s"
                ) from exc
            except Exception as exc:
                self._abort_connection(conn)
                raise self._disconnected_error("receiving policy server metadata", exc) from exc
            if isinstance(packed_metadata, str):
                self._abort_connection(conn)
                raise PolicyInvalidResponseError("Policy server metadata must be a binary MessagePack frame")
            try:
                metadata = msgpack_numpy.unpackb(packed_metadata)
            except Exception as exc:
                self._abort_connection(conn)
                raise PolicyInvalidResponseError(
                    f"Could not decode policy server metadata: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(metadata, dict):
                self._abort_connection(conn)
                raise PolicyInvalidResponseError("Policy server metadata must decode to a mapping")
            if self._protocol_version is not None:
                supported_versions = metadata.get(_protocol.PROTOCOL_METADATA_KEY, ())
                if not isinstance(supported_versions, (list, tuple)) or any(
                    isinstance(version, bool) or not isinstance(version, int) for version in supported_versions
                ):
                    self._abort_connection(conn)
                    raise PolicyInvalidResponseError(
                        f"Policy server protocol versions must be an integer list: {supported_versions!r}"
                    )
                if self._protocol_version not in supported_versions:
                    self._abort_connection(conn)
                    raise PolicyInvalidResponseError(
                        f"Policy server does not advertise protocol version {self._protocol_version}: "
                        f"{supported_versions!r}"
                    )
            return conn, metadata

    @staticmethod
    def _abort_connection(connection: Optional[websockets.sync.client.ClientConnection]) -> None:
        if connection is None:
            return
        raw_socket = getattr(connection, "socket", None)
        if raw_socket is not None:
            try:
                raw_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                raw_socket.close()
            except OSError:
                pass

    @staticmethod
    def _disconnected_error(
        operation: str, exc: BaseException, *, request_id: Optional[str] = None
    ) -> PolicyClientError:
        return PolicyDisconnectedError(
            f"Policy connection failed while {operation}: {type(exc).__name__}: {exc}",
            request_id=request_id,
        )

    def _set_fault(self, fault: PolicyClientError) -> None:
        with self._state_lock:
            self._state = PolicyClientState.FAULTED
            self._last_fault = fault
        self._abort_connection(self._ws)

    def _set_in_flight(self) -> None:
        self.ensure_ready()
        with self._state_lock:
            self._state = PolicyClientState.IN_FLIGHT

    def _set_ready(self) -> None:
        with self._state_lock:
            self._state = PolicyClientState.READY

    def _run_bounded_operation(self, operation: Callable[[], Any], timeout_s: float) -> Any:
        done = threading.Event()
        result = []

        def run() -> None:
            try:
                result.append((True, operation()))
            except BaseException as exc:
                result.append((False, exc))
            finally:
                done.set()

        thread = threading.Thread(target=run, name="openpi-policy-bounded-operation", daemon=True)
        thread.start()
        if not done.wait(timeout_s):
            self._abort_connection(self._ws)
            raise TimeoutError(f"operation exceeded {timeout_s:.3f} s")
        succeeded, value = result[0]
        if succeeded:
            return value
        raise value

    def _send(self, data: bytes, *, request_id: Optional[str]) -> None:
        assert self._ws is not None
        try:
            self._run_bounded_operation(lambda: self._ws.send(data), self._send_timeout_s)
        except TimeoutError as exc:
            raise PolicySendTimeoutError(
                f"Policy request send exceeded {self._send_timeout_s:.3f} s",
                request_id=request_id,
            ) from exc
        except Exception as exc:
            raise self._disconnected_error("sending a policy request", exc, request_id=request_id) from exc

    def _recv(self, *, request_id: Optional[str]) -> Any:
        assert self._ws is not None
        try:
            return self._ws.recv(timeout=self._inference_timeout_s)
        except TimeoutError as exc:
            raise PolicyInferenceTimeoutError(
                f"Policy inference exceeded {self._inference_timeout_s:.3f} s",
                request_id=request_id,
            ) from exc
        except Exception as exc:
            raise self._disconnected_error("receiving a policy response", exc, request_id=request_id) from exc

    def _remember_response_id(self, response_id: str) -> None:
        self._seen_response_ids.add(response_id)
        self._response_id_order.append(response_id)
        if len(self._response_id_order) > MAX_TRACKED_RESPONSE_IDS:
            oldest = self._response_id_order.pop(0)
            self._seen_response_ids.discard(oldest)

    def _validate_correlated_response(self, message: Dict, request_metadata: Dict) -> Dict:
        request_id = request_metadata["request_id"]
        try:
            payload, response_metadata = _protocol.unpack_response(message)
        except _protocol.ProtocolError as exc:
            raise PolicyInvalidResponseError(str(exc), request_id=request_id) from exc

        response_id = response_metadata["response_id"]
        if response_id in self._seen_response_ids:
            raise PolicyDuplicateResponseError(
                f"Policy response_id {response_id!r} was already accepted",
                request_id=request_id,
            )
        correlation_fields = (
            "request_id",
            "client_session_id",
            "connection_epoch",
            "observation_id",
            "request_created_unix_ns",
        )
        mismatches = {
            name: (request_metadata[name], response_metadata.get(name))
            for name in correlation_fields
            if response_metadata.get(name) != request_metadata[name]
        }
        if mismatches:
            raise PolicyStaleResponseError(
                f"Policy response does not match the in-flight request: {mismatches}",
                request_id=request_id,
            )
        self._remember_response_id(response_id)
        self._last_response_metadata = response_metadata
        if response_metadata["message_type"] == _protocol.ERROR_RESPONSE:
            raise PolicyServerError(
                f"Policy server error {payload.get('error_type')!r}: {payload.get('error_message')}",
                request_id=request_id,
            )
        return payload

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._request_lock:
            self.ensure_ready()
            request_metadata = None
            try:
                if self._protocol_version is None:
                    message = obs
                else:
                    message = _protocol.new_infer_request(
                        obs,
                        client_session_id=self._client_session_id,
                        connection_epoch=self._connection_epoch,
                    )
                    request_metadata = message[_protocol.ENVELOPE_KEY]
                data = self._packer.pack(message)
            except Exception as exc:
                raise PolicyInvalidRequestError(
                    f"Could not encode policy request: {type(exc).__name__}: {exc}"
                ) from exc

            self._last_request_metadata = None if request_metadata is None else dict(request_metadata)
            request_id = None if request_metadata is None else request_metadata["request_id"]
            self._set_in_flight()
            try:
                self._send(data, request_id=request_id)
                packed_response = self._recv(request_id=request_id)
                if isinstance(packed_response, str):
                    raise PolicyServerError(
                        f"Error in inference server:\n{packed_response}",
                        request_id=request_id,
                    )
                try:
                    response = msgpack_numpy.unpackb(packed_response)
                except Exception as exc:
                    raise PolicyInvalidResponseError(
                        f"Could not decode policy response: {type(exc).__name__}: {exc}",
                        request_id=request_id,
                    ) from exc
                if not isinstance(response, dict):
                    raise PolicyInvalidResponseError(
                        "Policy response must decode to a mapping",
                        request_id=request_id,
                    )
                if request_metadata is not None:
                    response = self._validate_correlated_response(response, request_metadata)
                self._set_ready()
                return response
            except PolicyClientError as exc:
                self._set_fault(exc)
                raise
            except Exception as exc:
                fault = self._disconnected_error("processing a policy request", exc, request_id=request_id)
                self._set_fault(fault)
                raise fault from exc

    def health_check(self, *, timeout_s: Optional[float] = None) -> None:
        """Send a WebSocket ping and fault the connection if no pong arrives."""

        heartbeat_timeout_s = (
            self._heartbeat_timeout_s if timeout_s is None else _validate_timeout(timeout_s, "timeout_s")
        )
        with self._request_lock:
            self._set_in_flight()
            assert self._ws is not None
            try:
                pong_waiter = self._run_bounded_operation(self._ws.ping, self._send_timeout_s)
                if not pong_waiter.wait(heartbeat_timeout_s):
                    raise PolicyHeartbeatTimeoutError(f"Policy heartbeat exceeded {heartbeat_timeout_s:.3f} s")
                self._set_ready()
            except PolicyClientError as exc:
                self._set_fault(exc)
                raise
            except TimeoutError as exc:
                fault = PolicySendTimeoutError(f"Policy heartbeat send exceeded {self._send_timeout_s:.3f} s")
                self._set_fault(fault)
                raise fault from exc
            except Exception as exc:
                fault = self._disconnected_error("checking policy heartbeat", exc)
                self._set_fault(fault)
                raise fault from exc

    def reconnect(self) -> None:
        """Explicitly replace the connection and advance its correlation epoch."""

        with self._request_lock:
            with self._state_lock:
                if self._state is PolicyClientState.CLOSED:
                    raise PolicyClientStateError("a closed policy client cannot reconnect")
                if self._state is PolicyClientState.IN_FLIGHT:
                    raise PolicyClientStateError("cannot reconnect while a request is in flight")
                self._state = PolicyClientState.CONNECTING
                self._last_fault = None
            self._abort_connection(self._ws)
            self._ws = None
            self._server_metadata = {}
            self._connection_epoch += 1
            try:
                self._ws, self._server_metadata = self._wait_for_server()
            except PolicyClientError as exc:
                self._set_fault(exc)
                raise
            self._set_ready()

    def close(self) -> None:
        """Close without allowing the WebSocket close handshake to block forever."""

        with self._request_lock:
            connection = self._ws
            if connection is not None:
                try:
                    self._run_bounded_operation(connection.close, self._close_timeout_s)
                except BaseException:
                    self._abort_connection(connection)
            self._ws = None
            with self._state_lock:
                self._state = PolicyClientState.CLOSED

    @override
    def reset(self) -> None:
        pass
