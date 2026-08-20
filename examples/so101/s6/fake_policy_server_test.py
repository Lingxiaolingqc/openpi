from __future__ import annotations

import asyncio

import numpy as np
import pytest

from examples.so101.s6 import fake_policy_server


@pytest.mark.parametrize(
    ("mode", "expected_shape", "expected_nonfinite_count"),
    [
        (fake_policy_server.FaultMode.NORMAL, (10, 6), 0),
        (fake_policy_server.FaultMode.BAD_SHAPE, (10, 7), 0),
        (fake_policy_server.FaultMode.NAN_ACTION, (10, 6), 1),
        (fake_policy_server.FaultMode.INF_ACTION, (10, 6), 1),
    ],
)
def test_make_action_chunk_faults(
    mode: fake_policy_server.FaultMode,
    expected_shape: tuple[int, int],
    expected_nonfinite_count: int,
) -> None:
    actions = fake_policy_server.make_action_chunk(mode, horizon=10, action_dim=6, action_value=0.0)
    assert actions.shape == expected_shape
    assert int(np.count_nonzero(~np.isfinite(actions))) == expected_nonfinite_count


def test_out_of_range_actions_are_not_clipped() -> None:
    actions = fake_policy_server.make_action_chunk(
        fake_policy_server.FaultMode.OUT_OF_RANGE_ACTION,
        horizon=10,
        action_dim=6,
        action_value=0.0,
    )
    np.testing.assert_array_equal(actions, np.full((10, 6), 999.0, dtype=np.float32))


def test_fake_server_rejects_invalid_jitter_range() -> None:
    args = fake_policy_server.build_parser().parse_args(["--jitter-min-s", "1", "--jitter-max-s", "0"])
    with pytest.raises(ValueError, match="jitter"):
        fake_policy_server.validate_args(args)


def test_fake_server_rejects_nonfinite_configuration() -> None:
    args = fake_policy_server.build_parser().parse_args(["--delay-s", "nan"])
    with pytest.raises(ValueError, match="finite"):
        fake_policy_server.validate_args(args)


def test_fake_server_rejects_nonpositive_mid_chunk_disconnect_delay() -> None:
    args = fake_policy_server.build_parser().parse_args(["--disconnect-after-response-s", "0"])
    with pytest.raises(ValueError, match="disconnect-after-response-s"):
        fake_policy_server.validate_args(args)


def test_duplicate_response_is_sent_only_after_next_request(capsys) -> None:
    args = fake_policy_server.build_parser().parse_args(["--fault", "duplicate-response"])
    server = fake_policy_server.FakePolicyServer(args)
    request_messages = [
        fake_policy_server.protocol.new_infer_request(
            {"state": [float(index)]},
            client_session_id="session-1",
            connection_epoch=0,
        )
        for index in range(2)
    ]
    packed_requests = [server._packer.pack(message) for message in request_messages]  # noqa: SLF001

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[bytes] = []
            self.request_index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.request_index >= len(packed_requests):
                raise StopAsyncIteration
            if self.request_index == 1:
                assert len(self.sent) == 2, "duplicate response must not be sent before the next request"
            packed_request = packed_requests[self.request_index]
            self.request_index += 1
            return packed_request

        async def send(self, data: bytes) -> None:
            self.sent.append(data)

    websocket = FakeWebSocket()
    asyncio.run(server.handler(websocket))

    assert len(websocket.sent) == 3
    first_response = fake_policy_server.msgpack_numpy.unpackb(websocket.sent[1])
    duplicate_response = fake_policy_server.msgpack_numpy.unpackb(websocket.sent[2])
    _, first_metadata = fake_policy_server.protocol.unpack_response(first_response)
    _, duplicate_metadata = fake_policy_server.protocol.unpack_response(duplicate_response)
    assert duplicate_metadata == first_metadata
    assert first_metadata["request_id"] == request_messages[0][fake_policy_server.protocol.ENVELOPE_KEY]["request_id"]
    assert (
        duplicate_metadata["request_id"] != request_messages[1][fake_policy_server.protocol.ENVELOPE_KEY]["request_id"]
    )

    injection_lines = [
        line.removeprefix(fake_policy_server.INJECTION_PREFIX)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(fake_policy_server.INJECTION_PREFIX)
    ]
    assert len(injection_lines) == 1
    assert '"injection_stage": "sent_on_next_request"' in injection_lines[0]
    assert '"request_count": 2' in injection_lines[0]
