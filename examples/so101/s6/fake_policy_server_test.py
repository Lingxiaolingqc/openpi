from __future__ import annotations

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
