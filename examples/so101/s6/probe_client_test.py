from __future__ import annotations

import numpy as np
import pytest

from examples.so101.s6 import probe_client


def test_validate_action_chunk_accepts_finite_in_range_actions() -> None:
    actions = np.zeros((10, 6), dtype=np.float32)
    assert (
        probe_client.validate_action_chunk(
            actions,
            expected_horizon=10,
            action_dim=6,
            max_abs_action=1.0,
        )
        is actions
    )


@pytest.mark.parametrize(
    ("actions", "fault_type"),
    [
        (np.zeros((10, 7), dtype=np.float32), probe_client.INVALID_ACTION_SHAPE),
        (np.full((10, 6), np.nan, dtype=np.float32), probe_client.INVALID_ACTION_NONFINITE),
        (np.full((10, 6), 2.0, dtype=np.float32), probe_client.INVALID_ACTION_OUT_OF_RANGE),
    ],
)
def test_validate_action_chunk_rejects_without_clipping(actions: np.ndarray, fault_type: str) -> None:
    original = actions.copy()
    with pytest.raises(probe_client.ProbeFaultError) as caught:
        probe_client.validate_action_chunk(
            actions,
            expected_horizon=10,
            action_dim=6,
            max_abs_action=1.0,
        )
    assert caught.value.fault_type == fault_type
    np.testing.assert_equal(actions, original)


def test_validate_action_chunk_rejects_unconvertible_payload() -> None:
    with pytest.raises(probe_client.ProbeFaultError) as caught:
        probe_client.validate_action_chunk(
            {"not": "an array"},
            expected_horizon=10,
            action_dim=6,
            max_abs_action=None,
        )
    assert caught.value.fault_type == probe_client.INVALID_ACTION_SHAPE
