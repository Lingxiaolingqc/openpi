from __future__ import annotations

import numpy as np
import pytest

from examples.so101 import audit_red_cube_to_box_policy_hdf5 as audit


def test_compare_action_chunks_reports_per_joint_and_global_errors() -> None:
    expert = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    predicted = np.asarray([[2.0, 2.0], [1.0, 6.0]])

    mae, maximum, rmse = audit.compare_action_chunks(predicted, expert)

    np.testing.assert_allclose(mae, [1.5, 1.0])
    np.testing.assert_allclose(maximum, [2.0, 2.0])
    assert rmse == pytest.approx(1.5)


def test_compare_action_chunks_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same 2D shape"):
        audit.compare_action_chunks(np.zeros((2, 6)), np.zeros((1, 6)))
