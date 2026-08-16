from __future__ import annotations

import numpy as np
import pytest

from examples.so101 import red_cube_to_box_expert_joint_replay as replay


def test_summarize_tracking_errors_reports_per_joint_and_global_values() -> None:
    errors = np.array([[1.0, -2.0], [-1.0, 0.0]], dtype=np.float32)

    mae, rmse, maximum, global_rmse = replay.summarize_tracking_errors(errors)

    np.testing.assert_allclose(mae, [1.0, 1.0])
    np.testing.assert_allclose(rmse, [1.0, np.sqrt(2.0)])
    np.testing.assert_allclose(maximum, [1.0, 2.0])
    assert global_rmse == pytest.approx(np.sqrt(1.5))


def test_summarize_tracking_errors_rejects_empty_or_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        replay.summarize_tracking_errors(np.zeros((0, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        replay.summarize_tracking_errors(np.zeros(6, dtype=np.float32))


def test_first_divergence_step_uses_maximum_absolute_joint_error() -> None:
    errors = np.array([[0.01, -0.02], [0.03, -0.051], [0.08, 0.0]], dtype=np.float32)

    assert replay.first_divergence_step(errors, 0.05) == 1
    assert replay.first_divergence_step(errors, 0.1) is None


def test_first_divergence_step_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        replay.first_divergence_step(np.zeros(6), 0.05)
    with pytest.raises(ValueError, match="positive"):
        replay.first_divergence_step(np.zeros((1, 6)), 0.0)


def test_clip_joint_targets_reports_joint_specific_corrections() -> None:
    targets = np.array([[-2.0, 0.0, 3.0], [-0.5, 2.0, 0.5]], dtype=np.float32)

    clipped, counts, maximum = replay.clip_joint_targets(
        targets,
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
    )

    np.testing.assert_allclose(clipped, [[-1.0, 0.0, 1.0], [-0.5, 1.0, 0.5]])
    np.testing.assert_array_equal(counts, [1, 1, 1])
    np.testing.assert_allclose(maximum, [1.0, 1.0, 2.0])


def test_clip_joint_targets_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        replay.clip_joint_targets(np.zeros((2, 6)), np.zeros(5), np.ones(5))
