from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from examples.so101.act import policy_utils

torch = pytest.importorskip("torch")


class _FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            temporal_ensemble_coeff=None,
            n_action_steps=2,
            chunk_size=3,
            image_features={},
        )
        self.model = self._model

    def eval(self) -> None:
        pass

    def normalize_inputs(self, batch):
        return batch

    def unnormalize_outputs(self, batch):
        return batch

    def _model(self, batch):
        assert "action" not in batch
        assert "action_is_pad" not in batch
        return torch.arange(18, dtype=torch.float32).reshape(1, 3, 6), (None, None)


def test_predict_action_chunk_does_not_condition_on_targets() -> None:
    actions = policy_utils.predict_action_chunk(
        _FakePolicy(),
        {
            "observation.state": torch.zeros((1, 6)),
            "action": torch.ones((1, 3, 6)),
            "action_is_pad": torch.zeros((1, 3), dtype=torch.bool),
        },
    )

    assert actions.shape == (1, 2, 6)


def test_request_batch_supports_checkpoint_discovered_cameras() -> None:
    feature = SimpleNamespace(shape=(3, 4, 5))
    policy = SimpleNamespace(
        config=SimpleNamespace(
            robot_state_feature=SimpleNamespace(shape=(6,)),
            image_features={"observation.images.front": feature},
        )
    )
    batch = policy_utils.request_to_policy_batch(
        policy,
        {
            "state": np.arange(6, dtype=np.float32),
            "images/front": np.zeros((4, 5, 3), dtype=np.uint8),
        },
        device="cpu",
    )

    assert batch["observation.state"].shape == (1, 6)
    assert batch["observation.images.front"].shape == (1, 3, 4, 5)
