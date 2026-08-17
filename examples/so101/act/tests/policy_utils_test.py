from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from examples.so101.act import policy_utils

try:
    import torch
except ImportError:
    torch = None


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
    if torch is None:
        pytest.skip("PyTorch is not installed")
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
    if torch is None:
        pytest.skip("PyTorch is not installed")
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


def test_load_act_config_uses_choice_registry_base(monkeypatch, tmp_path) -> None:
    class FakeACTConfig:
        device = "cpu"

    config = FakeACTConfig()

    class FakePreTrainedConfig:
        @classmethod
        def from_pretrained(cls, path, *, local_files_only):
            assert path == tmp_path
            assert local_files_only is True
            return config

    packages = (
        "lerobot",
        "lerobot.common",
        "lerobot.common.policies",
        "lerobot.common.policies.act",
        "lerobot.configs",
    )
    for name in packages:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    configuration = types.ModuleType("lerobot.common.policies.act.configuration_act")
    configuration.ACTConfig = FakeACTConfig
    monkeypatch.setitem(sys.modules, configuration.__name__, configuration)
    policies = types.ModuleType("lerobot.configs.policies")
    policies.PreTrainedConfig = FakePreTrainedConfig
    monkeypatch.setitem(sys.modules, policies.__name__, policies)

    loaded = policy_utils.load_act_config(tmp_path, device="cuda")

    assert loaded is config
    assert loaded.device == "cuda"
