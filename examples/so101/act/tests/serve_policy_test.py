from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from examples.so101.act import common
from examples.so101.act import serve_policy

torch = pytest.importorskip("torch")


def test_server_policy_marks_outputs_unclipped_and_supports_reset(monkeypatch) -> None:
    resets = []
    fake_policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={"observation.images.front": SimpleNamespace(shape=(3, 4, 5))},
            temporal_ensemble_coeff=None,
            chunk_size=10,
        ),
        reset=lambda: resets.append(True),
    )
    monkeypatch.setattr(serve_policy, "load_act_policy", lambda *args, **kwargs: fake_policy)
    monkeypatch.setattr(serve_policy, "request_to_policy_batch", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        serve_policy,
        "predict_action_chunk",
        lambda *args, **kwargs: torch.zeros((1, 10, 6), dtype=torch.float32),
    )
    adapter = serve_policy.ACTWebsocketPolicy(
        pretrained_model_dir=common.CONFIG_PATH.parent,
        device="cpu",
        actions_per_inference=10,
        marker={"deployment_scope": common.DEPLOYMENT_SCOPE},
        contract={
            "repo_id": "local/example",
            "metadata_fingerprint": "abc",
            "camera_keys": ["observation.images.front"],
        },
    )

    assert adapter.infer({"__reset__": True}) == {"reset_ack": True}
    response = adapter.infer({"state": np.zeros(6), "images/front": np.zeros((4, 5, 3), dtype=np.uint8)})

    assert resets == [True]
    assert response["actions"].shape == (10, 6)
    assert response["policy_diagnostics"]["actions_clipped"] is False
    assert adapter.metadata["real_robot_deployment_allowed"] is False
