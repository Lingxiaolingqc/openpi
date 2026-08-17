"""Pinned-LeRobot ACT loading and inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_act_config(pretrained_model_dir: Path, *, device: str | None = None):
    try:
        from lerobot.common.policies.act.configuration_act import ACTConfig
        from lerobot.configs.policies import PreTrainedConfig
    except ImportError as exc:
        raise RuntimeError("LeRobot ACT is required; run this command with `uv run`") from exc
    config = PreTrainedConfig.from_pretrained(pretrained_model_dir, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise ValueError(f"checkpoint policy must be ACT, got {type(config).__name__}")
    if device is not None:
        config.device = device
    return config


def load_act_policy(pretrained_model_dir: Path, *, device: str):
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:
        raise RuntimeError("LeRobot ACT is required; run this command with `uv run`") from exc
    config = load_act_config(pretrained_model_dir, device=device)
    policy = ACTPolicy.from_pretrained(pretrained_model_dir, config=config, local_files_only=True)
    policy.to(device)
    policy.eval()
    return policy


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _image_tensor(value: Any, *, expected_shape: tuple[int, ...], device: str):
    import torch

    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"ACT camera input must be rank-3, got {array.shape}")
    if tuple(array.shape) == expected_shape:
        channel_first = array
    elif tuple(array.shape) == (expected_shape[1], expected_shape[2], expected_shape[0]):
        channel_first = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"ACT camera input shape {array.shape} does not match checkpoint shape {expected_shape}")
    if not np.isfinite(channel_first).all():
        raise ValueError("ACT camera input contains NaN or infinity")
    tensor = torch.from_numpy(np.ascontiguousarray(channel_first)).to(device=device, dtype=torch.float32)
    if array.dtype == np.uint8:
        tensor = tensor / 255.0
    elif float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
        raise ValueError("floating-point ACT images must already be scaled to [0, 1]")
    return tensor.unsqueeze(0)


def request_to_policy_batch(policy: Any, request: dict[str, Any], *, device: str) -> dict[str, Any]:
    import torch

    if "state" not in request:
        raise ValueError("ACT inference request is missing 'state'")
    state = np.asarray(request["state"], dtype=np.float32)
    state_feature = policy.config.robot_state_feature
    if state_feature is None:
        raise ValueError("ACT checkpoint does not declare an observation.state feature")
    expected_state_shape = tuple(int(item) for item in state_feature.shape)
    if tuple(state.shape) != expected_state_shape:
        raise ValueError(f"ACT state shape {state.shape} does not match checkpoint {expected_state_shape}")
    if not np.isfinite(state).all():
        raise ValueError("ACT state contains NaN or infinity")
    batch: dict[str, Any] = {
        "observation.state": torch.from_numpy(state).to(device=device).unsqueeze(0),
    }
    for feature_key, feature in policy.config.image_features.items():
        suffix = feature_key.removeprefix("observation.images.")
        request_key = f"images/{suffix}"
        if request_key not in request:
            raise ValueError(
                f"ACT inference request is missing camera {request_key!r}; checkpoint expects "
                f"{sorted(policy.config.image_features)}"
            )
        batch[feature_key] = _image_tensor(
            request[request_key],
            expected_shape=tuple(int(item) for item in feature.shape),
            device=device,
        )
    return batch


def predict_action_chunk(
    policy: Any,
    batch: dict[str, Any],
    *,
    actions_per_inference: int | None = None,
    apply_temporal_ensemble: bool = True,
):
    """Predict unnormalized actions without conditioning the VAE on target actions."""
    import torch

    policy.eval()
    clean_batch = {key: value for key, value in batch.items() if key not in {"action", "action_is_pad"}}
    with torch.no_grad():
        if policy.config.temporal_ensemble_coeff is not None and apply_temporal_ensemble:
            if actions_per_inference not in {None, 1}:
                raise ValueError("temporal ensembling requires actions_per_inference=1")
            return policy.select_action(clean_batch).unsqueeze(1)

        normalized = policy.normalize_inputs(clean_batch)
        if policy.config.image_features:
            normalized = dict(normalized)
            normalized["observation.images"] = [normalized[key] for key in policy.config.image_features]
        actions = policy.model(normalized)[0]
        maximum = int(policy.config.n_action_steps)
        if actions_per_inference is not None:
            if actions_per_inference < 1 or actions_per_inference > int(policy.config.chunk_size):
                raise ValueError(
                    "actions_per_inference must be between 1 and the checkpoint chunk_size "
                    f"({policy.config.chunk_size})"
                )
            maximum = actions_per_inference
        actions = actions[:, :maximum]
        return policy.unnormalize_outputs({"action": actions})["action"]
