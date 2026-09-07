"""OpenPI transforms for the seven-dimensional PiPER MuJoCo contract."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model

PIPER_ACTION_DIM = 7
PIPER_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
PIPER_STATE_NAMES = (*PIPER_JOINT_NAMES, "gripper_open_fraction")


@dataclasses.dataclass(frozen=True)
class RepackPiperData(transforms.DataTransformFn):
    """Map a canonical PiPER LeRobot frame into the public policy contract."""

    def __call__(self, data: dict) -> dict:
        mapping = {
            "images/base": "observation.images.base",
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        missing = [source for source in mapping.values() if source not in data]
        if missing:
            raise ValueError(f"PiPER LeRobot frame is missing keys: {missing}")
        return {target: data[source] for target, source in mapping.items()}


def make_piper_example() -> dict:
    """Create one inference request following the PiPER simulation v1 contract."""

    return {
        "images/base": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.zeros(PIPER_ACTION_DIM, dtype=np.float32),
        "prompt": "place the red cube in the box",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"expected a 3D PiPER image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("PiPER image contains NaN or infinity")
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"PiPER image must be uint8 or floating point, got {image.dtype}")
    if image.shape[-1] == 3:
        return image
    if image.shape[0] == 3:
        return np.moveaxis(image, 0, -1)
    raise ValueError(f"expected PiPER RGB in HWC or CHW format, got shape {image.shape}")


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Map base RGB and seven-dimensional state to OpenPI model slots."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        camera_keys = {key for key in data if key.startswith("images/")}
        if unexpected := camera_keys - {"images/base"}:
            raise ValueError(f"unexpected PiPER camera keys: {sorted(unexpected)}")
        if "images/base" not in data:
            raise ValueError('PiPER observations require an "images/base" image')
        state = np.asarray(data["state"])
        if state.shape[-1:] != (PIPER_ACTION_DIM,):
            raise ValueError(f"expected PiPER state dimension {PIPER_ACTION_DIM}, got shape {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("PiPER state contains NaN or infinity")
        image = _parse_image(data["images/base"])
        missing_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        result = {
            "state": state,
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": missing_mask,
                "right_wrist_0_rgb": missing_mask,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1:] != (PIPER_ACTION_DIM,):
                raise ValueError(f"expected PiPER action dimension {PIPER_ACTION_DIM}, got shape {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("PiPER actions contain NaN or infinity")
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return result


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Extract only six arm targets and one normalized gripper target."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim == 0 or actions.shape[-1] < PIPER_ACTION_DIM:
            raise ValueError(f"model action dimension must be at least {PIPER_ACTION_DIM}, got shape {actions.shape}")
        actions = actions[..., :PIPER_ACTION_DIM]
        if not np.isfinite(actions).all():
            raise ValueError("PiPER model output contains NaN or infinity")
        return {"actions": actions}
