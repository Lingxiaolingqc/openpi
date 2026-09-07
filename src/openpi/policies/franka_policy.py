"""OpenPI input and output transforms for a single FR3 with Franka Hand."""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model

FRANKA_ACTION_DIM = 8
FRANKA_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
FRANKA_STATE_NAMES = (*FRANKA_JOINT_NAMES, "gripper_width_m")


@dataclasses.dataclass(frozen=True)
class RepackFrankaData(transforms.DataTransformFn):
    """Map a LeRobot frame while preserving the optional wrist camera."""

    def __call__(self, data: dict) -> dict:
        required = {
            "images/base": "observation.images.base",
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        missing = [source for source in required.values() if source not in data]
        if missing:
            raise ValueError(f"Franka LeRobot frame is missing keys: {missing}")
        result = {target: data[source] for target, source in required.items()}
        if "observation.images.wrist" in data:
            result["images/wrist"] = data["observation.images.wrist"]
        return result


def make_franka_example(*, include_wrist: bool = False) -> dict:
    """Create a random inference request that follows the Franka v1 contract."""

    example = {
        "images/base": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.zeros(FRANKA_ACTION_DIM, dtype=np.float32),
        "prompt": "Perform the instructed manipulation task.",
    }
    if include_wrist:
        example["images/wrist"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    return example


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Franka images must not contain NaN or infinity")
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"Franka images must be uint8 or floating point, got {image.dtype}")
    if image.shape[-1] == 3:
        return image
    if image.shape[0] == 3:
        return np.moveaxis(image, 0, -1)
    raise ValueError(f"Expected an RGB image in HWC or CHW format, got shape {image.shape}")


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Map the public Franka observation contract to OpenPI model slots."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        camera_keys = {key for key in data if key.startswith("images/")}
        supported_camera_keys = {"images/base", "images/wrist"}
        if unexpected_camera_keys := camera_keys - supported_camera_keys:
            raise ValueError(f"Unexpected Franka camera keys: {sorted(unexpected_camera_keys)}")
        if "images/base" not in data:
            raise ValueError('Franka observations require an "images/base" image')

        state = np.asarray(data["state"])
        if state.shape[-1:] != (FRANKA_ACTION_DIM,):
            raise ValueError(f"Expected Franka state dimension {FRANKA_ACTION_DIM}, got shape {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("Franka state must not contain NaN or infinity")

        base_image = _parse_image(data["images/base"])
        missing_image_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        if "images/wrist" in data:
            wrist_image = _parse_image(data["images/wrist"])
            wrist_mask = np.True_
        else:
            wrist_image = np.zeros_like(base_image)
            wrist_mask = missing_image_mask

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": missing_image_mask,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1:] != (FRANKA_ACTION_DIM,):
                raise ValueError(f"Expected Franka action dimension {FRANKA_ACTION_DIM}, got shape {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("Franka actions must not contain NaN or infinity")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Extract seven absolute joint targets and one absolute gripper width."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim == 0 or actions.shape[-1] < FRANKA_ACTION_DIM:
            raise ValueError(f"Model action dimension must be at least {FRANKA_ACTION_DIM}, got shape {actions.shape}")
        actions = actions[..., :FRANKA_ACTION_DIM]
        if not np.isfinite(actions).all():
            raise ValueError("Franka model output must not contain NaN or infinity")
        return {"actions": actions}
