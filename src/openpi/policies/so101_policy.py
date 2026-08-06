import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


SO101_ACTION_DIM = 6


def make_so101_example(*, include_wrist: bool = False) -> dict:
    """Creates a random input example matching the LeIsaac OpenPI client contract."""
    example = {
        "images/front": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(SO101_ACTION_DIM),
        "prompt": "Lift the cube.",
    }
    if include_wrist:
        example["images/wrist"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    return example


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    if image.shape[-1] == 3:
        return image
    if image.shape[0] == 3:
        return np.moveaxis(image, 0, -1)
    raise ValueError(f"Expected an RGB image in HWC or CHW format, got shape {image.shape}")


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Converts LeIsaac SO-101 observations into the fixed OpenPI model input slots.

    Expected inference keys are ``images/front``, optional ``images/wrist``,
    six-dimensional ``state``, and ``prompt``. Training data is repacked to the
    same structure by ``LeRobotSO101DataConfig``.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        camera_keys = {key for key in data if key.startswith("images/")}
        supported_camera_keys = {"images/front", "images/wrist"}
        if unexpected_camera_keys := camera_keys - supported_camera_keys:
            raise ValueError(f"Unexpected SO-101 camera keys: {sorted(unexpected_camera_keys)}")
        if "images/front" not in data:
            raise ValueError('SO-101 observations require an "images/front" image')

        state = np.asarray(data["state"])
        if state.shape[-1:] != (SO101_ACTION_DIM,):
            raise ValueError(f"Expected SO-101 state dimension {SO101_ACTION_DIM}, got shape {state.shape}")

        base_image = _parse_image(data["images/front"])
        missing_image_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        if "images/wrist" in data:
            left_wrist_image = _parse_image(data["images/wrist"])
            left_wrist_mask = np.True_
        else:
            left_wrist_image = np.zeros_like(base_image)
            left_wrist_mask = missing_image_mask

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_wrist_mask,
                "right_wrist_0_rgb": missing_image_mask,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1:] != (SO101_ACTION_DIM,):
                raise ValueError(f"Expected SO-101 action dimension {SO101_ACTION_DIM}, got shape {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Extracts the six absolute SO-101 motor targets expected by LeIsaac."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim == 0 or actions.shape[-1] < SO101_ACTION_DIM:
            raise ValueError(
                f"Model action dimension must be at least {SO101_ACTION_DIM}, got shape {actions.shape}"
            )
        return {"actions": actions[..., :SO101_ACTION_DIM]}
