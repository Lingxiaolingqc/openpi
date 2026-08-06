import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import so101_policy


def test_front_only_pi05_inputs() -> None:
    front = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
    state = np.arange(6, dtype=np.float32)
    actions = np.arange(60, dtype=np.float32).reshape(10, 6)

    result = so101_policy.SO101Inputs(_model.ModelType.PI05)(
        {
            "images/front": front,
            "state": state,
            "actions": actions,
            "prompt": "Lift the cube.",
        }
    )

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], front)
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], np.zeros_like(front))
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], np.zeros_like(front))
    assert bool(result["image_mask"]["base_0_rgb"])
    assert not bool(result["image_mask"]["left_wrist_0_rgb"])
    assert not bool(result["image_mask"]["right_wrist_0_rgb"])
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)
    assert result["prompt"] == "Lift the cube."


def test_optional_wrist_is_mapped_to_left_wrist_slot() -> None:
    front = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist_chw = np.ones((3, 224, 224), dtype=np.float32)

    result = so101_policy.SO101Inputs(_model.ModelType.PI05)(
        {
            "images/front": front,
            "images/wrist": wrist_chw,
            "state": np.zeros(6, dtype=np.float32),
        }
    )

    assert result["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert result["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert np.all(result["image"]["left_wrist_0_rgb"] == 255)
    assert bool(result["image_mask"]["left_wrist_0_rgb"])
    assert not bool(result["image_mask"]["right_wrist_0_rgb"])


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"state": np.zeros(6)}, "images/front"),
        ({"images/front": np.zeros((224, 224, 3)), "state": np.zeros(7)}, "state dimension 6"),
        (
            {
                "images/front": np.zeros((224, 224, 3)),
                "images/side": np.zeros((224, 224, 3)),
                "state": np.zeros(6),
            },
            "Unexpected SO-101 camera keys",
        ),
    ],
)
def test_invalid_inputs_fail_early(data: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        so101_policy.SO101Inputs(_model.ModelType.PI05)(data)


def test_outputs_extract_six_action_dimensions() -> None:
    model_actions = np.arange(320, dtype=np.float32).reshape(10, 32)

    result = so101_policy.SO101Outputs()({"actions": model_actions})

    assert result["actions"].shape == (10, 6)
    np.testing.assert_array_equal(result["actions"], model_actions[:, :6])


def test_absolute_motor_actions_round_trip_through_delta_space() -> None:
    state = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
    absolute_actions = np.array(
        [
            [11, 22, 33, 44, 55, 61],
            [12, 23, 34, 45, 56, 62],
        ],
        dtype=np.float32,
    )
    delta_mask = transforms.make_bool_mask(5, -1)

    encoded = transforms.DeltaActions(delta_mask)(
        {
            "state": state.copy(),
            "actions": absolute_actions.copy(),
        }
    )

    np.testing.assert_array_equal(encoded["actions"][..., :5], absolute_actions[..., :5] - state[:5])
    np.testing.assert_array_equal(encoded["actions"][..., 5], absolute_actions[..., 5])

    decoded = transforms.AbsoluteActions(delta_mask)(encoded)
    np.testing.assert_array_equal(decoded["actions"], absolute_actions)
