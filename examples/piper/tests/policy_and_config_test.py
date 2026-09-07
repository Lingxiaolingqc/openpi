import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import piper_policy
from openpi.training import config as _config


def test_pi05_base_only_input_masks_missing_wrist_slots() -> None:
    image = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    actions = np.arange(70, dtype=np.float32).reshape(10, 7)
    result = piper_policy.PiperInputs(_model.ModelType.PI05)(
        {"images/base": image, "state": state, "actions": actions, "prompt": b"place the cube"}
    )
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], image)
    assert bool(result["image_mask"]["base_0_rgb"])
    assert not bool(result["image_mask"]["left_wrist_0_rgb"])
    assert not bool(result["image_mask"]["right_wrist_0_rgb"])
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)
    assert result["prompt"] == "place the cube"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"state": np.zeros(7)}, "images/base"),
        ({"images/base": np.zeros((8, 8, 3)), "state": np.zeros(8)}, "state dimension 7"),
        (
            {"images/base": np.zeros((8, 8, 3)), "images/wrist": np.zeros((8, 8, 3)), "state": np.zeros(7)},
            "unexpected PiPER camera",
        ),
        ({"images/base": np.zeros((8, 8, 3)), "state": np.full(7, np.nan)}, "NaN or infinity"),
    ],
)
def test_invalid_policy_inputs_fail_early(data: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        piper_policy.PiperInputs(_model.ModelType.PI05)(data)


def test_outputs_extract_seven_physical_dimensions() -> None:
    model_actions = np.arange(320, dtype=np.float32).reshape(10, 32)
    result = piper_policy.PiperOutputs()({"actions": model_actions})
    assert result["actions"].shape == (10, 7)
    np.testing.assert_array_equal(result["actions"], model_actions[:, :7])


def test_six_joints_round_trip_through_delta_space() -> None:
    state = np.array([0.1, 1.0, -1.0, 0.2, -0.3, 0.4, 0.8], dtype=np.float32)
    absolute = np.stack([state, state + np.array([0.01] * 6 + [-0.2], dtype=np.float32)])
    mask = transforms.make_bool_mask(6, -1)
    encoded = transforms.DeltaActions(mask)({"state": state.copy(), "actions": absolute.copy()})
    np.testing.assert_allclose(encoded["actions"][..., :6], absolute[..., :6] - state[:6])
    np.testing.assert_array_equal(encoded["actions"][..., 6], absolute[..., 6])
    decoded = transforms.AbsoluteActions(mask)(encoded)
    np.testing.assert_allclose(decoded["actions"], absolute)


def test_pi05_piper_lora_config_contract() -> None:
    config = _config.get_config("pi05_piper_lora")
    assert config.model.model_type == _config.ModelType.PI05
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 10
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"
    assert isinstance(config.data, _config.LeRobotPiperDataConfig)
    assert config.data.repo_id == _config.PIPER_REPO_ID
    assert config.data.use_delta_joint_actions
    assert config.batch_size == 8
    assert config.num_workers == 0
    assert config.num_train_steps == 100
    assert config.ema_decay is None
    assert config.policy_metadata is not None
    assert config.policy_metadata["action_dim"] == 7
    assert config.policy_metadata["deployment_scope"] == "simulation-only"
    assert config.policy_metadata["real_robot_deployment_allowed"] is False


def test_lerobot_repack_has_one_required_camera() -> None:
    repack = _config.LeRobotPiperDataConfig(repo_id="test/piper").repack_transforms.inputs[0]
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(7, dtype=np.float32)
    actions = np.zeros((10, 7), dtype=np.float32)
    result = repack(
        {
            "observation.images.base": image,
            "observation.state": state,
            "action": actions,
            "prompt": "place the red cube in the box",
        }
    )
    np.testing.assert_array_equal(result["images/base"], image)
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)
