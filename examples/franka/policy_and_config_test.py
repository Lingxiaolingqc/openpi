import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import franka_policy
from openpi.training import config as _config


def test_base_only_pi05_inputs() -> None:
    base = np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
    state = np.arange(8, dtype=np.float32)
    actions = np.arange(128, dtype=np.float32).reshape(16, 8)

    result = franka_policy.FrankaInputs(_model.ModelType.PI05)(
        {"images/base": base, "state": state, "actions": actions, "prompt": "Move safely."}
    )

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], base)
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], np.zeros_like(base))
    assert bool(result["image_mask"]["base_0_rgb"])
    assert not bool(result["image_mask"]["left_wrist_0_rgb"])
    assert not bool(result["image_mask"]["right_wrist_0_rgb"])
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)


def test_optional_wrist_is_mapped_and_float_image_is_clamped() -> None:
    base = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist = np.full((3, 224, 224), 2.0, dtype=np.float32)

    result = franka_policy.FrankaInputs(_model.ModelType.PI05)(
        {"images/base": base, "images/wrist": wrist, "state": np.zeros(8, dtype=np.float32)}
    )

    assert result["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert np.all(result["image"]["left_wrist_0_rgb"] == 255)
    assert bool(result["image_mask"]["left_wrist_0_rgb"])


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"state": np.zeros(8)}, "images/base"),
        ({"images/base": np.zeros((8, 8, 3)), "state": np.zeros(7)}, "state dimension 8"),
        (
            {"images/base": np.zeros((8, 8, 3)), "images/side": np.zeros((8, 8, 3)), "state": np.zeros(8)},
            "Unexpected Franka camera keys",
        ),
        ({"images/base": np.zeros((8, 8, 3)), "state": np.full(8, np.nan)}, "NaN or infinity"),
    ],
)
def test_invalid_inputs_fail_early(data: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        franka_policy.FrankaInputs(_model.ModelType.PI05)(data)


def test_outputs_extract_eight_action_dimensions() -> None:
    model_actions = np.arange(512, dtype=np.float32).reshape(16, 32)

    result = franka_policy.FrankaOutputs()({"actions": model_actions})

    assert result["actions"].shape == (16, 8)
    np.testing.assert_array_equal(result["actions"], model_actions[:, :8])


def test_absolute_joint_actions_round_trip_through_delta_space() -> None:
    state = np.array([0.1, -0.2, 0.3, -1.0, 0.5, 1.0, -0.5, 0.08], dtype=np.float32)
    absolute_actions = np.stack([state, state + np.array([0.01] * 7 + [-0.02], dtype=np.float32)])
    delta_mask = transforms.make_bool_mask(7, -1)

    encoded = transforms.DeltaActions(delta_mask)({"state": state.copy(), "actions": absolute_actions.copy()})
    np.testing.assert_allclose(encoded["actions"][..., :7], absolute_actions[..., :7] - state[:7])
    np.testing.assert_array_equal(encoded["actions"][..., 7], absolute_actions[..., 7])

    decoded = transforms.AbsoluteActions(delta_mask)(encoded)
    np.testing.assert_allclose(decoded["actions"], absolute_actions)


def test_pi05_franka_lora_config_contract() -> None:
    config = _config.get_config("pi05_franka_lora")

    assert config.model.model_type == _config.ModelType.PI05
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 16
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"
    assert isinstance(config.data, _config.LeRobotFrankaDataConfig)
    assert config.data.repo_id == _config.FRANKA_REPO_ID
    assert config.data.use_delta_joint_actions
    assert config.batch_size == 8
    assert config.num_workers == 0
    assert config.num_train_steps == 100
    assert config.ema_decay is None
    assert config.policy_metadata is not None
    assert config.policy_metadata["action_semantics"] == "absolute_joint_position_and_gripper_width"
    assert config.policy_metadata["real_robot_deployment_allowed"] is False


def test_franka_lerobot_repack_supports_optional_wrist() -> None:
    factory = _config.LeRobotFrankaDataConfig(repo_id="test/franka")
    repack = factory.repack_transforms.inputs[0]
    base = np.zeros((32, 48, 3), dtype=np.uint8)
    wrist = np.ones((24, 32, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    actions = np.zeros((16, 8), dtype=np.float32)

    result = repack(
        {
            "observation.images.base": base,
            "observation.images.wrist": wrist,
            "observation.state": state,
            "action": actions,
            "prompt": "Demonstrate a task.",
        }
    )

    np.testing.assert_array_equal(result["images/base"], base)
    np.testing.assert_array_equal(result["images/wrist"], wrist)
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)

    base_only = repack(
        {
            "observation.images.base": base,
            "observation.state": state,
            "action": actions,
            "prompt": "Demonstrate a task.",
        }
    )
    assert "images/wrist" not in base_only
    np.testing.assert_array_equal(base_only["images/base"], base)
