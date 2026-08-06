import numpy as np

from openpi.training import config as _config


def test_pi05_so101_liftcube_config_contract() -> None:
    config = _config.get_config("pi05_so101_liftcube")

    assert config.model.model_type == _config.ModelType.PI05
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 10
    assert config.model.discrete_state_input
    assert isinstance(config.data, _config.LeRobotSO101DataConfig)
    assert config.data.action_sequence_keys == ("action",)
    assert config.data.use_delta_joint_actions
    assert config.data.base_config is not None
    assert config.data.base_config.prompt_from_task


def test_pi05_lora_so101_liftcube_config_contract() -> None:
    config = _config.get_config("pi05_lora_so101_liftcube")

    assert config.model.model_type == _config.ModelType.PI05
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 10
    assert config.model.discrete_state_input
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"
    assert isinstance(config.data, _config.LeRobotSO101DataConfig)
    assert config.batch_size == 8
    assert config.ema_decay is None


def test_so101_lerobot_repack_matches_leisaac_frame_keys() -> None:
    factory = _config.LeRobotSO101DataConfig(repo_id="test/leisaac_so101_liftcube")
    repack = factory.repack_transforms.inputs[0]
    front = np.zeros((480, 640, 3), dtype=np.uint8)
    state = np.arange(6, dtype=np.float32)
    actions = np.arange(60, dtype=np.float32).reshape(10, 6)

    result = repack(
        {
            "observation.images.front": front,
            "observation.state": state,
            "action": actions,
            "prompt": "Lift the red cube up.",
        }
    )

    np.testing.assert_array_equal(result["images/front"], front)
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)
    assert result["prompt"] == "Lift the red cube up."
