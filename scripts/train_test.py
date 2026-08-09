import dataclasses
import os
import pathlib
from unittest import mock

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


def test_checkpoint_manager_is_closed_when_training_setup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],  # noqa: SLF001
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="close_on_failure",
        wandb_enabled=False,
    )
    checkpoint_manager = mock.Mock()

    monkeypatch.setattr(
        train._checkpoints,  # noqa: SLF001
        "initialize_checkpoint_dir",
        lambda *args, **kwargs: (checkpoint_manager, False),
    )
    monkeypatch.setattr(train, "init_wandb", mock.Mock(side_effect=RuntimeError("setup failed")))

    with pytest.raises(RuntimeError, match="setup failed"):
        train.main(config)

    checkpoint_manager.close.assert_called_once_with()
