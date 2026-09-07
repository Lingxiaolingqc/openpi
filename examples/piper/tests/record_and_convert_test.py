import json
from pathlib import Path
import sys
from types import ModuleType

import h5py
import numpy as np
import pytest

from examples.piper import contract
from examples.piper import convert_hdf5_to_lerobot
from examples.piper import mark_simulation_only_checkpoint
from examples.piper import record
from examples.piper.expert import Phase


def _write_episode(path: Path, *, seed: int) -> None:
    metadata = record.EpisodeMetadata(seed=seed, model_sha256="a" * 64, menagerie_revision="deadbeef")
    state = np.concatenate([contract.HOME_ARM_Q_RAD, [1.0]]).astype(np.float32)
    image = np.zeros((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3), dtype=np.uint8)
    with record.PiperEpisodeRecorder(path, metadata) as recorder:
        for index in range(3):
            observation = contract.Observation(image=image, state=state, timestamp_ns=1_000_000 + index)
            recorder.append(observation, state, Phase.PREGRASP)
        recorder.finalize(success=True, final_phase=Phase.RELEASE.value, failure_reason=None)


def test_audit_and_episode_level_split(tmp_path: Path) -> None:
    for seed in range(5):
        _write_episode(tmp_path / f"episode_{seed}.h5", seed=seed)
    episodes = convert_hdf5_to_lerobot.discover_episodes(tmp_path)
    train, validation = convert_hdf5_to_lerobot.split_episodes(episodes, train_fraction=0.8, split_seed=7)
    assert len(train) == 4
    assert len(validation) == 1
    assert {episode.seed for episode in train}.isdisjoint(episode.seed for episode in validation)


def test_training_episode_limit_is_deterministic_and_strict(tmp_path: Path) -> None:
    for seed in range(6):
        _write_episode(tmp_path / f"episode_{seed}.h5", seed=seed)
    episodes = convert_hdf5_to_lerobot.discover_episodes(tmp_path)
    train, _ = convert_hdf5_to_lerobot.split_episodes(episodes, train_fraction=0.8, split_seed=7)
    selected = convert_hdf5_to_lerobot.limit_training_episodes(train, 2)
    assert selected == train[:2]
    with pytest.raises(ValueError, match="exceeds available"):
        convert_hdf5_to_lerobot.limit_training_episodes(train, len(train) + 1)


def test_audit_rejects_nonfinite_action(tmp_path: Path) -> None:
    path = tmp_path / "episode.h5"
    _write_episode(path, seed=1)
    with h5py.File(path, "r+") as file:
        file["actions/absolute_target"][1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        convert_hdf5_to_lerobot.validate_episode(path)


def test_conversion_writes_only_training_frames_and_audit_artifacts(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    _write_episode(source / "train.h5", seed=1)
    _write_episode(source / "validation.h5", seed=2)
    episodes = convert_hdf5_to_lerobot.discover_episodes(source)

    class FakeLeRobotDataset:
        last = None

        def __init__(self) -> None:
            self.frames: list[dict] = []
            self.saved_episodes = 0
            self.consolidated = False

        @classmethod
        def create(cls, **_kwargs):
            cls.last = cls()
            return cls.last

        def add_frame(self, frame: dict) -> None:
            self.frames.append(frame)

        def save_episode(self) -> None:
            self.saved_episodes += 1

        def consolidate(self) -> None:
            self.consolidated = True

    lerobot = ModuleType("lerobot")
    common = ModuleType("lerobot.common")
    datasets = ModuleType("lerobot.common.datasets")
    dataset_module = ModuleType("lerobot.common.datasets.lerobot_dataset")
    dataset_module.HF_LEROBOT_HOME = str(tmp_path / "lerobot-home")
    dataset_module.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.common", common)
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets.lerobot_dataset", dataset_module)

    output = convert_hdf5_to_lerobot.convert_training_split(
        [episodes[0]], [episodes[1]], repo_id="local/test-piper", split_seed=9, image_mode="image"
    )
    fake = FakeLeRobotDataset.last
    assert fake is not None
    assert len(fake.frames) == episodes[0].frames
    assert fake.saved_episodes == 1
    assert fake.consolidated
    assert all(frame["task"] == contract.TASK_PROMPT for frame in fake.frames)
    split = json.loads((output / "episode_split.json").read_text(encoding="utf-8"))
    assert [item["seed"] for item in split["train"]] == [1]
    assert [item["seed"] for item in split["validation"]] == [2]
    marker = json.loads((output / "SIMULATION_ONLY.json").read_text(encoding="utf-8"))
    assert marker["real_robot_deployment_allowed"] is False


def test_checkpoint_marker_is_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    dataset.mkdir()
    checkpoint.mkdir()
    artifacts = {
        "dataset_contract.json": {"robot_type": contract.ROBOT_TYPE},
        "episode_split.json": {"train": [], "validation": []},
        "SIMULATION_ONLY.json": {
            "deployment_scope": "simulation-only",
            "real_robot_deployment_allowed": False,
        },
    }
    for name, value in artifacts.items():
        (dataset / name).write_text(__import__("json").dumps(value), encoding="utf-8")
    mark_simulation_only_checkpoint.mark_checkpoint(checkpoint, dataset)
    mark_simulation_only_checkpoint.mark_checkpoint(checkpoint, dataset)
    (dataset / "dataset_contract.json").write_text('{"robot_type": "other"}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="conflicting"):
        mark_simulation_only_checkpoint.mark_checkpoint(checkpoint, dataset)
