from pathlib import Path
import sys
import time
from types import ModuleType

import h5py
import pytest

from examples.franka import backend
from examples.franka import contract
from examples.franka import convert_franka_hdf5_to_lerobot as converter
from examples.franka import record


def _write_episode(path: Path, *, success: bool = True, include_wrist: bool = True) -> None:
    robot = backend.MockFrankaBackend(include_wrist=include_wrist)
    metadata = record.EpisodeMetadata(
        task="Generic manipulation demonstration.",
        camera_calibration_sha256={"base": "a" * 64, **({"wrist": "b" * 64} if include_wrist else {})},
        robot_system_version="5.9.0",
        software_versions={"franka_ros2": "3.4.0", "libfranka": "0.20.4"},
    )
    recorder = record.FrankaEpisodeRecorder(path, metadata)
    for _ in range(3):
        snapshot = robot.read_snapshot()
        target = contract.FrankaTarget(snapshot.q_rad.copy(), snapshot.gripper_width_m)
        recorder.append(snapshot, target)
        time.sleep(0.001)
    recorder.finalize(success=success)


def test_recorder_and_dry_run_audit(tmp_path: Path) -> None:
    episode = tmp_path / "episode_000.h5"
    _write_episode(episode)

    episodes, skipped = converter.discover_episodes(episode)

    assert skipped == 0
    assert episodes == [
        converter.EpisodeRef(episode.resolve(), 3, (64, 96, 3), (48, 64, 3), "Generic manipulation demonstration.")
    ]
    with h5py.File(episode, "r") as h5_file:
        assert h5_file.attrs["observation_alignment"] == "pre_step"
        assert h5_file["actions/q_target"].shape == (3, 7)


def test_failed_episodes_are_skipped_by_default(tmp_path: Path) -> None:
    _write_episode(tmp_path / "success.h5", success=True)
    _write_episode(tmp_path / "failure.h5", success=False)

    episodes, skipped = converter.discover_episodes(tmp_path)

    assert len(episodes) == 1
    assert skipped == 1


def test_episode_metadata_and_observation_action_alignment_are_audited(tmp_path: Path) -> None:
    with pytest.raises(contract.ContractError, match="SHA-256"):
        record.EpisodeMetadata(
            task="Test.",
            camera_calibration_sha256={"base": "not-a-digest"},
            robot_system_version="5.9.0",
            software_versions={"franka_ros2": "3.4.0", "libfranka": "0.20.4"},
        )

    episode = tmp_path / "misaligned.h5"
    _write_episode(episode, include_wrist=False)
    with h5py.File(episode, "r+") as h5_file:
        h5_file["actions/q_target"][1, 0] = 1.0
    with pytest.raises(ValueError, match="continuity or alignment"):
        converter.discover_episodes(episode)


def test_converter_uses_pinned_lerobot_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode = tmp_path / "episode.h5"
    _write_episode(episode)
    episodes, _ = converter.discover_episodes(episode)
    lerobot_home = tmp_path / "lerobot"

    class FakeLeRobotDataset:
        instance = None
        create_kwargs = None

        def __init__(self) -> None:
            self.frames = []
            self.saved_episode_count = 0

        @classmethod
        def create(cls, **kwargs):
            cls.create_kwargs = kwargs
            cls.instance = cls()
            return cls.instance

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved_episode_count += 1

    for package_name in ("lerobot", "lerobot.common", "lerobot.common.datasets"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    dataset_module = ModuleType("lerobot.common.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = FakeLeRobotDataset
    dataset_module.HF_LEROBOT_HOME = lerobot_home
    monkeypatch.setitem(sys.modules, dataset_module.__name__, dataset_module)

    output = converter.convert_dataset(
        episodes,
        repo_id="local/franka-test",
        image_mode="image",
        image_writer_processes=0,
        image_writer_threads=1,
    )

    dataset = FakeLeRobotDataset.instance
    assert output == lerobot_home / "local/franka-test"
    assert dataset is not None
    assert FakeLeRobotDataset.create_kwargs["robot_type"] == "fr3"
    assert "observation.images.wrist" in FakeLeRobotDataset.create_kwargs["features"]
    assert len(dataset.frames) == 3
    assert dataset.frames[0]["observation.state"].shape == (8,)
    assert dataset.frames[0]["action"].shape == (8,)
    assert dataset.saved_episode_count == 1
    assert (output / "franka_conversion_manifest.json").is_file()
