import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import h5py
import numpy as np
import pytest

MODULE_PATH = Path(__file__).with_name("convert_leisaac_hdf5_to_lerobot.py")
MODULE_SPEC = importlib.util.spec_from_file_location("convert_leisaac_hdf5_to_lerobot", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load converter module from {MODULE_PATH}")
converter = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = converter
MODULE_SPEC.loader.exec_module(converter)


def test_joint_limit_endpoints_map_to_motor_limits() -> None:
    joint_limit_radians = np.deg2rad(converter.USD_JOINT_LIMITS_DEGREES.T)
    converted = converter.leisaac_radians_to_motor_degrees(joint_limit_radians)

    np.testing.assert_allclose(converted[0], converter.MOTOR_LIMITS_DEGREES[:, 0])
    np.testing.assert_allclose(converted[1], converter.MOTOR_LIMITS_DEGREES[:, 1])


def test_conversion_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="final dimension"):
        converter.leisaac_radians_to_motor_degrees(np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or infinity"):
        converter.leisaac_radians_to_motor_degrees(np.full((1, 6), np.nan, dtype=np.float32))


def test_audit_counts_but_does_not_clip_out_of_range_action_targets(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5_file:
        data = h5_file.create_group("data")
        _write_demo(data, "demo_0", success=True, num_samples=1)
        data["demo_0/actions"][0, 3] = np.deg2rad(120.0)
    episodes, _, _ = converter.discover_successful_episodes(source)

    *_, violations = converter.audit_ranges(episodes)

    assert tuple(violations) == (0, 0, 0, 1, 0, 0)
    converted = converter.leisaac_radians_to_motor_degrees(np.asarray([[0, 0, 0, np.deg2rad(120), 0, 0]]))
    assert converted[0, 3] > 100.0


def test_audit_start_frame_excludes_skipped_initial_action(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5_file:
        data = h5_file.create_group("data")
        _write_demo(data, "demo_0", success=True, num_samples=2)
        data["demo_0/actions"][0, 3] = np.deg2rad(120.0)
    episodes, _, _ = converter.discover_successful_episodes(source)

    *_, all_violations = converter.audit_ranges(episodes)
    *_, converted_violations = converter.audit_ranges(episodes, start_frame=1)

    assert tuple(all_violations) == (0, 0, 0, 1, 0, 0)
    assert tuple(converted_violations) == (0, 0, 0, 0, 0, 0)


def test_validate_start_frame_requires_a_retained_sample(tmp_path: Path) -> None:
    episodes = [converter.EpisodeRef(tmp_path / "source.hdf5", "demo_0", 1, (8, 12, 3), None)]

    converter.validate_start_frame(episodes, 0)
    with pytest.raises(ValueError, match="non-negative"):
        converter.validate_start_frame(episodes, -1)
    with pytest.raises(ValueError, match="leaves no samples"):
        converter.validate_start_frame(episodes, 1)


def _write_demo(
    data: h5py.Group, name: str, *, success: bool, num_samples: int = 3, include_wrist: bool = False
) -> None:
    demo = data.create_group(name)
    demo.attrs["success"] = success
    demo.attrs["num_samples"] = num_samples
    demo.create_dataset("actions", data=np.zeros((num_samples, 6), dtype=np.float32))
    obs = demo.create_group("obs")
    obs.create_dataset("joint_pos", data=np.zeros((num_samples, 6), dtype=np.float32))
    obs.create_dataset("front", data=np.zeros((num_samples, 8, 12, 3), dtype=np.uint8))
    if include_wrist:
        obs.create_dataset("wrist", data=np.zeros((num_samples, 6, 10, 3), dtype=np.uint8))


def test_discovery_validates_and_skips_failed_episodes(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5_file:
        data = h5_file.create_group("data")
        _write_demo(data, "demo_0", success=True)
        _write_demo(data, "demo_1", success=False)

    episodes, skipped_failures, file_count = converter.discover_successful_episodes(source)

    assert file_count == 1
    assert skipped_failures == 1
    assert episodes == [converter.EpisodeRef(source.resolve(), "demo_0", 3, (8, 12, 3), None)]


def test_discovery_and_conversion_contract_include_optional_wrist_camera(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5_file:
        data = h5_file.create_group("data")
        _write_demo(data, "demo_0", success=True, include_wrist=True)

    episodes, _, _ = converter.discover_successful_episodes(source)

    assert episodes[0].wrist_image_shape == (6, 10, 3)


def test_conversion_uses_openpi_pinned_lerobot_writer_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5_file:
        data = h5_file.create_group("data")
        _write_demo(data, "demo_0", success=True, include_wrist=True)
        data["demo_0/obs/front"][0] = 10
        data["demo_0/obs/front"][1] = 20
        data["demo_0/obs/front"][2] = 30
        data["demo_0/obs/joint_pos"][1] = np.deg2rad(np.arange(6, dtype=np.float32))
        data["demo_0/actions"][1] = np.deg2rad(np.arange(6, dtype=np.float32) + 1)

    episodes, _, _ = converter.discover_successful_episodes(source)
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

        def add_frame(self, frame: dict) -> None:
            self.frames.append(frame)

        def save_episode(self) -> None:
            self.saved_episode_count += 1

    package_names = ("lerobot", "lerobot.common", "lerobot.common.datasets")
    for package_name in package_names:
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    dataset_module = ModuleType("lerobot.common.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = FakeLeRobotDataset
    dataset_module.HF_LEROBOT_HOME = lerobot_home
    monkeypatch.setitem(sys.modules, dataset_module.__name__, dataset_module)

    output_path = converter.convert_dataset(
        episodes,
        repo_id="local/test-dataset",
        task="Lift the cube.",
        fps=60,
        image_mode="image",
        image_writer_processes=0,
        image_writer_threads=1,
        push_to_hub=False,
        private=True,
        start_frame=1,
    )

    dataset = FakeLeRobotDataset.instance
    assert dataset is not None
    assert output_path == lerobot_home / "local/test-dataset"
    assert FakeLeRobotDataset.create_kwargs["robot_type"] == "so101_follower"
    assert "observation.images.wrist" in FakeLeRobotDataset.create_kwargs["features"]
    assert "task" not in FakeLeRobotDataset.create_kwargs["features"]
    assert len(dataset.frames) == 2
    assert np.all(dataset.frames[0]["observation.images.front"] == 20)
    expected_state = converter.leisaac_radians_to_motor_degrees(
        np.deg2rad(np.arange(6, dtype=np.float32))[None]
    )[0]
    expected_action = converter.leisaac_radians_to_motor_degrees(
        np.deg2rad(np.arange(6, dtype=np.float32) + 1)[None]
    )[0]
    np.testing.assert_allclose(dataset.frames[0]["observation.state"], expected_state)
    np.testing.assert_allclose(dataset.frames[0]["action"], expected_action)
    assert all(frame["task"] == "Lift the cube." for frame in dataset.frames)
    assert all("observation.images.wrist" in frame for frame in dataset.frames)
    assert dataset.saved_episode_count == 1
