# ruff: noqa: PT009, PT018, PT027

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, ClassVar
import unittest

import h5py
import numpy as np

MODULE_PATH = Path(__file__).with_name("convert_real_hdf5_to_lerobot.py")
SPEC = importlib.util.spec_from_file_location("convert_real_hdf5_to_lerobot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
TEST_TEMP_ROOT = MODULE_PATH.parents[3] / "tmp" / "so101-real-tests"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def _write_episode(
    path: Path,
    contract: Any,
    *,
    box_id: str = "A",
    target_id: int = 1,
    action_value: float = 10.0,
    frame_age_s: float = 0.010,
) -> None:
    frames = 30
    control = 100.0 + np.arange(frames, dtype=np.float64) / MODULE.EXPECTED_CONTROL_RATE_HZ
    with h5py.File(path, "w") as file:
        file.attrs["schema_version"] = 1
        file.attrs["joint_order"] = json.dumps(MODULE.JOINT_ORDER)
        file.attrs["success"] = 1
        file.attrs["frame_count"] = frames
        file.attrs["action_mode"] = MODULE.EXPECTED_ACTION_MODE
        file.attrs["control_rate_hz"] = MODULE.EXPECTED_CONTROL_RATE_HZ
        file.attrs["calibration_freeze_id"] = contract.freeze_id
        file.attrs["calibration_hashes"] = json.dumps(contract.calibration_hashes)
        file.attrs["target_id"] = target_id
        file.attrs["task"] = "Put the sponge in the box."
        file.attrs["box_id"] = box_id
        obs = file.create_group("obs")
        obs.create_dataset("joint_pos", data=np.zeros((frames, 6), dtype=np.float32))
        obs.create_dataset("front", shape=(frames, *MODULE.EXPECTED_IMAGE_SHAPE), dtype="uint8")
        obs.create_dataset("wrist", shape=(frames, *MODULE.EXPECTED_IMAGE_SHAPE), dtype="uint8")
        file.create_dataset("actions", data=np.full((frames, 6), action_value, dtype=np.float32))
        timestamps = file.create_group("timestamps")
        timestamps.create_dataset("control", data=control)
        timestamps.create_dataset("front", data=control - frame_age_s)
        timestamps.create_dataset("wrist", data=control - frame_age_s / 2)


class FakeLeRobotDataset:
    output_root: ClassVar[Path]
    instances: ClassVar[list[FakeLeRobotDataset]] = []

    def __init__(self, *, repo_id: str, create_kwargs: dict[str, Any]) -> None:
        self.repo_id = repo_id
        self.create_kwargs = create_kwargs
        self.first_frame: dict[str, Any] | None = None
        self.current_frame_count = 0
        self.episode_frame_counts: list[int] = []
        self.consolidated = False
        (self.output_root / repo_id).mkdir(parents=True)
        self.__class__.instances.append(self)

    @classmethod
    def create(cls, **kwargs: Any) -> FakeLeRobotDataset:
        repo_id = kwargs.pop("repo_id")
        return cls(repo_id=repo_id, create_kwargs=kwargs)

    def add_frame(self, frame: dict[str, Any]) -> None:
        if self.first_frame is None:
            self.first_frame = {
                key: np.asarray(value).copy() if key != "task" else value for key, value in frame.items()
            }
        self.current_frame_count += 1

    def save_episode(self) -> None:
        self.episode_frame_counts.append(self.current_frame_count)
        self.current_frame_count = 0

    def consolidate(self) -> None:
        self.consolidated = True


class ConvertRealHDF5ToLeRobotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = MODULE.load_calibration_contract()

    def setUp(self) -> None:
        FakeLeRobotDataset.instances.clear()

    def test_conversion_preserves_motor_degrees_and_episode_conditions(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            episodes_dir = root / "episodes"
            episodes_dir.mkdir()
            _write_episode(episodes_dir / "episode.h5", self.contract, action_value=10.0)
            episodes, skipped, file_count = MODULE.discover_real_episodes(root, self.contract)
            FakeLeRobotDataset.output_root = root / "lerobot"

            output = MODULE.convert_dataset(
                episodes,
                repo_id="local/so101_real_test",
                image_mode="image",
                image_writer_processes=0,
                image_writer_threads=1,
                push_to_hub=False,
                private=True,
                contract=self.contract,
                dataset_class=FakeLeRobotDataset,
                output_root=FakeLeRobotDataset.output_root,
            )

            self.assertEqual(skipped, 0)
            self.assertEqual(file_count, 1)
            dataset = FakeLeRobotDataset.instances[0]
            self.assertEqual(dataset.create_kwargs["fps"], 30)
            self.assertEqual(dataset.create_kwargs["features"]["observation.state"]["shape"], (6,))
            self.assertEqual(dataset.episode_frame_counts, [30])
            self.assertTrue(dataset.consolidated)
            np.testing.assert_array_equal(dataset.first_frame["action"], np.full(6, 10.0, dtype=np.float32))
            np.testing.assert_array_equal(dataset.first_frame["target_id"], np.asarray([1], dtype=np.int64))
            np.testing.assert_array_equal(dataset.first_frame["box_id"], np.asarray([0], dtype=np.int64))
            self.assertEqual(dataset.first_frame["task"], "Put the sponge in the box.")
            manifest = json.loads((output / "real_conversion_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_format"], "so101_real_hdf5_v1")
            self.assertFalse(manifest["box_id_is_model_input"])
            self.assertEqual(manifest["episodes"][0]["box_id"], "A")

    def test_box_c_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            source = Path(directory) / "box-c.h5"
            _write_episode(source, self.contract, box_id="C")
            with self.assertRaisesRegex(ValueError, "forbidden box_id"):
                MODULE.discover_real_episodes(source, self.contract)

    def test_motor_values_outside_frozen_calibration_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            source = Path(directory) / "bad-action.h5"
            _write_episode(source, self.contract, action_value=500.0)
            with self.assertRaisesRegex(ValueError, "action exceeds frozen calibration"):
                MODULE.discover_real_episodes(source, self.contract)

    def test_stale_camera_timestamp_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            source = Path(directory) / "stale-camera.h5"
            _write_episode(source, self.contract, frame_age_s=0.101)
            with self.assertRaisesRegex(ValueError, "maximum camera frame age"):
                MODULE.discover_real_episodes(source, self.contract)

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            root = Path(directory)
            source = root / "episode.h5"
            _write_episode(source, self.contract)
            episodes, _, _ = MODULE.discover_real_episodes(source, self.contract)
            output_root = root / "lerobot"
            (output_root / "local" / "existing").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                MODULE.convert_dataset(
                    episodes,
                    repo_id="local/existing",
                    image_mode="image",
                    image_writer_processes=0,
                    image_writer_threads=1,
                    push_to_hub=False,
                    private=True,
                    contract=self.contract,
                    dataset_class=FakeLeRobotDataset,
                    output_root=output_root,
                )


if __name__ == "__main__":
    unittest.main()
