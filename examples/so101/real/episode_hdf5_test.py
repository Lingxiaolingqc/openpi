# ruff: noqa: PT009, PT018, PT027

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

MODULE_PATH = Path(__file__).with_name("episode_hdf5.py")
SPEC = importlib.util.spec_from_file_location("episode_hdf5", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
TEST_TEMP_ROOT = MODULE_PATH.parents[3] / "tmp" / "so101-real-tests"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def _sample(value: int) -> object:
    return MODULE.EpisodeSample(
        joint_pos=np.full(6, value, dtype=np.float32),
        front_rgb=np.full((4, 5, 3), value, dtype=np.uint8),
        wrist_rgb=np.full((4, 5, 3), value + 1, dtype=np.uint8),
        action=np.full(6, value + 2, dtype=np.float32),
        control_timestamp_s=10.0 + value,
        front_timestamp_s=10.1 + value,
        wrist_timestamp_s=10.2 + value,
    )


class EpisodeHDF5Test(unittest.TestCase):
    def test_success_is_incremental_and_atomically_published(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            writer = MODULE.IncrementalEpisodeWriter(
                dataset_root=Path(directory),
                episode_id="episode-001",
                metadata={"target_id": 1, "camera_config": {"front": 2}},
                image_shape=(4, 5, 3),
                compression=None,
                flush_every=1,
            )
            for value in range(3):
                writer.append(_sample(value))
            path = writer.finish(success=True, final_metadata={"quality": "ok"})

            self.assertEqual(path.parent.name, "episodes")
            self.assertFalse(writer.partial_path.exists())
            with h5py.File(path, "r") as file:
                self.assertEqual(file["obs/joint_pos"].shape, (3, 6))
                self.assertEqual(file["obs/front"].shape, (3, 4, 5, 3))
                self.assertEqual(file["obs/wrist"].dtype, np.dtype("uint8"))
                self.assertEqual(file["actions"].shape, (3, 6))
                self.assertEqual(file["timestamps/control"].shape, (3,))
                self.assertEqual(file.attrs["success"], 1)
                self.assertEqual(file.attrs["frame_count"], 3)
                self.assertEqual(file.attrs["quality"], "ok")

    def test_rejected_episode_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            writer = MODULE.IncrementalEpisodeWriter(
                dataset_root=Path(directory),
                episode_id="episode-002",
                metadata={},
                image_shape=(4, 5, 3),
                compression=None,
            )
            writer.append(_sample(0))
            path = writer.finish(success=False, abort_reason="operator_discard")

            self.assertEqual(path.parent.name, "rejected")
            with h5py.File(path, "r") as file:
                self.assertEqual(file.attrs["success"], 0)
                self.assertEqual(file.attrs["abort_reason"], "operator_discard")

    def test_invalid_sample_is_rejected_before_queueing(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            writer = MODULE.IncrementalEpisodeWriter(
                dataset_root=Path(directory),
                episode_id="episode-003",
                metadata={},
                image_shape=(4, 5, 3),
                compression=None,
            )
            invalid = MODULE.EpisodeSample(
                joint_pos=np.zeros(5, dtype=np.float32),
                front_rgb=np.zeros((4, 5, 3), dtype=np.uint8),
                wrist_rgb=np.zeros((4, 5, 3), dtype=np.uint8),
                action=np.zeros(6, dtype=np.float32),
                control_timestamp_s=0.0,
                front_timestamp_s=0.0,
                wrist_timestamp_s=0.0,
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                writer.append(invalid)
            path = writer.finish(success=False, abort_reason="invalid_test")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
