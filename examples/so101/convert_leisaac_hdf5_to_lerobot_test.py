from pathlib import Path

import h5py
import numpy as np
import pytest

from examples.so101 import convert_leisaac_hdf5_to_lerobot as converter


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


def _write_demo(data: h5py.Group, name: str, *, success: bool, num_samples: int = 3) -> None:
    demo = data.create_group(name)
    demo.attrs["success"] = success
    demo.attrs["num_samples"] = num_samples
    demo.create_dataset("actions", data=np.zeros((num_samples, 6), dtype=np.float32))
    obs = demo.create_group("obs")
    obs.create_dataset("joint_pos", data=np.zeros((num_samples, 6), dtype=np.float32))
    obs.create_dataset("front", data=np.zeros((num_samples, 8, 12, 3), dtype=np.uint8))


def test_discovery_validates_and_skips_failed_episodes(tmp_path: Path) -> None:
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as h5_file:
        data = h5_file.create_group("data")
        _write_demo(data, "demo_0", success=True)
        _write_demo(data, "demo_1", success=False)

    episodes, skipped_failures, file_count = converter.discover_successful_episodes(source)

    assert file_count == 1
    assert skipped_failures == 1
    assert episodes == [converter.EpisodeRef(source.resolve(), "demo_0", 3, (8, 12, 3))]
