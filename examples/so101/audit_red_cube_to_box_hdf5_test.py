from pathlib import Path

import h5py
import numpy as np
import pytest

from examples.so101 import audit_red_cube_to_box_hdf5 as audit


def test_audit_is_read_only_and_rejects_incomplete_staging(tmp_path: Path) -> None:
    path = tmp_path / "part-000.hdf5"
    with h5py.File(path, "w") as target:
        target.attrs["schema_version"] = "so101-red-cube-v1"
        target.create_group("data")
        target.create_group("attempts")
        target.create_group("_staging").create_group("interrupted")
    before = path.read_bytes()

    with pytest.raises(ValueError, match="incomplete staging"):
        audit.audit_dataset(tmp_path)

    assert path.read_bytes() == before


def test_audit_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "part-000.hdf5"
    with h5py.File(path, "w") as target:
        target.attrs["schema_version"] = "so101-red-cube-v1"
        data = target.create_group("data")
        target.create_group("attempts")
        target.create_group("_staging")
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 2
        demo.create_dataset("actions", data=np.zeros((2, 6), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("joint_pos", data=np.zeros((2, 6), dtype=np.float32))
        obs.create_dataset("front", data=np.zeros((2, 8, 12, 3), dtype=np.uint8))
        demo.create_dataset("timestamps", data=np.zeros(2, dtype=np.float64))

    with pytest.raises(ValueError, match="timestamps"):
        audit.audit_dataset(tmp_path)
