"""Read-only integrity audit for sharded SO-101 RedCube HDF5 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.so101.utils.red_cube_to_box_hdf5 import JOINT_NAMES
from examples.so101.utils.red_cube_to_box_hdf5 import SCHEMA_VERSION


def audit_dataset(root: Path) -> dict[str, int | tuple[int, ...]]:
    root = Path(root).expanduser().resolve()
    shards = sorted(root.glob("part-*.hdf5"))
    if not shards:
        raise ValueError(f"No part-*.hdf5 shards found in {root}")
    successes = attempts = failures = frames = staging = 0
    image_shape: tuple[int, ...] | None = None
    for path in shards:
        with h5py.File(path, "r") as h5_file:
            if h5_file.attrs.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"{path} has an unsupported schema version")
            staging += len(h5_file.get("_staging", {}))
            attempt_group = h5_file.get("attempts", {})
            attempts += len(attempt_group)
            failures += sum(not bool(group.attrs.get("success", False)) for group in attempt_group.values())
            for name, demo in h5_file.get("data", {}).items():
                required = ("actions", "obs/joint_pos", "obs/front", "timestamps")
                missing = [field for field in required if field not in demo]
                if missing:
                    raise ValueError(f"{path}:{name} missing {missing}")
                length = int(demo.attrs.get("num_samples", -1))
                if length <= 0 or demo["actions"].shape != (length, len(JOINT_NAMES)):
                    raise ValueError(f"{path}:{name} has invalid action length/shape")
                if demo["obs/joint_pos"].shape != demo["actions"].shape:
                    raise ValueError(f"{path}:{name} state/action shape mismatch")
                front = demo["obs/front"]
                if front.shape[0] != length or front.shape[-1] != 3 or front.dtype != np.uint8:
                    raise ValueError(f"{path}:{name} has invalid front images")
                if "obs/wrist" in demo:
                    wrist = demo["obs/wrist"]
                    if wrist.shape[0] != length or wrist.ndim != 4 or wrist.shape[-1] != 3 or wrist.dtype != np.uint8:
                        raise ValueError(f"{path}:{name} has invalid wrist images")
                current_shape = tuple(front.shape[1:])
                if image_shape is not None and current_shape != image_shape:
                    raise ValueError("Front image shapes differ between episodes")
                image_shape = current_shape
                timestamps = demo["timestamps"][:]
                if not np.isfinite(demo["actions"][:]).all() or not np.isfinite(demo["obs/joint_pos"][:]).all():
                    raise ValueError(f"{path}:{name} contains non-finite joint data")
                if (
                    timestamps.shape != (length,)
                    or not np.isfinite(timestamps).all()
                    or np.any(np.diff(timestamps) <= 0)
                ):
                    raise ValueError(f"{path}:{name} has invalid timestamps")
                successes += 1
                frames += length
    if staging:
        raise ValueError(f"Dataset contains {staging} incomplete staging groups")
    return {
        "shards": len(shards),
        "successful_episodes": successes,
        "attempts": attempts,
        "failed_attempts": failures,
        "frames": frames,
        "front_image_shape": image_shape or (),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    result = audit_dataset(args.dataset_dir)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("RED_CUBE_TO_BOX_HDF5_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
