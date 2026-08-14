import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hdf5 = _load("red_cube_to_box_hdf5")
audit = _load("audit_red_cube_to_box_hdf5")


def _sample(index: int = 0) -> dict:
    return {
        "actions": np.full(6, index + 0.5, dtype=np.float32),
        "obs/joint_pos": np.full(6, index, dtype=np.float32),
        "obs/front": np.full((8, 12, 3), index, dtype=np.uint8),
        "diagnostics/joint_vel": np.zeros(6, dtype=np.float32),
        "timestamps": np.float64(index / 60),
    }


def test_success_is_promoted_and_failure_retains_only_summary(tmp_path: Path) -> None:
    writer = hdf5.ShardedDatasetWriter(tmp_path, shard_size=2, metadata={"expert": "polar"})
    success = writer.begin_attempt({"seed": 42})
    success.append(_sample(0))
    success.append(_sample(1))
    assert success.finish(success=True, summary={"reason": None}) == "demo_0"

    failure = writer.begin_attempt({"seed": 43})
    failure.append(_sample(0))
    assert failure.finish(success=False, summary={"reason": "abort"}) is None

    with h5py.File(tmp_path / "part-000.hdf5", "r") as source:
        assert list(source["data"]) == ["demo_0"]
        assert len(source["attempts"]) == 2
        assert list(source["_staging"]) == []
        assert source["attempts/attempt_1"].attrs["num_samples"] == 1
        assert "obs" not in source["attempts/attempt_1"]
    manifest = json.loads((tmp_path / "collection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["successful_episodes"] == 1
    assert manifest["attempts"] == 2
    assert audit.audit_dataset(tmp_path)["failed_attempts"] == 1


def test_resume_removes_incomplete_staging_and_preserves_counts(tmp_path: Path) -> None:
    writer = hdf5.ShardedDatasetWriter(tmp_path)
    episode = writer.begin_attempt()
    episode.append(_sample())
    episode._h5_file.close()  # noqa: SLF001 - deliberately simulates process death

    resumed = hdf5.ShardedDatasetWriter(tmp_path)

    assert resumed.recovery.removed_staging_groups == 1
    assert resumed.attempts == 0
    assert resumed.successful_episodes == 0


def test_shards_roll_after_configured_success_count(tmp_path: Path) -> None:
    writer = hdf5.ShardedDatasetWriter(tmp_path, shard_size=1)
    for _ in range(2):
        episode = writer.begin_attempt()
        episode.append(_sample())
        episode.finish(success=True, summary={})

    assert sorted(path.name for path in tmp_path.glob("*.hdf5")) == ["part-000.hdf5", "part-001.hdf5"]
