from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.so101.act import common


def _contract(tmp_path: Path, *, episodes: int = 20) -> common.DatasetContract:
    location = common.resolve_dataset_location(tmp_path, "local/example")
    meta = location.dataset_path / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}\n", encoding="utf-8")
    metadata = SimpleNamespace(
        features={
            "observation.state": {
                "shape": [6],
                "names": list(common.JOINT_NAMES),
            },
            "action": {"shape": [6], "names": list(common.JOINT_NAMES)},
            "observation.images.front": {"shape": [480, 640, 3]},
        },
        camera_keys=["observation.images.front"],
        info={"fps": 60, "total_episodes": episodes, "total_frames": episodes * 100},
    )
    return common.build_dataset_contract(
        location,
        metadata,
        action_semantics=common.ACTION_SEMANTICS,
        expected_camera_keys=("observation.images.front",),
    )


def test_dataset_contract_and_split_are_metadata_driven(tmp_path: Path) -> None:
    contract = _contract(tmp_path, episodes=20)
    split = common.deterministic_episode_split(contract, validation_fraction=0.2, seed=42)

    assert contract.num_episodes == 20
    assert contract.num_frames == 2000
    assert contract.fps == 60
    assert contract.camera_keys == ("observation.images.front",)
    assert len(split.train_episodes) == 16
    assert len(split.validation_episodes) == 4
    common.validate_split(split, contract)


def test_contract_accepts_dynamic_multicamera_schema(tmp_path: Path) -> None:
    location = common.resolve_dataset_location(tmp_path, "local/multicam")
    (location.dataset_path / "meta").mkdir(parents=True)
    (location.dataset_path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    metadata = SimpleNamespace(
        features={
            "observation.state": {"shape": [6], "names": list(common.JOINT_NAMES)},
            "action": {"shape": [6], "names": list(common.JOINT_NAMES)},
            "observation.images.front": {"shape": [480, 640, 3]},
            "observation.images.wrist": {"shape": [480, 640, 3]},
        },
        camera_keys=["observation.images.wrist", "observation.images.front"],
        total_episodes=100,
        total_frames=50000,
        fps=30,
    )

    contract = common.build_dataset_contract(
        location,
        metadata,
        action_semantics=common.ACTION_SEMANTICS,
    )

    assert contract.camera_keys == (
        "observation.images.front",
        "observation.images.wrist",
    )
    assert contract.num_episodes == 100


def test_dataset_location_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative path"):
        common.resolve_dataset_location(tmp_path, "../outside")


def test_error_and_motor_limit_statistics() -> None:
    target = np.zeros((1, 2, 6), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0] = 2.0
    accumulator = common.ErrorAccumulator()
    accumulator.update(prediction, target, np.array([[True, False]]))

    result = accumulator.result()
    assert result["mae"] == pytest.approx(2.0)
    assert result["rmse"] == pytest.approx(2.0)

    limits = common.motor_limit_statistics(np.array([[101.0, 0.0, 0.0, -101.0, 0.0, -1.0]], dtype=np.float32))
    assert limits["count_by_joint"] == [1, 0, 0, 1, 0, 1]
    assert limits["total_count"] == 3


def test_safety_marker_forbids_real_robot(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    output = tmp_path / "run"
    common.write_contract_and_safety(output, contract)

    marker = common.require_simulation_only_marker(output)

    assert marker["deployment_scope"] == "simulation-only"
    assert marker["real_robot_deployment_allowed"] is False
    serialized = json.loads((output / common.CONTRACT_FILENAME).read_text(encoding="utf-8"))
    assert serialized["repo_id"] == contract.repo_id
