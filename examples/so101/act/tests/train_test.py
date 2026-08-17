from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.so101.act import common
from examples.so101.act import train


class _FakeTensor(list):
    def new_zeros(self, size: int) -> _FakeTensor:
        return _FakeTensor([0] * size)


def _contract() -> common.DatasetContract:
    return common.DatasetContract(
        repo_id="local/example",
        dataset_root="/datasets",
        dataset_path="/datasets/local/example",
        metadata_fingerprint="fingerprint",
        fps=60,
        num_episodes=20,
        num_frames=1000,
        camera_keys=("observation.images.front",),
        state_feature="observation.state",
        action_feature="action",
        state_shape=(6,),
        action_shape=(6,),
        joint_names=common.JOINT_NAMES,
        action_semantics=common.ACTION_SEMANTICS,
    )


def test_full_training_gate_binds_dataset_and_validation_split(tmp_path: Path) -> None:
    contract = _contract()
    split = common.deterministic_episode_split(contract, validation_fraction=0.2, seed=42)
    report = tmp_path / "overfit_gate.json"
    report.write_text(
        json.dumps(
            {
                "passed": True,
                "repo_id": contract.repo_id,
                "metadata_fingerprint": contract.metadata_fingerprint,
                "validation_episodes": split.validation_episodes,
            }
        ),
        encoding="utf-8",
    )

    train._require_overfit_gate(report, contract, split)  # noqa: SLF001

    different_split = common.deterministic_episode_split(
        contract,
        validation_fraction=0.2,
        seed=7,
    )
    with pytest.raises(ValueError, match="held-out"):
        train._require_overfit_gate(report, contract, different_split)  # noqa: SLF001


def test_non_contiguous_lerobot_subset_uses_original_episode_ids() -> None:
    dataset = SimpleNamespace(
        meta=SimpleNamespace(total_episodes=8),
        episode_data_index={
            "from": _FakeTensor([0, 10, 30]),
            "to": _FakeTensor([10, 30, 45]),
        },
    )

    changed = common.ensure_original_episode_index_lookup(dataset, (2, 5, 7))

    assert changed is True
    assert dataset.episode_data_index["from"] == [0, 0, 0, 0, 0, 10, 0, 30]
    assert dataset.episode_data_index["to"] == [0, 0, 10, 0, 0, 30, 0, 45]


def test_full_sized_lerobot_episode_lookup_is_unchanged() -> None:
    original = {
        "from": _FakeTensor([0, 10, 20]),
        "to": _FakeTensor([10, 20, 30]),
    }
    dataset = SimpleNamespace(meta=SimpleNamespace(total_episodes=3), episode_data_index=original)

    changed = common.ensure_original_episode_index_lookup(dataset, (0, 1, 2))

    assert changed is False
    assert dataset.episode_data_index is original
