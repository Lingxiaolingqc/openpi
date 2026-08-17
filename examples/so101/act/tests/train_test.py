from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.so101.act import common
from examples.so101.act import train


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
