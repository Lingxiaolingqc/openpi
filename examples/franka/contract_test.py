import json
from pathlib import Path
import time

import numpy as np
import pytest

from examples.franka import backend
from examples.franka import contract


def _snapshot(*, wrist: bool = True) -> contract.FrankaSnapshot:
    now = time.monotonic_ns()
    return contract.FrankaSnapshot(
        q_rad=np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]),
        dq_rad_s=np.zeros(7),
        gripper_width_m=0.08,
        base_rgb=np.zeros((8, 12, 3), dtype=np.uint8),
        captured_monotonic_ns=now,
        base_image_monotonic_ns=now,
        wrist_rgb=np.zeros((6, 10, 3), dtype=np.uint8) if wrist else None,
        wrist_image_monotonic_ns=now if wrist else None,
    )


def test_snapshot_and_server_metadata_contract() -> None:
    config = contract.SafetyConfig()
    snapshot = _snapshot()

    contract.validate_snapshot(snapshot, config)
    observation = snapshot.policy_observation("Test task")
    assert observation["state"].shape == (8,)
    assert set(observation) == {"images/base", "images/wrist", "state", "prompt"}

    metadata = contract.expected_server_metadata()
    metadata["openpi_protocol_versions"] = [1]
    metadata["real_robot_deployment_allowed"] = False
    contract.validate_server_metadata(metadata)


def test_snapshot_rejects_stale_and_unsynchronized_images() -> None:
    config = contract.SafetyConfig(max_sensor_age_s=0.01, max_image_skew_s=0.001)
    snapshot = _snapshot()
    stale_now = snapshot.captured_monotonic_ns + 20_000_000
    with pytest.raises(contract.ContractError, match="stale"):
        contract.validate_snapshot(snapshot, config, now_monotonic_ns=stale_now)
    skewed = contract.FrankaSnapshot(
        **{**snapshot.__dict__, "wrist_image_monotonic_ns": snapshot.base_image_monotonic_ns + 2_000_000}
    )
    with pytest.raises(contract.ContractError, match="synchronization"):
        contract.validate_snapshot(skewed, config, now_monotonic_ns=skewed.wrist_image_monotonic_ns)
    state_skewed = contract.FrankaSnapshot(
        **{**snapshot.__dict__, "base_image_monotonic_ns": snapshot.captured_monotonic_ns + 2_000_000}
    )
    with pytest.raises(contract.ContractError, match="state/camera"):
        contract.validate_snapshot(state_skewed, config, now_monotonic_ns=state_skewed.base_image_monotonic_ns)


def test_training_step_formula_and_real_approval(tmp_path: Path) -> None:
    assert contract.training_steps_for_passes(101, batch_size=8, passes=3.0) == 38
    metadata = contract.expected_server_metadata()
    metadata["openpi_protocol_versions"] = [1]
    metadata["real_robot_deployment_allowed"] = True
    approval = {
        "schema_version": 1,
        "robot_type": "fr3",
        "real_robot_deployment_allowed": True,
        "approved_server_metadata_sha256": contract.metadata_digest(metadata),
        "max_speed_scale": 0.1,
        "approval_id": "review-001",
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval), encoding="utf-8")

    assert contract.validate_real_robot_approval(path, metadata, requested_speed_scale=0.1) == approval
    with pytest.raises(contract.ContractError, match="exceeds"):
        contract.validate_real_robot_approval(path, metadata, requested_speed_scale=0.2)


def test_franka_jazzy_software_compatibility_gate() -> None:
    versions = backend.FrankaSoftwareVersions(
        ubuntu="24.04",
        ros_distro="jazzy",
        franka_ros2="3.4.0",
        libfranka="0.20.4",
        franka_description="2.8.0",
        robot_system="5.9.0",
    )
    backend.validate_software_versions(versions)
    with pytest.raises(RuntimeError, match="Ubuntu"):
        backend.validate_software_versions(backend.FrankaSoftwareVersions(**{**versions.__dict__, "ubuntu": "22.04"}))
