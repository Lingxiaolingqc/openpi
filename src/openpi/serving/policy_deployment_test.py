from __future__ import annotations

import pytest

from openpi.serving import policy_deployment


def test_prepare_metadata_preserves_default_behavior() -> None:
    original = {"model": "fake"}

    metadata = policy_deployment.prepare_metadata(
        original,
        deployment_scope=None,
        confirm_simulation_only="",
    )

    assert metadata == original
    assert metadata is not original


@pytest.mark.parametrize("confirmation", ["", "yes", "simulation-only"])
def test_prepare_metadata_requires_exact_simulation_confirmation(confirmation: str) -> None:
    with pytest.raises(ValueError, match="confirm-simulation-only SIMULATION_ONLY"):
        policy_deployment.prepare_metadata(
            {},
            deployment_scope="simulation-only",
            confirm_simulation_only=confirmation,
        )


def test_prepare_metadata_advertises_simulation_only_boundary() -> None:
    metadata = policy_deployment.prepare_metadata(
        {"model": "pi05"},
        deployment_scope="simulation-only",
        confirm_simulation_only="SIMULATION_ONLY",
    )

    assert metadata == {
        "model": "pi05",
        "deployment_scope": "simulation-only",
        "real_robot_deployment_allowed": False,
    }


def test_prepare_metadata_rejects_conflicting_real_robot_metadata() -> None:
    with pytest.raises(ValueError, match="allows real-robot deployment"):
        policy_deployment.prepare_metadata(
            {"real_robot_deployment_allowed": True},
            deployment_scope="simulation-only",
            confirm_simulation_only="SIMULATION_ONLY",
        )


def test_prepare_metadata_rejects_conflicting_scope() -> None:
    with pytest.raises(ValueError, match="conflicting deployment scope"):
        policy_deployment.prepare_metadata(
            {"deployment_scope": "real"},
            deployment_scope="simulation-only",
            confirm_simulation_only="SIMULATION_ONLY",
        )


def test_prepare_metadata_rejects_unsupported_scope() -> None:
    with pytest.raises(ValueError, match="unsupported deployment scope"):
        policy_deployment.prepare_metadata(
            {},
            deployment_scope="real",
            confirm_simulation_only="SIMULATION_ONLY",
        )


def test_prepare_metadata_rejects_confirmation_without_scope() -> None:
    with pytest.raises(ValueError, match="requires --deployment-scope simulation-only"):
        policy_deployment.prepare_metadata(
            {},
            deployment_scope=None,
            confirm_simulation_only="SIMULATION_ONLY",
        )
