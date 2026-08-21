"""Explicit deployment-boundary metadata for policy servers."""

from __future__ import annotations

from typing import Any


SIMULATION_ONLY_SCOPE = "simulation-only"
SIMULATION_ONLY_CONFIRMATION = "SIMULATION_ONLY"


def prepare_metadata(
    metadata: dict[str, Any] | None,
    *,
    deployment_scope: str | None,
    confirm_simulation_only: str,
) -> dict[str, Any]:
    """Apply an explicitly confirmed deployment boundary without changing default server behavior."""
    prepared = dict(metadata or {})
    if deployment_scope is None:
        if confirm_simulation_only:
            raise ValueError("--confirm-simulation-only requires --deployment-scope simulation-only")
        return prepared
    if deployment_scope != SIMULATION_ONLY_SCOPE:
        raise ValueError(f"unsupported deployment scope: {deployment_scope!r}")
    if confirm_simulation_only != SIMULATION_ONLY_CONFIRMATION:
        raise ValueError(
            "refusing to advertise a simulation-only policy without "
            f"--confirm-simulation-only {SIMULATION_ONLY_CONFIRMATION}"
        )

    existing_scope = prepared.get("deployment_scope")
    if existing_scope not in {None, SIMULATION_ONLY_SCOPE}:
        raise ValueError(f"policy metadata has conflicting deployment scope: {existing_scope!r}")
    real_robot_allowed = prepared.get("real_robot_deployment_allowed")
    if real_robot_allowed is not None and real_robot_allowed is not False:
        raise ValueError("policy metadata allows real-robot deployment and cannot be served as simulation-only")

    prepared["deployment_scope"] = SIMULATION_ONLY_SCOPE
    prepared["real_robot_deployment_allowed"] = False
    return prepared
