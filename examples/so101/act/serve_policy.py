"""Serve a simulation-only ACT checkpoint over OpenPI's WebSocket protocol."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.so101.act import common
from examples.so101.act.policy_utils import load_act_policy
from examples.so101.act.policy_utils import predict_action_chunk
from examples.so101.act.policy_utils import request_to_policy_batch

CONFIRMATION = "SIMULATION_ONLY"


class ACTWebsocketPolicy:
    """Translate OpenPI SO-101 requests into pinned-LeRobot ACT batches."""

    def __init__(
        self,
        *,
        pretrained_model_dir: Path,
        device: str,
        actions_per_inference: int,
        marker: dict[str, Any],
        contract: dict[str, Any],
    ) -> None:
        self._policy = load_act_policy(pretrained_model_dir, device=device)
        self._device = device
        self._actions_per_inference = actions_per_inference
        self._marker = marker
        self._contract = contract
        self._camera_features = tuple(sorted(self._policy.config.image_features))
        contract_cameras = tuple(sorted(contract["camera_keys"]))
        if self._camera_features != contract_cameras:
            raise ValueError(f"checkpoint cameras {self._camera_features} do not match contract {contract_cameras}")
        if self._policy.config.temporal_ensemble_coeff is not None and actions_per_inference != 1:
            raise ValueError("temporal-ensemble ACT checkpoints require actions_per_inference=1")
        if actions_per_inference > int(self._policy.config.chunk_size):
            raise ValueError(
                f"actions_per_inference={actions_per_inference} exceeds chunk_size={self._policy.config.chunk_size}"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": "lerobot-act",
            "deployment_scope": common.DEPLOYMENT_SCOPE,
            "real_robot_deployment_allowed": False,
            "supports_remote_reset": True,
            "repo_id": self._contract["repo_id"],
            "metadata_fingerprint": self._contract["metadata_fingerprint"],
            "camera_keys": self._camera_features,
            "joint_names": common.JOINT_NAMES,
            "action_semantics": common.ACTION_SEMANTICS,
            "chunk_size": int(self._policy.config.chunk_size),
            "actions_per_inference": self._actions_per_inference,
            "temporal_ensemble_coeff": self._policy.config.temporal_ensemble_coeff,
        }

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("__reset__") is True:
            self._policy.reset()
            return {"reset_ack": True}
        expected_requests = {f"images/{key.removeprefix('observation.images.')}": key for key in self._camera_features}
        unexpected = sorted(key for key in request if key.startswith("images/") and key not in expected_requests)
        if unexpected:
            raise ValueError(f"ACT request contains unexpected camera key(s): {unexpected}")
        batch = request_to_policy_batch(self._policy, request, device=self._device)
        actions = predict_action_chunk(
            self._policy,
            batch,
            actions_per_inference=self._actions_per_inference,
        )[0]
        actions_np = actions.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(actions_np).all():
            raise RuntimeError("ACT checkpoint returned NaN or infinity")
        limits = common.motor_limit_statistics(actions_np)
        return {
            "actions": actions_np,
            "policy_diagnostics": {
                "deployment_scope": common.DEPLOYMENT_SCOPE,
                "actions_clipped": False,
                "motor_limit_violation_count_by_joint": limits["count_by_joint"],
                "motor_limit_violation_total": limits["total_count"],
                "motor_limit_max_violation_degrees_by_joint": limits["max_violation_by_joint"],
            },
        }


def _config_preparse() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=common.CONFIG_PATH)
    args, _ = parser.parse_known_args()
    return args.config


def _build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    serving = config.get("serving", {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(config["_path"]))
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("OPENPI_ACT_CHECKPOINT"))
    parser.add_argument("--device", default=config.get("policy", {}).get("device", "cuda"))
    parser.add_argument("--host", default=serving.get("host", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(serving.get("port", 18000)))
    parser.add_argument(
        "--actions-per-inference",
        type=int,
        default=int(serving.get("actions_per_inference", 10)),
    )
    parser.add_argument("--confirm-simulation-only", default="")
    return parser


def main() -> int:
    config = common.load_config(_config_preparse())
    args = _build_parser(config).parse_args()
    try:
        if args.checkpoint is None:
            raise ValueError("--checkpoint or OPENPI_ACT_CHECKPOINT is required")
        if args.confirm_simulation_only != CONFIRMATION:
            raise ValueError(
                f"refusing to serve without --confirm-simulation-only {CONFIRMATION}; "
                "this checkpoint is forbidden on real hardware"
            )
        if not 1 <= args.port <= 65535 or args.actions_per_inference <= 0:
            raise ValueError("port and actions_per_inference must be positive and valid")
        pretrained_dir, _ = common.resolve_pretrained_model_path(args.checkpoint)
        marker = common.require_simulation_only_marker(pretrained_dir)
        contract_path = pretrained_dir / common.CONTRACT_FILENAME
        with contract_path.open("r", encoding="utf-8") as stream:
            contract = json.load(stream)
        if contract.get("deployment_scope") != common.DEPLOYMENT_SCOPE:
            raise ValueError(f"invalid checkpoint dataset contract: {contract_path}")
        policy = ACTWebsocketPolicy(
            pretrained_model_dir=pretrained_dir,
            device=args.device,
            actions_per_inference=args.actions_per_inference,
            marker=marker,
            contract=contract,
        )
        from openpi.serving import websocket_policy_server

        hostname = socket.gethostname()
        logging.info(
            "Starting simulation-only ACT server host=%s machine=%s port=%d checkpoint=%s",
            args.host,
            hostname,
            args.port,
            pretrained_dir,
        )
        print(f"act_server_metadata: {json.dumps(policy.metadata, sort_keys=True)}", flush=True)
        print("ACT_POLICY_SERVER_SIMULATION_ONLY", flush=True)
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=policy.metadata,
        )
        server.serve_forever()
        return 0
    except Exception as exc:
        print(f"ACT_POLICY_SERVER_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
