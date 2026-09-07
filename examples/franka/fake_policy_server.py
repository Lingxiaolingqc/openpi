"""Protocol-v1 fake Franka policy server for interface and fault testing."""

from __future__ import annotations

import dataclasses
from enum import Enum
import logging
from typing import Literal

import numpy as np
from openpi_client import base_policy
import tyro

from examples.franka import contract
from openpi.serving import websocket_policy_server


class FaultMode(str, Enum):
    NORMAL = "normal"
    BAD_SHAPE = "bad-shape"
    NONFINITE = "nonfinite"
    JOINT_LIMIT = "joint-limit"


class FakeFrankaPolicy(base_policy.BasePolicy):
    def __init__(self, fault_mode: FaultMode = FaultMode.NORMAL) -> None:
        self._fault_mode = fault_mode

    def infer(self, obs: dict) -> dict:
        state = np.asarray(obs["state"], dtype=np.float32)
        actions = np.repeat(state[None, :], contract.ACTION_HORIZON, axis=0)
        if self._fault_mode is FaultMode.BAD_SHAPE:
            actions = actions[:, :-1]
        elif self._fault_mode is FaultMode.NONFINITE:
            actions[0, 0] = np.nan
        elif self._fault_mode is FaultMode.JOINT_LIMIT:
            actions[0, 0] = contract.Q_MAX_RAD[0] + 1.0
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class Args:
    port: int = 8000
    fault_mode: Literal["normal", "bad-shape", "nonfinite", "joint-limit"] = "normal"


def main(args: Args) -> None:
    metadata = contract.expected_server_metadata()
    metadata["real_robot_deployment_allowed"] = False
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=FakeFrankaPolicy(FaultMode(args.fault_mode)),
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
