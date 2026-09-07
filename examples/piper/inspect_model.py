"""Inspect the PiPER MJCF contract and optionally open MuJoCo's passive viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from examples.piper import contract
from examples.piper import mujoco_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    with mujoco_env.PiperRedCubeToBoxEnv(model_dir=args.model_dir, render=not args.headless) as env:
        observation, reset_info = env.reset(seed=0)
        names = {
            "arm_joints": list(contract.ARM_JOINT_NAMES),
            "finger_joints": ["joint7", "joint8"],
            "actuators": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
            "nq": env.model.nq,
            "nv": env.model.nv,
            "nu": env.model.nu,
            "physics_hz": contract.PHYSICS_HZ,
            "control_hz": contract.CONTROL_HZ,
            "state": observation.state.tolist(),
            "cube_position_m": reset_info.cube_position_m.tolist(),
            "target_position_m": reset_info.target_position_m.tolist(),
            "model_sha256": env.model_sha256,
        }
        print(json.dumps(names, indent=2, sort_keys=True))
        if args.headless:
            print("PIPER_MODEL_INSPECTION_OK")
            return 0
        import mujoco.viewer

        deadline = time.monotonic() + args.seconds
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running() and time.monotonic() < deadline:
                env.step(observation.state)
                viewer.sync()
                time.sleep(1.0 / contract.CONTROL_HZ)
        print("PIPER_MODEL_INSPECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
