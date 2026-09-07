"""Attach audited PiPER dataset metadata and a simulation-only marker to a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FILENAMES = ("dataset_contract.json", "episode_split.json", "SIMULATION_ONLY.json")


def mark_checkpoint(checkpoint_dir: Path, dataset_dir: Path) -> None:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    missing = [name for name in FILENAMES if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset directory is missing audit artifacts: {missing}")
    for name in FILENAMES:
        source = json.loads((dataset_dir / name).read_text(encoding="utf-8"))
        destination = checkpoint_dir / name
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != source:
                raise FileExistsError(f"refusing to replace conflicting checkpoint metadata: {destination}")
            continue
        destination.write_text(json.dumps(source, indent=2, sort_keys=True), encoding="utf-8")
    marker = json.loads((checkpoint_dir / "SIMULATION_ONLY.json").read_text(encoding="utf-8"))
    if marker.get("deployment_scope") != "simulation-only" or marker.get("real_robot_deployment_allowed") is not False:
        raise ValueError("SIMULATION_ONLY.json does not enforce the required deployment boundary")
    print("PIPER_CHECKPOINT_MARKED_SIMULATION_ONLY")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    mark_checkpoint(args.checkpoint_dir, args.dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
