"""Plot PiPER state and absolute action trajectories from one HDF5 episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from examples.piper import contract
from examples.piper import convert_hdf5_to_lerobot


def plot(episode_path: Path, output_path: Path) -> None:
    reference = convert_hdf5_to_lerobot.validate_episode(episode_path)
    with h5py.File(reference.path, "r") as file:
        states = np.asarray(file["observations/state"][:], dtype=np.float32)
        actions = np.asarray(file["actions/absolute_target"][:], dtype=np.float32)
    time_s = np.arange(reference.frames) / contract.CONTROL_HZ
    figure, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(time_s, states[:, index], label="measured state")
        axis.plot(time_s, actions[:, index], label="absolute target", alpha=0.8)
        axis.set_ylabel(contract.STATE_LAYOUT[index])
        axis.grid(alpha=0.25)
    axes[-1, -1].set_visible(False)
    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("time [s]")
    figure.suptitle(f"PiPER episode seed={reference.seed}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"plot:{output_path}")
    print("PIPER_EPISODE_PLOT_OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.episode, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
