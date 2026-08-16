"""Read-only Leader/Follower pose alignment monitor for SO-101.

Both arms must remain torque-disabled. Move only the passive Leader by hand
until all six calibrated motor-angle differences stay within the threshold for
the required stable duration. This tool never writes a motor register.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any


REAL_DIR = Path(__file__).resolve().parent
SO101_DIR = REAL_DIR.parent
REPO_ROOT = REAL_DIR.parents[2]
sys.path.insert(0, str(REAL_DIR))
sys.path.insert(0, str(SO101_DIR))

import hold_joint_angles_windows as hold  # noqa: E402
from calibration.verify_frozen_calibrations import verify  # noqa: E402


ARM_CONFIGS = (
    ("leader", "COM7", "leader_arm"),
    ("follower", "COM8", "follower_arm"),
)
DEFAULT_RATE_HZ = 5.0
DEFAULT_THRESHOLD_DEG = 3.0
DEFAULT_STABLE_S = 2.0
DEFAULT_MAX_TEMPERATURE_C = 65


def _range_fraction(angle_deg: float, calibration: dict[str, int]) -> float:
    range_min = calibration["range_min"]
    range_max = calibration["range_max"]
    midpoint = (range_min + range_max) / 2
    raw_position = midpoint + angle_deg * hold.MAX_RESOLUTION_VALUE / 360
    return (raw_position - range_min) / (range_max - range_min)


def mapped_follower_target(
    leader: dict[str, float],
    leader_calibration: dict[str, dict[str, int]],
    follower_calibration: dict[str, dict[str, int]],
) -> dict[str, float]:
    target: dict[str, float] = {}
    for name in hold.MOTOR_NAMES:
        fraction = _range_fraction(leader[name], leader_calibration[name])
        follower_min = follower_calibration[name]["range_min"]
        follower_max = follower_calibration[name]["range_max"]
        follower_midpoint = (follower_min + follower_max) / 2
        follower_raw_target = follower_min + fraction * (follower_max - follower_min)
        target[name] = (
            (follower_raw_target - follower_midpoint) * 360 / hold.MAX_RESOLUTION_VALUE
        )
    return target


def alignment_errors(
    mapped_target: dict[str, float], follower: dict[str, float]
) -> dict[str, float]:
    return {name: mapped_target[name] - follower[name] for name in hold.MOTOR_NAMES}


def _validate_pose(values: dict[str, float], label: str) -> None:
    if set(values) != set(hold.MOTOR_NAMES):
        raise RuntimeError(f"{label} does not contain the required six joints")
    invalid = [name for name in hold.MOTOR_NAMES if not math.isfinite(float(values[name]))]
    if invalid:
        raise RuntimeError(f"{label} contains non-finite values: {invalid}")


def _check_read_only_state(bus: Any, role: str, max_temperature_c: int) -> float:
    torque = bus.sync_read("Torque_Enable", normalize=False)
    enabled = {name: value for name, value in torque.items() if value != 0}
    if enabled:
        raise RuntimeError(f"{role} torque must remain disabled: {enabled}")

    modes = bus.sync_read("Operating_Mode", normalize=False)
    invalid_modes = {name: value for name, value in modes.items() if value != 0}
    if invalid_modes:
        raise RuntimeError(f"{role} expected Operating_Mode=0: {invalid_modes}")

    temperatures = bus.sync_read("Present_Temperature", normalize=False)
    hottest_joint = max(temperatures, key=temperatures.get)
    hottest = float(temperatures[hottest_joint])
    if hottest >= max_temperature_c:
        raise RuntimeError(f"{role} over-temperature: {hottest_joint}={hottest:.0f} C")
    return hottest


def _print_sample(
    sample: int,
    leader: dict[str, float],
    mapped_target: dict[str, float],
    follower: dict[str, float],
    errors: dict[str, float],
    hottest: dict[str, float],
    threshold_deg: float,
    streak: int,
    required_streak: int,
) -> None:
    maximum_joint = max(errors, key=lambda name: abs(errors[name]))
    maximum_error = abs(errors[maximum_joint])
    print(
        f"\nsample={sample} max_abs_error={maximum_error:.2f}deg joint={maximum_joint} "
        f"aligned_streak={streak}/{required_streak} "
        f"hottest=leader:{hottest['leader']:.0f}C,follower:{hottest['follower']:.0f}C"
    )
    print(
        f"{'joint':14s} {'leader':>9s} {'mapped_F':>9s} "
        f"{'follower':>9s} {'target-F':>9s} {'status':>8s}"
    )
    for name in hold.MOTOR_NAMES:
        status = "OK" if abs(errors[name]) <= threshold_deg else "ADJUST"
        print(
            f"{name:14s} {leader[name]:9.2f} {mapped_target[name]:9.2f} "
            f"{follower[name]:9.2f} {errors[name]:9.2f} {status:>8s}"
        )


def monitor(
    buses: dict[str, Any],
    calibrations: dict[str, dict[str, dict[str, int]]],
    *,
    rate_hz: float,
    threshold_deg: float,
    stable_s: float,
    samples: int,
    max_temperature_c: int,
) -> int:
    connected: list[str] = []
    try:
        for role, _, _ in ARM_CONFIGS:
            buses[role].connect()
            connected.append(role)
            _check_read_only_state(buses[role], role, max_temperature_c)

        interval_s = 1.0 / rate_hz
        required_streak = max(1, math.ceil(stable_s * rate_hz))
        aligned_streak = 0
        sample = 0
        print("READ_ONLY_MONITOR_STARTED: move only the passive Leader; Ctrl-C stops", flush=True)
        while samples <= 0 or sample < samples:
            loop_start = time.monotonic()
            sample += 1
            leader = buses["leader"].sync_read("Present_Position")
            follower = buses["follower"].sync_read("Present_Position")
            _validate_pose(leader, "leader pose")
            _validate_pose(follower, "follower pose")
            mapped_target = mapped_follower_target(
                leader,
                calibrations["leader"],
                calibrations["follower"],
            )
            errors = alignment_errors(mapped_target, follower)
            all_aligned = all(abs(value) <= threshold_deg for value in errors.values())
            aligned_streak = aligned_streak + 1 if all_aligned else 0
            hottest = {
                role: _check_read_only_state(buses[role], role, max_temperature_c)
                for role, _, _ in ARM_CONFIGS
            }
            _print_sample(
                sample,
                leader,
                mapped_target,
                follower,
                errors,
                hottest,
                threshold_deg,
                aligned_streak,
                required_streak,
            )
            if aligned_streak >= required_streak:
                print(
                    f"DUAL_ARM_ALIGNMENT_READY threshold_deg={threshold_deg:.2f} "
                    f"stable_s={stable_s:.2f}",
                    flush=True,
                )
                return 0

            sleep_time = interval_s - (time.monotonic() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("ALIGNMENT_NOT_READY: sample limit reached", flush=True)
        return 1
    except KeyboardInterrupt:
        print("\nMONITOR_STOPPED_BY_OPERATOR", flush=True)
        return 130
    finally:
        for role in reversed(connected):
            try:
                # Read-only invariant: do not write Torque_Enable on exit.
                buses[role].disconnect(disable_torque=False)
            except Exception as exc:  # noqa: BLE001
                print(f"{role}_port_close_error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if connected:
            print("READ_ONLY_PORTS_CLOSED", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--threshold-deg", type=float, default=DEFAULT_THRESHOLD_DEG)
    parser.add_argument("--stable-s", type=float, default=DEFAULT_STABLE_S)
    parser.add_argument("--samples", type=int, default=0, help="0 means run until aligned or Ctrl-C")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1.0 <= args.rate_hz <= 10.0:
        raise SystemExit("--rate-hz must be between 1 and 10 Hz")
    if not 1.0 <= args.threshold_deg <= 5.0:
        raise SystemExit("--threshold-deg must be between 1 and 5 degrees")
    if not 0.5 <= args.stable_s <= 10.0:
        raise SystemExit("--stable-s must be between 0.5 and 10 seconds")
    if args.samples < 0:
        raise SystemExit("--samples must not be negative")

    lock_path = REAL_DIR / "calibration" / "calibration_lock.json"
    if verify(lock_path, REPO_ROOT) != 0:
        return 2

    leisaac_root = REPO_ROOT / "tmp" / "leisaac-v0.4.0"
    buses = {}
    calibrations = {}
    for role, port, calibration_id in ARM_CONFIGS:
        calibration_path = hold._calibration_path(leisaac_root, calibration_id)
        calibration_data = hold._load_calibration_data(calibration_path)
        calibrations[role] = calibration_data
        buses[role] = hold._build_bus(port, leisaac_root, calibration_data)

    return monitor(
        buses,
        calibrations,
        rate_hz=args.rate_hz,
        threshold_deg=args.threshold_deg,
        stable_s=args.stable_s,
        samples=args.samples,
        max_temperature_c=DEFAULT_MAX_TEMPERATURE_C,
    )


if __name__ == "__main__":
    raise SystemExit(main())
