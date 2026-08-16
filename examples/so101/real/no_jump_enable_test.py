"""Perform a bounded, no-jump torque-enable test on the SO-101 Follower.

The only commanded goal is the Follower pose measured immediately before
torque is enabled. The test holds that pose for a short, fixed duration while
monitoring displacement, tracking error, temperature, torque state, and
communication. Torque is disabled and COM8 is closed on every exit path.
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


PORT = "COM8"
CALIBRATION_ID = "follower_arm"
CONFIRMATION = "ENABLE_FOLLOWER_HOLD_CURRENT"
DEFAULT_DURATION_S = 5.0
DEFAULT_MONITOR_RATE_HZ = 10.0
# This test commands only the pose measured immediately before torque enable.
# It may therefore start at a frozen calibrated endpoint; ordinary motion uses
# the recorder's separate inward margin.
DEFAULT_MARGIN_DEG = 0.0
DEFAULT_MAX_DISPLACEMENT_DEG = 3.0
DEFAULT_MAX_TRACKING_ERROR_DEG = 3.0
DEFAULT_MAX_TEMPERATURE_C = 65
LIMIT_EPSILON_DEG = 1e-6


def _print_pose(label: str, values: dict[str, float]) -> None:
    formatted = " ".join(f"{name}={values[name]:7.2f}" for name in hold.MOTOR_NAMES)
    print(f"{label}: {formatted}", flush=True)


def _check_complete_finite(values: dict[str, float], label: str) -> None:
    if set(values) != set(hold.MOTOR_NAMES):
        raise RuntimeError(f"{label} joint names do not match the required six-joint order")
    invalid = [name for name in hold.MOTOR_NAMES if not math.isfinite(float(values[name]))]
    if invalid:
        raise RuntimeError(f"{label} contains non-finite values: {invalid}")


def _check_inside_limits(
    current: dict[str, float], limits: dict[str, tuple[float, float]]
) -> None:
    errors = []
    for name in hold.MOTOR_NAMES:
        low, high = limits[name]
        if current[name] < low - LIMIT_EPSILON_DEG or current[name] > high + LIMIT_EPSILON_DEG:
            errors.append(
                f"{name}={current[name]:.2f} deg is outside the allowed calibrated range "
                f"{low:.2f}..{high:.2f} deg"
            )
    if errors:
        raise RuntimeError(
            "Current pose is outside the allowed calibrated range. "
            "With torque off, move the listed joint(s) inside the frozen range and retry:\n  "
            + "\n  ".join(errors)
        )


def run_no_jump_test(
    bus: Any,
    limits: dict[str, tuple[float, float]],
    *,
    duration_s: float,
    monitor_rate_hz: float,
    max_displacement_deg: float,
    max_tracking_error_deg: float,
    max_temperature_c: int,
) -> dict[str, float | str]:
    """Run the hardware test on an injected bus and always disable torque."""
    connected = False
    torque_enabled = False
    result: dict[str, float | str] = {}
    try:
        bus.connect()
        connected = True

        operating_modes = bus.sync_read("Operating_Mode", normalize=False)
        invalid_modes = {name: value for name, value in operating_modes.items() if value != 0}
        if invalid_modes:
            raise RuntimeError(f"Expected position-control Operating_Mode=0: {invalid_modes}")

        torque_states = bus.sync_read("Torque_Enable", normalize=False)
        enabled_before = {name: value for name, value in torque_states.items() if value != 0}
        if enabled_before:
            raise RuntimeError(f"Torque was already enabled before the test: {enabled_before}")

        temperatures = bus.sync_read("Present_Temperature", normalize=False)
        hottest_before = max(temperatures, key=temperatures.get)
        if temperatures[hottest_before] >= max_temperature_c:
            raise RuntimeError(
                f"Preflight over-temperature: {hottest_before}={temperatures[hottest_before]} C"
            )

        initial = bus.sync_read("Present_Position")
        _check_complete_finite(initial, "initial pose")
        _check_inside_limits(initial, limits)
        _print_pose("initial_deg", initial)
        print(
            f"preflight_hottest: {hottest_before}={temperatures[hottest_before]} C; "
            "preflight_torque: all disabled",
            flush=True,
        )

        # This is the safety-critical no-jump ordering: measured pose first,
        # identical goal second, and only then torque enable.
        bus.sync_write("Goal_Position", initial)
        bus.enable_torque(num_retry=2)
        torque_enabled = True
        print("TORQUE_ENABLED_AT_MEASURED_POSE", flush=True)

        interval_s = 1.0 / monitor_rate_hz
        start = time.monotonic()
        deadline = start + duration_s
        next_report = start
        maximum_displacement = 0.0
        maximum_tracking_error = 0.0
        maximum_temperature = float(temperatures[hottest_before])

        while True:
            loop_start = time.monotonic()
            present = bus.sync_read("Present_Position")
            current_temperatures = bus.sync_read("Present_Temperature", normalize=False)
            current_torque = bus.sync_read("Torque_Enable", normalize=False)
            _check_complete_finite(present, "monitored pose")

            lost_torque = {name: value for name, value in current_torque.items() if value != 1}
            if lost_torque:
                raise RuntimeError(f"Torque state changed unexpectedly: {lost_torque}")

            displacement = {name: abs(present[name] - initial[name]) for name in hold.MOTOR_NAMES}
            tracking_error = {name: abs(present[name] - initial[name]) for name in hold.MOTOR_NAMES}
            displacement_joint = max(displacement, key=displacement.get)
            tracking_joint = max(tracking_error, key=tracking_error.get)
            sample_displacement = displacement[displacement_joint]
            sample_tracking_error = tracking_error[tracking_joint]
            hottest = max(current_temperatures, key=current_temperatures.get)
            hottest_value = current_temperatures[hottest]
            maximum_displacement = max(maximum_displacement, sample_displacement)
            maximum_tracking_error = max(maximum_tracking_error, sample_tracking_error)
            maximum_temperature = max(maximum_temperature, float(hottest_value))

            if hottest_value >= max_temperature_c:
                raise RuntimeError(f"Over-temperature: {hottest}={hottest_value} C")
            if sample_displacement > max_displacement_deg:
                raise RuntimeError(
                    f"No-jump failure: {displacement_joint} moved {sample_displacement:.2f} deg "
                    f"(limit {max_displacement_deg:.2f} deg)"
                )
            if sample_tracking_error > max_tracking_error_deg:
                raise RuntimeError(
                    f"Tracking error: {tracking_joint}={sample_tracking_error:.2f} deg "
                    f"(limit {max_tracking_error_deg:.2f} deg)"
                )

            if loop_start >= next_report:
                print(
                    f"hold_elapsed_s={loop_start - start:.2f} "
                    f"max_displacement_deg={maximum_displacement:.2f} "
                    f"hottest={hottest}:{hottest_value}C",
                    flush=True,
                )
                next_report += 1.0

            if loop_start >= deadline:
                break
            sleep_time = min(interval_s, max(0.0, deadline - time.monotonic()))
            if sleep_time > 0:
                time.sleep(sleep_time)

        result = {
            "status": "PASS",
            "duration_s": duration_s,
            "max_displacement_deg": maximum_displacement,
            "max_tracking_error_deg": maximum_tracking_error,
            "max_temperature_c": maximum_temperature,
        }
        print(
            "NO_JUMP_TEST_PASS "
            f"max_displacement_deg={maximum_displacement:.2f} "
            f"max_temperature_c={maximum_temperature:.0f}",
            flush=True,
        )
        return result
    finally:
        if connected:
            try:
                bus.disconnect(disable_torque=True)
                torque_enabled = False
                print("TORQUE_DISABLED_AND_COM8_CLOSED", flush=True)
            except Exception as exc:  # noqa: BLE001
                if torque_enabled:
                    print(
                        f"CRITICAL_TORQUE_DISABLE_FAILURE: {type(exc).__name__}: {exc}\n"
                        "Disconnect Follower servo power immediately.",
                        file=sys.stderr,
                        flush=True,
                    )
                raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Allow the bounded physical torque test")
    parser.add_argument("--confirm", default="", help=f"Required confirmation: {CONFIRMATION}")
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--monitor-rate-hz", type=float, default=DEFAULT_MONITOR_RATE_HZ)
    parser.add_argument("--max-displacement-deg", type=float, default=DEFAULT_MAX_DISPLACEMENT_DEG)
    parser.add_argument(
        "--margin-deg",
        type=float,
        default=DEFAULT_MARGIN_DEG,
        help=(
            "Startup endpoint margin in degrees. Default 0 permits an exact frozen endpoint because "
            "the test holds only the measured pose."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 1.0 <= args.duration_s <= 30.0:
        raise SystemExit("--duration-s must be between 1 and 30 seconds")
    if not 5.0 <= args.monitor_rate_hz <= 30.0:
        raise SystemExit("--monitor-rate-hz must be between 5 and 30 Hz")
    if not 0.5 <= args.max_displacement_deg <= 3.0:
        raise SystemExit("--max-displacement-deg must be between 0.5 and 3 degrees")
    if not 0.0 <= args.margin_deg <= 2.0:
        raise SystemExit("--margin-deg must be between 0 and 2 degrees")

    lock_path = REAL_DIR / "calibration" / "calibration_lock.json"
    if verify(lock_path, REPO_ROOT) != 0:
        return 2
    print("CALIBRATION_GATE_PASS", flush=True)

    if not args.execute or args.confirm != CONFIRMATION:
        print(
            "DRY_RUN_ONLY: COM8 was not opened. To run the physical test, pass "
            f"--execute --confirm {CONFIRMATION}"
        )
        return 0

    leisaac_root = REPO_ROOT / "tmp" / "leisaac-v0.4.0"
    calibration_path = hold._calibration_path(leisaac_root, CALIBRATION_ID)
    calibration_data = hold._load_calibration_data(calibration_path)
    limits = hold._angle_limits(calibration_data, args.margin_deg)
    bus = hold._build_bus(PORT, leisaac_root, calibration_data)

    print("PHYSICAL_TEST_STARTING_IN_3_SECONDS", flush=True)
    for remaining in (3, 2, 1):
        print(f"countdown={remaining}", flush=True)
        time.sleep(1.0)

    try:
        result = run_no_jump_test(
            bus,
            limits,
            duration_s=args.duration_s,
            monitor_rate_hz=args.monitor_rate_hz,
            max_displacement_deg=args.max_displacement_deg,
            max_tracking_error_deg=DEFAULT_MAX_TRACKING_ERROR_DEG,
            max_temperature_c=DEFAULT_MAX_TEMPERATURE_C,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"NO_JUMP_TEST_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
