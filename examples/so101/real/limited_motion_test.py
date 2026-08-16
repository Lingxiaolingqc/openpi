"""Run one bounded, reversible SO-101 Follower joint motion.

The test starts from the measured pose, moves exactly one selected joint by a
small relative delta, holds briefly, returns to the measured start pose, and
then disables torque. It is intentionally separate from Leader mirroring and
policy deployment.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


REAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = REAL_DIR.parents[2]
sys.path.insert(0, str(REAL_DIR))

import no_jump_enable_test as no_jump  # noqa: E402


CONFIRMATION = "MOVE_FOLLOWER_SMALL_DELTA"
DEFAULT_DELTA_DEG = 3.0
DEFAULT_SPEED_DEG_S = 15.0
DEFAULT_CONTROL_RATE_HZ = 30.0
DEFAULT_HOLD_S = 1.0
DEFAULT_MARGIN_DEG = 2.0
DEFAULT_MAX_TRACKING_ERROR_DEG = 5.0
DEFAULT_MAX_NON_TARGET_DRIFT_DEG = 3.0
DEFAULT_MAX_FINAL_ERROR_DEG = 3.0


def _read_monitored_sample(
    bus: Any,
    *,
    command: dict[str, float],
    initial: dict[str, float],
    moving_joint: str,
    max_temperature_c: int,
    max_tracking_error_deg: float,
    max_non_target_drift_deg: float,
) -> tuple[float, float, float]:
    present = bus.sync_read("Present_Position")
    temperatures = bus.sync_read("Present_Temperature", normalize=False)
    torque_states = bus.sync_read("Torque_Enable", normalize=False)
    no_jump._check_complete_finite(present, "monitored pose")

    invalid_torque = {name: value for name, value in torque_states.items() if value != 1}
    if invalid_torque:
        raise RuntimeError(f"Torque state changed unexpectedly: {invalid_torque}")

    hottest_joint = max(temperatures, key=temperatures.get)
    hottest_value = float(temperatures[hottest_joint])
    if hottest_value >= max_temperature_c:
        raise RuntimeError(f"Over-temperature: {hottest_joint}={hottest_value:.0f} C")

    tracking_errors = {name: abs(present[name] - command[name]) for name in no_jump.hold.MOTOR_NAMES}
    tracking_joint = max(tracking_errors, key=tracking_errors.get)
    maximum_tracking_error = tracking_errors[tracking_joint]
    if maximum_tracking_error > max_tracking_error_deg:
        raise RuntimeError(
            f"Tracking error: {tracking_joint}={maximum_tracking_error:.2f} deg "
            f"(limit {max_tracking_error_deg:.2f} deg)"
        )

    non_target_drifts = {
        name: abs(present[name] - initial[name])
        for name in no_jump.hold.MOTOR_NAMES
        if name != moving_joint
    }
    drift_joint = max(non_target_drifts, key=non_target_drifts.get)
    maximum_non_target_drift = non_target_drifts[drift_joint]
    if maximum_non_target_drift > max_non_target_drift_deg:
        raise RuntimeError(
            f"Non-target drift: {drift_joint}={maximum_non_target_drift:.2f} deg "
            f"(limit {max_non_target_drift_deg:.2f} deg)"
        )

    return maximum_tracking_error, maximum_non_target_drift, hottest_value


def _command_segment(
    bus: Any,
    *,
    start: dict[str, float],
    end: dict[str, float],
    initial: dict[str, float],
    moving_joint: str,
    speed_deg_s: float,
    control_rate_hz: float,
    max_temperature_c: int,
    sleep: Callable[[float], None],
) -> tuple[float, float, float, float]:
    maximum_delta = max(abs(end[name] - start[name]) for name in no_jump.hold.MOTOR_NAMES)
    maximum_step = speed_deg_s / control_rate_hz
    steps = max(1, math.ceil(maximum_delta / maximum_step))
    interval_s = 1.0 / control_rate_hz
    previous = start.copy()
    observed_max_step = 0.0
    observed_max_tracking = 0.0
    observed_max_drift = 0.0
    observed_max_temperature = 0.0

    for step in range(1, steps + 1):
        ratio = step / steps
        command = {
            name: start[name] + (end[name] - start[name]) * ratio
            for name in no_jump.hold.MOTOR_NAMES
        }
        command_step = max(abs(command[name] - previous[name]) for name in no_jump.hold.MOTOR_NAMES)
        if command_step > maximum_step + 1e-9:
            raise RuntimeError(
                f"Internal command-step violation: {command_step:.3f} deg exceeds {maximum_step:.3f} deg"
            )
        bus.sync_write("Goal_Position", command)
        sleep(interval_s)
        tracking, drift, temperature = _read_monitored_sample(
            bus,
            command=command,
            initial=initial,
            moving_joint=moving_joint,
            max_temperature_c=max_temperature_c,
            max_tracking_error_deg=DEFAULT_MAX_TRACKING_ERROR_DEG,
            max_non_target_drift_deg=DEFAULT_MAX_NON_TARGET_DRIFT_DEG,
        )
        observed_max_step = max(observed_max_step, command_step)
        observed_max_tracking = max(observed_max_tracking, tracking)
        observed_max_drift = max(observed_max_drift, drift)
        observed_max_temperature = max(observed_max_temperature, temperature)
        previous = command

    return observed_max_step, observed_max_tracking, observed_max_drift, observed_max_temperature


def _hold_pose(
    bus: Any,
    *,
    pose: dict[str, float],
    initial: dict[str, float],
    moving_joint: str,
    hold_s: float,
    control_rate_hz: float,
    max_temperature_c: int,
    sleep: Callable[[float], None],
) -> tuple[float, float, float]:
    cycles = max(1, math.ceil(hold_s * control_rate_hz))
    interval_s = 1.0 / control_rate_hz
    maximum_tracking = 0.0
    maximum_drift = 0.0
    maximum_temperature = 0.0
    for _ in range(cycles):
        sleep(interval_s)
        tracking, drift, temperature = _read_monitored_sample(
            bus,
            command=pose,
            initial=initial,
            moving_joint=moving_joint,
            max_temperature_c=max_temperature_c,
            max_tracking_error_deg=DEFAULT_MAX_TRACKING_ERROR_DEG,
            max_non_target_drift_deg=DEFAULT_MAX_NON_TARGET_DRIFT_DEG,
        )
        maximum_tracking = max(maximum_tracking, tracking)
        maximum_drift = max(maximum_drift, drift)
        maximum_temperature = max(maximum_temperature, temperature)
    return maximum_tracking, maximum_drift, maximum_temperature


def run_limited_motion_test(
    bus: Any,
    limits: dict[str, tuple[float, float]],
    *,
    joint: str,
    delta_deg: float,
    speed_deg_s: float,
    control_rate_hz: float,
    hold_s: float,
    max_temperature_c: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, float | str]:
    connected = False
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
        no_jump._check_complete_finite(initial, "initial pose")
        no_jump._check_inside_limits(initial, limits)
        target = initial.copy()
        target[joint] += delta_deg
        no_jump._check_inside_limits(target, limits)
        no_jump._print_pose("initial_deg", initial)
        no_jump._print_pose("target_deg", target)

        # No-jump ordering: write the measured pose before torque enable.
        bus.sync_write("Goal_Position", initial)
        bus.enable_torque(num_retry=2)
        print("TORQUE_ENABLED_AT_MEASURED_POSE", flush=True)

        initial_tracking, initial_drift, initial_temperature = _hold_pose(
            bus,
            pose=initial,
            initial=initial,
            moving_joint=joint,
            hold_s=hold_s,
            control_rate_hz=control_rate_hz,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )
        print(f"MOVING_OUTBOUND joint={joint} delta_deg={delta_deg:.2f}", flush=True)
        outbound = _command_segment(
            bus,
            start=initial,
            end=target,
            initial=initial,
            moving_joint=joint,
            speed_deg_s=speed_deg_s,
            control_rate_hz=control_rate_hz,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )
        target_tracking, target_drift, target_temperature = _hold_pose(
            bus,
            pose=target,
            initial=initial,
            moving_joint=joint,
            hold_s=hold_s,
            control_rate_hz=control_rate_hz,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )
        print("RETURNING_TO_MEASURED_START", flush=True)
        inbound = _command_segment(
            bus,
            start=target,
            end=initial,
            initial=initial,
            moving_joint=joint,
            speed_deg_s=speed_deg_s,
            control_rate_hz=control_rate_hz,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )
        final_tracking, final_drift, final_temperature = _hold_pose(
            bus,
            pose=initial,
            initial=initial,
            moving_joint=joint,
            hold_s=hold_s,
            control_rate_hz=control_rate_hz,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )

        final_pose = bus.sync_read("Present_Position")
        final_errors = {name: abs(final_pose[name] - initial[name]) for name in no_jump.hold.MOTOR_NAMES}
        final_error_joint = max(final_errors, key=final_errors.get)
        maximum_final_error = final_errors[final_error_joint]
        if maximum_final_error > DEFAULT_MAX_FINAL_ERROR_DEG:
            raise RuntimeError(
                f"Return-to-start error: {final_error_joint}={maximum_final_error:.2f} deg "
                f"(limit {DEFAULT_MAX_FINAL_ERROR_DEG:.2f} deg)"
            )

        maximum_step = max(outbound[0], inbound[0])
        maximum_tracking = max(
            initial_tracking,
            outbound[1],
            target_tracking,
            inbound[1],
            final_tracking,
        )
        maximum_drift = max(initial_drift, outbound[2], target_drift, inbound[2], final_drift)
        maximum_temperature = max(
            initial_temperature,
            outbound[3],
            target_temperature,
            inbound[3],
            final_temperature,
        )
        print(
            "LIMITED_MOTION_TEST_PASS "
            f"joint={joint} delta_deg={delta_deg:.2f} "
            f"max_command_step_deg={maximum_step:.2f} "
            f"max_tracking_error_deg={maximum_tracking:.2f} "
            f"max_non_target_drift_deg={maximum_drift:.2f} "
            f"max_final_error_deg={maximum_final_error:.2f} "
            f"max_temperature_c={maximum_temperature:.0f}",
            flush=True,
        )
        return {
            "status": "PASS",
            "max_command_step_deg": maximum_step,
            "max_tracking_error_deg": maximum_tracking,
            "max_non_target_drift_deg": maximum_drift,
            "max_final_error_deg": maximum_final_error,
            "max_temperature_c": maximum_temperature,
        }
    finally:
        if connected:
            try:
                bus.disconnect(disable_torque=True)
                print("TORQUE_DISABLED_AND_COM8_CLOSED", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"CRITICAL_TORQUE_DISABLE_FAILURE: {type(exc).__name__}: {exc}\n"
                    "Disconnect Follower servo power immediately.",
                    file=sys.stderr,
                    flush=True,
                )
                raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", choices=no_jump.hold.MOTOR_NAMES, required=True)
    parser.add_argument("--delta-deg", type=float, default=DEFAULT_DELTA_DEG)
    parser.add_argument("--speed-deg-s", type=float, default=DEFAULT_SPEED_DEG_S)
    parser.add_argument("--execute", action="store_true", help="Allow the bounded physical motion")
    parser.add_argument("--confirm", default="", help=f"Required confirmation: {CONFIRMATION}")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 0.5 <= abs(args.delta_deg) <= 5.0:
        raise SystemExit("absolute --delta-deg must be between 0.5 and 5 degrees")
    if not 1.0 <= args.speed_deg_s <= DEFAULT_SPEED_DEG_S:
        raise SystemExit(f"--speed-deg-s must be between 1 and {DEFAULT_SPEED_DEG_S:g}")

    lock_path = REAL_DIR / "calibration" / "calibration_lock.json"
    if no_jump.verify(lock_path, REPO_ROOT) != 0:
        return 2
    print("CALIBRATION_GATE_PASS", flush=True)

    maximum_step = args.speed_deg_s / DEFAULT_CONTROL_RATE_HZ
    print(
        f"requested_joint={args.joint} requested_delta_deg={args.delta_deg:.2f} "
        f"speed_deg_s={args.speed_deg_s:.2f} control_rate_hz={DEFAULT_CONTROL_RATE_HZ:.0f} "
        f"maximum_command_step_deg={maximum_step:.2f}",
        flush=True,
    )
    if not args.execute or args.confirm != CONFIRMATION:
        print(
            "DRY_RUN_ONLY: COM8 was not opened. To run the physical test, pass "
            f"--execute --confirm {CONFIRMATION}"
        )
        return 0

    leisaac_root = REPO_ROOT / "tmp" / "leisaac-v0.4.0"
    calibration_path = no_jump.hold._calibration_path(leisaac_root, no_jump.CALIBRATION_ID)
    calibration_data = no_jump.hold._load_calibration_data(calibration_path)
    limits = no_jump.hold._angle_limits(calibration_data, DEFAULT_MARGIN_DEG)
    bus = no_jump.hold._build_bus(no_jump.PORT, leisaac_root, calibration_data)

    print("PHYSICAL_TEST_STARTING_IN_3_SECONDS", flush=True)
    for remaining in (3, 2, 1):
        print(f"countdown={remaining}", flush=True)
        time.sleep(1.0)

    try:
        run_limited_motion_test(
            bus,
            limits,
            joint=args.joint,
            delta_deg=args.delta_deg,
            speed_deg_s=args.speed_deg_s,
            control_rate_hz=DEFAULT_CONTROL_RATE_HZ,
            hold_s=DEFAULT_HOLD_S,
            max_temperature_c=no_jump.DEFAULT_MAX_TEMPERATURE_C,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"LIMITED_MOTION_TEST_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
