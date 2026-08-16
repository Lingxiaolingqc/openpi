"""Run the first bounded, single-joint Leader-to-Follower validation.

The Leader always remains torque-disabled. The Follower is enabled at its
measured pose, then only one selected joint follows the Leader relative to the
validated starting pose. All other Follower joints hold their measured start
positions. Every exit path disables Follower torque and closes both ports.
"""

# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from collections.abc import Callable
import math
from pathlib import Path
import sys
import time
from typing import Any

REAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = REAL_DIR.parents[2]
sys.path.insert(0, str(REAL_DIR))

import dual_arm_alignment_monitor as alignment  # noqa: E402
import limited_motion_test as limited  # noqa: E402
import no_jump_enable_test as no_jump  # noqa: E402

CONFIRMATION = "RUN_BOUNDED_LEADER_FOLLOW"
DEFAULT_JOINT = "shoulder_pan"
DEFAULT_CONTROL_RATE_HZ = 30.0
DEFAULT_DURATION_S = 8.0
DEFAULT_READY_S = 3.0
DEFAULT_FINISH_HOLD_S = 3.0
DEFAULT_MAX_EXCURSION_DEG = 3.0
DEFAULT_MIN_EXCURSION_DEG = 1.0
DEFAULT_MAX_OTHER_LEADER_DRIFT_DEG = 3.0
DEFAULT_MAX_SPEED_DEG_S = 15.0
DEFAULT_MAX_TRACKING_ERROR_DEG = 5.0
DEFAULT_ALIGNMENT_THRESHOLD_DEG = 3.0
DEFAULT_MARGIN_DEG = 2.0
DEFAULT_FAULT_HOLD_S = 1.0


def _slew(current: dict[str, float], desired: dict[str, float], maximum_step_deg: float) -> dict[str, float]:
    result = {}
    for name in no_jump.hold.MOTOR_NAMES:
        delta = desired[name] - current[name]
        result[name] = current[name] + max(-maximum_step_deg, min(maximum_step_deg, delta))
    return result


def _check_modes_and_torque(bus: Any, role: str, expected_torque: int) -> None:
    modes = bus.sync_read("Operating_Mode", normalize=False)
    invalid_modes = {name: value for name, value in modes.items() if value != 0}
    if invalid_modes:
        raise RuntimeError(f"{role} expected Operating_Mode=0: {invalid_modes}")
    torque = bus.sync_read("Torque_Enable", normalize=False)
    invalid_torque = {name: value for name, value in torque.items() if value != expected_torque}
    if invalid_torque:
        raise RuntimeError(f"{role} unexpected torque state; expected {expected_torque}: {invalid_torque}")


def _read_temperatures(bus: Any, role: str, max_temperature_c: int) -> float:
    temperatures = bus.sync_read("Present_Temperature", normalize=False)
    hottest_joint = max(temperatures, key=temperatures.get)
    hottest = float(temperatures[hottest_joint])
    if hottest >= max_temperature_c:
        raise RuntimeError(f"{role} over-temperature: {hottest_joint}={hottest:.0f} C")
    return hottest


def _read_follower_sample(
    bus: Any,
    *,
    command: dict[str, float],
    initial: dict[str, float],
    moving_joint: str,
    limits: dict[str, tuple[float, float]],
    max_temperature_c: int,
    max_tracking_error_deg: float,
) -> tuple[dict[str, float], float, float, float]:
    present = bus.sync_read("Present_Position")
    no_jump._check_complete_finite(present, "Follower pose")
    no_jump._check_inside_limits(present, limits)
    _check_modes_and_torque(bus, "Follower", 1)
    hottest = _read_temperatures(bus, "Follower", max_temperature_c)

    tracking = {name: abs(present[name] - command[name]) for name in no_jump.hold.MOTOR_NAMES}
    tracking_joint = max(tracking, key=tracking.get)
    maximum_tracking = tracking[tracking_joint]
    if maximum_tracking > max_tracking_error_deg:
        raise RuntimeError(
            f"Follower tracking error: {tracking_joint}={maximum_tracking:.2f} deg "
            f"(limit {max_tracking_error_deg:.2f} deg)"
        )

    drift = {name: abs(present[name] - initial[name]) for name in no_jump.hold.MOTOR_NAMES if name != moving_joint}
    drift_joint = max(drift, key=drift.get)
    maximum_drift = drift[drift_joint]
    if maximum_drift > limited.DEFAULT_MAX_NON_TARGET_DRIFT_DEG:
        raise RuntimeError(
            f"Follower non-target drift: {drift_joint}={maximum_drift:.2f} deg "
            f"(limit {limited.DEFAULT_MAX_NON_TARGET_DRIFT_DEG:.2f} deg)"
        )
    return present, maximum_tracking, maximum_drift, hottest


def _fault_hold(bus: Any, *, hold_s: float, sleep: Callable[[float], None]) -> None:
    """Best-effort measured-pose hold before the mandatory torque-off."""
    if hold_s <= 0:
        return
    try:
        measured = bus.sync_read("Present_Position")
        no_jump._check_complete_finite(measured, "fault-hold pose")
        bus.sync_write("Goal_Position", measured)
        print("FAULT_HOLDING_MEASURED_POSE", flush=True)
        cycles = max(1, math.ceil(hold_s * 10.0))
        for _ in range(cycles):
            _check_modes_and_torque(bus, "Follower", 1)
            _read_temperatures(bus, "Follower", no_jump.DEFAULT_MAX_TEMPERATURE_C)
            sleep(0.1)
    except Exception as exc:
        print(
            f"FAULT_HOLD_UNAVAILABLE: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _hold_start_pose(
    leader_bus: Any,
    follower_bus: Any,
    *,
    initial_follower: dict[str, float],
    joint: str,
    limits: dict[str, tuple[float, float]],
    duration_s: float,
    control_rate_hz: float,
    max_temperature_c: int,
    sleep: Callable[[float], None],
) -> tuple[float, float, float]:
    maximum_tracking = 0.0
    maximum_drift = 0.0
    maximum_temperature = 0.0
    cycles = max(0, math.ceil(duration_s * control_rate_hz))
    for _ in range(cycles):
        _check_modes_and_torque(leader_bus, "Leader", 0)
        leader_pose = leader_bus.sync_read("Present_Position")
        no_jump._check_complete_finite(leader_pose, "Leader pose")
        leader_temperature = _read_temperatures(leader_bus, "Leader", max_temperature_c)
        follower_bus.sync_write("Goal_Position", initial_follower)
        sleep(1.0 / control_rate_hz)
        _, tracking, drift, follower_temperature = _read_follower_sample(
            follower_bus,
            command=initial_follower,
            initial=initial_follower,
            moving_joint=joint,
            limits=limits,
            max_temperature_c=max_temperature_c,
            max_tracking_error_deg=DEFAULT_MAX_TRACKING_ERROR_DEG,
        )
        maximum_tracking = max(maximum_tracking, tracking)
        maximum_drift = max(maximum_drift, drift)
        maximum_temperature = max(maximum_temperature, leader_temperature, follower_temperature)
    return maximum_tracking, maximum_drift, maximum_temperature


def run_bounded_leader_follow_test(
    leader_bus: Any,
    follower_bus: Any,
    calibrations: dict[str, dict[str, dict[str, int]]],
    follower_limits: dict[str, tuple[float, float]],
    *,
    joint: str,
    duration_s: float,
    ready_s: float,
    finish_hold_s: float,
    control_rate_hz: float,
    max_excursion_deg: float,
    min_excursion_deg: float,
    max_other_leader_drift_deg: float,
    max_speed_deg_s: float,
    max_tracking_error_deg: float,
    max_temperature_c: int,
    fault_hold_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, float | str]:
    connected: list[tuple[str, Any]] = []
    follower_enabled = False
    failed = False
    try:
        leader_bus.connect()
        connected.append(("Leader", leader_bus))
        follower_bus.connect()
        connected.append(("Follower", follower_bus))
        _check_modes_and_torque(leader_bus, "Leader", 0)
        _check_modes_and_torque(follower_bus, "Follower", 0)
        leader_temperature = _read_temperatures(leader_bus, "Leader", max_temperature_c)
        follower_temperature = _read_temperatures(follower_bus, "Follower", max_temperature_c)

        initial_leader = leader_bus.sync_read("Present_Position")
        initial_follower = follower_bus.sync_read("Present_Position")
        no_jump._check_complete_finite(initial_leader, "initial Leader pose")
        no_jump._check_complete_finite(initial_follower, "initial Follower pose")
        no_jump._check_inside_limits(initial_follower, follower_limits)
        initial_mapped = alignment.mapped_follower_target(
            initial_leader, calibrations["leader"], calibrations["follower"]
        )
        initial_errors = alignment.alignment_errors(initial_mapped, initial_follower)
        misaligned = {
            name: error for name, error in initial_errors.items() if abs(error) > DEFAULT_ALIGNMENT_THRESHOLD_DEG
        }
        if misaligned:
            formatted = ", ".join(f"{name}={value:+.2f}" for name, value in misaligned.items())
            raise RuntimeError(f"Initial mapped alignment exceeds 3 degrees: {formatted}")

        no_jump._print_pose("initial_leader_deg", initial_leader)
        no_jump._print_pose("initial_follower_deg", initial_follower)
        print(
            f"PREFLIGHT_PASS joint={joint} initial_alignment_max_deg="
            f"{max(abs(value) for value in initial_errors.values()):.2f}",
            flush=True,
        )

        follower_bus.sync_write("Goal_Position", initial_follower)
        follower_bus.enable_torque(num_retry=2)
        follower_enabled = True
        print("TORQUE_ENABLED_AT_MEASURED_FOLLOWER_POSE", flush=True)
        print(
            f"READY_WINDOW_{ready_s:.0f}S: steady the non-rolling support from below; "
            "keep hands outside every joint and link trajectory",
            flush=True,
        )
        ready_metrics = _hold_start_pose(
            leader_bus,
            follower_bus,
            initial_follower=initial_follower,
            joint=joint,
            limits=follower_limits,
            duration_s=ready_s,
            control_rate_hz=control_rate_hz,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )

        print(
            f"MOVE_ONLY_{joint.upper()}_NOW: slowly move out and back; "
            f"do not exceed +/-{max_excursion_deg:.1f} degrees",
            flush=True,
        )
        cycles = max(1, math.ceil(duration_s * control_rate_hz))
        maximum_step = max_speed_deg_s / control_rate_hz
        command = initial_follower.copy()
        peak_leader_excursion = 0.0
        peak_follower_excursion = 0.0
        maximum_tracking = ready_metrics[0]
        maximum_drift = ready_metrics[1]
        maximum_temperature = max(ready_metrics[2], leader_temperature, follower_temperature)
        report_every = max(1, round(control_rate_hz))

        for cycle in range(cycles):
            _check_modes_and_torque(leader_bus, "Leader", 0)
            leader = leader_bus.sync_read("Present_Position")
            no_jump._check_complete_finite(leader, "Leader pose")
            leader_temperature = _read_temperatures(leader_bus, "Leader", max_temperature_c)
            mapped = alignment.mapped_follower_target(leader, calibrations["leader"], calibrations["follower"])
            selected_delta = mapped[joint] - initial_mapped[joint]
            if abs(selected_delta) > max_excursion_deg + 1e-6:
                raise RuntimeError(
                    f"Leader {joint} excursion {selected_delta:+.2f} deg exceeds +/-{max_excursion_deg:.2f} deg"
                )
            other_drift = {
                name: abs(leader[name] - initial_leader[name]) for name in no_jump.hold.MOTOR_NAMES if name != joint
            }
            other_joint = max(other_drift, key=other_drift.get)
            if other_drift[other_joint] > max_other_leader_drift_deg:
                raise RuntimeError(
                    f"Leader non-target motion: {other_joint}={other_drift[other_joint]:.2f} deg "
                    f"(limit {max_other_leader_drift_deg:.2f} deg)"
                )

            desired = initial_follower.copy()
            desired[joint] += selected_delta
            no_jump._check_inside_limits(desired, follower_limits)
            next_command = _slew(command, desired, maximum_step)
            actual_step = max(abs(next_command[name] - command[name]) for name in no_jump.hold.MOTOR_NAMES)
            if actual_step > maximum_step + 1e-9:
                raise RuntimeError("Internal command-step limit violation")
            follower_bus.sync_write("Goal_Position", next_command)
            command = next_command
            sleep(1.0 / control_rate_hz)
            follower, tracking, drift, follower_temperature = _read_follower_sample(
                follower_bus,
                command=command,
                initial=initial_follower,
                moving_joint=joint,
                limits=follower_limits,
                max_temperature_c=max_temperature_c,
                max_tracking_error_deg=max_tracking_error_deg,
            )
            peak_leader_excursion = max(peak_leader_excursion, abs(selected_delta))
            peak_follower_excursion = max(peak_follower_excursion, abs(follower[joint] - initial_follower[joint]))
            maximum_tracking = max(maximum_tracking, tracking)
            maximum_drift = max(maximum_drift, drift)
            maximum_temperature = max(maximum_temperature, leader_temperature, follower_temperature)
            if cycle % report_every == 0:
                print(
                    f"elapsed_s={cycle / control_rate_hz:.1f} "
                    f"leader_delta_deg={selected_delta:+.2f} "
                    f"follower_delta_deg={follower[joint] - initial_follower[joint]:+.2f} "
                    f"tracking_deg={tracking:.2f} hottest_c={maximum_temperature:.0f}",
                    flush=True,
                )

        print("RETURNING_FOLLOWER_TO_MEASURED_START", flush=True)
        return_metrics = limited._command_segment(
            follower_bus,
            start=command,
            end=initial_follower,
            initial=initial_follower,
            moving_joint=joint,
            speed_deg_s=max_speed_deg_s,
            control_rate_hz=control_rate_hz,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )
        maximum_tracking = max(maximum_tracking, return_metrics[1])
        maximum_drift = max(maximum_drift, return_metrics[2])
        maximum_temperature = max(maximum_temperature, return_metrics[3])

        if peak_leader_excursion < min_excursion_deg:
            raise RuntimeError(
                f"Leader movement was only {peak_leader_excursion:.2f} deg; "
                f"at least {min_excursion_deg:.2f} deg is required"
            )
        if peak_follower_excursion < min_excursion_deg * 0.5:
            raise RuntimeError(f"Follower movement was only {peak_follower_excursion:.2f} deg")

        print(
            f"FINISH_HOLD_{finish_hold_s:.0f}S: support the Follower from below; torque will then switch off",
            flush=True,
        )
        finish_metrics = _hold_start_pose(
            leader_bus,
            follower_bus,
            initial_follower=initial_follower,
            joint=joint,
            limits=follower_limits,
            duration_s=finish_hold_s,
            control_rate_hz=control_rate_hz,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )
        maximum_tracking = max(maximum_tracking, finish_metrics[0])
        maximum_drift = max(maximum_drift, finish_metrics[1])
        maximum_temperature = max(maximum_temperature, finish_metrics[2])

        print(
            "BOUNDED_LEADER_FOLLOW_PASS "
            f"joint={joint} peak_leader_excursion_deg={peak_leader_excursion:.2f} "
            f"peak_follower_excursion_deg={peak_follower_excursion:.2f} "
            f"max_command_step_deg={maximum_step:.2f} "
            f"max_tracking_error_deg={maximum_tracking:.2f} "
            f"max_non_target_drift_deg={maximum_drift:.2f} "
            f"max_temperature_c={maximum_temperature:.0f}",
            flush=True,
        )
        return {
            "status": "PASS",
            "peak_leader_excursion_deg": peak_leader_excursion,
            "peak_follower_excursion_deg": peak_follower_excursion,
            "max_command_step_deg": maximum_step,
            "max_tracking_error_deg": maximum_tracking,
            "max_non_target_drift_deg": maximum_drift,
            "max_temperature_c": maximum_temperature,
        }
    except BaseException:
        failed = True
        if follower_enabled:
            _fault_hold(follower_bus, hold_s=fault_hold_s, sleep=sleep)
        raise
    finally:
        for role, bus in reversed(connected):
            try:
                bus.disconnect(disable_torque=(role == "Follower"))
            except Exception as exc:
                print(
                    f"CRITICAL_{role.upper()}_CLOSE_FAILURE: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if role == "Follower":
                    print("Disconnect Follower servo power immediately.", file=sys.stderr)
                    if not failed:
                        raise
        if connected:
            print("FOLLOWER_TORQUE_DISABLED_AND_PORTS_CLOSED", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", choices=no_jump.hold.MOTOR_NAMES, default=DEFAULT_JOINT)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--max-excursion-deg", type=float, default=DEFAULT_MAX_EXCURSION_DEG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Required: {CONFIRMATION}")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 3.0 <= args.duration_s <= 15.0:
        raise SystemExit("--duration-s must be between 3 and 15 seconds")
    if not 1.0 <= args.max_excursion_deg <= DEFAULT_MAX_EXCURSION_DEG:
        raise SystemExit("--max-excursion-deg must be between 1 and 3 degrees")

    lock_path = REAL_DIR / "calibration" / "calibration_lock.json"
    if no_jump.verify(lock_path, REPO_ROOT) != 0:
        return 2
    print("CALIBRATION_GATE_PASS", flush=True)
    print(
        f"joint={args.joint} duration_s={args.duration_s:.1f} "
        f"max_excursion_deg={args.max_excursion_deg:.1f} "
        f"speed_limit_deg_s={DEFAULT_MAX_SPEED_DEG_S:.1f} "
        f"control_rate_hz={DEFAULT_CONTROL_RATE_HZ:.0f}",
        flush=True,
    )
    if not args.execute or args.confirm != CONFIRMATION:
        print(f"DRY_RUN_ONLY: no serial port was opened. To run, pass --execute --confirm {CONFIRMATION}")
        return 0

    leisaac_root = REPO_ROOT / "tmp" / "leisaac-v0.4.0"
    buses = {}
    calibrations = {}
    for role, port, calibration_id in alignment.ARM_CONFIGS:
        calibration_path = no_jump.hold._calibration_path(leisaac_root, calibration_id)
        calibration = no_jump.hold._load_calibration_data(calibration_path)
        calibrations[role] = calibration
        buses[role] = no_jump.hold._build_bus(port, leisaac_root, calibration)
    follower_limits = no_jump.hold._angle_limits(calibrations["follower"], DEFAULT_MARGIN_DEG)

    print(
        "OPERATOR_READY: steady only the support below the Follower; keep fingers "
        "outside joints, links, gripper, and the motion path; keep power cutoff reachable",
        flush=True,
    )
    print("PHYSICAL_TEST_STARTING_IN_5_SECONDS", flush=True)
    for remaining in (5, 4, 3, 2, 1):
        print(f"countdown={remaining}", flush=True)
        time.sleep(1.0)
    try:
        run_bounded_leader_follow_test(
            buses["leader"],
            buses["follower"],
            calibrations,
            follower_limits,
            joint=args.joint,
            duration_s=args.duration_s,
            ready_s=DEFAULT_READY_S,
            finish_hold_s=DEFAULT_FINISH_HOLD_S,
            control_rate_hz=DEFAULT_CONTROL_RATE_HZ,
            max_excursion_deg=args.max_excursion_deg,
            min_excursion_deg=DEFAULT_MIN_EXCURSION_DEG,
            max_other_leader_drift_deg=DEFAULT_MAX_OTHER_LEADER_DRIFT_DEG,
            max_speed_deg_s=DEFAULT_MAX_SPEED_DEG_S,
            max_tracking_error_deg=DEFAULT_MAX_TRACKING_ERROR_DEG,
            max_temperature_c=no_jump.DEFAULT_MAX_TEMPERATURE_C,
            fault_hold_s=DEFAULT_FAULT_HOLD_S,
        )
    except BaseException as exc:
        print(
            f"BOUNDED_LEADER_FOLLOW_FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
