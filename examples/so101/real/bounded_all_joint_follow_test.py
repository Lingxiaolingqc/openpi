"""Run a short, bounded six-joint SO-101 Leader/Follower validation.

All six Follower joints track relative Leader motion around the validated start
pose. Each joint has an independent excursion limit, every command is slew-rate
limited, and the Follower returns to its measured start pose before torque is
disabled. This is a validation gate, not unrestricted teleoperation.
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

import bounded_leader_follow_test as single  # noqa: E402
import dual_arm_alignment_monitor as alignment  # noqa: E402
import no_jump_enable_test as no_jump  # noqa: E402

CONFIRMATION = "RUN_BOUNDED_ALL_JOINT_FOLLOW"
DEFAULT_CONTROL_RATE_HZ = 30.0
DEFAULT_DURATION_S = 20.0
DEFAULT_READY_S = 3.0
DEFAULT_FINISH_HOLD_S = 3.0
DEFAULT_MAX_EXCURSION_DEG = 3.0
DEFAULT_MIN_EXCURSION_DEG = 0.5
DEFAULT_FINAL_LEADER_TOLERANCE_DEG = 1.0
DEFAULT_MAX_SPEED_DEG_S = 10.0
DEFAULT_MAX_TRACKING_ERROR_DEG = 5.0
DEFAULT_MARGIN_DEG = 2.0
DEFAULT_FAULT_HOLD_S = 1.0


def _read_follower(
    bus: Any,
    *,
    command: dict[str, float],
    limits: dict[str, tuple[float, float]],
    max_tracking_error_deg: float,
    max_temperature_c: int,
) -> tuple[dict[str, float], float, float]:
    present = bus.sync_read("Present_Position")
    no_jump._check_complete_finite(present, "Follower pose")
    no_jump._check_inside_limits(present, limits)
    single._check_modes_and_torque(bus, "Follower", 1)
    hottest = single._read_temperatures(bus, "Follower", max_temperature_c)
    errors = {name: abs(present[name] - command[name]) for name in no_jump.hold.MOTOR_NAMES}
    error_joint = max(errors, key=errors.get)
    maximum_error = errors[error_joint]
    if maximum_error > max_tracking_error_deg:
        raise RuntimeError(
            f"Follower tracking error: {error_joint}={maximum_error:.2f} deg (limit {max_tracking_error_deg:.2f} deg)"
        )
    return present, maximum_error, hottest


def _hold_pose(
    leader_bus: Any,
    follower_bus: Any,
    *,
    pose: dict[str, float],
    limits: dict[str, tuple[float, float]],
    duration_s: float,
    control_rate_hz: float,
    max_tracking_error_deg: float,
    max_temperature_c: int,
    sleep: Callable[[float], None],
) -> tuple[float, float]:
    maximum_tracking = 0.0
    maximum_temperature = 0.0
    for _ in range(max(0, math.ceil(duration_s * control_rate_hz))):
        single._check_modes_and_torque(leader_bus, "Leader", 0)
        leader_pose = leader_bus.sync_read("Present_Position")
        no_jump._check_complete_finite(leader_pose, "Leader pose")
        leader_temperature = single._read_temperatures(leader_bus, "Leader", max_temperature_c)
        follower_bus.sync_write("Goal_Position", pose)
        sleep(1.0 / control_rate_hz)
        _, tracking, follower_temperature = _read_follower(
            follower_bus,
            command=pose,
            limits=limits,
            max_tracking_error_deg=max_tracking_error_deg,
            max_temperature_c=max_temperature_c,
        )
        maximum_tracking = max(maximum_tracking, tracking)
        maximum_temperature = max(maximum_temperature, leader_temperature, follower_temperature)
    return maximum_tracking, maximum_temperature


def _return_to_start(
    follower_bus: Any,
    *,
    command: dict[str, float],
    start_pose: dict[str, float],
    limits: dict[str, tuple[float, float]],
    maximum_step_deg: float,
    control_rate_hz: float,
    max_tracking_error_deg: float,
    max_temperature_c: int,
    sleep: Callable[[float], None],
) -> tuple[float, float]:
    maximum_tracking = 0.0
    maximum_temperature = 0.0
    while max(abs(command[name] - start_pose[name]) for name in no_jump.hold.MOTOR_NAMES) > 1e-9:
        command = single._slew(command, start_pose, maximum_step_deg)
        follower_bus.sync_write("Goal_Position", command)
        sleep(1.0 / control_rate_hz)
        _, tracking, temperature = _read_follower(
            follower_bus,
            command=command,
            limits=limits,
            max_tracking_error_deg=max_tracking_error_deg,
            max_temperature_c=max_temperature_c,
        )
        maximum_tracking = max(maximum_tracking, tracking)
        maximum_temperature = max(maximum_temperature, temperature)
    return maximum_tracking, maximum_temperature


def run_bounded_all_joint_test(
    leader_bus: Any,
    follower_bus: Any,
    calibrations: dict[str, dict[str, dict[str, int]]],
    follower_limits: dict[str, tuple[float, float]],
    *,
    duration_s: float,
    ready_s: float,
    finish_hold_s: float,
    control_rate_hz: float,
    max_excursion_deg: float,
    min_excursion_deg: float,
    final_leader_tolerance_deg: float,
    max_speed_deg_s: float,
    max_tracking_error_deg: float,
    max_temperature_c: int,
    fault_hold_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, float | str | dict[str, float]]:
    connected: list[tuple[str, Any]] = []
    follower_enabled = False
    failed = False
    try:
        leader_bus.connect()
        connected.append(("Leader", leader_bus))
        follower_bus.connect()
        connected.append(("Follower", follower_bus))
        single._check_modes_and_torque(leader_bus, "Leader", 0)
        single._check_modes_and_torque(follower_bus, "Follower", 0)
        leader_temperature = single._read_temperatures(leader_bus, "Leader", max_temperature_c)
        follower_temperature = single._read_temperatures(follower_bus, "Follower", max_temperature_c)

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
            name: error for name, error in initial_errors.items() if abs(error) > single.DEFAULT_ALIGNMENT_THRESHOLD_DEG
        }
        if misaligned:
            formatted = ", ".join(f"{name}={value:+.2f}" for name, value in misaligned.items())
            raise RuntimeError(f"Initial mapped alignment exceeds 3 degrees: {formatted}")

        no_jump._print_pose("initial_leader_deg", initial_leader)
        no_jump._print_pose("initial_follower_deg", initial_follower)
        print(
            f"PREFLIGHT_PASS initial_alignment_max_deg={max(abs(value) for value in initial_errors.values()):.2f}",
            flush=True,
        )

        follower_bus.sync_write("Goal_Position", initial_follower)
        follower_bus.enable_torque(num_retry=2)
        follower_enabled = True
        print("TORQUE_ENABLED_AT_MEASURED_FOLLOWER_POSE", flush=True)
        print(
            f"READY_WINDOW_{ready_s:.0f}S: steady the support from below; keep hands outside all motion paths",
            flush=True,
        )
        ready_tracking, ready_temperature = _hold_pose(
            leader_bus,
            follower_bus,
            pose=initial_follower,
            limits=follower_limits,
            duration_s=ready_s,
            control_rate_hz=control_rate_hz,
            max_tracking_error_deg=max_tracking_error_deg,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )

        print(
            "MOVE_ALL_JOINTS_NOW: move each Leader joint gently, including the gripper, "
            f"then return near the start; each joint must stay within +/-{max_excursion_deg:.1f} deg",
            flush=True,
        )
        maximum_step = max_speed_deg_s / control_rate_hz
        command = initial_follower.copy()
        peak_leader = dict.fromkeys(no_jump.hold.MOTOR_NAMES, 0.0)
        peak_follower = dict.fromkeys(no_jump.hold.MOTOR_NAMES, 0.0)
        final_deltas = dict.fromkeys(no_jump.hold.MOTOR_NAMES, 0.0)
        maximum_tracking = ready_tracking
        maximum_temperature = max(ready_temperature, leader_temperature, follower_temperature)
        cycles = max(1, math.ceil(duration_s * control_rate_hz))
        report_every = max(1, round(control_rate_hz))

        for cycle in range(cycles):
            single._check_modes_and_torque(leader_bus, "Leader", 0)
            leader = leader_bus.sync_read("Present_Position")
            no_jump._check_complete_finite(leader, "Leader pose")
            leader_temperature = single._read_temperatures(leader_bus, "Leader", max_temperature_c)
            mapped = alignment.mapped_follower_target(leader, calibrations["leader"], calibrations["follower"])
            deltas = {name: mapped[name] - initial_mapped[name] for name in no_jump.hold.MOTOR_NAMES}
            exceeded = {name: delta for name, delta in deltas.items() if abs(delta) > max_excursion_deg + 1e-6}
            if exceeded:
                formatted = ", ".join(f"{name}={value:+.2f}" for name, value in exceeded.items())
                raise RuntimeError(f"Leader excursion exceeds +/-{max_excursion_deg:.2f} deg: {formatted}")

            desired = {name: initial_follower[name] + deltas[name] for name in no_jump.hold.MOTOR_NAMES}
            no_jump._check_inside_limits(desired, follower_limits)
            command = single._slew(command, desired, maximum_step)
            follower_bus.sync_write("Goal_Position", command)
            sleep(1.0 / control_rate_hz)
            follower, tracking, follower_temperature = _read_follower(
                follower_bus,
                command=command,
                limits=follower_limits,
                max_tracking_error_deg=max_tracking_error_deg,
                max_temperature_c=max_temperature_c,
            )
            for name in no_jump.hold.MOTOR_NAMES:
                peak_leader[name] = max(peak_leader[name], abs(deltas[name]))
                peak_follower[name] = max(peak_follower[name], abs(follower[name] - initial_follower[name]))
            final_deltas = deltas
            maximum_tracking = max(maximum_tracking, tracking)
            maximum_temperature = max(maximum_temperature, leader_temperature, follower_temperature)
            if cycle % report_every == 0:
                delta_text = " ".join(f"{name}={deltas[name]:+.1f}" for name in no_jump.hold.MOTOR_NAMES)
                print(
                    f"elapsed_s={cycle / control_rate_hz:.1f} deltas_deg[{delta_text}] "
                    f"tracking_deg={tracking:.2f} hottest_c={maximum_temperature:.0f}",
                    flush=True,
                )

        print("RETURNING_FOLLOWER_TO_MEASURED_START", flush=True)
        return_tracking, return_temperature = _return_to_start(
            follower_bus,
            command=command,
            start_pose=initial_follower,
            limits=follower_limits,
            maximum_step_deg=maximum_step,
            control_rate_hz=control_rate_hz,
            max_tracking_error_deg=max_tracking_error_deg,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )
        maximum_tracking = max(maximum_tracking, return_tracking)
        maximum_temperature = max(maximum_temperature, return_temperature)

        insufficient = {name: value for name, value in peak_leader.items() if value < min_excursion_deg}
        if insufficient:
            formatted = ", ".join(f"{name}={value:.2f}" for name, value in insufficient.items())
            raise RuntimeError(f"These Leader joints did not reach {min_excursion_deg:.2f} deg: {formatted}")
        not_returned = {name: value for name, value in final_deltas.items() if abs(value) > final_leader_tolerance_deg}
        if not_returned:
            formatted = ", ".join(f"{name}={value:+.2f}" for name, value in not_returned.items())
            raise RuntimeError(f"Leader was not returned within +/-{final_leader_tolerance_deg:.2f} deg: {formatted}")

        print(
            f"FINISH_HOLD_{finish_hold_s:.0f}S: support the Follower from below; torque will then switch off",
            flush=True,
        )
        finish_tracking, finish_temperature = _hold_pose(
            leader_bus,
            follower_bus,
            pose=initial_follower,
            limits=follower_limits,
            duration_s=finish_hold_s,
            control_rate_hz=control_rate_hz,
            max_tracking_error_deg=max_tracking_error_deg,
            max_temperature_c=max_temperature_c,
            sleep=sleep,
        )
        maximum_tracking = max(maximum_tracking, finish_tracking)
        maximum_temperature = max(maximum_temperature, finish_temperature)
        peaks = ",".join(f"{name}:{peak_leader[name]:.2f}" for name in no_jump.hold.MOTOR_NAMES)
        print(
            "BOUNDED_ALL_JOINT_FOLLOW_PASS "
            f"peak_leader_excursions_deg={peaks} "
            f"max_command_step_deg={maximum_step:.2f} "
            f"max_tracking_error_deg={maximum_tracking:.2f} "
            f"max_temperature_c={maximum_temperature:.0f}",
            flush=True,
        )
        return {
            "status": "PASS",
            "peak_leader_excursions_deg": peak_leader,
            "peak_follower_excursions_deg": peak_follower,
            "max_command_step_deg": maximum_step,
            "max_tracking_error_deg": maximum_tracking,
            "max_temperature_c": maximum_temperature,
        }
    except BaseException:
        failed = True
        if follower_enabled:
            single._fault_hold(follower_bus, hold_s=fault_hold_s, sleep=sleep)
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
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--max-excursion-deg", type=float, default=DEFAULT_MAX_EXCURSION_DEG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Required: {CONFIRMATION}")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 10.0 <= args.duration_s <= 30.0:
        raise SystemExit("--duration-s must be between 10 and 30 seconds")
    if not 1.0 <= args.max_excursion_deg <= DEFAULT_MAX_EXCURSION_DEG:
        raise SystemExit("--max-excursion-deg must be between 1 and 3 degrees")

    lock_path = REAL_DIR / "calibration" / "calibration_lock.json"
    if no_jump.verify(lock_path, REPO_ROOT) != 0:
        return 2
    print("CALIBRATION_GATE_PASS", flush=True)
    print(
        f"duration_s={args.duration_s:.1f} max_excursion_deg={args.max_excursion_deg:.1f} "
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
        "OPERATOR_READY: move one Leader joint at a time; steady only the support below "
        "the Follower; keep fingers outside every motion path; keep power cutoff reachable",
        flush=True,
    )
    print("PHYSICAL_TEST_STARTING_IN_5_SECONDS", flush=True)
    for remaining in (5, 4, 3, 2, 1):
        print(f"countdown={remaining}", flush=True)
        time.sleep(1.0)
    try:
        run_bounded_all_joint_test(
            buses["leader"],
            buses["follower"],
            calibrations,
            follower_limits,
            duration_s=args.duration_s,
            ready_s=DEFAULT_READY_S,
            finish_hold_s=DEFAULT_FINISH_HOLD_S,
            control_rate_hz=DEFAULT_CONTROL_RATE_HZ,
            max_excursion_deg=args.max_excursion_deg,
            min_excursion_deg=DEFAULT_MIN_EXCURSION_DEG,
            final_leader_tolerance_deg=DEFAULT_FINAL_LEADER_TOLERANCE_DEG,
            max_speed_deg_s=DEFAULT_MAX_SPEED_DEG_S,
            max_tracking_error_deg=DEFAULT_MAX_TRACKING_ERROR_DEG,
            max_temperature_c=no_jump.DEFAULT_MAX_TEMPERATURE_C,
            fault_hold_s=DEFAULT_FAULT_HOLD_S,
        )
    except BaseException as exc:
        print(
            f"BOUNDED_ALL_JOINT_FOLLOW_FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
