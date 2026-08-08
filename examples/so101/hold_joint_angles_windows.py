"""Move a locally connected SO-101 arm to six joint angles and hold the pose.

The six angles are expressed in calibrated joint-space degrees. Zero degrees is
the midpoint of each joint's recorded calibration range; it is not raw servo
position zero. The motor order is::

    shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper

This script is audit-only unless ``--execute`` is supplied. During execution it
sets the current pose as the initial goal before enabling torque, ramps to the
requested pose at a bounded speed, monitors temperature and tracking error,
and disables torque on Ctrl-C, normal exit, or an exception.

It writes only runtime position/torque registers. It does not change motor IDs,
baud rate, PID gains, homing offsets, or calibration files.

Examples from the OpenPI repository root on Windows::

    tmp\leisaac-remote-env\python.exe examples\so101\hold_joint_angles_windows.py \
        --port COM7 --angles 0 0 0 0 0 0

    tmp\leisaac-remote-env\python.exe examples\so101\hold_joint_angles_windows.py \
        --port COM7 --angles 0 -20 35 0 0 10 --execute
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MOTOR_IDS = dict(zip(MOTOR_NAMES, range(1, 7), strict=True))
MODEL_RESOLUTION = 4096
MAX_RESOLUTION_VALUE = MODEL_RESOLUTION - 1
EXECUTION_CONFIRMATION = "MOVE_AND_HOLD"
TARGET_CONFIRMATION = "APPLY_TARGET"
DEFAULT_PORT = "COM7"
DEFAULT_CALIBRATION_ID = "leader_arm"
DEFAULT_SPEED_DEG_S = 15.0
MAX_ALLOWED_SPEED_DEG_S = 30.0
DEFAULT_COMMAND_RATE_HZ = 50.0
DEFAULT_MONITOR_RATE_HZ = 5.0
DEFAULT_MAX_TEMPERATURE_C = 65
DEFAULT_MAX_TRACKING_ERROR_DEG = 15.0
DEFAULT_ERROR_GRACE_S = 2.0


def _default_leisaac_root() -> Path:
    return Path(__file__).resolve().parents[2] / "tmp" / "leisaac-v0.4.0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port connected to the arm (default: COM7).")
    parser.add_argument("--id", default=DEFAULT_CALIBRATION_ID, help="Calibration file ID (default: leader_arm).")
    parser.add_argument(
        "--leisaac-root",
        type=Path,
        default=_default_leisaac_root(),
        help="LeIsaac v0.4.0 checkout containing source/leisaac and the calibration cache.",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs=6,
        metavar=("PAN", "LIFT", "ELBOW", "WRIST_FLEX", "WRIST_ROLL", "GRIPPER"),
        help="Six calibrated joint angles in degrees. If omitted, they are requested interactively.",
    )
    parser.add_argument(
        "--speed-deg-s",
        type=float,
        default=DEFAULT_SPEED_DEG_S,
        help=f"Ramp speed in degrees/second (default: {DEFAULT_SPEED_DEG_S:g}, maximum: {MAX_ALLOWED_SPEED_DEG_S:g}).",
    )
    parser.add_argument(
        "--margin-deg",
        type=float,
        default=2.0,
        help="Keep every target this many degrees inside the recorded calibration limits (default: 2).",
    )
    parser.add_argument(
        "--max-temperature-c",
        type=int,
        default=DEFAULT_MAX_TEMPERATURE_C,
        help=f"Disable torque at or above this temperature (default: {DEFAULT_MAX_TEMPERATURE_C}).",
    )
    parser.add_argument(
        "--max-tracking-error-deg",
        type=float,
        default=DEFAULT_MAX_TRACKING_ERROR_DEG,
        help="Disable torque if pose error stays above this threshold (default: 15 degrees).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow physical motion. Without this flag the script only validates and prints the request.",
    )
    return parser


def _calibration_path(leisaac_root: Path, calibration_id: str) -> Path:
    return leisaac_root / "scripts" / "environments" / "teleoperation" / ".cache" / f"{calibration_id}.json"


def _load_calibration_data(path: Path) -> dict[str, dict[str, int]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Calibration file does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read calibration file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError("Calibration JSON must contain an object at the top level.")

    missing = [name for name in MOTOR_NAMES if name not in raw]
    extra = [name for name in raw if name not in MOTOR_NAMES]
    if missing or extra:
        raise RuntimeError(f"Calibration motor names do not match: missing={missing}, extra={extra}")

    calibration: dict[str, dict[str, int]] = {}
    required_fields = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
    for name in MOTOR_NAMES:
        item = raw[name]
        if not isinstance(item, dict) or any(field not in item for field in required_fields):
            raise RuntimeError(f"Calibration entry for {name!r} is incomplete.")
        values = {field: int(item[field]) for field in required_fields}
        if values["id"] != MOTOR_IDS[name]:
            raise RuntimeError(f"Calibration ID mismatch for {name}: expected {MOTOR_IDS[name]}, got {values['id']}")
        if values["drive_mode"] != 0:
            raise RuntimeError(
                f"Unsupported drive_mode={values['drive_mode']} for {name}; "
                "this script refuses ambiguous angle directions."
            )
        if not 0 <= values["range_min"] < values["range_max"] <= MAX_RESOLUTION_VALUE:
            raise RuntimeError(
                f"Invalid calibrated range for {name}: {values['range_min']}..{values['range_max']}"
            )
        calibration[name] = values
    return calibration


def _angle_limits(calibration: dict[str, dict[str, int]], margin_deg: float) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for name in MOTOR_NAMES:
        range_min = calibration[name]["range_min"]
        range_max = calibration[name]["range_max"]
        midpoint = (range_min + range_max) / 2
        low = (range_min - midpoint) * 360 / MAX_RESOLUTION_VALUE + margin_deg
        high = (range_max - midpoint) * 360 / MAX_RESOLUTION_VALUE - margin_deg
        if low >= high:
            raise RuntimeError(f"Safety margin leaves no usable range for {name}: {low:.2f}..{high:.2f} degrees")
        limits[name] = (low, high)
    return limits


def _read_angles(values: list[float] | None) -> dict[str, float]:
    if values is None:
        print("Enter six angles in this order:")
        print("  shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper")
        text = input("angles_deg> ").strip().replace(",", " ")
        parts = text.split()
        if len(parts) != len(MOTOR_NAMES):
            raise RuntimeError(f"Expected exactly six angles, received {len(parts)}.")
        try:
            values = [float(value) for value in parts]
        except ValueError as exc:
            raise RuntimeError("Every angle must be a finite number.") from exc

    targets = dict(zip(MOTOR_NAMES, values, strict=True))
    non_finite = [name for name, value in targets.items() if not math.isfinite(value)]
    if non_finite:
        raise RuntimeError(f"Non-finite target angles: {non_finite}")
    return targets


def _validate_targets(targets: dict[str, float], limits: dict[str, tuple[float, float]]) -> None:
    errors = []
    for name in MOTOR_NAMES:
        low, high = limits[name]
        if not low <= targets[name] <= high:
            errors.append(f"{name}={targets[name]:.2f} is outside {low:.2f}..{high:.2f} degrees")
    if errors:
        raise RuntimeError("Unsafe target request:\n  " + "\n  ".join(errors))


def _print_pose(label: str, values: dict[str, float]) -> None:
    formatted = " ".join(f"{name}={values[name]:7.2f}" for name in MOTOR_NAMES)
    print(f"{label}: {formatted}", flush=True)


def _print_limits(limits: dict[str, tuple[float, float]]) -> None:
    print("calibrated_safe_angle_limits_deg:")
    for name in MOTOR_NAMES:
        low, high = limits[name]
        print(f"  {name:14s}: {low:8.2f} .. {high:8.2f}")


def _load_motor_api(leisaac_root: Path) -> tuple[Any, Any, Any, Any]:
    source_root = leisaac_root / "source" / "leisaac"
    if not source_root.is_dir():
        raise RuntimeError(f"LeIsaac Python source directory does not exist: {source_root}")
    sys.path.insert(0, str(source_root))

    try:
        from leisaac.devices.lerobot.common.motors import FeetechMotorsBus
        from leisaac.devices.lerobot.common.motors import Motor
        from leisaac.devices.lerobot.common.motors import MotorCalibration
        from leisaac.devices.lerobot.common.motors import MotorNormMode
    except ImportError as exc:
        raise RuntimeError(
            "Could not import the LeIsaac Feetech motor API. Run this script with "
            "tmp\\leisaac-remote-env\\python.exe and ensure scservo_sdk is installed."
        ) from exc

    return FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode


def _build_bus(port: str, leisaac_root: Path, calibration_data: dict[str, dict[str, int]]) -> Any:
    FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode = _load_motor_api(leisaac_root)
    motors = {
        name: Motor(id=MOTOR_IDS[name], model="sts3215", norm_mode=MotorNormMode.DEGREES) for name in MOTOR_NAMES
    }
    calibration = {
        name: MotorCalibration(
            id=calibration_data[name]["id"],
            drive_mode=calibration_data[name]["drive_mode"],
            homing_offset=calibration_data[name]["homing_offset"],
            range_min=calibration_data[name]["range_min"],
            range_max=calibration_data[name]["range_max"],
        )
        for name in MOTOR_NAMES
    }
    return FeetechMotorsBus(port=port, motors=motors, calibration=calibration)


def _ramp_to_target(bus: Any, start: dict[str, float], target: dict[str, float], speed_deg_s: float) -> None:
    max_delta = max(abs(target[name] - start[name]) for name in MOTOR_NAMES)
    duration = max_delta / speed_deg_s
    steps = max(1, math.ceil(duration * DEFAULT_COMMAND_RATE_HZ))
    interval = 1.0 / DEFAULT_COMMAND_RATE_HZ
    next_tick = time.monotonic()

    print(f"ramp_duration_s: {duration:.2f}")
    print(f"ramp_steps: {steps}")
    for step in range(1, steps + 1):
        ratio = step / steps
        command = {name: start[name] + (target[name] - start[name]) * ratio for name in MOTOR_NAMES}
        bus.sync_write("Goal_Position", command)
        next_tick += interval
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


def _hold_and_monitor(bus: Any, target: dict[str, float], max_temperature_c: int, max_error_deg: float) -> None:
    interval = 1.0 / DEFAULT_MONITOR_RATE_HZ
    allowed_error_samples = max(1, math.ceil(DEFAULT_ERROR_GRACE_S * DEFAULT_MONITOR_RATE_HZ))
    excessive_error_samples = 0
    next_report = 0.0

    print("HOLDING_POSE: press Ctrl-C to disable torque and exit", flush=True)
    while True:
        loop_start = time.monotonic()
        present = bus.sync_read("Present_Position")
        temperatures = bus.sync_read("Present_Temperature", normalize=False)
        hottest_name = max(MOTOR_NAMES, key=lambda name: temperatures[name])
        hottest_temperature = temperatures[hottest_name]
        errors = {name: abs(present[name] - target[name]) for name in MOTOR_NAMES}
        max_error_name = max(MOTOR_NAMES, key=lambda name: errors[name])
        max_error = errors[max_error_name]

        if hottest_temperature >= max_temperature_c:
            raise RuntimeError(
                f"Over-temperature: {hottest_name} reached {hottest_temperature} C "
                f"(limit {max_temperature_c} C)."
            )

        excessive_error_samples = excessive_error_samples + 1 if max_error > max_error_deg else 0
        if excessive_error_samples >= allowed_error_samples:
            raise RuntimeError(
                f"Tracking error remained too large: {max_error_name} error={max_error:.2f} degrees "
                f"(limit {max_error_deg:.2f}). Check for a collision, obstruction, or insufficient power."
            )

        if loop_start >= next_report:
            _print_pose("present_deg", present)
            print(
                f"hottest_motor: {hottest_name} {hottest_temperature} C; "
                f"max_tracking_error: {max_error_name} {max_error:.2f} deg",
                flush=True,
            )
            next_report = loop_start + 1.0

        sleep_time = interval - (time.monotonic() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    leisaac_root = args.leisaac_root.expanduser().resolve()

    if not 0 < args.speed_deg_s <= MAX_ALLOWED_SPEED_DEG_S:
        parser.error(f"--speed-deg-s must be in (0, {MAX_ALLOWED_SPEED_DEG_S:g}].")
    if args.margin_deg < 0:
        parser.error("--margin-deg must not be negative.")
    if not 40 <= args.max_temperature_c <= 75:
        parser.error("--max-temperature-c must be between 40 and 75.")
    if args.max_tracking_error_deg <= 0:
        parser.error("--max-tracking-error-deg must be positive.")

    calibration_path = _calibration_path(leisaac_root, args.id)
    calibration_data = _load_calibration_data(calibration_path)
    limits = _angle_limits(calibration_data, args.margin_deg)
    targets = _read_angles(args.angles)
    _validate_targets(targets, limits)

    print(f"leisaac_root: {leisaac_root}")
    print(f"calibration_file: {calibration_path}")
    print(f"port: {args.port}")
    print("baudrate: 1000000 (LeIsaac Feetech default)")
    print("motor_model: sts3215; ids: 1..6")
    print("persistent_configuration_writes: NONE")
    _print_limits(limits)
    _print_pose("requested_target_deg", targets)

    if not args.execute:
        print("DRY_RUN_OK: no serial port was opened and torque was not enabled.")
        print("Re-run with --execute only after checking the pose, clearing the workspace, and supporting the arm.")
        return 0

    print()
    print("WARNING: this will actively torque and move all six servos.")
    print("Stop the Leader publisher first; only one process may use COM7.")
    print("Keep hands, cables, and objects outside the arm's swept volume.")
    confirmation = input(f"Type {EXECUTION_CONFIRMATION} to continue: ").strip()
    if confirmation != EXECUTION_CONFIRMATION:
        print("ABORTED: confirmation did not match; no serial port was opened.")
        return 2

    bus = _build_bus(args.port, leisaac_root, calibration_data)
    connected = False
    try:
        bus.connect()
        connected = True
        operating_modes = bus.sync_read("Operating_Mode", normalize=False)
        invalid_modes = {name: mode for name, mode in operating_modes.items() if mode != 0}
        if invalid_modes:
            raise RuntimeError(
                f"Motors are not all in position-control mode (expected Operating_Mode=0): {invalid_modes}. "
                "This script will not modify operating mode automatically."
            )

        current = bus.sync_read("Present_Position")
        _print_pose("current_deg", current)
        requested_move = {name: targets[name] - current[name] for name in MOTOR_NAMES}
        _print_pose("requested_move_deg", requested_move)
        max_move_name = max(MOTOR_NAMES, key=lambda name: abs(requested_move[name]))
        print(f"max_requested_move: {max_move_name} {abs(requested_move[max_move_name]):.2f} deg")
        target_confirmation = input(f"Type {TARGET_CONFIRMATION} to enable torque and move: ").strip()
        if target_confirmation != TARGET_CONFIRMATION:
            print("ABORTED: target confirmation did not match; torque was not enabled.")
            return 2

        # Establish a no-jump goal before torque is enabled.
        bus.sync_write("Goal_Position", current)
        bus.enable_torque(num_retry=2)
        print("TORQUE_ENABLED", flush=True)

        _ramp_to_target(bus, current, targets, args.speed_deg_s)
        _print_pose("holding_target_deg", targets)
        _hold_and_monitor(bus, targets, args.max_temperature_c, args.max_tracking_error_deg)
    except KeyboardInterrupt:
        print("\nSTOP_REQUESTED: disabling torque", flush=True)
    finally:
        if connected:
            try:
                bus.disconnect(disable_torque=True)
                print("TORQUE_DISABLED_AND_PORT_CLOSED", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"CRITICAL: automatic torque disable failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                print("Disconnect servo power immediately.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
