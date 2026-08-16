"""Record a torque-off SO-101 range candidate without changing calibration.

This tool only reads servo registers. It never calls motor setup, homing,
calibration writes, torque writes, or goal-position writes. The active
calibration remains unchanged; a successful run creates a candidate JSON and
an audit JSON for manual review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any


REAL_DIR = Path(__file__).resolve().parent
SO101_DIR = REAL_DIR.parent
REPO_ROOT = REAL_DIR.parents[2]
sys.path.insert(0, str(SO101_DIR))

import hold_joint_angles_windows as hold  # noqa: E402
from calibration.verify_frozen_calibrations import verify  # noqa: E402


ARM_CONFIG = {
    "leader": {"port": "COM7", "calibration_id": "leader_arm"},
    "follower": {"port": "COM8", "calibration_id": "follower_arm"},
}
CONFIRMATION = "RECORD_TORQUE_OFF_RANGES"
DEFAULT_DURATION_S = 120.0
DEFAULT_SAMPLE_RATE_HZ = 10.0
DEFAULT_MAX_TEMPERATURE_C = 65
DEFAULT_MIN_SPAN_RAW = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _check_six_values(values: dict[str, Any], label: str) -> dict[str, int]:
    if tuple(values) != hold.MOTOR_NAMES:
        raise RuntimeError(f"{label} joint order does not match the required six-joint order")
    parsed = {name: int(values[name]) for name in hold.MOTOR_NAMES}
    invalid = [name for name, value in parsed.items() if not math.isfinite(float(value))]
    if invalid:
        raise RuntimeError(f"{label} contains non-finite values: {invalid}")
    return parsed


def _verify_read_only_preflight(
    bus: Any,
    calibration: dict[str, dict[str, int]],
    *,
    max_temperature_c: int,
) -> dict[str, int]:
    operating_modes = _check_six_values(
        bus.sync_read("Operating_Mode", normalize=False), "operating modes"
    )
    invalid_modes = {name: value for name, value in operating_modes.items() if value != 0}
    if invalid_modes:
        raise RuntimeError(f"Expected position-control Operating_Mode=0: {invalid_modes}")

    torque_states = _check_six_values(
        bus.sync_read("Torque_Enable", normalize=False), "torque states"
    )
    enabled = {name: value for name, value in torque_states.items() if value != 0}
    if enabled:
        raise RuntimeError(f"Torque must already be disabled before manual range recording: {enabled}")

    offsets = _check_six_values(
        bus.sync_read("Homing_Offset", normalize=False), "homing offsets"
    )
    offset_mismatches = {
        name: {"active_json": calibration[name]["homing_offset"], "servo": offsets[name]}
        for name in hold.MOTOR_NAMES
        if offsets[name] != calibration[name]["homing_offset"]
    }
    if offset_mismatches:
        raise RuntimeError(f"Servo Homing_Offset differs from the frozen calibration: {offset_mismatches}")

    temperatures = _check_six_values(
        bus.sync_read("Present_Temperature", normalize=False), "temperatures"
    )
    hottest = max(temperatures, key=temperatures.get)
    if temperatures[hottest] >= max_temperature_c:
        raise RuntimeError(f"Preflight over-temperature: {hottest}={temperatures[hottest]} C")
    return temperatures


def collect_ranges(
    bus: Any,
    *,
    display_joints: tuple[str, ...] = hold.MOTOR_NAMES,
    duration_s: float,
    sample_rate_hz: float,
    max_temperature_c: int,
    stop_requested: Callable[[], bool],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect raw ranges from an already connected, torque-off bus."""
    start = clock()
    deadline = start + duration_s
    next_tick = start
    next_report = start
    mins = {name: hold.MAX_RESOLUTION_VALUE for name in hold.MOTOR_NAMES}
    maxes = {name: 0 for name in hold.MOTOR_NAMES}
    maximum_temperatures = {name: 0 for name in hold.MOTOR_NAMES}
    samples = 0

    while True:
        positions = _check_six_values(
            bus.sync_read("Present_Position", normalize=False), "raw positions"
        )
        temperatures = _check_six_values(
            bus.sync_read("Present_Temperature", normalize=False), "temperatures"
        )
        torque_states = _check_six_values(
            bus.sync_read("Torque_Enable", normalize=False), "torque states"
        )

        enabled = {name: value for name, value in torque_states.items() if value != 0}
        if enabled:
            raise RuntimeError(f"Torque became enabled during manual recording: {enabled}")
        out_of_range = {
            name: value
            for name, value in positions.items()
            if not 0 <= value <= hold.MAX_RESOLUTION_VALUE
        }
        if out_of_range:
            raise RuntimeError(f"Raw positions are outside 0..4095: {out_of_range}")

        for name in hold.MOTOR_NAMES:
            mins[name] = min(mins[name], positions[name])
            maxes[name] = max(maxes[name], positions[name])
            maximum_temperatures[name] = max(maximum_temperatures[name], temperatures[name])
        samples += 1

        hottest = max(temperatures, key=temperatures.get)
        if temperatures[hottest] >= max_temperature_c:
            raise RuntimeError(f"Over-temperature: {hottest}={temperatures[hottest]} C")

        now = clock()
        if now >= next_report:
            print("\njoint                min     pos     max    span", flush=True)
            for name in display_joints:
                print(
                    f"{name:16s} {mins[name]:7d} {positions[name]:7d} "
                    f"{maxes[name]:7d} {maxes[name] - mins[name]:7d}",
                    flush=True,
                )
            print(
                f"samples={samples} remaining_s={max(0.0, deadline - now):.1f} "
                "press_ENTER_to_finish",
                flush=True,
            )
            next_report = now + 1.0

        if stop_requested() or now >= deadline:
            break
        next_tick += 1.0 / sample_rate_hz
        delay = next_tick - clock()
        if delay > 0:
            sleep(delay)

    return {
        "range_min": mins,
        "range_max": maxes,
        "maximum_temperature_c": maximum_temperatures,
        "sample_count": samples,
        "elapsed_s": clock() - start,
    }


def build_candidate(
    active: dict[str, dict[str, int]],
    result: dict[str, Any],
    *,
    min_span_raw: int,
    selected_joints: tuple[str, ...] = hold.MOTOR_NAMES,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, float | int]]]:
    candidate = copy.deepcopy(active)
    comparison: dict[str, dict[str, float | int]] = {}
    incomplete = []
    for name in selected_joints:
        new_min = int(result["range_min"][name])
        new_max = int(result["range_max"][name])
        new_span = new_max - new_min
        old_min = active[name]["range_min"]
        old_max = active[name]["range_max"]
        old_span = old_max - old_min
        if new_span < min_span_raw:
            incomplete.append(f"{name} span={new_span}")
        candidate[name]["range_min"] = new_min
        candidate[name]["range_max"] = new_max
        comparison[name] = {
            "old_min": old_min,
            "new_min": new_min,
            "min_delta": new_min - old_min,
            "old_max": old_max,
            "new_max": new_max,
            "max_delta": new_max - old_max,
            "old_span": old_span,
            "new_span": new_span,
            "span_ratio": new_span / old_span,
        }
    if incomplete:
        raise RuntimeError(
            "Incomplete sweep; no candidate was saved. Move every joint through its gentle usable "
            f"range and retry: {', '.join(incomplete)}"
        )
    return candidate, comparison


def _enter_pressed() -> bool:
    try:
        import msvcrt
    except ImportError:
        return False
    if not msvcrt.kbhit():
        return False
    key = msvcrt.getwch()
    if key == "\x03":
        raise KeyboardInterrupt
    return key in {"\r", "\n"}


def _unique_output_paths(output_dir: Path, arm: str) -> tuple[Path, Path]:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    for suffix in ("", *[f"-{index}" for index in range(1, 1000)]):
        base = output_dir / f"{stamp}-{arm}{suffix}"
        candidate_path = base.with_suffix(".candidate.json")
        audit_path = base.with_suffix(".audit.json")
        if not candidate_path.exists() and not audit_path.exists():
            return candidate_path, audit_path
    raise RuntimeError("Could not allocate a unique candidate filename")


def _write_json_exclusive(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=4)
        stream.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_CONFIG))
    parser.add_argument(
        "--joints",
        nargs="+",
        choices=hold.MOTOR_NAMES,
        default=list(hold.MOTOR_NAMES),
        help="Joint ranges to place in the candidate; unselected joints remain unchanged.",
    )
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--min-span-raw", type=int, default=DEFAULT_MIN_SPAN_RAW)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REAL_DIR / "calibration" / "candidates",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not 10.0 <= args.duration_s <= 300.0:
        raise SystemExit("--duration-s must be between 10 and 300 seconds")
    if not 2.0 <= args.sample_rate_hz <= 30.0:
        raise SystemExit("--sample-rate-hz must be between 2 and 30 Hz")
    if not 1 <= args.min_span_raw <= 1000:
        raise SystemExit("--min-span-raw must be between 1 and 1000")

    lock_path = REAL_DIR / "calibration" / "calibration_lock.json"
    if verify(lock_path, REPO_ROOT) != 0:
        return 2
    print("CALIBRATION_GATE_PASS", flush=True)

    config = ARM_CONFIG[args.arm]
    selected_joints = tuple(dict.fromkeys(args.joints))
    print(
        f"arm={args.arm} port={config['port']} calibration_id={config['calibration_id']}\n"
        f"selected_joints={','.join(selected_joints)}\n"
        "SERVO_WRITES=DISABLED active_calibration_will_not_change",
        flush=True,
    )
    if not args.execute or args.confirm != CONFIRMATION:
        print(
            "DRY_RUN_ONLY: no serial port was opened. To record, add "
            f"--execute --confirm {CONFIRMATION}"
        )
        return 0

    leisaac_root = REPO_ROOT / "tmp" / "leisaac-v0.4.0"
    calibration_path = hold._calibration_path(leisaac_root, str(config["calibration_id"]))
    active = hold._load_calibration_data(calibration_path)
    bus = hold._build_bus(str(config["port"]), leisaac_root, active)

    print(
        "Move every selected joint slowly through its gentle usable range. Do not force a mechanical stop.\n"
        "Press ENTER when every selected joint has reached both endpoints.",
        flush=True,
    )
    for remaining in (3, 2, 1):
        print(f"read_only_capture_starts_in={remaining}", flush=True)
        time.sleep(1.0)

    connected = False
    try:
        bus.connect()
        connected = True
        preflight_temperatures = _verify_read_only_preflight(
            bus,
            active,
            max_temperature_c=DEFAULT_MAX_TEMPERATURE_C,
        )
        result = collect_ranges(
            bus,
            display_joints=selected_joints,
            duration_s=args.duration_s,
            sample_rate_hz=args.sample_rate_hz,
            max_temperature_c=DEFAULT_MAX_TEMPERATURE_C,
            stop_requested=_enter_pressed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"RANGE_AUDIT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if connected:
            bus.disconnect(disable_torque=False)
            print(f"READ_ONLY_PORT_CLOSED port={config['port']}", flush=True)

    try:
        candidate, comparison = build_candidate(
            active,
            result,
            min_span_raw=args.min_span_raw,
            selected_joints=selected_joints,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"RANGE_AUDIT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    candidate_path, audit_path = _unique_output_paths(args.output_dir, args.arm)
    audit = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "CANDIDATE_ONLY_NOT_ACTIVE",
        "arm": args.arm,
        "port": config["port"],
        "calibration_id": config["calibration_id"],
        "selected_joints": selected_joints,
        "active_calibration_path": str(calibration_path),
        "active_calibration_sha256": _sha256(calibration_path),
        "servo_writes": False,
        "preflight_temperature_c": preflight_temperatures,
        "maximum_temperature_c": result["maximum_temperature_c"],
        "sample_count": result["sample_count"],
        "elapsed_s": result["elapsed_s"],
        "comparison": comparison,
    }
    try:
        _write_json_exclusive(candidate_path, candidate)
        _write_json_exclusive(audit_path, audit)
    except Exception as exc:  # noqa: BLE001
        print(f"CANDIDATE_WRITE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    print("\nrange comparison:")
    print("joint              old range       new range    span ratio")
    for name in selected_joints:
        row = comparison[name]
        print(
            f"{name:16s} {row['old_min']:4d}..{row['old_max']:<4d}  "
            f"{row['new_min']:4d}..{row['new_max']:<4d}  {row['span_ratio']:9.3f}"
        )
    print(f"RANGE_CANDIDATE_SAVED path={candidate_path}")
    print(f"RANGE_AUDIT_SAVED path={audit_path}")
    print("ACTIVE_CALIBRATION_UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
