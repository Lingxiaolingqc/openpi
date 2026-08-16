"""Verify that active SO-101 calibration files match a frozen lock file.

This command is read-only: it never opens a serial port and never changes a
calibration file.  A non-zero exit code means real-hardware motion, recording,
and policy deployment must remain blocked until the mismatch is reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
REQUIRED_FIELDS = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
DEFAULT_LOCK = Path(__file__).with_name("calibration_lock.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing file: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read JSON {path}: {exc}") from exc


def _validate_calibration(data: Any, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"{label}: top level must be an object"]

    if tuple(data) != MOTOR_NAMES:
        errors.append(f"{label}: motor order/names do not match {MOTOR_NAMES}")

    for expected_id, name in enumerate(MOTOR_NAMES, start=1):
        entry = data.get(name)
        if not isinstance(entry, dict):
            errors.append(f"{label}: missing or invalid entry {name!r}")
            continue
        if set(entry) != REQUIRED_FIELDS:
            errors.append(f"{label}: {name} fields are {sorted(entry)}, expected {sorted(REQUIRED_FIELDS)}")
            continue
        if entry["id"] != expected_id:
            errors.append(f"{label}: {name} id={entry['id']}, expected {expected_id}")
        if entry["drive_mode"] != 0:
            errors.append(f"{label}: {name} drive_mode={entry['drive_mode']}, expected 0")
        if not 0 <= entry["range_min"] < entry["range_max"] <= 4095:
            errors.append(
                f"{label}: {name} has invalid range "
                f"{entry['range_min']}..{entry['range_max']}"
            )
    return errors


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def verify(lock_path: Path, repo_root: Path) -> int:
    lock = _load_json(lock_path)
    errors: list[str] = []

    for arm in lock["arms"]:
        role = arm["role"]
        active_path = _resolve(repo_root, arm["active_path"])
        snapshot_path = _resolve(repo_root, arm["snapshot_path"])
        expected_active_hash = arm["active_sha256"].upper()
        expected_snapshot_hash = arm["snapshot_sha256"].upper()

        try:
            active_hash = _sha256(active_path)
            snapshot_hash = _sha256(snapshot_path)
            active_data = _load_json(active_path)
            snapshot_data = _load_json(snapshot_path)
        except RuntimeError as exc:
            errors.append(f"{role}: {exc}")
            continue

        print(f"{role}_active_sha256={active_hash}")
        print(f"{role}_snapshot_sha256={snapshot_hash}")
        if active_hash != expected_active_hash:
            errors.append(f"{role}: active hash changed (expected {expected_active_hash})")
        if snapshot_hash != expected_snapshot_hash:
            errors.append(f"{role}: frozen snapshot hash changed (expected {expected_snapshot_hash})")
        if active_data != snapshot_data:
            errors.append(f"{role}: active and frozen JSON contents differ")
        errors.extend(_validate_calibration(active_data, f"{role} active"))
        errors.extend(_validate_calibration(snapshot_data, f"{role} snapshot"))

    if errors:
        print("CALIBRATION_LOCK_MISMATCH", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"CALIBRATION_LOCK_OK freeze_id={lock['freeze_id']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="OpenPI repository root (normally detected automatically).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    return verify(args.lock.resolve(), args.repo_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
