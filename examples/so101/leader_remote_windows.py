"""Safe Windows helper for an SO-101 Leader used with LeIsaac remote teleoperation.

The Leader is passive in this workflow: its servos stay torque-disabled while
LeIsaac reads and publishes joint positions. Therefore this helper deliberately
does not write custom PID gains.

Typical workflow on the Windows machine connected to the Leader::

    python examples/so101/leader_remote_windows.py audit \
        --port COM7 --leisaac-root D:\\path\\to\\leisaac

    python examples/so101/leader_remote_windows.py calibrate \
        --port COM7 --leisaac-root D:\\path\\to\\leisaac

    python examples/so101/leader_remote_windows.py publish \
        --port COM7 --leisaac-root D:\\path\\to\\leisaac

``setup-motors`` is only for brand-new or reset motors. It invokes LeRobot's
official ID/baud-rate setup and requires two explicit confirmations. It does
not tune PID gains.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_PORT = "COM7"
DEFAULT_ID = "leader_arm"
DEFAULT_BIND = "tcp://127.0.0.1:5556"
EEPROM_CONFIRMATION = "SET_IDS_AND_BAUDRATE"
PUBLISHER_RELATIVE_PATH = Path(
    "scripts/environments/teleoperation/so101_joint_state_server.py"
)
CALIBRATION_RELATIVE_DIR = Path("scripts/environments/teleoperation/.cache")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _print_serial_ports() -> set[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        print("serial_ports: unavailable (pyserial is not installed)")
        return set()

    ports = sorted(list_ports.comports(), key=lambda item: item.device)
    if not ports:
        print("serial_ports: none detected")
        return set()

    print("serial_ports:")
    for item in ports:
        print(f"  {item.device}: {item.description} [{item.hwid}]")
    return {item.device.upper() for item in ports}


def _leisaac_root(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(path).expanduser().resolve()


def _publisher_path(root: Path | None) -> Path | None:
    return None if root is None else root / PUBLISHER_RELATIVE_PATH


def _calibration_path(root: Path | None, calibration_id: str) -> Path | None:
    return None if root is None else root / CALIBRATION_RELATIVE_DIR / f"{calibration_id}.json"


def _audit_calibration(path: Path | None) -> None:
    if path is None:
        return

    print(f"calibration_file: {path}")
    print(f"calibration_file_exists: {path.is_file()}")
    if not path.is_file():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"calibration_json_valid: False ({type(exc).__name__}: {exc})")
        return

    print("calibration_json_valid: True")
    if isinstance(data, dict):
        print(f"calibration_top_level_keys: {sorted(data)}")
    else:
        print(f"calibration_top_level_type: {type(data).__name__}")


def audit(args: argparse.Namespace) -> int:
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"platform: {sys.platform}")
    print("custom_pid_writes: DISABLED")
    print("leader_runtime_mode: passive position readout with torque disabled")

    for module in ("serial", "zmq", "scservo_sdk"):
        print(f"module_{module}: {'available' if _module_available(module) else 'missing'}")

    for command in ("lerobot-setup-motors", "lerobot-calibrate"):
        print(f"optional_command_{command}: {shutil.which(command) or 'not found'}")

    detected_ports = _print_serial_ports()
    print(f"requested_port: {args.port}")
    print(f"requested_port_detected: {args.port.upper() in detected_ports}")

    root = _leisaac_root(args.leisaac_root)
    publisher = _publisher_path(root)
    if publisher is not None:
        print(f"leisaac_root: {root}")
        print(f"leisaac_publisher: {publisher}")
        print(f"leisaac_publisher_exists: {publisher.is_file()}")
    _audit_calibration(_calibration_path(root, args.id))

    print("AUDIT_COMPLETE")
    return 0


def setup_motors(args: argparse.Namespace) -> int:
    """Run LeRobot's one-time ID/baud-rate setup, never custom PID writes."""
    if args.confirm != EEPROM_CONFIRMATION:
        print(
            "STOP: this operation writes motor IDs and baud rate to EEPROM. "
            f"Re-run with --confirm {EEPROM_CONFIRMATION} only for brand-new or reset motors, "
            "and only when each motor can be connected individually as prompted.",
            file=sys.stderr,
        )
        return 2

    executable = shutil.which("lerobot-setup-motors")
    if executable is None:
        print("STOP: lerobot-setup-motors is not installed in this environment.", file=sys.stderr)
        return 2

    print("This does NOT tune PID gains.")
    print("It uses LeRobot's official routine to set motor IDs 1-6 and a common baud rate.")
    print("Only the single motor requested by the official prompt may be connected to the bus.")
    final_confirmation = input(f"Type {EEPROM_CONFIRMATION} again to continue: ").strip()
    if final_confirmation != EEPROM_CONFIRMATION:
        print("Cancelled without starting the EEPROM setup.")
        return 2

    command = [
        executable,
        "--teleop.type=so101_leader",
        f"--teleop.port={args.port}",
    ]
    print("running:", subprocess.list2cmdline(command))
    return subprocess.call(command)


def _run_publisher(args: argparse.Namespace, *, recalibrate: bool) -> int:
    root = _leisaac_root(args.leisaac_root)
    publisher = _publisher_path(root)
    if publisher is None or not publisher.is_file():
        print(
            "STOP: LeIsaac publisher was not found. Pass --leisaac-root pointing to "
            "the Windows checkout of the pinned LeIsaac version.",
            file=sys.stderr,
        )
        return 2

    calibration = _calibration_path(root, args.id)
    if not recalibrate and (calibration is None or not calibration.is_file()):
        print(
            f"STOP: LeIsaac calibration file is missing: {calibration}. "
            "Run the calibrate subcommand first.",
            file=sys.stderr,
        )
        return 2

    python_executable = args.python or sys.executable
    command = [
        python_executable,
        str(publisher),
        "--port",
        args.port,
        "--id",
        args.id,
        "--rate",
        str(args.rate),
        "--bind",
        args.bind,
    ]
    if recalibrate:
        command.append("--recalibrate")

    print("custom_pid_writes: DISABLED")
    print("running:", subprocess.list2cmdline(command))
    if recalibrate:
        print("Calibration is interactive: the operator must position and move every joint.")
        print("After calibration, the same process continues publishing joint states.")
    return subprocess.call(command)


def calibrate(args: argparse.Namespace) -> int:
    return _run_publisher(args, recalibrate=True)


def publish(args: argparse.Namespace) -> int:
    return _run_publisher(args, recalibrate=False)


def _add_publisher_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"Windows serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--id", default=DEFAULT_ID, help="Stable LeIsaac calibration identifier")
    parser.add_argument("--rate", type=int, default=50, choices=range(1, 101), metavar="1..100")
    parser.add_argument(
        "--bind",
        default=DEFAULT_BIND,
        help=f"ZMQ publisher bind endpoint (default: {DEFAULT_BIND}; intended for an SSH tunnel)",
    )
    parser.add_argument("--leisaac-root", required=True, help="Windows LeIsaac checkout root")
    parser.add_argument(
        "--python",
        help="Python executable from the LeIsaac remote environment (default: current Python)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit, initialize, calibrate, and publish an SO-101 Leader safely on Windows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Read-only dependency, port, and calibration audit")
    audit_parser.add_argument("--port", default=DEFAULT_PORT, help=f"Expected port (default: {DEFAULT_PORT})")
    audit_parser.add_argument("--id", default=DEFAULT_ID, help="LeIsaac calibration identifier")
    audit_parser.add_argument("--leisaac-root", help="Optional Windows LeIsaac checkout root")
    audit_parser.set_defaults(handler=audit)

    setup_parser = subparsers.add_parser(
        "setup-motors",
        help="One-time official motor ID/baud-rate setup (EEPROM write; no PID tuning)",
    )
    setup_parser.add_argument("--port", default=DEFAULT_PORT, help=f"Windows port (default: {DEFAULT_PORT})")
    setup_parser.add_argument(
        "--confirm",
        default="",
        help=f"Required EEPROM confirmation phrase: {EEPROM_CONFIRMATION}",
    )
    setup_parser.set_defaults(handler=setup_motors)

    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Run LeIsaac's interactive calibration, then publish joint states"
    )
    _add_publisher_arguments(calibrate_parser)
    calibrate_parser.set_defaults(handler=calibrate)

    publish_parser = subparsers.add_parser(
        "publish", help="Publish joint states using an existing LeIsaac calibration"
    )
    _add_publisher_arguments(publish_parser)
    publish_parser.set_defaults(handler=publish)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
