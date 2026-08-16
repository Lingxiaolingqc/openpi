from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).with_name("no_jump_enable_test.py")
SPEC = importlib.util.spec_from_file_location("no_jump_enable_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeBus:
    def __init__(self, monitored_offset: float = 0.0) -> None:
        self.events: list[Any] = []
        self.monitored_offset = monitored_offset
        self.position_reads = 0

    def connect(self) -> None:
        self.events.append("connect")

    def sync_read(self, register: str, normalize: bool = True) -> dict[str, float]:
        self.events.append(("read", register, normalize))
        if register == "Operating_Mode":
            return dict.fromkeys(MODULE.hold.MOTOR_NAMES, 0)
        if register == "Torque_Enable":
            enabled = int("enable" in self.events)
            return dict.fromkeys(MODULE.hold.MOTOR_NAMES, enabled)
        if register == "Present_Temperature":
            return dict.fromkeys(MODULE.hold.MOTOR_NAMES, 30)
        if register == "Present_Position":
            self.position_reads += 1
            pose = dict.fromkeys(MODULE.hold.MOTOR_NAMES, 0.0)
            if self.position_reads > 1:
                pose["elbow_flex"] = self.monitored_offset
            return pose
        raise AssertionError(register)

    def sync_write(self, register: str, values: dict[str, float]) -> None:
        self.events.append(("write", register, values.copy()))

    def enable_torque(self, num_retry: int) -> None:
        assert num_retry == 2
        self.events.append("enable")

    def disconnect(self, disable_torque: bool) -> None:
        self.events.append(("disconnect", disable_torque))


LIMITS = dict.fromkeys(MODULE.hold.MOTOR_NAMES, (-90.0, 90.0))


def _run(bus: FakeBus) -> dict[str, float | str]:
    return MODULE.run_no_jump_test(
        bus,
        LIMITS,
        duration_s=0.0,
        monitor_rate_hz=10.0,
        max_displacement_deg=3.0,
        max_tracking_error_deg=3.0,
        max_temperature_c=65,
    )


class NoJumpEnableTest(unittest.TestCase):
    def test_default_startup_margin_allows_exact_frozen_endpoint(self) -> None:
        calibration = {
            name: {"range_min": 0, "range_max": 4096}
            for name in MODULE.hold.MOTOR_NAMES
        }
        limits = MODULE.hold._angle_limits(calibration, MODULE.DEFAULT_MARGIN_DEG)
        endpoint_pose = {name: limits[name][0] for name in MODULE.hold.MOTOR_NAMES}

        MODULE._check_inside_limits(endpoint_pose, limits)

        self.assertEqual(MODULE.DEFAULT_MARGIN_DEG, 0.0)

    def test_goal_is_measured_pose_before_enable(self) -> None:
        bus = FakeBus()
        result = _run(bus)

        write_index = next(
            index
            for index, event in enumerate(bus.events)
            if isinstance(event, tuple) and event[0] == "write"
        )
        enable_index = bus.events.index("enable")
        write_event = bus.events[write_index]
        self.assertEqual(write_event[1], "Goal_Position")
        self.assertEqual(write_event[2], dict.fromkeys(MODULE.hold.MOTOR_NAMES, 0.0))
        self.assertLess(write_index, enable_index)
        self.assertEqual(bus.events[-1], ("disconnect", True))
        self.assertEqual(result["status"], "PASS")

    def test_excessive_displacement_fails_and_disables_torque(self) -> None:
        bus = FakeBus(monitored_offset=3.1)

        with self.assertRaisesRegex(RuntimeError, "No-jump failure"):
            _run(bus)

        self.assertEqual(bus.events[-1], ("disconnect", True))


if __name__ == "__main__":
    unittest.main()
