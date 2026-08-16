from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).with_name("limited_motion_test.py")
SPEC = importlib.util.spec_from_file_location("limited_motion_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeBus:
    def __init__(self, tracking_offset: float = 0.0) -> None:
        self.events: list[Any] = []
        self.pose = dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, 0.0)
        self.tracking_offset = tracking_offset
        self.enabled = False

    def connect(self) -> None:
        self.events.append("connect")

    def sync_read(self, register: str, normalize: bool = True) -> dict[str, float]:
        self.events.append(("read", register, normalize))
        if register == "Operating_Mode":
            return dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, 0)
        if register == "Torque_Enable":
            return dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, int(self.enabled))
        if register == "Present_Temperature":
            return dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, 30)
        if register == "Present_Position":
            pose = self.pose.copy()
            if self.enabled:
                pose["shoulder_pan"] += self.tracking_offset
            return pose
        raise AssertionError(register)

    def sync_write(self, register: str, values: dict[str, float]) -> None:
        self.events.append(("write", register, values.copy()))
        self.pose = values.copy()

    def enable_torque(self, num_retry: int) -> None:
        self.events.append("enable")
        self.enabled = True

    def disconnect(self, disable_torque: bool) -> None:
        self.events.append(("disconnect", disable_torque))
        self.enabled = False


LIMITS = dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, (-90.0, 90.0))


def _run(bus: FakeBus) -> dict[str, float | str]:
    return MODULE.run_limited_motion_test(
        bus,
        LIMITS,
        joint="shoulder_pan",
        delta_deg=3.0,
        speed_deg_s=15.0,
        control_rate_hz=30.0,
        hold_s=0.0,
        max_temperature_c=65,
        sleep=lambda _: None,
    )


class LimitedMotionTest(unittest.TestCase):
    def test_bounded_steps_return_to_start_and_disable(self) -> None:
        bus = FakeBus()
        result = _run(bus)

        writes = [event for event in bus.events if isinstance(event, tuple) and event[0] == "write"]
        enable_index = bus.events.index("enable")
        first_write_index = bus.events.index(writes[0])
        commanded_pan = [event[2]["shoulder_pan"] for event in writes]
        command_steps = [
            abs(current - previous)
            for previous, current in zip(commanded_pan, commanded_pan[1:], strict=False)
        ]
        self.assertLess(first_write_index, enable_index)
        self.assertLessEqual(max(command_steps), 0.5 + 1e-9)
        self.assertAlmostEqual(commanded_pan[-1], 0.0)
        self.assertEqual(bus.events[-1], ("disconnect", True))
        self.assertEqual(result["status"], "PASS")

    def test_tracking_fault_still_disables(self) -> None:
        bus = FakeBus(tracking_offset=5.1)

        with self.assertRaisesRegex(RuntimeError, "Tracking error"):
            _run(bus)

        self.assertEqual(bus.events[-1], ("disconnect", True))


if __name__ == "__main__":
    unittest.main()
