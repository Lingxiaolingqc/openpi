# ruff: noqa: FBT001, FBT002, PT009, PT018, PT027

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
import unittest

MODULE_PATH = Path(__file__).with_name("bounded_all_joint_follow_test.py")
SPEC = importlib.util.spec_from_file_location("bounded_all_joint_follow_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeBus:
    def __init__(self, *, leader: bool, tracking_offset: float = 0.0) -> None:
        self.leader = leader
        self.tracking_offset = tracking_offset
        self.events: list[Any] = []
        self.pose = dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, 0.0)
        self.enabled = False
        self.position_reads = 0

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
            self.position_reads += 1
            pose = self.pose.copy()
            if self.leader and 2 <= self.position_reads <= 7:
                pose[MODULE.no_jump.hold.MOTOR_NAMES[self.position_reads - 2]] = 1.0
            if not self.leader and self.enabled:
                pose["shoulder_pan"] += self.tracking_offset
            return pose
        raise AssertionError(register)

    def sync_write(self, register: str, values: dict[str, float]) -> None:
        if self.leader:
            raise AssertionError("Leader must never receive a register write")
        self.events.append(("write", register, values.copy()))
        self.pose = values.copy()

    def enable_torque(self, num_retry: int) -> None:
        if self.leader:
            raise AssertionError("Leader torque must never be enabled")
        self.events.append("enable")
        self.enabled = True

    def disconnect(self, disable_torque: bool) -> None:
        self.events.append(("disconnect", disable_torque))
        if disable_torque:
            self.enabled = False


CALIBRATION = {name: {"range_min": 0, "range_max": 4096} for name in MODULE.no_jump.hold.MOTOR_NAMES}
CALIBRATIONS = {"leader": CALIBRATION, "follower": CALIBRATION}
LIMITS = dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, (-90.0, 90.0))


def _run(leader: FakeBus, follower: FakeBus) -> dict[str, Any]:
    return MODULE.run_bounded_all_joint_test(
        leader,
        follower,
        CALIBRATIONS,
        LIMITS,
        duration_s=0.7,
        ready_s=0.0,
        finish_hold_s=0.0,
        control_rate_hz=10.0,
        max_excursion_deg=3.0,
        min_excursion_deg=0.5,
        final_leader_tolerance_deg=1.0,
        max_speed_deg_s=5.0,
        max_tracking_error_deg=5.0,
        max_temperature_c=65,
        fault_hold_s=0.0,
        sleep=lambda _: None,
    )


class BoundedAllJointFollowTest(unittest.TestCase):
    def test_all_joints_move_bounded_and_ports_close(self) -> None:
        leader = FakeBus(leader=True)
        follower = FakeBus(leader=False)
        result = _run(leader, follower)

        writes = [event for event in follower.events if isinstance(event, tuple) and event[0] == "write"]
        self.assertLess(follower.events.index(writes[0]), follower.events.index("enable"))
        for event in writes:
            self.assertLessEqual(max(abs(value) for value in event[2].values()), 3.0)
        self.assertEqual(follower.events[-1], ("disconnect", True))
        self.assertEqual(leader.events[-1], ("disconnect", False))
        self.assertEqual(result["status"], "PASS")

    def test_tracking_fault_disables_follower(self) -> None:
        leader = FakeBus(leader=True)
        follower = FakeBus(leader=False, tracking_offset=5.1)

        with self.assertRaisesRegex(RuntimeError, "tracking error"):
            _run(leader, follower)

        self.assertEqual(follower.events[-1], ("disconnect", True))
        self.assertEqual(leader.events[-1], ("disconnect", False))


if __name__ == "__main__":
    unittest.main()
