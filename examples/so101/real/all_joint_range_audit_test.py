from __future__ import annotations

import copy
import unittest

import all_joint_range_audit as audit


class FakeBus:
    def __init__(self) -> None:
        self.positions = [
            {name: 1000 for name in audit.hold.MOTOR_NAMES},
            {name: 1200 for name in audit.hold.MOTOR_NAMES},
        ]
        self.position_index = 0

    def sync_read(self, register: str, normalize: bool = True):
        del normalize
        if register == "Present_Position":
            value = self.positions[min(self.position_index, len(self.positions) - 1)]
            self.position_index += 1
            return value
        if register == "Present_Temperature":
            return {name: 30 for name in audit.hold.MOTOR_NAMES}
        if register == "Torque_Enable":
            return {name: 0 for name in audit.hold.MOTOR_NAMES}
        raise AssertionError(register)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class AllJointRangeAuditTest(unittest.TestCase):
    def test_collects_raw_extrema_without_writes(self) -> None:
        bus = FakeBus()
        clock = FakeClock()
        result = audit.collect_ranges(
            bus,
            duration_s=10.0,
            sample_rate_hz=10.0,
            max_temperature_c=65,
            stop_requested=lambda: bus.position_index >= 2,
            clock=clock,
            sleep=clock.sleep,
        )
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["range_min"]["shoulder_pan"], 1000)
        self.assertEqual(result["range_max"]["shoulder_pan"], 1200)

    def test_candidate_changes_only_ranges(self) -> None:
        active = {
            name: {
                "id": index,
                "drive_mode": 0,
                "homing_offset": index * 10,
                "range_min": 900,
                "range_max": 1300,
            }
            for index, name in enumerate(audit.hold.MOTOR_NAMES, start=1)
        }
        original = copy.deepcopy(active)
        result = {
            "range_min": {name: 800 for name in audit.hold.MOTOR_NAMES},
            "range_max": {name: 1400 for name in audit.hold.MOTOR_NAMES},
        }
        candidate, comparison = audit.build_candidate(active, result, min_span_raw=100)
        self.assertEqual(active, original)
        for name in audit.hold.MOTOR_NAMES:
            self.assertEqual(candidate[name]["homing_offset"], original[name]["homing_offset"])
            self.assertEqual(candidate[name]["range_min"], 800)
            self.assertEqual(candidate[name]["range_max"], 1400)
            self.assertEqual(comparison[name]["span_ratio"], 1.5)

    def test_incomplete_sweep_is_rejected(self) -> None:
        active = {
            name: {
                "id": index,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 900,
                "range_max": 1300,
            }
            for index, name in enumerate(audit.hold.MOTOR_NAMES, start=1)
        }
        result = {
            "range_min": {name: 1000 for name in audit.hold.MOTOR_NAMES},
            "range_max": {name: 1000 for name in audit.hold.MOTOR_NAMES},
        }
        with self.assertRaisesRegex(RuntimeError, "Incomplete sweep"):
            audit.build_candidate(active, result, min_span_raw=100)

    def test_unselected_ranges_remain_unchanged(self) -> None:
        active = {
            name: {
                "id": index,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 900,
                "range_max": 1300,
            }
            for index, name in enumerate(audit.hold.MOTOR_NAMES, start=1)
        }
        result = {
            "range_min": {name: 800 for name in audit.hold.MOTOR_NAMES},
            "range_max": {name: 1400 for name in audit.hold.MOTOR_NAMES},
        }
        candidate, comparison = audit.build_candidate(
            active,
            result,
            min_span_raw=100,
            selected_joints=("gripper",),
        )
        self.assertEqual(tuple(comparison), ("gripper",))
        self.assertEqual(candidate["gripper"]["range_min"], 800)
        self.assertEqual(candidate["shoulder_pan"]["range_min"], 900)


if __name__ == "__main__":
    unittest.main()
