from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("dual_arm_alignment_monitor.py")
SPEC = importlib.util.spec_from_file_location("dual_arm_alignment_monitor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AlignmentErrorsTest(unittest.TestCase):
    def test_signed_mapped_target_minus_follower_error(self) -> None:
        mapped_target = dict.fromkeys(MODULE.hold.MOTOR_NAMES, 2.0)
        follower = dict.fromkeys(MODULE.hold.MOTOR_NAMES, -1.0)

        errors = MODULE.alignment_errors(mapped_target, follower)

        self.assertEqual(errors, dict.fromkeys(MODULE.hold.MOTOR_NAMES, 3.0))

    def test_leader_minimum_maps_to_different_follower_minimum(self) -> None:
        leader_calibration = {
            name: {"range_min": 1000, "range_max": 2000}
            for name in MODULE.hold.MOTOR_NAMES
        }
        follower_calibration = {
            name: {"range_min": 500, "range_max": 2500}
            for name in MODULE.hold.MOTOR_NAMES
        }
        leader_min_deg = (1000 - 1500) * 360 / MODULE.hold.MAX_RESOLUTION_VALUE
        follower_min_deg = (500 - 1500) * 360 / MODULE.hold.MAX_RESOLUTION_VALUE
        leader = dict.fromkeys(MODULE.hold.MOTOR_NAMES, leader_min_deg)

        mapped = MODULE.mapped_follower_target(
            leader,
            leader_calibration,
            follower_calibration,
        )

        for value in mapped.values():
            self.assertAlmostEqual(value, follower_min_deg)


if __name__ == "__main__":
    unittest.main()
