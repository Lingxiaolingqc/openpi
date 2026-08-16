# ruff: noqa: FBT001, FBT002, PT009, PT018, PT027

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any, ClassVar
import unittest

import numpy as np

MODULE_PATH = Path(__file__).with_name("teleop_recorder.py")
SPEC = importlib.util.spec_from_file_location("teleop_recorder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
TEST_TEMP_ROOT = MODULE_PATH.parents[3] / "tmp" / "so101-real-tests"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


class FakeBus:
    def __init__(self, *, leader: bool) -> None:
        self.leader = leader
        self.events: list[Any] = []
        self.pose = dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, 0.0)
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
            return self.pose.copy()
        raise AssertionError(register)

    def sync_write(self, register: str, values: dict[str, float]) -> None:
        if self.leader:
            raise AssertionError("Leader must never receive a register write")
        self.events.append(("write", register, values.copy()))
        self.pose = values.copy()

    def enable_torque(self, num_retry: int) -> None:
        if self.leader:
            raise AssertionError("Leader torque must remain disabled")
        self.events.append("enable")
        self.enabled = True

    def disconnect(self, disable_torque: bool) -> None:
        self.events.append(("disconnect", disable_torque))
        if disable_torque:
            self.enabled = False


class FakeCameras:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.image = np.full((480, 640, 3), 50, dtype=np.uint8)
        self.sequence = 0

    def get_latest_pair(self, **_: Any) -> tuple[Any, Any]:
        self.sequence += 1
        timestamp = self.clock()
        return (
            MODULE.camera_capture.CameraFrame(
                role="front",
                index=1,
                sequence=self.sequence,
                timestamp_s=timestamp,
                rgb=self.image,
            ),
            MODULE.camera_capture.CameraFrame(
                role="wrist",
                index=2,
                sequence=self.sequence,
                timestamp_s=timestamp,
                rgb=self.image,
            ),
        )

    @staticmethod
    def config() -> dict[str, Any]:
        return {"front": {"index": 1}, "wrist": {"index": 2}}


class FakeKeys:
    def poll(self) -> list[str]:
        # R is ignored during auto-align and accepted only by the explicit
        # ready gate. The writer exists only after that gate has passed.
        if not FakeWriter.instances:
            return ["r"]
        if FakeWriter.instances[-1].frames_enqueued >= MODULE.MIN_SUCCESS_FRAMES:
            return ["y"]
        return []


class SequenceKeys:
    def __init__(self, *responses: list[str]) -> None:
        self.responses = list(responses)

    def poll(self) -> list[str]:
        return self.responses.pop(0) if self.responses else []


class LargeAlignKeys:
    def __init__(self) -> None:
        self.confirmed = False

    def poll(self) -> list[str]:
        if not self.confirmed:
            self.confirmed = True
            return ["a"]
        if not FakeWriter.instances:
            return ["r"]
        if FakeWriter.instances[-1].frames_enqueued >= MODULE.MIN_SUCCESS_FRAMES:
            return ["y"]
        return []


class MoveThenRecordKeys:
    def __init__(self, leader: FakeBus) -> None:
        self.leader = leader
        self.poll_count = 0

    def poll(self) -> list[str]:
        self.poll_count += 1
        if self.poll_count == 1:
            self.leader.pose["shoulder_pan"] = 2.0
            return []
        return ["r"]


class FakeWriter:
    instances: ClassVar[list[FakeWriter]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.samples: list[Any] = []
        self.finish_args: dict[str, Any] | None = None
        self.frames_enqueued = 0
        self.queue_depth = 0
        self.__class__.instances.append(self)

    def append(self, sample: Any) -> None:
        self.samples.append(sample)
        self.frames_enqueued += 1

    def finish(self, **kwargs: Any) -> Path:
        self.finish_args = kwargs
        return Path("fake-episode.h5")


CALIBRATION = {name: {"range_min": 0, "range_max": 4096} for name in MODULE.no_jump.hold.MOTOR_NAMES}
CALIBRATIONS = {"leader": CALIBRATION, "follower": CALIBRATION}


def make_config(
    dataset_root: Path,
    *,
    max_auto_align_deg: float = 10.0,
    large_auto_align_speed_deg_s: float = MODULE.DEFAULT_LARGE_AUTO_ALIGN_SPEED_DEG_S,
) -> Any:
    return MODULE.RecorderConfig(
        dataset_root=dataset_root,
        target_id=1,
        box_id="A",
        task=MODULE.TASKS[1],
        operator="test",
        object_inventory_version="v1",
        front_camera=1,
        wrist_camera=2,
        speed_deg_s=15.0,
        max_episode_s=30.0,
        motion_margin_deg=1.0,
        max_auto_align_deg=max_auto_align_deg,
        large_auto_align_speed_deg_s=large_auto_align_speed_deg_s,
    )


class TeleopRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeWriter.instances.clear()

    def test_relative_controller_slew_limits_absolute_action(self) -> None:
        limits = dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, (-90.0, 90.0))
        zeros = dict.fromkeys(MODULE.no_jump.hold.MOTOR_NAMES, 0.0)
        controller = MODULE.RelativeJointController(
            anchor_mapped=zeros,
            anchor_follower=zeros,
            follower_limits=limits,
            maximum_step_deg=0.5,
        )
        mapped = zeros.copy()
        mapped["shoulder_pan"] = 2.0
        command = controller.compute(mapped)
        self.assertEqual(command["shoulder_pan"], 0.5)
        self.assertTrue(all(command[name] == 0.0 for name in MODULE.no_jump.hold.MOTOR_NAMES[1:]))

    def test_complete_episode_writes_before_motion_and_always_unloads(self) -> None:
        clock = FakeClock()
        leader = FakeBus(leader=True)
        follower = FakeBus(leader=False)
        # A folded arm may start exactly at the frozen endpoint. No-jump holds
        # that measured pose, then auto-align moves it into the 1-degree interior.
        leader.pose["shoulder_pan"] = -180.0
        follower.pose["shoulder_pan"] = -180.0
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            config = make_config(Path(directory))
            recorder = MODULE.RealTeleopRecorder(
                config=config,
                cameras=FakeCameras(clock),
                calibrations=CALIBRATIONS,
                key_source=FakeKeys(),
                bus_factory=lambda: {"leader": leader, "follower": follower},
                writer_factory=FakeWriter,
                clock=clock,
                sleep=clock.sleep,
            )
            result = recorder.collect_one_episode()

        writes = [event for event in follower.events if isinstance(event, tuple) and event[0] == "write"]
        self.assertLess(follower.events.index(writes[0]), follower.events.index("enable"))
        self.assertEqual(writes[0][2]["shoulder_pan"], -180.0)
        safe_low = recorder.follower_limits["shoulder_pan"][0]
        self.assertTrue(any(event[2]["shoulder_pan"] == safe_low for event in writes))
        maximum_step = max(
            abs(current[2]["shoulder_pan"] - previous[2]["shoulder_pan"])
            for previous, current in zip(writes, writes[1:], strict=False)
        )
        self.assertLessEqual(maximum_step, 0.5)
        self.assertEqual(follower.events[-1], ("disconnect", True))
        self.assertEqual(leader.events[-1], ("disconnect", False))
        self.assertTrue(result.success)
        self.assertEqual(len(FakeWriter.instances), 1)
        writer = FakeWriter.instances[0]
        self.assertGreaterEqual(len(writer.samples), 1)
        self.assertTrue(writer.finish_args["success"])
        self.assertEqual(
            writer.samples[0].joint_pos.shape if hasattr(writer.samples[0].joint_pos, "shape") else (6,), (6,)
        )

    def test_large_auto_align_is_refused_before_torque_enable(self) -> None:
        clock = FakeClock()
        leader = FakeBus(leader=True)
        follower = FakeBus(leader=False)
        leader.pose["shoulder_pan"] = 11.0
        recorder = MODULE.RealTeleopRecorder(
            config=make_config(Path(".")),
            cameras=FakeCameras(clock),
            calibrations=CALIBRATIONS,
            key_source=SequenceKeys(),
            bus_factory=lambda: {"leader": leader, "follower": follower},
            writer_factory=FakeWriter,
            clock=clock,
            sleep=clock.sleep,
        )

        result = recorder.collect_one_episode()

        self.assertFalse(result.success)
        self.assertIn("exceeding the configured 10.00 deg", result.abort_reason)
        self.assertNotIn("enable", follower.events)
        self.assertEqual(follower.events[-1], ("disconnect", True))
        self.assertEqual(leader.events[-1], ("disconnect", False))

    def test_large_wrist_roll_and_gripper_alignment_bypasses_angle_gate_but_stays_slow(self) -> None:
        clock = FakeClock()
        leader = FakeBus(leader=True)
        follower = FakeBus(leader=False)
        leader.pose["wrist_roll"] = 15.0
        leader.pose["gripper"] = -12.0
        recorder = MODULE.RealTeleopRecorder(
            config=make_config(Path(".")),
            cameras=FakeCameras(clock),
            calibrations=CALIBRATIONS,
            # FakeKeys never emits A. Success therefore proves that these two
            # joints do not enter the catch-up angle confirmation gate.
            key_source=FakeKeys(),
            bus_factory=lambda: {"leader": leader, "follower": follower},
            writer_factory=FakeWriter,
            clock=clock,
            sleep=clock.sleep,
        )

        result = recorder.collect_one_episode()

        writes = [event for event in follower.events if isinstance(event, tuple) and event[0] == "write"]
        for joint in MODULE.CATCH_UP_ANGLE_EXEMPT_JOINTS:
            maximum_step = max(
                abs(current[2][joint] - previous[2][joint])
                for previous, current in zip(writes, writes[1:], strict=False)
            )
            self.assertLessEqual(
                maximum_step,
                MODULE.DEFAULT_LARGE_AUTO_ALIGN_SPEED_DEG_S / MODULE.CONTROL_RATE_HZ + 1e-9,
            )
        self.assertTrue(result.success)
        self.assertIn("enable", follower.events)
        self.assertEqual(follower.events[-1], ("disconnect", True))

    def test_angle_gate_exemption_does_not_bypass_frozen_calibration_range(self) -> None:
        for joint in MODULE.CATCH_UP_ANGLE_EXEMPT_JOINTS:
            with self.subTest(joint=joint):
                clock = FakeClock()
                leader = FakeBus(leader=True)
                follower = FakeBus(leader=False)
                leader.pose[joint] = 200.0
                recorder = MODULE.RealTeleopRecorder(
                    config=make_config(Path(".")),
                    cameras=FakeCameras(clock),
                    calibrations=CALIBRATIONS,
                    key_source=SequenceKeys(),
                    bus_factory=lambda: {"leader": leader, "follower": follower},
                    writer_factory=FakeWriter,
                    clock=clock,
                    sleep=clock.sleep,
                )

                result = recorder.collect_one_episode()

                self.assertFalse(result.success)
                self.assertIn("outside the allowed calibrated range", result.abort_reason)
                self.assertIn(joint, result.abort_reason)
                self.assertNotIn("enable", follower.events)
                self.assertEqual(follower.events[-1], ("disconnect", True))
                self.assertEqual(leader.events[-1], ("disconnect", False))

    def test_confirmed_large_auto_align_uses_configured_speed_cap(self) -> None:
        clock = FakeClock()
        leader = FakeBus(leader=True)
        follower = FakeBus(leader=False)
        leader.pose["shoulder_pan"] = 15.0
        keys = LargeAlignKeys()
        recorder = MODULE.RealTeleopRecorder(
            config=make_config(
                Path("."),
                max_auto_align_deg=30.0,
                large_auto_align_speed_deg_s=8.0,
            ),
            cameras=FakeCameras(clock),
            calibrations=CALIBRATIONS,
            key_source=keys,
            bus_factory=lambda: {"leader": leader, "follower": follower},
            writer_factory=FakeWriter,
            clock=clock,
            sleep=clock.sleep,
        )

        result = recorder.collect_one_episode()

        writes = [event for event in follower.events if isinstance(event, tuple) and event[0] == "write"]
        maximum_step = max(
            abs(current[2]["shoulder_pan"] - previous[2]["shoulder_pan"])
            for previous, current in zip(writes, writes[1:], strict=False)
        )
        self.assertTrue(keys.confirmed)
        self.assertTrue(result.success)
        self.assertLessEqual(
            maximum_step,
            8.0 / MODULE.CONTROL_RATE_HZ + 1e-9,
        )
        self.assertLess(follower.events.index(writes[0]), follower.events.index("enable"))
        self.assertEqual(follower.events[-1], ("disconnect", True))

    def test_faster_large_auto_align_keeps_conservative_timeout_budget(self) -> None:
        self.assertAlmostEqual(
            MODULE._auto_align_timeout_s(110.11, 8.0),
            110.11 / MODULE.DEFAULT_LARGE_AUTO_ALIGN_SPEED_DEG_S
            + MODULE.AUTO_ALIGN_TIMEOUT_MARGIN_S,
        )

    def test_large_auto_align_confirmation_handles_q_and_x(self) -> None:
        for key in ("q", "x"):
            with self.subTest(key=key):
                clock = FakeClock()
                leader = FakeBus(leader=True)
                follower = FakeBus(leader=False)
                recorder = MODULE.RealTeleopRecorder(
                    config=make_config(Path("."), max_auto_align_deg=30.0),
                    cameras=FakeCameras(clock),
                    calibrations=CALIBRATIONS,
                    key_source=SequenceKeys([key]),
                    bus_factory=lambda: {"leader": leader, "follower": follower},
                    writer_factory=FakeWriter,
                    clock=clock,
                    sleep=clock.sleep,
                )
                mapped = leader.pose.copy()
                mapped["shoulder_pan"] = 15.0
                buses = {"leader": leader, "follower": follower}

                if key == "x":
                    with self.assertRaises(MODULE.OperatorEmergencyStopError):
                        recorder._wait_for_large_auto_align_confirmation(
                            buses,
                            initial_follower=follower.pose,
                            mapped_leader=mapped,
                        )
                else:
                    self.assertFalse(
                        recorder._wait_for_large_auto_align_confirmation(
                            buses,
                            initial_follower=follower.pose,
                            mapped_leader=mapped,
                        )
                    )

    def test_large_confirmation_cancel_closes_ports_without_enabling_torque(self) -> None:
        for key in ("q", "x"):
            with self.subTest(key=key):
                clock = FakeClock()
                leader = FakeBus(leader=True)
                follower = FakeBus(leader=False)
                leader.pose["shoulder_pan"] = 15.0
                recorder = MODULE.RealTeleopRecorder(
                    config=make_config(Path("."), max_auto_align_deg=30.0),
                    cameras=FakeCameras(clock),
                    calibrations=CALIBRATIONS,
                    key_source=SequenceKeys([key]),
                    bus_factory=lambda: {"leader": leader, "follower": follower},
                    writer_factory=FakeWriter,
                    clock=clock,
                    sleep=clock.sleep,
                )

                result = recorder.collect_one_episode()

                self.assertFalse(result.success)
                self.assertNotIn("enable", follower.events)
                self.assertEqual(follower.events[-1], ("disconnect", True))
                self.assertEqual(leader.events[-1], ("disconnect", False))
                self.assertEqual(result.exit_requested, key == "x")

    def test_wait_for_record_start_handles_r_q_and_x(self) -> None:
        for key in ("r", "q", "x"):
            with self.subTest(key=key):
                clock = FakeClock()
                leader = FakeBus(leader=True)
                follower = FakeBus(leader=False)
                follower.enabled = True
                leader.pose["shoulder_pan"] = 2.0
                follower.pose["shoulder_pan"] = 1.0
                recorder = MODULE.RealTeleopRecorder(
                    config=make_config(Path(".")),
                    cameras=FakeCameras(clock),
                    calibrations=CALIBRATIONS,
                    key_source=SequenceKeys([key]),
                    bus_factory=lambda: {"leader": leader, "follower": follower},
                    writer_factory=FakeWriter,
                    clock=clock,
                    sleep=clock.sleep,
                )
                buses = {"leader": leader, "follower": follower}

                if key == "x":
                    with self.assertRaises(MODULE.OperatorEmergencyStopError):
                        recorder._wait_for_record_start(buses)
                else:
                    result = recorder._wait_for_record_start(buses)
                    if key == "q":
                        self.assertIsNone(result)
                    else:
                        self.assertIsNotNone(result)
                        assert result is not None
                        follower_anchor, mapped_anchor, _ = result
                        self.assertEqual(follower_anchor["shoulder_pan"], 1.0)
                        self.assertEqual(mapped_anchor["shoulder_pan"], 2.0)

    def test_pre_record_stage_follows_leader_before_r_without_writing_an_episode(self) -> None:
        clock = FakeClock()
        leader = FakeBus(leader=True)
        follower = FakeBus(leader=False)
        follower.enabled = True
        keys = MoveThenRecordKeys(leader)
        recorder = MODULE.RealTeleopRecorder(
            config=make_config(Path(".")),
            cameras=FakeCameras(clock),
            calibrations=CALIBRATIONS,
            key_source=keys,
            bus_factory=lambda: {"leader": leader, "follower": follower},
            writer_factory=FakeWriter,
            clock=clock,
            sleep=clock.sleep,
        )

        result = recorder._wait_for_record_start({"leader": leader, "follower": follower})

        self.assertIsNotNone(result)
        assert result is not None
        follower_anchor, mapped_anchor, _ = result
        writes = [event for event in follower.events if isinstance(event, tuple) and event[0] == "write"]
        self.assertGreaterEqual(len(writes), 3)
        self.assertEqual(writes[1][2]["shoulder_pan"], 0.5)
        self.assertEqual(follower_anchor["shoulder_pan"], 0.5)
        self.assertEqual(mapped_anchor["shoulder_pan"], 2.0)
        self.assertEqual(FakeWriter.instances, [])


if __name__ == "__main__":
    unittest.main()
