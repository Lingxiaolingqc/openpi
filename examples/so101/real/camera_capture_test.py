# ruff: noqa: FBT001, FBT002, N802, PT009, PT018, PT027

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

MODULE_PATH = Path(__file__).with_name("camera_capture.py")
SPEC = importlib.util.spec_from_file_location("camera_capture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
TEST_TEMP_ROOT = MODULE_PATH.parents[3] / "tmp" / "so101-real-tests"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class FakeCapture:
    def __init__(self, index: int, *, black: bool = False) -> None:
        self.index = index
        self.black = black
        self.opened = True
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.opened

    def set(self, key: int, value: float) -> bool:
        self.properties[key] = value
        return True

    def get(self, key: int) -> float:
        return self.properties.get(key, 0.0)

    def read(self) -> tuple[bool, np.ndarray]:
        time.sleep(0.002)
        value = 0 if self.black else 10 + self.index
        return True, np.full((480, 640, 3), value, dtype=np.uint8)

    def release(self) -> None:
        self.opened = False


class FakeCV2:
    CAP_DSHOW = 700
    CAP_PROP_FOURCC = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_BUFFERSIZE = 5
    COLOR_BGR2RGB = 6
    COLOR_RGB2BGR = 7

    def __init__(self, *, black: bool = False) -> None:
        self.black = black

    def VideoCapture(self, index: int, backend: int) -> FakeCapture:
        assert backend == self.CAP_DSHOW
        return FakeCapture(index, black=self.black)

    @staticmethod
    def VideoWriter_fourcc(*characters: str) -> int:
        assert characters == ("M", "J", "P", "G")
        return 123

    @staticmethod
    def cvtColor(frame: np.ndarray, code: int) -> np.ndarray:
        assert code in {FakeCV2.COLOR_BGR2RGB, FakeCV2.COLOR_RGB2BGR}
        return frame[..., ::-1].copy()

    @staticmethod
    def imwrite(path: str, frame: np.ndarray) -> bool:
        Path(path).write_bytes(frame[:1, :1].tobytes())
        return True


class CameraCaptureTest(unittest.TestCase):
    def test_dual_capture_uses_explicit_distinct_roles(self) -> None:
        cameras = MODULE.DualCameraCapture(
            front_index=1,
            wrist_index=2,
            cv2_module=FakeCV2(),
        )
        try:
            cameras.start()
            front, wrist = cameras.get_latest_pair()
            self.assertEqual((front.role, front.index), ("front", 1))
            self.assertEqual((wrist.role, wrist.index), ("wrist", 2))
            self.assertEqual(front.rgb.shape, (480, 640, 3))
            with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
                paths = cameras.save_startup_frames(Path(directory))
                self.assertTrue(all(path.exists() for path in paths))
        finally:
            cameras.close()

    def test_same_index_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be different"):
            MODULE.DualCameraCapture(front_index=1, wrist_index=1)

    def test_three_black_frames_fail_warmup(self) -> None:
        camera = MODULE.OpenCVCameraCapture(role="front", index=1, cv2_module=FakeCV2(black=True))
        with self.assertRaisesRegex(RuntimeError, "black frames"):
            camera.start(warmup_timeout_s=1.0)


if __name__ == "__main__":
    unittest.main()
