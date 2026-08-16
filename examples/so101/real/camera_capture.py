"""Asynchronous OpenCV capture for two explicitly assigned USB cameras."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import threading
import time
from typing import Any


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    role: str
    index: int
    sequence: int
    timestamp_s: float
    rgb: Any


class OpenCVCameraCapture:
    """Continuously retain the newest valid RGB frame from one camera."""

    def __init__(
        self,
        *,
        role: str,
        index: int,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        cv2_module: Any | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if role not in {"front", "wrist"}:
            raise ValueError(f"unsupported camera role: {role}")
        if index < 0:
            raise ValueError("camera index must not be negative")
        self.role = role
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cv2 = cv2_module
        self._clock = clock
        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._latest: CameraFrame | None = None
        self._failure: BaseException | None = None
        self._sequence = 0
        self._consecutive_read_failures = 0
        self._consecutive_black_frames = 0

    def cv2_module(self) -> Any:
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        return self._cv2

    def start(self, *, warmup_timeout_s: float = 5.0) -> None:
        if self._thread is not None:
            raise RuntimeError(f"{self.role} camera is already started")
        cv2 = self.cv2_module()
        backend = getattr(cv2, "CAP_DSHOW", 0)
        capture = cv2.VideoCapture(self.index, backend)
        self._capture = capture
        if not capture.isOpened():
            capture.release()
            self._capture = None
            raise RuntimeError(f"could not open {self.role} camera index {self.index}")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"so101-camera-{self.role}-{self.index}",
            daemon=True,
        )
        self._thread.start()
        if not self._frame_ready.wait(timeout=warmup_timeout_s):
            failure = self._failure
            self.close()
            if failure is not None:
                raise RuntimeError(
                    f"{self.role} camera index {self.index} failed during warmup: {failure}"
                ) from failure
            raise RuntimeError(
                f"{self.role} camera index {self.index} produced no valid frame within {warmup_timeout_s:.1f} seconds"
            )
        try:
            self.get_latest(max_age_s=1.0)
        except BaseException:
            self.close()
            raise

    def _capture_loop(self) -> None:
        assert self._capture is not None
        cv2 = self.cv2_module()
        try:
            while not self._stop.is_set():
                ok, bgr = self._capture.read()
                timestamp_s = self._clock()
                if not ok or bgr is None:
                    self._consecutive_read_failures += 1
                    if self._consecutive_read_failures >= 10:
                        raise RuntimeError(f"{self.role} camera returned 10 consecutive failed reads")
                    time.sleep(0.01)
                    continue
                self._consecutive_read_failures = 0
                if bgr.shape != (self.height, self.width, 3):
                    raise RuntimeError(
                        f"{self.role} frame shape {bgr.shape} does not match ({self.height}, {self.width}, 3)"
                    )
                if str(bgr.dtype) != "uint8":
                    raise RuntimeError(f"{self.role} frame dtype {bgr.dtype} is not uint8")
                if int(bgr.max()) <= 1:
                    self._consecutive_black_frames += 1
                    if self._consecutive_black_frames >= 3:
                        raise RuntimeError(f"{self.role} camera produced 3 consecutive black frames")
                    continue
                self._consecutive_black_frames = 0
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self._sequence += 1
                frame = CameraFrame(
                    role=self.role,
                    index=self.index,
                    sequence=self._sequence,
                    timestamp_s=timestamp_s,
                    rgb=rgb,
                )
                with self._lock:
                    self._latest = frame
                self._frame_ready.set()
        except BaseException as exc:
            with self._lock:
                self._failure = exc
            self._frame_ready.set()

    def get_latest(self, *, max_age_s: float, copy: bool = True) -> CameraFrame:
        with self._lock:
            failure = self._failure
            frame = self._latest
        if failure is not None:
            raise RuntimeError(f"{self.role} camera index {self.index} capture failed: {failure}") from failure
        if frame is None:
            raise RuntimeError(f"{self.role} camera index {self.index} has no frame")
        age_s = self._clock() - frame.timestamp_s
        if age_s < -0.001:
            raise RuntimeError(f"{self.role} camera timestamp is in the future")
        if age_s > max_age_s:
            raise RuntimeError(
                f"{self.role} camera frame is stale: {age_s * 1000:.1f} ms (limit {max_age_s * 1000:.1f} ms)"
            )
        rgb = frame.rgb.copy() if copy else frame.rgb
        return dataclasses.replace(frame, rgb=rgb)

    def config(self) -> dict[str, Any]:
        capture = self._capture
        cv2 = self.cv2_module()
        return {
            "role": self.role,
            "index": self.index,
            "requested_width": self.width,
            "requested_height": self.height,
            "requested_fps": self.fps,
            "actual_width": None if capture is None else float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "actual_height": None if capture is None else float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "actual_fps": None if capture is None else float(capture.get(cv2.CAP_PROP_FPS)),
        }

    def close(self) -> None:
        self._stop.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        self._capture = None
        self._thread = None


class DualCameraCapture:
    def __init__(
        self,
        *,
        front_index: int,
        wrist_index: int,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        cv2_module: Any | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        if front_index == wrist_index:
            raise ValueError("front and wrist camera indices must be different")
        common = {
            "width": width,
            "height": height,
            "fps": fps,
            "cv2_module": cv2_module,
            "clock": clock,
        }
        self.front = OpenCVCameraCapture(role="front", index=front_index, **common)
        self.wrist = OpenCVCameraCapture(role="wrist", index=wrist_index, **common)
        self._cv2 = cv2_module

    def start(self) -> None:
        try:
            self.front.start()
            self.wrist.start()
        except BaseException:
            self.close()
            raise

    def get_latest_pair(
        self, *, max_age_s: float = 0.100, max_skew_s: float = 0.100
    ) -> tuple[CameraFrame, CameraFrame]:
        front = self.front.get_latest(max_age_s=max_age_s)
        wrist = self.wrist.get_latest(max_age_s=max_age_s)
        skew_s = abs(front.timestamp_s - wrist.timestamp_s)
        if skew_s > max_skew_s:
            raise RuntimeError(f"camera timestamp skew is {skew_s * 1000:.1f} ms (limit {max_skew_s * 1000:.1f} ms)")
        return front, wrist

    def save_startup_frames(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        front, wrist = self.get_latest_pair(max_age_s=0.100, max_skew_s=0.100)
        cv2 = self.front.cv2_module()
        paths = (
            output_dir / f"front_index_{front.index}_startup.jpg",
            output_dir / f"wrist_index_{wrist.index}_startup.jpg",
        )
        for path, frame in zip(paths, (front, wrist), strict=True):
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError(f"failed to save startup frame: {path}")
        return paths

    def config(self) -> dict[str, Any]:
        return {"front": self.front.config(), "wrist": self.wrist.config()}

    def close(self) -> None:
        self.wrist.close()
        self.front.close()

    def __enter__(self) -> DualCameraCapture:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
