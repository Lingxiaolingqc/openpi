"""Camera freshness and deterministic freeze injection for S6 simulation rollouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any

import numpy as np


class CameraSafetyError(RuntimeError):
    """Typed camera fault consumed by the rollout's existing S6 safe-state path."""

    def __init__(
        self,
        fault_type: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        detection_started_monotonic_ns: int | None = None,
    ) -> None:
        super().__init__(message)
        self.fault_type = fault_type
        self.details = details or {}
        self.detection_started_monotonic_ns = detection_started_monotonic_ns


@dataclass(frozen=True)
class CameraFrameToken:
    """One camera sensor update token and the audited field that supplied it."""

    source: str
    values: tuple[int | float, ...]

    def as_json(self) -> dict[str, Any]:
        return {"source": self.source, "values": list(self.values)}


@dataclass(frozen=True)
class _CameraSample:
    frame_token: CameraFrameToken
    fingerprint: str


def _numeric_token(value: Any, *, camera_name: str, source: str) -> tuple[int | float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0 or array.dtype.kind not in "iuf":
        raise CameraSafetyError(
            "camera_frame_token_invalid",
            f"Camera {camera_name!r} frame token {source!r} is not a non-empty numeric value",
            details={"camera_name": camera_name, "frame_token_source": source},
        )
    flat = array.reshape(-1)
    if not bool(np.isfinite(flat).all()):
        raise CameraSafetyError(
            "camera_frame_token_invalid",
            f"Camera {camera_name!r} frame token {source!r} contains NaN or infinity",
            details={"camera_name": camera_name, "frame_token_source": source},
        )
    return tuple(int(item) if array.dtype.kind in "iu" else float(item) for item in flat)


def read_camera_frame_tokens(
    sensors: Mapping[str, Any],
    camera_names: tuple[str, ...],
) -> dict[str, CameraFrameToken]:
    """Read per-camera update tokens without forcing or modifying a sensor update.

    IsaacLab 2.x keeps the timestamp of the last completed sensor buffer update in
    ``_timestamp_last_update``. Public frame counters are preferred when a future
    sensor implementation exposes one; the audited timestamp remains the fallback
    used by the current LeIsaac environment.
    """

    tokens: dict[str, CameraFrameToken] = {}
    candidate_fields = ("frame_count", "frame_id", "frame", "_frame", "_timestamp_last_update")
    for camera_name in camera_names:
        if camera_name not in sensors:
            raise CameraSafetyError(
                "camera_frame_token_unavailable",
                f"Camera sensor {camera_name!r} is missing from the scene",
                details={"camera_name": camera_name, "max_undetected_old_action_steps": 0},
            )
        sensor = sensors[camera_name]
        for field in candidate_fields:
            if not hasattr(sensor, field):
                continue
            value = getattr(sensor, field)
            if callable(value):
                continue
            tokens[camera_name] = CameraFrameToken(
                source=field,
                values=_numeric_token(value, camera_name=camera_name, source=field),
            )
            break
        else:
            raise CameraSafetyError(
                "camera_frame_token_unavailable",
                f"Camera sensor {camera_name!r} exposes no supported frame/update token",
                details={
                    "camera_name": camera_name,
                    "supported_frame_token_fields": list(candidate_fields),
                    "max_undetected_old_action_steps": 0,
                },
            )
    return tokens


def camera_observation_fingerprint(image: Any, *, grid_size: int = 16) -> str:
    """Hash a small deterministic RGB grid without copying a full GPU frame."""

    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("camera fingerprint grid size must be an integer of at least 2")
    shape = tuple(int(value) for value in getattr(image, "shape", ()))
    if len(shape) == 4:
        if shape[0] != 1:
            raise CameraSafetyError(
                "camera_observation_invalid",
                f"Expected a single-environment camera image, got shape {shape}",
                details={"image_shape": list(shape)},
            )
        image = image[0]
        shape = shape[1:]
    if len(shape) != 3 or shape[0] < 1 or shape[1] < 1 or shape[2] < 1:
        raise CameraSafetyError(
            "camera_observation_invalid",
            f"Expected camera image shape (H, W, C) or (1, H, W, C), got {shape}",
            details={"image_shape": list(shape)},
        )
    row_stride = max(1, math.ceil(shape[0] / grid_size))
    column_stride = max(1, math.ceil(shape[1] / grid_size))
    sample = image[::row_stride, ::column_stride, :]
    sample = sample[:grid_size, :grid_size, :]
    if hasattr(sample, "detach"):
        sample = sample.detach().cpu().numpy()
    sample_array = np.ascontiguousarray(np.asarray(sample))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(shape).encode())
    digest.update(sample_array.dtype.str.encode())
    digest.update(sample_array.tobytes())
    return digest.hexdigest()


class CameraFreshnessTracker:
    """Reject cameras whose update token and sampled RGB remain unchanged too long."""

    def __init__(self, camera_names: tuple[str, ...], *, max_stale_steps: int, fingerprint_grid_size: int = 16) -> None:
        if not camera_names:
            raise ValueError("at least one camera is required for S6 freshness monitoring")
        if isinstance(max_stale_steps, bool) or not isinstance(max_stale_steps, int) or max_stale_steps < 0:
            raise ValueError("camera max stale steps must be a non-negative integer")
        self.camera_names = camera_names
        self.max_stale_steps = max_stale_steps
        self.fingerprint_grid_size = fingerprint_grid_size
        self._previous: dict[str, _CameraSample] = {}
        self._stale_steps = dict.fromkeys(camera_names, 0)
        self._stale_started_monotonic_ns: dict[str, int | None] = dict.fromkeys(camera_names)

    def reset(
        self,
        policy_observation: Mapping[str, Any],
        frame_tokens: Mapping[str, CameraFrameToken],
    ) -> None:
        self._previous = self._samples(policy_observation, frame_tokens)
        self._stale_steps = dict.fromkeys(self.camera_names, 0)
        self._stale_started_monotonic_ns = dict.fromkeys(self.camera_names)

    def observe(
        self,
        policy_observation: Mapping[str, Any],
        frame_tokens: Mapping[str, CameraFrameToken],
        *,
        environment_step_after: int,
        injection_active: bool = False,
        observed_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        if not self._previous:
            raise RuntimeError("camera freshness tracker must be reset before observations are checked")
        if observed_monotonic_ns is None:
            observed_monotonic_ns = time.monotonic_ns()
        if (
            isinstance(observed_monotonic_ns, bool)
            or not isinstance(observed_monotonic_ns, int)
            or observed_monotonic_ns <= 0
        ):
            raise ValueError("camera observation monotonic timestamp must be a positive integer")
        current = self._samples(policy_observation, frame_tokens)
        for camera_name in self.camera_names:
            previous_sample = self._previous[camera_name]
            current_sample = current[camera_name]
            unchanged = (
                current_sample.frame_token == previous_sample.frame_token
                and current_sample.fingerprint == previous_sample.fingerprint
            )
            self._stale_steps[camera_name] = self._stale_steps[camera_name] + 1 if unchanged else 0
            stale_steps = self._stale_steps[camera_name]
            if unchanged and stale_steps == 1:
                self._stale_started_monotonic_ns[camera_name] = observed_monotonic_ns
            elif not unchanged:
                self._stale_started_monotonic_ns[camera_name] = None
            if stale_steps > self.max_stale_steps:
                stale_started_monotonic_ns = self._stale_started_monotonic_ns[camera_name]
                if stale_started_monotonic_ns is None:
                    raise RuntimeError("camera stale interval lost its start timestamp")
                raise CameraSafetyError(
                    "camera_freeze",
                    (
                        f"Camera {camera_name!r} update token and RGB fingerprint were unchanged for "
                        f"{stale_steps} consecutive environment steps"
                    ),
                    details={
                        "camera_name": camera_name,
                        "environment_step_after": environment_step_after,
                        "stale_observation_steps": stale_steps,
                        "max_camera_stale_steps": self.max_stale_steps,
                        "max_undetected_old_action_steps": self.max_stale_steps + 1,
                        "frame_token": current_sample.frame_token.as_json(),
                        "fingerprint_prefix": current_sample.fingerprint[:12],
                        "detection_source": "sensor_update_token_and_rgb_fingerprint",
                        "freeze_injected": injection_active,
                    },
                    detection_started_monotonic_ns=stale_started_monotonic_ns,
                )
        self._previous = current
        return {
            "environment_step_after": environment_step_after,
            "camera_names": list(self.camera_names),
            "stale_steps_by_camera": dict(self._stale_steps),
            "maximum_stale_steps_observed": max(self._stale_steps.values(), default=0),
            "frame_tokens": {name: current[name].frame_token.as_json() for name in self.camera_names},
            "fingerprint_prefixes": {name: current[name].fingerprint[:12] for name in self.camera_names},
            "freeze_injected": injection_active,
        }

    def _samples(
        self,
        policy_observation: Mapping[str, Any],
        frame_tokens: Mapping[str, CameraFrameToken],
    ) -> dict[str, _CameraSample]:
        samples: dict[str, _CameraSample] = {}
        for camera_name in self.camera_names:
            if camera_name not in policy_observation:
                raise CameraSafetyError(
                    "camera_observation_missing",
                    f"Policy observation is missing camera {camera_name!r}",
                    details={"camera_name": camera_name, "max_undetected_old_action_steps": 0},
                )
            if camera_name not in frame_tokens:
                raise CameraSafetyError(
                    "camera_frame_token_unavailable",
                    f"No frame token was provided for camera {camera_name!r}",
                    details={"camera_name": camera_name, "max_undetected_old_action_steps": 0},
                )
            samples[camera_name] = _CameraSample(
                frame_token=frame_tokens[camera_name],
                fingerprint=camera_observation_fingerprint(
                    policy_observation[camera_name],
                    grid_size=self.fingerprint_grid_size,
                ),
            )
        return samples


def _clone_image(image: Any) -> Any:
    if hasattr(image, "clone"):
        return image.clone()
    return np.array(image, copy=True)


class CameraFreezeInjector:
    """Freeze policy camera images and monitor tokens after a chosen simulation step."""

    def __init__(self, camera_names: tuple[str, ...], *, freeze_after_step: int) -> None:
        if isinstance(freeze_after_step, bool) or not isinstance(freeze_after_step, int) or freeze_after_step < 1:
            raise ValueError("camera freeze injection step must be a positive integer")
        self.camera_names = camera_names
        self.freeze_after_step = freeze_after_step
        self._frozen_images: dict[str, Any] = {}
        self._frozen_tokens: dict[str, CameraFrameToken] = {}
        self._active = False

    def reset(
        self,
        policy_observation: Mapping[str, Any],
        frame_tokens: Mapping[str, CameraFrameToken],
    ) -> None:
        self._capture(policy_observation, frame_tokens)
        self._active = False

    def apply(
        self,
        policy_observation: Mapping[str, Any],
        frame_tokens: Mapping[str, CameraFrameToken],
        *,
        environment_step_after: int,
    ) -> tuple[dict[str, Any], dict[str, CameraFrameToken], bool, bool]:
        if not self._frozen_images:
            raise RuntimeError("camera freeze injector must be reset before it is applied")
        if environment_step_after < self.freeze_after_step:
            self._capture(policy_observation, frame_tokens)
            return dict(policy_observation), dict(frame_tokens), False, False
        newly_activated = not self._active
        self._active = True
        injected_observation = dict(policy_observation)
        injected_tokens = dict(frame_tokens)
        for camera_name in self.camera_names:
            injected_observation[camera_name] = self._frozen_images[camera_name]
            injected_tokens[camera_name] = self._frozen_tokens[camera_name]
        return injected_observation, injected_tokens, True, newly_activated

    def _capture(
        self,
        policy_observation: Mapping[str, Any],
        frame_tokens: Mapping[str, CameraFrameToken],
    ) -> None:
        self._frozen_images = {name: _clone_image(policy_observation[name]) for name in self.camera_names}
        self._frozen_tokens = {name: frame_tokens[name] for name in self.camera_names}
