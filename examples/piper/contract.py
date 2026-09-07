"""Shared PiPER MuJoCo observation, action, timing, and dataset contract."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

INTERFACE_VERSION = 1
ROBOT_TYPE = "agilex_piper"
END_EFFECTOR = "menagerie_parallel_gripper"
TASK_PROMPT = "place the red cube in the box"

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
STATE_LAYOUT = (*ARM_JOINT_NAMES, "gripper_open_fraction")
ACTION_LAYOUT = STATE_LAYOUT
ACTION_DIM = 7
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 10

PHYSICS_HZ = 500
CONTROL_HZ = 20
PHYSICS_TIMESTEP_S = 1.0 / PHYSICS_HZ
PHYSICS_STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# MuJoCo Menagerie agilex_piper/piper.xml limits, in joint order.
ARM_Q_MIN_RAD = np.array([-2.618, 0.0, -2.697, -1.832, -1.22, -3.14], dtype=np.float64)
ARM_Q_MAX_RAD = np.array([2.618, 3.14, 0.0, 1.832, 1.22, 3.14], dtype=np.float64)
ARM_Q_APP_MARGIN_RAD = np.array([0.02, 0.02, 0.02, 0.02, 0.08, 0.02], dtype=np.float64)
ARM_Q_MIN_APP_RAD = ARM_Q_MIN_RAD + ARM_Q_APP_MARGIN_RAD
ARM_Q_MAX_APP_RAD = ARM_Q_MAX_RAD - ARM_Q_APP_MARGIN_RAD
GRIPPER_JOINT_MAX_M = 0.035

# Conservative per-control-tick changes used by the expert and dataset audit.
ARM_MAX_STEP_RAD = np.array([0.06, 0.06, 0.06, 0.08, 0.08, 0.10], dtype=np.float64)
GRIPPER_MAX_STEP = 0.20
HOME_ARM_Q_RAD = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)


class ContractError(ValueError):
    """Raised when an observation, action, or dataset violates the PiPER contract."""


@dataclass(frozen=True)
class Observation:
    """One pre-step policy observation."""

    image: np.ndarray
    state: np.ndarray
    timestamp_ns: int


def normalize_gripper_joint(joint_position_m: float) -> float:
    """Map the independent Menagerie finger joint from metres to [0, 1]."""

    return float(np.clip(joint_position_m / GRIPPER_JOINT_MAX_M, 0.0, 1.0))


def denormalize_gripper(open_fraction: float) -> float:
    """Map the public [0, 1] gripper command to the Menagerie actuator target."""

    if not math.isfinite(open_fraction):
        raise ContractError("gripper command must be finite")
    if not 0.0 <= open_fraction <= 1.0:
        raise ContractError(f"gripper command must be in [0, 1], got {open_fraction}")
    return open_fraction * GRIPPER_JOINT_MAX_M


def validate_state(state: np.ndarray) -> np.ndarray:
    """Return a float32 state after enforcing the seven-dimensional public contract."""

    state = np.asarray(state, dtype=np.float64)
    if state.shape != (ACTION_DIM,):
        raise ContractError(f"expected state shape ({ACTION_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ContractError("state contains NaN or infinity")
    if np.any(state[:6] < ARM_Q_MIN_RAD) or np.any(state[:6] > ARM_Q_MAX_RAD):
        raise ContractError(f"state arm joints exceed Menagerie limits: {state[:6].tolist()}")
    if not 0.0 <= state[6] <= 1.0:
        raise ContractError("state gripper_open_fraction must be in [0, 1]")
    return state.astype(np.float32)


def validate_action(action: np.ndarray) -> np.ndarray:
    """Validate one absolute six-joint plus normalized-gripper action."""

    try:
        action = validate_state(action)
    except ContractError as exc:
        raise ContractError(str(exc).replace("state", "action")) from exc
    if np.any(action[:6] < ARM_Q_MIN_APP_RAD - 1e-5) or np.any(action[:6] > ARM_Q_MAX_APP_RAD + 1e-5):
        raise ContractError("action arm joints exceed application soft limits")
    return action


def limit_action_step(previous: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Rate-limit an absolute action without changing its public semantics."""

    previous = validate_state(previous).astype(np.float64)
    requested = validate_action(requested).astype(np.float64)
    maximum_step = np.concatenate([ARM_MAX_STEP_RAD, [GRIPPER_MAX_STEP]])
    limited = previous + np.clip(requested - previous, -maximum_step, maximum_step)
    limited[:6] = np.clip(limited[:6], ARM_Q_MIN_APP_RAD, ARM_Q_MAX_APP_RAD)
    limited[6] = np.clip(limited[6], 0.0, 1.0)
    return limited.astype(np.float32)


def training_steps_for_passes(train_frames: int, *, batch_size: int = 8, passes: int = 3) -> int:
    """Compute fixed OpenPI steps from audited frame count and requested equivalent passes."""

    if train_frames < 1 or batch_size < 1 or passes < 1:
        raise ValueError("train_frames, batch_size, and passes must be positive")
    return math.ceil(train_frames / batch_size) * passes
