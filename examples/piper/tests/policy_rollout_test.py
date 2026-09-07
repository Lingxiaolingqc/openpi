from types import SimpleNamespace

import numpy as np
import pytest

from examples.piper import contract
from examples.piper import offline_policy_eval
from examples.piper import policy_rollout


def test_action_chunk_extracts_exact_physical_dimensions() -> None:
    action = np.concatenate([contract.HOME_ARM_Q_RAD, [1.0]]).astype(np.float32)
    actions = np.repeat(action[None], contract.ACTION_HORIZON, axis=0)
    result = policy_rollout.validate_action_chunk({"actions": actions}, minimum_actions=contract.ACTION_HORIZON)
    np.testing.assert_array_equal(result, actions)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((9, 7), dtype=np.float32),
        np.zeros((10, 8), dtype=np.float32),
        np.full((10, 7), np.nan, dtype=np.float32),
    ],
)
def test_bad_policy_chunks_are_rejected(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="expected actions shape|NaN or infinity"):
        policy_rollout.validate_action_chunk({"actions": actions}, minimum_actions=contract.ACTION_HORIZON)


def test_offline_action_metrics_are_explicitly_per_dimension() -> None:
    targets = np.zeros((2, contract.ACTION_DIM), dtype=np.float32)
    predictions = np.array([[1, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    metrics = offline_policy_eval.action_metrics(predictions, targets)
    assert metrics["samples"] == 2
    assert metrics["mae"] == pytest.approx(1 / contract.ACTION_DIM)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / contract.ACTION_DIM))
    assert metrics["mae_by_action"]["joint1"] == 1.0
    assert metrics["mae_by_action"]["gripper_open_fraction"] == 0.0


def test_policy_fault_cancels_chunk_and_holds_fresh_measurement() -> None:
    action = np.concatenate([contract.HOME_ARM_Q_RAD, [1.0]]).astype(np.float32)
    observation = contract.Observation(
        image=np.zeros((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3), dtype=np.uint8),
        state=action.copy(),
        timestamp_ns=1,
    )

    class FakeEnv:
        maximum_steps = 20

        def __init__(self) -> None:
            self.step_count = 0
            self.executed: list[np.ndarray] = []
            self.hold_count = 0

        def reset(self, *, seed: int):
            del seed
            self.step_count = 0
            return observation, None

        def step(self, requested: np.ndarray):
            self.executed.append(requested.copy())
            self.step_count += 1
            return SimpleNamespace(observation=observation, terminated=False, truncated=False)

        def hold_measured_pose(self):
            self.hold_count += 1
            self.step_count += 1
            return observation

    class FaultAfterOneChunk:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def metadata(self) -> dict:
            return {}

        def infer(self, _observation: dict) -> dict:
            self.calls += 1
            if self.calls == 1:
                return {"actions": np.repeat(action[None], contract.ACTION_HORIZON, axis=0)}
            raise ConnectionError("test disconnect")

        def close(self) -> None:
            pass

    env = FakeEnv()
    report = policy_rollout.run_policy_episode(env, FaultAfterOneChunk(), seed=7)
    assert not report.success
    assert report.failure_reason == "policy_fault:ConnectionError:test disconnect"
    assert len(env.executed) == contract.ACTION_HORIZON
    assert env.hold_count == 1
