import numpy as np
import pytest

from examples.piper import contract


def test_gripper_normalization_round_trip() -> None:
    assert contract.normalize_gripper_joint(0.0) == 0.0
    assert contract.normalize_gripper_joint(contract.GRIPPER_JOINT_MAX_M) == 1.0
    assert contract.denormalize_gripper(0.5) == pytest.approx(0.0175)


def test_action_contract_and_rate_limit() -> None:
    previous = np.concatenate([contract.HOME_ARM_Q_RAD, [1.0]])
    requested = previous.copy()
    requested[0] += 0.5
    requested[6] = 0.0
    limited = contract.limit_action_step(previous, requested)
    assert limited[0] - previous[0] == pytest.approx(contract.ARM_MAX_STEP_RAD[0])
    assert previous[6] - limited[6] == pytest.approx(contract.GRIPPER_MAX_STEP)


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(6),
        np.full(7, np.nan),
        np.concatenate([contract.HOME_ARM_Q_RAD, [1.1]]),
        np.concatenate([[contract.ARM_Q_MAX_APP_RAD[0] + 0.1], contract.HOME_ARM_Q_RAD[1:], [1.0]]),
    ],
)
def test_invalid_actions_fail_closed(action: np.ndarray) -> None:
    with pytest.raises(contract.ContractError):
        contract.validate_action(action)


def test_training_steps_are_dataset_sized() -> None:
    assert contract.training_steps_for_passes(1600, batch_size=8, passes=1) == 200
    assert contract.training_steps_for_passes(1601, batch_size=8, passes=3) == 603
