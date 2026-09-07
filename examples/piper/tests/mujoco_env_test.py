import os
from pathlib import Path

import numpy as np
import pytest

from examples.piper import contract
from examples.piper import expert
from examples.piper import ik
from examples.piper import mujoco_env

mujoco = pytest.importorskip("mujoco")


def _model_dir() -> Path:
    root = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if not root:
        pytest.skip("MUJOCO_MENAGERIE_PATH is not set")
    return mujoco_env.resolve_model_dir(Path(root))


def test_model_contract_and_deterministic_reset() -> None:
    with mujoco_env.PiperRedCubeToBoxEnv(model_dir=_model_dir(), render=False) as env:
        first, first_info = env.reset(seed=9)
        second, second_info = env.reset(seed=9)
        assert env.model.nq == 15
        assert env.model.nu == 7
        assert first.state.shape == (contract.ACTION_DIM,)
        np.testing.assert_allclose(first.state, second.state)
        np.testing.assert_allclose(first_info.cube_position_m, second_info.cube_position_m)
        np.testing.assert_allclose(first_info.target_position_m, second_info.target_position_m)


def test_ik_does_not_teleport_live_simulation() -> None:
    with mujoco_env.PiperRedCubeToBoxEnv(model_dir=_model_dir(), render=False) as env:
        env.reset(seed=4)
        before = env.data.qpos.copy()
        result = ik.solve_site_ik(env, env.cube_position + np.array([0.0, 0.0, 0.08]))
        assert result.converged
        np.testing.assert_array_equal(env.data.qpos, before)


def test_reference_expert_episode_succeeds() -> None:
    with mujoco_env.PiperRedCubeToBoxEnv(model_dir=_model_dir(), render=False) as env:
        report = expert.run_expert_episode(env, seed=42)
    assert report.success, report


def test_fault_hold_uses_fresh_measured_pose() -> None:
    with mujoco_env.PiperRedCubeToBoxEnv(model_dir=_model_dir(), render=False) as env:
        observation, _ = env.reset(seed=11)
        step_before = env.step_count
        held = env.hold_measured_pose()
        assert env.step_count == step_before + 1
        np.testing.assert_allclose(held.state, observation.state, atol=2e-3)
