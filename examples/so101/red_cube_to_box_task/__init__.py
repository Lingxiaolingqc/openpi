"""Runtime registration for the OpenPI RedCubeToBox LeIsaac task."""

import gymnasium as gym

TASK_ID = "OpenPI-LeIsaac-SO101-RedCubeToBox-v0"


if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "red_cube_to_box_task.env_cfg:RedCubeToBoxEnvCfg",
        },
    )
