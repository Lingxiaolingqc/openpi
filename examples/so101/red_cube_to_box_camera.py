"""Camera refresh helpers shared by SO-101 collection and policy rollout."""

from __future__ import annotations


def refresh_camera_observations_without_control(
    env,
    observations,
    *,
    camera_names: tuple[str, ...],
    refreshes: int,
):
    """Refresh RTX camera buffers without advancing physics or applying an action.

    Reset observations can precede the first usable RTX frame.  Advancing an
    environment step fixes the image but also changes the robot state and loses
    the true ``(obs_0, action_0)`` training pair.  Rendering and force-refreshing
    only the named sensors keeps the reset state unchanged.
    """

    if refreshes < 0:
        raise ValueError("camera refreshes must be non-negative")
    for _ in range(refreshes):
        env.sim.render()
        for camera_name in camera_names:
            env.scene.sensors[camera_name].update(env.physics_dt, force_recompute=True)
        observations = env.observation_manager.compute(update_history=False)
    return observations
