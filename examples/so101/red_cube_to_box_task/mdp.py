"""MDP terms for the RedCubeToBox task."""

from isaaclab.managers import SceneEntityCfg
import torch


def cube_inside_target_box(
    env,
    cube_cfg: SceneEntityCfg | None = None,
    floor_cfg: SceneEntityCfg | None = None,
    inner_half_extent: float = 0.045,
    floor_thickness: float = 0.008,
    wall_height: float = 0.060,
    maximum_speed: float = 0.15,
) -> torch.Tensor:
    """Return whether the cube is fully inside the target tray and nearly settled.

    The target floor is a tracked kinematic rigid object, so the predicate is
    expressed relative to its current world pose. This keeps the success test
    valid if target-box position randomization is added later.
    """

    cube = env.scene[(cube_cfg or SceneEntityCfg("cube")).name]
    floor = env.scene[(floor_cfg or SceneEntityCfg("target_box_floor")).name]

    offset = cube.data.root_pos_w - floor.data.root_pos_w
    inside_xy = torch.logical_and(
        torch.abs(offset[:, 0]) <= inner_half_extent,
        torch.abs(offset[:, 1]) <= inner_half_extent,
    )

    cube_center_above_floor = offset[:, 2] - floor_thickness / 2.0
    inside_z = torch.logical_and(
        cube_center_above_floor > 0.0,
        cube_center_above_floor < wall_height,
    )
    settled = torch.linalg.vector_norm(cube.data.root_lin_vel_w, dim=-1) <= maximum_speed
    return inside_xy & inside_z & settled
