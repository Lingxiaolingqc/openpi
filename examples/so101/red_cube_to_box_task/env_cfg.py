"""LeIsaac environment configuration for moving a red cube into a tray."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeEnvCfg
from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeSceneCfg
from leisaac.tasks.lift_cube.lift_cube_env_cfg import TerminationsCfg as LiftCubeTerminationsCfg

from . import mdp

TARGET_BOX_CENTER_XY = (0.520, -0.36161)
TABLE_SURFACE_Z = 0.04146
TARGET_BOX_INNER_SIZE = 0.110
TARGET_BOX_WALL_THICKNESS = 0.012
TARGET_BOX_FLOOR_THICKNESS = 0.008
TARGET_BOX_WALL_HEIGHT = 0.060
TARGET_BOX_INNER_HALF_EXTENT_FOR_SUCCESS = 0.045
STATE_MACHINE_GRIPPER_CLOSE_POSITION = 0.05


def _target_box_part(
    prim_name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        # Each tracked object uses its own top-level prim. IsaacLab's shape
        # spawner treats the parent of a nested regex path as a source prim and
        # requires it to exist before the first RigidObject is constructed.
        prim_path=f"{{ENV_REGEX_NS}}/TargetBox{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.55, 0.16),
                roughness=0.6,
            ),
        ),
    )


_BOX_X, _BOX_Y = TARGET_BOX_CENTER_XY
_FLOOR_Z = TABLE_SURFACE_Z + TARGET_BOX_FLOOR_THICKNESS / 2.0
_WALL_Z = TABLE_SURFACE_Z + TARGET_BOX_FLOOR_THICKNESS + TARGET_BOX_WALL_HEIGHT / 2.0
_OUTER_SIZE = TARGET_BOX_INNER_SIZE + 2.0 * TARGET_BOX_WALL_THICKNESS
_WALL_OFFSET = TARGET_BOX_INNER_SIZE / 2.0 + TARGET_BOX_WALL_THICKNESS / 2.0


@configclass
class RedCubeToBoxSceneCfg(LiftCubeSceneCfg):
    """LiftCube scene plus a five-piece static green target tray."""

    target_box_floor: RigidObjectCfg = _target_box_part(
        "Floor",
        (_OUTER_SIZE, _OUTER_SIZE, TARGET_BOX_FLOOR_THICKNESS),
        (_BOX_X, _BOX_Y, _FLOOR_Z),
    )
    target_box_left: RigidObjectCfg = _target_box_part(
        "LeftWall",
        (TARGET_BOX_WALL_THICKNESS, _OUTER_SIZE, TARGET_BOX_WALL_HEIGHT),
        (_BOX_X - _WALL_OFFSET, _BOX_Y, _WALL_Z),
    )
    target_box_right: RigidObjectCfg = _target_box_part(
        "RightWall",
        (TARGET_BOX_WALL_THICKNESS, _OUTER_SIZE, TARGET_BOX_WALL_HEIGHT),
        (_BOX_X + _WALL_OFFSET, _BOX_Y, _WALL_Z),
    )
    target_box_front: RigidObjectCfg = _target_box_part(
        "FrontWall",
        (TARGET_BOX_INNER_SIZE, TARGET_BOX_WALL_THICKNESS, TARGET_BOX_WALL_HEIGHT),
        (_BOX_X, _BOX_Y - _WALL_OFFSET, _WALL_Z),
    )
    target_box_back: RigidObjectCfg = _target_box_part(
        "BackWall",
        (TARGET_BOX_INNER_SIZE, TARGET_BOX_WALL_THICKNESS, TARGET_BOX_WALL_HEIGHT),
        (_BOX_X, _BOX_Y + _WALL_OFFSET, _WALL_Z),
    )


@configclass
class RedCubeToBoxTerminationsCfg(LiftCubeTerminationsCfg):
    """Terminate successfully only after the cube is settled inside the tray."""

    success = DoneTerm(
        func=mdp.cube_inside_target_box,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "floor_cfg": SceneEntityCfg("target_box_floor"),
            "inner_half_extent": TARGET_BOX_INNER_HALF_EXTENT_FOR_SUCCESS,
            "floor_thickness": TARGET_BOX_FLOOR_THICKNESS,
            "wall_height": TARGET_BOX_WALL_HEIGHT,
            "maximum_speed": 0.15,
        },
    )


@configclass
class RedCubeToBoxEnvCfg(LiftCubeEnvCfg):
    """SO-101 RedCubeToBox environment built on the validated LiftCube task."""

    scene: RedCubeToBoxSceneCfg = RedCubeToBoxSceneCfg(env_spacing=8.0)
    terminations: RedCubeToBoxTerminationsCfg = RedCubeToBoxTerminationsCfg()
    task_description: str = "Pick up the red cube and place it inside the green box."

    def use_teleop_device(self, teleop_device) -> None:
        super().use_teleop_device(teleop_device)
        if teleop_device == "so101_state_machine":
            self.actions.gripper_action.close_command_expr = {
                "gripper": STATE_MACHINE_GRIPPER_CLOSE_POSITION,
            }
