"""Directly control the closed-gripper jaw detection frame during high alignment."""

from __future__ import annotations

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import subtract_frame_transforms
import torch

from .env_cfg import TABLE_SURFACE_Z
from .legacy_gripper_anchor_safe_xyz_pan_nullspace_align_then_lower_state_machine import (
    RedCubeToBoxLegacyGripperAnchorSafeXyzPanNullspaceAlignThenLowerStateMachine,
)

_CONTROLLED_POINT_RADIUS = 0.010
_TABLE_PROJECTION_RADIUS = 0.008


class RedCubeToBoxLegacyGripperAnchorSafeDirectJawXyzPanNullspaceAlignThenLowerStateMachine(
    RedCubeToBoxLegacyGripperAnchorSafeXyzPanNullspaceAlignThenLowerStateMachine
):
    """Use the closed jaw point itself as the four-row alignment control frame."""

    def __init__(self) -> None:
        super().__init__()
        self._direct_jaw_offset_pos: torch.Tensor | None = None
        self._direct_jaw_offset_quat: torch.Tensor | None = None
        self._controlled_jaw_point_w: torch.Tensor | None = None
        self._table_projection_point_w: torch.Tensor | None = None
        self._point_visualizer: VisualizationMarkers | None = None

    def setup(self, env) -> None:
        super().setup(env)
        self._point_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/RedCubeToBox/DirectJawControlPoints",
                markers={
                    "controlled_jaw": sim_utils.SphereCfg(
                        radius=_CONTROLLED_POINT_RADIUS,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.15)),
                    ),
                    "table_projection": sim_utils.SphereCfg(
                        radius=_TABLE_PROJECTION_RADIUS,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.9, 1.0)),
                    ),
                },
            )
        )
        self._update_point_visualization(env)

    def get_action(self, env) -> torch.Tensor:
        assert self._arm_action_term is not None
        phase_name, _, _ = self._phase_state()
        self._update_point_visualization(env)

        if phase_name == "align_over_box":
            if self._direct_jaw_offset_pos is None or self._direct_jaw_offset_quat is None:
                self._capture_direct_jaw_offset(env)
            self._arm_action_term.set_control_frame_offset(
                position=self._direct_jaw_offset_pos,
                orientation=self._direct_jaw_offset_quat,
            )
        else:
            self._arm_action_term.set_identity_control_frame_offset()

        action = super().get_action(env)
        if phase_name == "align_over_box":
            assert self._last_desired_jaw_w is not None
            robot = env.scene["robot"]
            desired_jaw_b = quat_apply(
                quat_inv(robot.data.root_quat_w),
                self._last_desired_jaw_w - robot.data.root_pos_w,
            )
            action[:, :3] = desired_jaw_b
        return action

    def reset(self) -> None:
        super().reset()
        self._direct_jaw_offset_pos = None
        self._direct_jaw_offset_quat = None
        self._controlled_jaw_point_w = None
        self._table_projection_point_w = None
        if self._arm_action_term is not None:
            self._arm_action_term.set_identity_control_frame_offset()

    def _capture_direct_jaw_offset(self, env) -> None:
        ee_frame = env.scene["ee_frame"]
        offset_pos, offset_quat = subtract_frame_transforms(
            ee_frame.data.target_pos_w[:, 0, :],
            ee_frame.data.target_quat_w[:, 0, :],
            ee_frame.data.target_pos_w[:, 1, :],
            ee_frame.data.target_quat_w[:, 1, :],
        )
        self._direct_jaw_offset_pos = offset_pos.detach().clone()
        self._direct_jaw_offset_quat = offset_quat.detach().clone()

    def _update_point_visualization(self, env) -> None:
        jaw_point_w = env.scene["ee_frame"].data.target_pos_w[:, 1, :].clone()
        table_projection_w = jaw_point_w.clone()
        table_projection_w[:, 2] = TABLE_SURFACE_Z
        self._controlled_jaw_point_w = jaw_point_w
        self._table_projection_point_w = table_projection_w
        if self._point_visualizer is None:
            return
        positions = torch.cat((jaw_point_w, table_projection_w), dim=0)
        marker_indices = torch.cat(
            (
                torch.zeros(env.num_envs, device=env.device, dtype=torch.int32),
                torch.ones(env.num_envs, device=env.device, dtype=torch.int32),
            )
        )
        self._point_visualizer.visualize(translations=positions, marker_indices=marker_indices)

    @property
    def _high_align_safety_mode(self) -> str:
        return "direct_jaw_xyz_plus_shoulder_pan_nullspace_joint_limit_avoidance_physical_z_gates"

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "high_align_control_frame": "closed_gripper_jaw_detection_frame",
                "high_align_target_conversion": "direct_jaw_target_no_gripper_position_compensation",
                "jaw_marker_color": "red",
                "table_projection_marker_color": "cyan",
                "table_projection_z": TABLE_SURFACE_Z,
            }
        )
        return parameters

    @property
    def direct_jaw_offset_pos(self) -> torch.Tensor | None:
        return self._direct_jaw_offset_pos

    @property
    def direct_jaw_offset_quat(self) -> torch.Tensor | None:
        return self._direct_jaw_offset_quat

    @property
    def controlled_jaw_point_w(self) -> torch.Tensor | None:
        return self._controlled_jaw_point_w

    @property
    def table_projection_point_w(self) -> torch.Tensor | None:
        return self._table_projection_point_w
