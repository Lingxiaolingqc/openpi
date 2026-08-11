"""Dynamic-grasp legacy expert with one frozen transfer-residual correction."""

from __future__ import annotations

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
import torch

from .env_cfg import TABLE_SURFACE_Z
from .env_cfg import TARGET_BOX_WALL_TOP_Z
from .legacy_dynamic_grasp_offset_state_machine import RedCubeToBoxLegacyDynamicGraspOffsetStateMachine

_MAXIMUM_RESIDUAL_CORRECTION = 0.10
_LOWER_TARGET_TOLERANCE = 0.015
_LOWER_TARGET_STABLE_STEPS = 10
_MAXIMUM_LOWER_HOLD_STEPS = 300
_EE_POINT_RADIUS = 0.010
_TABLE_PROJECTION_RADIUS = 0.008
_MINIMUM_JAW_CLEARANCE_ABOVE_WALL = 0.030


class RedCubeToBoxLegacyDynamicGraspOffsetResidualCorrectedStateMachine(
    RedCubeToBoxLegacyDynamicGraspOffsetStateMachine
):
    """Correct the placement target once from the measured post-transfer cube error.

    The parent expert is unchanged through transfer. On the first lower-into-box
    control step, this comparison measures ``desired_cube_xy - actual_cube_xy``,
    caps that vector at 10 cm, adds it once to the parent gripper target, and
    freezes the corrected target through release and retraction. It deliberately
    does not accumulate feedback on every step.
    """

    def __init__(self) -> None:
        super().__init__()
        self._raw_transfer_residual_xy: torch.Tensor | None = None
        self._applied_transfer_residual_xy: torch.Tensor | None = None
        self._corrected_gripper_target_xy: torch.Tensor | None = None
        self._measured_gripper_above_jaw_z: torch.Tensor | None = None
        self._safe_jaw_target_z: torch.Tensor | None = None
        self._safe_release_gripper_target_z: torch.Tensor | None = None
        self._lower_target_error: float | None = None
        self._lower_target_stable_streak = 0
        self._lower_hold_steps = 0
        self._release_authorized = False
        self._release_block_reason: str | None = None
        self._ee_point_w: torch.Tensor | None = None
        self._ee_table_projection_w: torch.Tensor | None = None
        self._point_visualizer: VisualizationMarkers | None = None

    def setup(self, env) -> None:
        super().setup(env)
        self._point_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/RedCubeToBox/ResidualCorrectedEePoints",
                markers={
                    "ee_point": sim_utils.SphereCfg(
                        radius=_EE_POINT_RADIUS,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.8)),
                    ),
                    "table_projection": sim_utils.SphereCfg(
                        radius=_TABLE_PROJECTION_RADIUS,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 1.0, 0.2)),
                    ),
                },
            )
        )
        self._update_point_visualization(env)

    def get_action(self, env) -> torch.Tensor:
        phase_name, phase_step, phase_duration = self._phase_state()
        action = super().get_action(env)
        self._update_point_visualization(env)

        if phase_name == "lower_into_box" and phase_step == phase_duration - 1:
            assert self._last_gripper_target_w is not None
            actual_ee_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
            error = torch.linalg.vector_norm(self._last_gripper_target_w - actual_ee_w, dim=-1)
            self._lower_target_error = float(error.max().item())
            if self._lower_target_error <= _LOWER_TARGET_TOLERANCE:
                self._lower_target_stable_streak += 1
            else:
                self._lower_target_stable_streak = 0
            self._release_authorized = self._lower_target_stable_streak >= _LOWER_TARGET_STABLE_STEPS

        return action

    def advance(self) -> None:
        phase_name, phase_step, phase_duration = self._phase_state()
        at_lower_target = phase_name == "lower_into_box" and phase_step == phase_duration - 1
        if at_lower_target and not self._release_authorized:
            self._lower_hold_steps += 1
            if self._lower_hold_steps >= _MAXIMUM_LOWER_HOLD_STEPS:
                self._release_block_reason = (
                    "lower_target_not_reached_before_timeout:"
                    f"error={self._lower_target_error}:tolerance={_LOWER_TARGET_TOLERANCE}"
                )
                self._episode_done = True
            return
        super().advance()

    def reset(self) -> None:
        super().reset()
        self._raw_transfer_residual_xy = None
        self._applied_transfer_residual_xy = None
        self._corrected_gripper_target_xy = None
        self._measured_gripper_above_jaw_z = None
        self._safe_jaw_target_z = None
        self._safe_release_gripper_target_z = None
        self._lower_target_error = None
        self._lower_target_stable_streak = 0
        self._lower_hold_steps = 0
        self._release_authorized = False
        self._release_block_reason = None
        self._ee_point_w = None
        self._ee_table_projection_w = None

    def _placement_gripper_target_xy(self, env, phase_name: str) -> torch.Tensor:
        base_target_xy = super()._placement_gripper_target_xy(env, phase_name)
        if phase_name == "transfer_to_box":
            return base_target_xy

        if self._corrected_gripper_target_xy is None:
            assert self._desired_cube_xy is not None
            actual_cube_xy = env.scene["cube"].data.root_pos_w[:, :2]
            raw_residual_xy = self._desired_cube_xy - actual_cube_xy
            residual_norm = torch.linalg.vector_norm(raw_residual_xy, dim=-1, keepdim=True)
            correction_scale = torch.clamp(
                _MAXIMUM_RESIDUAL_CORRECTION / torch.clamp(residual_norm, min=1e-8),
                max=1.0,
            )
            applied_residual_xy = raw_residual_xy * correction_scale

            self._raw_transfer_residual_xy = raw_residual_xy.detach().clone()
            self._applied_transfer_residual_xy = applied_residual_xy.detach().clone()
            self._corrected_gripper_target_xy = (base_target_xy + applied_residual_xy).detach().clone()

        return self._corrected_gripper_target_xy

    def _placement_release_z(self, env, floor_center_z: torch.Tensor, phase_name: str) -> torch.Tensor:
        legacy_release_z = super()._placement_release_z(env, floor_center_z, phase_name)
        if phase_name == "transfer_to_box":
            return legacy_release_z

        if self._safe_release_gripper_target_z is None:
            ee_frame = env.scene["ee_frame"]
            gripper_z = ee_frame.data.target_pos_w[:, 0, 2]
            jaw_z = ee_frame.data.target_pos_w[:, 1, 2]
            measured_gripper_above_jaw_z = torch.clamp(gripper_z - jaw_z, min=0.0)
            safe_jaw_target_z = torch.full_like(jaw_z, TARGET_BOX_WALL_TOP_Z + _MINIMUM_JAW_CLEARANCE_ABOVE_WALL)
            safe_gripper_target_z = safe_jaw_target_z + measured_gripper_above_jaw_z

            self._measured_gripper_above_jaw_z = measured_gripper_above_jaw_z.detach().clone()
            self._safe_jaw_target_z = safe_jaw_target_z.detach().clone()
            self._safe_release_gripper_target_z = (
                torch.maximum(
                    legacy_release_z,
                    safe_gripper_target_z,
                )
                .detach()
                .clone()
            )

        return self._safe_release_gripper_target_z

    def _update_point_visualization(self, env) -> None:
        ee_point_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].clone()
        table_projection_w = ee_point_w.clone()
        table_projection_w[:, 2] = TABLE_SURFACE_Z
        self._ee_point_w = ee_point_w
        self._ee_table_projection_w = table_projection_w
        if self._point_visualizer is None:
            return
        positions = torch.cat((ee_point_w, table_projection_w), dim=0)
        marker_indices = torch.cat(
            (
                torch.zeros(env.num_envs, device=env.device, dtype=torch.int32),
                torch.ones(env.num_envs, device=env.device, dtype=torch.int32),
            )
        )
        self._point_visualizer.visualize(translations=positions, marker_indices=marker_indices)

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        parameters.update(
            {
                "residual_correction_phase": "lower_into_box_entry",
                "residual_correction_policy": "single_frozen_xy_update",
                "maximum_residual_correction": _MAXIMUM_RESIDUAL_CORRECTION,
                "lower_release_gate": "actual_gripper_ee_to_corrected_release_target_3d",
                "lower_target_tolerance": _LOWER_TARGET_TOLERANCE,
                "lower_target_stable_steps": _LOWER_TARGET_STABLE_STEPS,
                "maximum_lower_hold_steps": _MAXIMUM_LOWER_HOLD_STEPS,
                "safe_release_height_policy": "measured_gripper_above_jaw_plus_wall_top_clearance",
                "minimum_jaw_clearance_above_wall": _MINIMUM_JAW_CLEARANCE_ABOVE_WALL,
                "target_box_wall_top_z": TARGET_BOX_WALL_TOP_Z,
                "ee_marker_color": "magenta",
                "ee_table_projection_marker_color": "green",
                "ee_table_projection_z": TABLE_SURFACE_Z,
            }
        )
        return parameters

    @property
    def raw_transfer_residual_xy(self) -> torch.Tensor | None:
        return self._raw_transfer_residual_xy

    @property
    def applied_transfer_residual_xy(self) -> torch.Tensor | None:
        return self._applied_transfer_residual_xy

    @property
    def corrected_gripper_target_xy(self) -> torch.Tensor | None:
        return self._corrected_gripper_target_xy

    @property
    def lower_target_error(self) -> float | None:
        return self._lower_target_error

    @property
    def lower_target_stable_streak(self) -> int:
        return self._lower_target_stable_streak

    @property
    def lower_hold_steps(self) -> int:
        return self._lower_hold_steps

    @property
    def release_authorized(self) -> bool:
        return self._release_authorized

    @property
    def release_block_reason(self) -> str | None:
        return self._release_block_reason

    @property
    def measured_gripper_above_jaw_z(self) -> torch.Tensor | None:
        return self._measured_gripper_above_jaw_z

    @property
    def safe_jaw_target_z(self) -> torch.Tensor | None:
        return self._safe_jaw_target_z

    @property
    def safe_release_gripper_target_z(self) -> torch.Tensor | None:
        return self._safe_release_gripper_target_z

    @property
    def ee_point_w(self) -> torch.Tensor | None:
        return self._ee_point_w

    @property
    def ee_table_projection_w(self) -> torch.Tensor | None:
        return self._ee_table_projection_w
