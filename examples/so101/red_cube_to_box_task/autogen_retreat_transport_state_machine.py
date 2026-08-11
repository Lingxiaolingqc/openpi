"""AutoGen-inspired retreat-then-transport expert for RedCubeToBox."""

from __future__ import annotations

import torch

from .legacy_dynamic_grasp_offset_residual_corrected_state_machine import (
    RedCubeToBoxLegacyDynamicGraspOffsetResidualCorrectedStateMachine,
)
from .phase_aware_ik_action import PhaseAwareDifferentialInverseKinematicsAction
from .phase_aware_ik_action import resolve_action_term

_GRIPPER_CLOSE = -1.0
_RETREAT_RADIAL_SCALE = 5.0 / 7.0
_RETREAT_HEIGHT_ABOVE_FLOOR_CENTER = 0.30
_BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER = 0.25
_RETREAT_CARTESIAN_STEP = 0.0025
_TRANSPORT_CARTESIAN_STEP = 0.0025


class RedCubeToBoxAutogenRetreatTransportStateMachine(
    RedCubeToBoxLegacyDynamicGraspOffsetResidualCorrectedStateMachine
):
    """Keep legacy pickup, then retreat inward before transporting to the box.

    This adapts ``so101-autogen`` rather than copying its Isaac Sim runtime.
    Pickup and lift remain the validated legacy sequence. At retreat entry the
    live gripper position is captured, moved to five sevenths of its horizontal
    radius from the robot root, and raised above the scene. Transport captures
    the live gripper position again and moves from there to the box-floor
    center without a grasp offset or residual placement correction. Both moves
    are bounded to 2.5 mm per control step. Full 6D pose IK is retained so the
    grasp orientation cannot drift during retreat and transport.
    """

    _PHASES = (
        ("approach_cube", 120),
        ("descend_to_cube", 120),
        ("close_gripper", 80),
        ("lift_cube", 120),
        ("retreat_to_safe", 120),
        ("transfer_to_box", 240),
        ("lower_into_box", 120),
        ("release_cube", 100),
        ("retract_gripper", 100),
        ("settle", 180),
    )
    MAX_STEPS = sum(duration for _, duration in _PHASES)

    def __init__(self) -> None:
        super().__init__()
        self._arm_action_term: PhaseAwareDifferentialInverseKinematicsAction | None = None
        self._retreat_start_gripper_w: torch.Tensor | None = None
        self._retreat_safe_target_w: torch.Tensor | None = None
        self._transport_start_gripper_w: torch.Tensor | None = None
        self._transport_target_w: torch.Tensor | None = None

    def setup(self, env) -> None:
        super().setup(env)
        arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise TypeError("autogen_retreat_transport requires PhaseAwareDifferentialInverseKinematicsAction")
        self._arm_action_term = arm_action_term

    def get_action(self, env) -> torch.Tensor:
        if self._arm_action_term is None:
            raise RuntimeError("Call setup(env) before requesting an AutoGen-style expert action")

        phase_name, phase_step, _ = self._phase_state()
        self._arm_action_term.set_orientation_weight(weight=1.0)

        if phase_name == "retreat_to_safe":
            self._initialize_retreat(env)
            assert self._retreat_start_gripper_w is not None
            assert self._retreat_safe_target_w is not None
            target_pos_w = self._constant_speed_target(
                self._retreat_start_gripper_w,
                self._retreat_safe_target_w,
                phase_step,
                _RETREAT_CARTESIAN_STEP,
            )
            self._last_gripper_target_w = target_pos_w.detach().clone()
            return self._compose_legacy_pose_action(env, target_pos_w, _GRIPPER_CLOSE)

        if phase_name == "transfer_to_box":
            self._initialize_transport(env)
            assert self._transport_start_gripper_w is not None
            assert self._transport_target_w is not None
            target_pos_w = self._constant_speed_target(
                self._transport_start_gripper_w,
                self._transport_target_w,
                phase_step,
                _TRANSPORT_CARTESIAN_STEP,
            )
            self._last_gripper_target_w = target_pos_w.detach().clone()
            return self._compose_legacy_pose_action(env, target_pos_w, _GRIPPER_CLOSE)

        return super().get_action(env)

    def reset(self) -> None:
        super().reset()
        self._retreat_start_gripper_w = None
        self._retreat_safe_target_w = None
        self._transport_start_gripper_w = None
        self._transport_target_w = None
        if self._arm_action_term is not None:
            self._arm_action_term.set_orientation_weight(weight=1.0)

    def _initialize_retreat(self, env) -> None:
        if self._retreat_start_gripper_w is not None:
            return

        self._initialize_anchors(env)
        self._finalize_grasp_offset(env)
        assert self._floor_anchor is not None

        gripper_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].clone()
        robot_root_w = env.scene["robot"].data.root_pos_w
        safe_target_w = gripper_pos_w.clone()
        safe_target_w[:, :2] = robot_root_w[:, :2] + _RETREAT_RADIAL_SCALE * (
            gripper_pos_w[:, :2] - robot_root_w[:, :2]
        )
        safe_target_w[:, 2] = torch.maximum(
            gripper_pos_w[:, 2],
            self._floor_anchor[:, 2] + _RETREAT_HEIGHT_ABOVE_FLOOR_CENTER,
        )

        self._retreat_start_gripper_w = gripper_pos_w.detach().clone()
        self._retreat_safe_target_w = safe_target_w.detach().clone()

    def _initialize_transport(self, env) -> None:
        if self._transport_start_gripper_w is not None:
            return

        assert self._floor_anchor is not None
        transport_target_w = self._floor_anchor.clone()
        transport_target_w[:, 2] += _BOX_HOVER_HEIGHT_ABOVE_FLOOR_CENTER

        self._transport_start_gripper_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].detach().clone()
        self._transport_target_w = transport_target_w.detach().clone()

    def _sample_grasp_offset(self, env, phase_step: int) -> None:
        """Disable the inherited late-lift placement-offset measurement."""
        del env, phase_step

    def _sample_transfer_grasp_offset(self, env, phase_step: int) -> None:
        """Disable the inherited late-transfer residual measurement."""
        del env, phase_step

    def _finalize_grasp_offset(self, env) -> None:
        """Populate parent fields with an explicit box-center zero offset."""
        if self._dynamic_gripper_target_xy is not None:
            return
        self._initialize_anchors(env)
        assert self._floor_anchor is not None
        zero_xy = torch.zeros_like(self._floor_anchor[:, :2])
        self._gripper_to_cube_xy = zero_xy.clone()
        self._desired_cube_xy = self._floor_anchor[:, :2].detach().clone()
        self._dynamic_gripper_target_xy = self._floor_anchor[:, :2].detach().clone()
        self._dynamic_place_offset_xy = zero_xy.clone()

    def _placement_gripper_target_xy(self, env, phase_name: str) -> torch.Tensor:
        """Use the box-floor center directly in every placement phase."""
        del env, phase_name
        assert self._floor_anchor is not None
        return self._floor_anchor[:, :2]

    @staticmethod
    def _constant_speed_target(
        start_w: torch.Tensor,
        end_w: torch.Tensor,
        phase_step: int,
        maximum_step: float,
    ) -> torch.Tensor:
        displacement = end_w - start_w
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        traveled = torch.full_like(distance, (phase_step + 1) * maximum_step)
        progress = torch.clamp(traveled / torch.clamp(distance, min=1e-8), max=1.0)
        return start_w + progress * displacement

    @property
    def servo_parameters(self) -> dict[str, float | int | str]:
        parameters = super().servo_parameters
        for key in (
            "grasp_offset_sample_start_step",
            "minimum_captured_lift",
            "box_to_root_safety_distance",
            "residual_correction_phase",
            "maximum_residual_correction",
            "transfer_grasp_offset_measurement",
            "transfer_grasp_offset_sample_start_step",
            "minimum_transfer_cube_height",
        ):
            parameters.pop(key, None)
        parameters.update(
            {
                "changed_component": "retreat_transport_path_and_box_center_placement",
                "placement_xy_policy": "target_box_floor_center_without_offset",
                "grasp_offset_measurement": "disabled",
                "residual_correction_policy": "disabled",
                "transport_reference": "so101-autogen-retreat-pattern",
                "retreat_policy": "live_gripper_to_root_centered_five_sevenths_radius",
                "retreat_radial_scale": _RETREAT_RADIAL_SCALE,
                "retreat_height_above_floor_center": _RETREAT_HEIGHT_ABOVE_FLOOR_CENTER,
                "retreat_cartesian_step": _RETREAT_CARTESIAN_STEP,
                "transport_start": "live_gripper_at_transfer_entry",
                "transport_cartesian_step": _TRANSPORT_CARTESIAN_STEP,
                "post_lift_ik_mode": "full_6d_pose",
            }
        )
        return parameters

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "uninitialized"
        return self._arm_action_term.runtime_mode

    @property
    def retreat_start_gripper_w(self) -> torch.Tensor | None:
        return self._retreat_start_gripper_w

    @property
    def retreat_safe_target_w(self) -> torch.Tensor | None:
        return self._retreat_safe_target_w

    @property
    def transport_start_gripper_w(self) -> torch.Tensor | None:
        return self._transport_start_gripper_w

    @property
    def transport_target_w(self) -> torch.Tensor | None:
        return self._transport_target_w
