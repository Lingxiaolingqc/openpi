"""Behavioral port of the bundled so101-autogen state machine to Isaac Lab.

The state order and motion constants intentionally follow
``autogen/so101-autogen-main/src/state_machine/simple_state_machine.py``.
Only the simulator interfaces are adapted: targets live in the robot-base
frame, source height constants retain their world-Z meaning, the original
green-ray test is evaluated against the cube OBB, and the original
position-only wrist IK is expressed through the phase-aware Isaac Lab action
term.
"""

from __future__ import annotations

import math
import random
from typing import ClassVar

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_inv
from isaaclab.utils.math import quat_mul
from leisaac.datagen.state_machine.base import StateMachineBase
import torch

from . import mdp
from .env_cfg import CUBE_HALF_HEIGHT
from .phase_aware_ik_action import PhaseAwareDifferentialInverseKinematicsAction
from .phase_aware_ik_action import resolve_action_term


class RedCubeToBoxAutogenReferenceStateMachine(StateMachineBase):
    """Port the original Autogen controller without inheriting our legacy expert."""

    APPROACH_HEIGHT = 0.25
    LIFT_HEIGHT = 0.20
    SAFE_HEIGHT = 0.30
    TRANSPORT_HEIGHT = 0.25
    RELEASE_HEIGHT = 0.21
    INITIAL_POSITION = (0.25, 0.0, 0.25)

    TRAVEL_STEP = 0.003
    DESCEND_STEP = 0.001
    LIFT_STEP = 0.002
    MAX_DESCEND_STEPS = 600
    MAX_LIFT_STEPS = 300
    GRASP_CHECK_INTERVAL = 30
    GRASP_DURATION_STEPS = 80
    GRASP_SETTLE_STEPS = 21
    GRASP_SETTLE_MAX_STEPS = 180
    GRASP_SETTLE_STABLE_STEPS = 8
    GRIPPER_TARGET_TOLERANCE = 0.03
    GRIPPER_STALL_VELOCITY_TOLERANCE = 0.01
    RELEASE_DURATION_STEPS = 180

    GRIPPER_OPEN_POSITION = 1.74533
    GRIPPER_CLOSED_POSITION = -0.174533
    CLOSE_OPENNESS_RANGE = (0.18, 0.235)
    MIN_CONFIRMED_LIFT = 0.005

    GREEN_RAY_ORIGIN_OFFSET = (CUBE_HALF_HEIGHT, 0.0, -0.04)
    GREEN_RAY_DIRECTION = (0.0, 0.0, -1.0)
    GREEN_RAY_MAX_HIT_DISTANCE = CUBE_HALF_HEIGHT + 0.048
    LOCAL_RAY_AXES: ClassVar[dict[str, tuple[float, float, float]]] = {
        "+x": (1.0, 0.0, 0.0),
        "-x": (-1.0, 0.0, 0.0),
        "+y": (0.0, 1.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "+z": (0.0, 0.0, 1.0),
        "-z": (0.0, 0.0, -1.0),
    }

    GROUND_GUARD_HEIGHT_W = 0.01
    MIN_WRIST_HEIGHT_W = 0.03
    MAX_DESCENT_WRIST_XY_ERROR = 0.05
    APPROACH_TRACKING_TOLERANCE = 0.01
    APPROACH_SETTLE_TIMEOUT_STEPS = 120
    RAY_ALIGNMENT_KP = 0.2
    RAY_ALIGNMENT_MAX_XY_STEP = 0.001
    RAY_ALIGNMENT_XY_TOLERANCE = 0.008
    RAY_ALIGNMENT_STABLE_STEPS = 5
    GRASP_REACH_MARGIN_BEYOND_JAW = CUBE_HALF_HEIGHT
    GREEN_RAY_VISUAL_LENGTH = 0.35
    GREEN_RAY_VISUAL_POINT_COUNT = 36

    MAX_STEPS = 2500

    def __init__(self, green_ray_axis: str = "-z") -> None:
        if green_ray_axis not in self.LOCAL_RAY_AXES:
            raise ValueError(f"Unsupported gripper ray axis: {green_ray_axis!r}")
        self._green_ray_axis = green_ray_axis
        self._arm_action_term = None
        self._rng: random.Random | None = None
        self._wrist_body_index: int | None = None
        self._green_ray_visualizer: VisualizationMarkers | None = None
        self.reset()

    def setup(self, env) -> None:
        """Resolve the adapted wrist IK action and reproduce Autogen damping."""

        env.scene["robot"].write_joint_damping_to_sim(damping=10.0)
        self._arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(self._arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise RuntimeError("autogen_reference requires PhaseAwareDifferentialInverseKinematicsAction")
        body_names = list(env.scene["robot"].data.body_names)
        self._wrist_body_index = body_names.index("wrist")
        self._green_ray_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/RedCubeToBox/AutogenWristGripperRay",
                markers={
                    "ray_miss": sim_utils.SphereCfg(
                        radius=0.0025,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.72, 0.02)),
                    ),
                    "ray_hit": sim_utils.SphereCfg(
                        radius=0.003,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 1.0, 0.12)),
                    ),
                    "ray_origin": sim_utils.SphereCfg(
                        radius=0.006,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.25, 1.0)),
                    ),
                    "gripper_point": sim_utils.SphereCfg(
                        radius=0.006,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.05, 1.0)),
                    ),
                    "axis_pos_x": sim_utils.SphereCfg(
                        radius=0.002,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
                    ),
                    "axis_neg_x": sim_utils.SphereCfg(
                        radius=0.002,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
                    ),
                    "axis_pos_y": sim_utils.SphereCfg(
                        radius=0.002,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 1.0, 0.05)),
                    ),
                    "axis_neg_y": sim_utils.SphereCfg(
                        radius=0.002,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.05)),
                    ),
                    "axis_pos_z": sim_utils.SphereCfg(
                        radius=0.002,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.25, 1.0)),
                    ),
                    "axis_neg_z": sim_utils.SphereCfg(
                        radius=0.002,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.95, 0.95)),
                    ),
                },
            )
        )
        if self._rng is None:
            self._rng = random.Random(int(env.cfg.seed))

    def reset(self) -> None:
        self._state = "approach"
        self._state_step = 0
        self._step_count = 0
        self._episode_done = False
        self._initialized = False
        self._command_pos_b: torch.Tensor | None = None
        self._move_start_b: torch.Tensor | None = None
        self._move_end_b: torch.Tensor | None = None
        self._move_duration = 1
        self._gripper_command = self.GRIPPER_OPEN_POSITION
        self._grasp_start_position = self.GRIPPER_OPEN_POSITION
        self._grasp_end_position = self.GRIPPER_CLOSED_POSITION
        self._gripper_settle_streak = 0
        self._gripper_settle_reason: str | None = None
        self._gripper_target_error: torch.Tensor | None = None
        self._gripper_joint_velocity: torch.Tensor | None = None
        self._initial_cube_z_w: torch.Tensor | None = None
        self._posture_target: torch.Tensor | None = None
        self._held_gripper_angle: float | None = None
        self._held_gripper_angle_capture_step: int | None = None
        self._wrist_position_w: torch.Tensor | None = None
        self._approach_tracking_error: torch.Tensor | None = None
        self._descent_wrist_xy_error: torch.Tensor | None = None
        self._descent_ray_xy_error: torch.Tensor | None = None
        self._descent_xy_correction_w: torch.Tensor | None = None
        self._ray_alignment_streak = 0
        self.green_ray_hit = False
        self.green_ray_obb_hit = False
        self.green_ray_within_grasp_reach = False
        self.green_ray_origin_w: torch.Tensor | None = None
        self.green_ray_direction_w: torch.Tensor | None = None
        self.green_ray_hit_distance: torch.Tensor | None = None
        self.wrist_to_gripper_length: torch.Tensor | None = None
        self.wrist_to_jaw_length: torch.Tensor | None = None
        self.cube_distance_to_green_ray: torch.Tensor | None = None
        self.cube_projection_on_green_ray: torch.Tensor | None = None
        self.cube_to_green_ray_error_w: torch.Tensor | None = None
        self.gripper_frame_position_w: torch.Tensor | None = None
        self.jaw_detection_position_w: torch.Tensor | None = None
        self.wrist_height_above_cube: torch.Tensor | None = None
        self.gripper_frame_height_above_cube: torch.Tensor | None = None
        self.jaw_height_above_cube: torch.Tensor | None = None
        self.retreat_target_b: torch.Tensor | None = None
        self.transport_target_b: torch.Tensor | None = None
        self.grasp_confirmed = False
        self.grasp_lost_before_release = False
        self.retry_used = False
        self.box_aligned_before_release = False
        self.servo_timeout_phase = None
        self.servo_abort_reason = None
        self.release_block_reason = None

    def get_action(self, env) -> torch.Tensor:
        """Advance the Autogen controller and return its adapted 8D command."""

        if not self._initialized:
            self._initialize_episode(env)

        self._update_posture_target(env)
        self._update_state(env)

        assert self._command_pos_b is not None
        robot = env.scene["robot"]
        wrist_quat_w = robot.data.body_quat_w[:, self._wrist_body_index]
        target_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), wrist_quat_w)
        if self._held_gripper_angle is not None and self._state not in {"release", "return_home", "success"}:
            self._gripper_command = self._held_gripper_angle
        gripper = torch.full(
            (env.num_envs, 1),
            self._gripper_command,
            device=env.device,
            dtype=self._command_pos_b.dtype,
        )
        return torch.cat((self._command_pos_b, target_quat_b, gripper), dim=-1)

    def observe_pick_cube(self, pick_cube: bool | torch.Tensor, env) -> bool:
        """Latch the measured gripper angle on the first environment-confirmed pick."""

        picked = bool(pick_cube.all().item()) if isinstance(pick_cube, torch.Tensor) else bool(pick_cube)
        if not picked or self._held_gripper_angle is not None:
            return False
        if self._state not in {"grasp", "grasp_settle", "ik_handoff", "lift", "retreat", "transport"}:
            return False

        robot = env.scene["robot"]
        gripper_joint_index = list(robot.data.joint_names).index("gripper")
        measured_angle = robot.data.joint_pos[:, gripper_joint_index]
        if measured_angle.numel() != 1:
            raise RuntimeError("autogen_reference gripper-angle hold currently requires exactly one environment")
        self._held_gripper_angle = float(measured_angle.item())
        self._held_gripper_angle_capture_step = self._step_count
        self._grasp_end_position = self._held_gripper_angle
        self._gripper_command = self._held_gripper_angle
        return True

    def advance(self) -> None:
        if self._episode_done:
            return
        self._step_count += 1
        self._state_step += 1
        if self._step_count >= self.MAX_STEPS:
            self._fail("autogen_reference exceeded its global step limit")

    def check_success(self, env) -> bool:
        return bool(mdp.cube_inside_target_box(env).all().item())

    def _initialize_episode(self, env) -> None:
        robot = env.scene["robot"]
        wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
        self._command_pos_b = self._world_position_to_base(robot, wrist_pos_w)
        self._initial_cube_z_w = env.scene["cube"].data.root_pos_w[:, 2].clone()
        self._gripper_command = self.GRIPPER_OPEN_POSITION

        cube_pos_b = self._world_position_to_base(robot, env.scene["cube"].data.root_pos_w)
        approach_target_b = self._set_world_height(robot, cube_pos_b, self.APPROACH_HEIGHT)
        self._start_move(approach_target_b, self.TRAVEL_STEP)
        self._initialized = True

    def _update_state(self, env) -> None:
        if self._episode_done:
            return

        if self._state == "approach":
            if self._update_move():
                robot = env.scene["robot"]
                wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
                command_pos_w = self._base_position_to_world(robot, self._command_pos_b)
                self._approach_tracking_error = torch.linalg.vector_norm(
                    wrist_pos_w - command_pos_w,
                    dim=-1,
                ).detach()
                if bool((self._approach_tracking_error <= self.APPROACH_TRACKING_TOLERANCE).all().item()):
                    self._transition("descend")
                elif self._state_step > self._move_duration + self.APPROACH_SETTLE_TIMEOUT_STEPS:
                    self._fail("actual wrist did not settle at the Autogen approach target")
        elif self._state == "descend":
            self.green_ray_hit = self._green_ray_intersects_cube(env)
            robot = env.scene["robot"]
            wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
            command_pos_w = self._base_position_to_world(robot, self._command_pos_b)
            self._wrist_position_w = wrist_pos_w.detach().clone()
            self._descent_wrist_xy_error = torch.linalg.vector_norm(
                wrist_pos_w[:, :2] - command_pos_w[:, :2], dim=-1
            ).detach()
            if self.green_ray_hit:
                self._on_grasp_pose_reached(env)
            elif bool((self._descent_wrist_xy_error > self.MAX_DESCENT_WRIST_XY_ERROR).any().item()):
                self._fail("actual wrist XY drifted more than 50 mm during Autogen descent")
            elif bool((wrist_pos_w[:, 2] < self.MIN_WRIST_HEIGHT_W).any().item()):
                self._fail("actual wrist reached the Autogen descent safety height")
            elif self._state_step > self.MAX_DESCEND_STEPS:
                self._fail("descent timed out before the Autogen green ray intersected the cube")
            else:
                assert self._command_pos_b is not None
                assert self.cube_to_green_ray_error_w is not None
                ray_xy_error_w = self.cube_to_green_ray_error_w[:, :2]
                ray_xy_error_norm = torch.linalg.vector_norm(ray_xy_error_w, dim=-1)
                self._descent_ray_xy_error = ray_xy_error_norm.detach()
                if bool((ray_xy_error_norm > self.RAY_ALIGNMENT_XY_TOLERANCE).any().item()):
                    self._ray_alignment_streak = 0
                    raw_correction_w = self.RAY_ALIGNMENT_KP * ray_xy_error_w
                    raw_norm = torch.linalg.vector_norm(raw_correction_w, dim=-1, keepdim=True)
                    correction_scale = torch.clamp(
                        self.RAY_ALIGNMENT_MAX_XY_STEP / torch.clamp(raw_norm, min=1.0e-8),
                        max=1.0,
                    )
                    correction_w = raw_correction_w * correction_scale
                    # Rate-limit the reference itself. Rebasing every step on the measured
                    # wrist leaves only a 1 mm tracking error, which this damped IK follows
                    # too slowly to remove the ray error before the descent deadline.
                    command_pos_w[:, :2] += correction_w
                    self._descent_xy_correction_w = correction_w.detach()
                else:
                    self._ray_alignment_streak += 1
                    self._descent_xy_correction_w = torch.zeros_like(ray_xy_error_w)
                    if self._ray_alignment_streak >= self.RAY_ALIGNMENT_STABLE_STEPS:
                        command_pos_w[:, 2] -= self.DESCEND_STEP
                self._command_pos_b = self._world_position_to_base(robot, command_pos_w)
                if bool((command_pos_w[:, 2] < self.GROUND_GUARD_HEIGHT_W).any().item()):
                    self._fail("descent reached the Autogen 10 mm ground guard")
        elif self._state == "grasp":
            progress = min(self._state_step / self.GRASP_DURATION_STEPS, 1.0)
            self._gripper_command = (1.0 - progress) * self._grasp_start_position + progress * self._grasp_end_position
            if self._state_step > self.GRASP_DURATION_STEPS:
                self._transition("grasp_settle")
        elif self._state == "grasp_settle":
            robot = env.scene["robot"]
            gripper_position = robot.data.joint_pos[:, -1]
            gripper_velocity = torch.abs(robot.data.joint_vel[:, -1])
            target_error = torch.abs(gripper_position - self._grasp_end_position)
            halfway_closed = gripper_position <= (self.GRIPPER_OPEN_POSITION + self._grasp_end_position) / 2.0
            target_reached = (target_error <= self.GRIPPER_TARGET_TOLERANCE) & (
                gripper_velocity <= self.GRIPPER_STALL_VELOCITY_TOLERANCE
            )
            contact_stalled = halfway_closed & (gripper_velocity <= self.GRIPPER_STALL_VELOCITY_TOLERANCE)
            settled = target_reached | contact_stalled
            self._gripper_target_error = target_error.detach()
            self._gripper_joint_velocity = gripper_velocity.detach()
            if self._state_step >= self.GRASP_SETTLE_STEPS and bool(settled.all().item()):
                self._gripper_settle_streak += 1
                self._gripper_settle_reason = (
                    "target_reached" if bool(target_reached.all().item()) else "contact_stalled"
                )
            else:
                self._gripper_settle_streak = 0
                self._gripper_settle_reason = None
            if self._gripper_settle_streak >= self.GRASP_SETTLE_STABLE_STEPS:
                self._transition("lift")
            elif self._state_step > self.GRASP_SETTLE_MAX_STEPS:
                self._fail("gripper did not settle before the Autogen lift")
        elif self._state == "lift":
            if self._state_step >= self.GRASP_CHECK_INTERVAL and self._state_step % self.GRASP_CHECK_INTERVAL == 0:
                if not self._object_grasped(env):
                    self._fail("Autogen grasp check failed during lift")
                    return
                self.grasp_confirmed = True
                assert self._command_pos_b is not None
                command_pos_w = self._base_position_to_world(env.scene["robot"], self._command_pos_b)
                if bool((command_pos_w[:, 2] >= self.LIFT_HEIGHT).all().item()):
                    self._transition("retreat", env)
                    return
            if self._state_step > self.MAX_LIFT_STEPS:
                self._fail("Autogen lift timed out")
                return
            assert self._command_pos_b is not None
            robot = env.scene["robot"]
            command_pos_w = self._base_position_to_world(robot, self._command_pos_b)
            command_pos_w[:, 2] = torch.clamp(command_pos_w[:, 2] + self.LIFT_STEP, max=self.LIFT_HEIGHT)
            self._command_pos_b = self._world_position_to_base(robot, command_pos_w)
        elif self._state == "retreat":
            if self._update_move():
                self._transition("transport", env)
        elif self._state == "transport":
            if self._state_step % self.GRASP_CHECK_INTERVAL == 0 and not self._object_grasped(env):
                self.grasp_lost_before_release = True
                self._fail("object lost during Autogen transport")
                return
            if self._update_move():
                self.box_aligned_before_release = True
                self._transition("release")
        elif self._state == "release":
            if self.check_success(env):
                self._transition("return_home")
            elif self._state_step > self.RELEASE_DURATION_STEPS:
                self._fail("Autogen placement detection timed out")
        elif self._state == "return_home" and self._update_move():
            self._transition("success")

    def _on_grasp_pose_reached(self, env) -> None:
        """Enter grasp after descent reaches its ray/OBB gate.

        Variants may override this hook to insert a bounded pre-grasp phase.
        The reference expert deliberately preserves its original direct
        ``descend -> grasp`` transition.
        """

        del env
        self._transition("grasp")

    def _transition(self, state: str, env=None) -> None:
        self._state = state
        self._state_step = 0

        if state == "grasp":
            assert self._rng is not None
            openness = self._rng.uniform(*self.CLOSE_OPENNESS_RANGE)
            self._grasp_start_position = self._gripper_command
            self._grasp_end_position = (
                self.GRIPPER_CLOSED_POSITION + (self.GRIPPER_OPEN_POSITION - self.GRIPPER_CLOSED_POSITION) * openness
            )
        elif state == "lift":
            self._gripper_command = self._grasp_end_position
        elif state == "retreat":
            assert self._command_pos_b is not None
            if env is None:
                raise RuntimeError("retreat initialization requires the environment")
            robot = env.scene["robot"]
            target = self._command_pos_b.clone()
            distance_from_origin = torch.linalg.vector_norm(target[:, :2], dim=-1)
            target[:, :2] *= 5.0 / 7.0
            default_xy = torch.tensor((0.15, 0.0), device=target.device, dtype=target.dtype)
            target[:, :2] = torch.where(
                (distance_from_origin < 0.05).unsqueeze(-1),
                default_xy.unsqueeze(0),
                target[:, :2],
            )
            target = self._set_world_height(robot, target, self.SAFE_HEIGHT)
            self.retreat_target_b = target.detach().clone()
            self._start_move(target, self.TRAVEL_STEP)
        elif state == "transport":
            if env is None:
                raise RuntimeError("transport initialization requires the environment")
            robot = env.scene["robot"]
            target = self._world_position_to_base(robot, env.scene["target_box_floor"].data.root_pos_w)
            # The original _start_transport first creates a 0.25 m target, but
            # _update_transport replaces it on the next frame with release_height.
            # Preserve the target that is actually executed by that implementation.
            target = self._set_world_height(robot, target, self.RELEASE_HEIGHT)
            self.transport_target_b = target.detach().clone()
            self._start_move(target, self.TRAVEL_STEP)
        elif state == "release":
            self._gripper_command = self.GRIPPER_OPEN_POSITION
        elif state == "return_home":
            assert self._command_pos_b is not None
            target = torch.tensor(
                self.INITIAL_POSITION,
                device=self._command_pos_b.device,
                dtype=self._command_pos_b.dtype,
            ).repeat(self._command_pos_b.shape[0], 1)
            self._start_move(target, self.TRAVEL_STEP)
        elif state == "success":
            self._episode_done = True

    def _start_move(self, target_b: torch.Tensor, speed: float) -> None:
        assert self._command_pos_b is not None
        self._move_start_b = self._command_pos_b.detach().clone()
        self._move_end_b = target_b.detach().clone()
        distance = torch.linalg.vector_norm(self._move_end_b - self._move_start_b, dim=-1).max().item()
        self._move_duration = max(1, int(distance / speed))

    def _update_move(self) -> bool:
        assert self._move_start_b is not None
        assert self._move_end_b is not None
        progress = min((self._state_step + 1) / self._move_duration, 1.0)
        self._command_pos_b = torch.lerp(self._move_start_b, self._move_end_b, progress)
        return progress >= 1.0

    def _update_posture_target(self, env) -> None:
        """Restore the second-back port's wrist-flex correction within XYZ IK."""

        assert self._arm_action_term is not None
        target = self._compute_wrist_flex_target(env)
        self._posture_target = target.detach().clone()
        self._arm_action_term.set_xyz_joint_nullspace_target(
            joint_name="wrist_flex",
            joint_target=target,
            damping=0.04,
            posture_gain=0.0,
            max_posture_step=0.03,
        )

    def _compute_wrist_flex_target(self, env) -> torch.Tensor:
        """Compute the original wrist-flex correction that points the tool down."""

        robot = env.scene["robot"]
        wrist_quat_w = robot.data.body_quat_w[:, self._wrist_body_index]
        dtype = wrist_quat_w.dtype
        local_forward = torch.tensor((0.0, -1.0, 0.0), device=env.device, dtype=dtype).repeat(env.num_envs, 1)
        local_flex_axis = torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=dtype).repeat(env.num_envs, 1)
        desired_down = torch.tensor((0.0, 0.0, -1.0), device=env.device, dtype=dtype).repeat(env.num_envs, 1)
        forward_w = quat_apply(wrist_quat_w, local_forward)
        flex_axis_w = quat_apply(wrist_quat_w, local_flex_axis)
        angle = torch.acos(torch.clamp(torch.sum(forward_w * desired_down, dim=-1), -1.0, 1.0))
        rotation_axis = torch.linalg.cross(forward_w, desired_down, dim=-1)
        correction_sign = -torch.sign(torch.sum(rotation_axis * flex_axis_w, dim=-1))
        target = math.pi / 2.0 + correction_sign * angle

        wrist_joint_index = list(robot.data.joint_names).index("wrist_flex")
        limits = robot.data.soft_joint_pos_limits[:, wrist_joint_index]
        return torch.clamp(target, min=limits[:, 0], max=limits[:, 1])

    def _green_ray_intersects_cube(self, env) -> bool:
        """Test the selected gripper-local axis against the cube OBB."""

        ee_frame = env.scene["ee_frame"]
        cube = env.scene["cube"]
        robot = env.scene["robot"]
        wrist_pos_w = robot.data.body_pos_w[:, self._wrist_body_index]
        gripper_pos_w = ee_frame.data.target_pos_w[:, 0]
        gripper_quat_w = ee_frame.data.target_quat_w[:, 0]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1]
        wrist_to_gripper_w = gripper_pos_w - wrist_pos_w
        wrist_to_gripper_length = torch.linalg.vector_norm(wrist_to_gripper_w, dim=-1)
        wrist_to_jaw_length = torch.linalg.vector_norm(jaw_pos_w - wrist_pos_w, dim=-1)
        local_direction = torch.tensor(
            self.LOCAL_RAY_AXES[self._green_ray_axis],
            device=env.device,
            dtype=gripper_pos_w.dtype,
        ).repeat(env.num_envs, 1)
        local_origin_offset = torch.tensor(
            self.GREEN_RAY_ORIGIN_OFFSET,
            device=env.device,
            dtype=gripper_pos_w.dtype,
        ).repeat(env.num_envs, 1)
        direction_w = quat_apply(gripper_quat_w, local_direction)
        direction_w = direction_w / torch.clamp(torch.linalg.vector_norm(direction_w, dim=-1, keepdim=True), min=1.0e-8)
        origin_w = gripper_pos_w + quat_apply(gripper_quat_w, local_origin_offset)
        self.gripper_frame_position_w = gripper_pos_w.detach().clone()
        self.jaw_detection_position_w = jaw_pos_w.detach().clone()
        self.green_ray_origin_w = origin_w.detach().clone()
        self.green_ray_direction_w = direction_w.detach().clone()
        self.wrist_to_gripper_length = wrist_to_gripper_length.detach()
        self.wrist_to_jaw_length = wrist_to_jaw_length.detach()
        cube_from_origin_w = cube.data.root_pos_w - origin_w
        cube_projection = torch.sum(cube_from_origin_w * direction_w, dim=-1)
        clamped_projection = torch.clamp(cube_projection, min=0.0)
        closest_ray_point_w = origin_w + clamped_projection.unsqueeze(-1) * direction_w
        self.cube_projection_on_green_ray = cube_projection.detach()
        cube_to_ray_error_w = cube.data.root_pos_w - closest_ray_point_w
        self.cube_to_green_ray_error_w = cube_to_ray_error_w.detach()
        self.cube_distance_to_green_ray = torch.linalg.vector_norm(cube_to_ray_error_w, dim=-1).detach()
        cube_z_w = cube.data.root_pos_w[:, 2]
        wrist_z_w = env.scene["robot"].data.body_pos_w[:, self._wrist_body_index, 2]
        self.wrist_height_above_cube = (wrist_z_w - cube_z_w).detach()
        self.gripper_frame_height_above_cube = (gripper_pos_w[:, 2] - cube_z_w).detach()
        self.jaw_height_above_cube = (jaw_pos_w[:, 2] - cube_z_w).detach()

        cube_quat_inv = quat_inv(cube.data.root_quat_w)
        origin_cube = quat_apply(cube_quat_inv, origin_w - cube.data.root_pos_w)
        direction_cube = quat_apply(cube_quat_inv, direction_w)
        half_extents = torch.full_like(origin_cube, CUBE_HALF_HEIGHT)

        parallel = torch.abs(direction_cube) <= 1.0e-6
        safe_direction = torch.where(parallel, torch.ones_like(direction_cube), direction_cube)
        t1 = (-half_extents - origin_cube) / safe_direction
        t2 = (half_extents - origin_cube) / safe_direction
        near = torch.minimum(t1, t2)
        far = torch.maximum(t1, t2)
        near = torch.where(parallel, torch.full_like(near, -torch.inf), near)
        far = torch.where(parallel, torch.full_like(far, torch.inf), far)
        parallel_outside = torch.any(parallel & (torch.abs(origin_cube) > half_extents), dim=-1)
        t_near = torch.max(near, dim=-1).values
        t_far = torch.min(far, dim=-1).values
        obb_hit = (~parallel_outside) & (t_far >= torch.clamp(t_near, min=0.0))
        nearest_forward_hit = torch.clamp(t_near, min=0.0)
        # The bundled assessor uses an infinite ray, which closes too early in
        # this USD.  Range-gate the first OBB hit from the offset ray origin;
        # do not use the open jaw-frame origin as a distance reference.
        hit_within_grasp_reach = obb_hit & (nearest_forward_hit <= self.GREEN_RAY_MAX_HIT_DISTANCE)
        hit = hit_within_grasp_reach
        self.green_ray_obb_hit = bool(obb_hit.all().item())
        self.green_ray_within_grasp_reach = bool(hit_within_grasp_reach.all().item())
        self.green_ray_hit_distance = torch.where(
            obb_hit,
            nearest_forward_hit,
            torch.full_like(nearest_forward_hit, torch.nan),
        ).detach()
        self._update_green_ray_visualization(
            env,
            origin_w,
            direction_w,
            gripper_pos_w,
            gripper_quat_w,
            hit,
        )
        return bool(hit.all().item())

    def _update_green_ray_visualization(
        self,
        env,
        origin_w: torch.Tensor,
        direction_w: torch.Tensor,
        gripper_pos_w: torch.Tensor,
        gripper_quat_w: torch.Tensor,
        hit: torch.Tensor,
    ) -> None:
        """Render the active ray and all six gripper-local axis directions."""

        if self._green_ray_visualizer is None:
            return
        distances = torch.linspace(
            0.0,
            self.GREEN_RAY_VISUAL_LENGTH,
            self.GREEN_RAY_VISUAL_POINT_COUNT,
            device=env.device,
            dtype=origin_w.dtype,
        )
        ray_points_w = origin_w[:, None, :] + distances[None, :, None] * direction_w[:, None, :]
        ray_points_w = ray_points_w.reshape(-1, 3)
        ray_marker_indices = torch.where(
            hit,
            torch.ones_like(hit, dtype=torch.int32),
            torch.zeros_like(hit, dtype=torch.int32),
        )
        ray_marker_indices = ray_marker_indices[:, None].expand(-1, self.GREEN_RAY_VISUAL_POINT_COUNT).reshape(-1)
        endpoint_marker_indices = torch.cat(
            (
                torch.full((env.num_envs,), 2, device=env.device, dtype=torch.int32),
                torch.full((env.num_envs,), 3, device=env.device, dtype=torch.int32),
            )
        )
        local_axes = torch.tensor(
            tuple(self.LOCAL_RAY_AXES.values()),
            device=env.device,
            dtype=origin_w.dtype,
        )
        local_axes = local_axes.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3)
        repeated_quat_w = gripper_quat_w[:, None, :].expand(-1, len(self.LOCAL_RAY_AXES), -1).reshape(-1, 4)
        axes_w = quat_apply(repeated_quat_w, local_axes).reshape(env.num_envs, len(self.LOCAL_RAY_AXES), 3)
        axis_distances = torch.linspace(0.0, 0.10, 11, device=env.device, dtype=origin_w.dtype)
        axis_points_w = gripper_pos_w[:, None, None, :] + (axis_distances[None, None, :, None] * axes_w[:, :, None, :])
        axis_points_w = axis_points_w.reshape(-1, 3)
        axis_marker_indices = (
            torch.arange(4, 10, device=env.device, dtype=torch.int32)
            .repeat_interleave(axis_distances.numel())
            .repeat(env.num_envs)
        )
        positions_w = torch.cat((ray_points_w, origin_w, gripper_pos_w, axis_points_w), dim=0)
        marker_indices = torch.cat((ray_marker_indices, endpoint_marker_indices, axis_marker_indices), dim=0)
        self._green_ray_visualizer.visualize(translations=positions_w, marker_indices=marker_indices)

    def _object_grasped(self, env) -> bool:
        """Confirm transport from cube lift after feedback-gated closure."""

        if self._initial_cube_z_w is None:
            return False
        cube_z_w = env.scene["cube"].data.root_pos_w[:, 2]
        lifted = cube_z_w >= self._initial_cube_z_w + self.MIN_CONFIRMED_LIFT
        return bool(lifted.all().item())

    @staticmethod
    def _world_position_to_base(robot, position_w: torch.Tensor) -> torch.Tensor:
        return quat_apply(quat_inv(robot.data.root_quat_w), position_w - robot.data.root_pos_w)

    @staticmethod
    def _base_position_to_world(robot, position_b: torch.Tensor) -> torch.Tensor:
        return robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, position_b)

    @classmethod
    def _set_world_height(cls, robot, position_b: torch.Tensor, height_w: float) -> torch.Tensor:
        position_w = cls._base_position_to_world(robot, position_b)
        position_w[:, 2] = height_w
        return cls._world_position_to_base(robot, position_w)

    def _fail(self, reason: str) -> None:
        self.servo_abort_reason = reason
        self._state = "failed"
        self._state_step = 0
        self._episode_done = True

    @property
    def phase_name(self) -> str:
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def phase_step(self) -> int:
        return self._state_step

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def posture_target(self) -> torch.Tensor | None:
        return self._posture_target

    @property
    def wrist_position_w(self) -> torch.Tensor | None:
        return self._wrist_position_w

    @property
    def descent_wrist_xy_error(self) -> torch.Tensor | None:
        return self._descent_wrist_xy_error

    @property
    def approach_tracking_error(self) -> torch.Tensor | None:
        return self._approach_tracking_error

    @property
    def descent_ray_xy_error(self) -> torch.Tensor | None:
        return self._descent_ray_xy_error

    @property
    def descent_xy_correction_w(self) -> torch.Tensor | None:
        return self._descent_xy_correction_w

    @property
    def ray_alignment_streak(self) -> int:
        return self._ray_alignment_streak

    @property
    def ik_runtime_mode(self) -> str:
        if self._arm_action_term is None:
            return "uninitialized"
        return self._arm_action_term.runtime_mode

    @property
    def command_position_b(self) -> torch.Tensor | None:
        return self._command_pos_b

    @property
    def gripper_command(self) -> float:
        return self._gripper_command

    @property
    def gripper_settle_streak(self) -> int:
        return self._gripper_settle_streak

    @property
    def gripper_settle_reason(self) -> str | None:
        return self._gripper_settle_reason

    @property
    def gripper_target_error(self) -> torch.Tensor | None:
        return self._gripper_target_error

    @property
    def gripper_joint_velocity(self) -> torch.Tensor | None:
        return self._gripper_joint_velocity

    @property
    def held_gripper_angle(self) -> float | None:
        return self._held_gripper_angle

    @property
    def held_gripper_angle_capture_step(self) -> int | None:
        return self._held_gripper_angle_capture_step

    @property
    def servo_parameters(self) -> dict[str, object]:
        return {
            "source": "bundled_so101_autogen_simple_state_machine",
            "control_frame": "wrist",
            "ik_adapter": "xyz_plus_autogen_wrist_flex_correction,restored_from_c1295cb",
            "grasp_trigger": "range_gated_gripper_local_axis_cube_obb",
            "green_ray_frame": "ee_frame.target[0]:gripper_frame",
            "green_ray_local_axis": self._green_ray_axis,
            "green_ray_local_origin_offset": self.GREEN_RAY_ORIGIN_OFFSET,
            "green_ray_max_hit_distance": self.GREEN_RAY_MAX_HIT_DISTANCE,
            "grasp_confirmation": "feedback_settled_gripper_then_cube_lift_above_episode_initial_z",
            "gripper_target_reached_requires_low_velocity": True,
            "gripper_hold_trigger": "first_observed_pick_cube_true",
            "gripper_hold_valid_phases": "grasp_through_transport",
            "gripper_hold_value": "measured_gripper_joint_angle_at_trigger",
            "gripper_hold_until": "release",
            "minimum_confirmed_lift": self.MIN_CONFIRMED_LIFT,
            "gripper_settle_min_steps": self.GRASP_SETTLE_STEPS,
            "gripper_settle_max_steps": self.GRASP_SETTLE_MAX_STEPS,
            "gripper_settle_stable_steps": self.GRASP_SETTLE_STABLE_STEPS,
            "gripper_target_tolerance": self.GRIPPER_TARGET_TOLERANCE,
            "gripper_stall_velocity_tolerance": self.GRIPPER_STALL_VELOCITY_TOLERANCE,
            "approach_height": self.APPROACH_HEIGHT,
            "lift_height": self.LIFT_HEIGHT,
            "safe_height": self.SAFE_HEIGHT,
            "transport_height": self.TRANSPORT_HEIGHT,
            "effective_transport_target_height": self.RELEASE_HEIGHT,
            "travel_step": self.TRAVEL_STEP,
            "descend_step": self.DESCEND_STEP,
            "lift_step": self.LIFT_STEP,
            "close_openness_range": self.CLOSE_OPENNESS_RANGE,
            "height_coordinate": "world_z",
            "descent_wrist_xy_guard": self.MAX_DESCENT_WRIST_XY_ERROR,
            "descent_wrist_height_guard_w": self.MIN_WRIST_HEIGHT_W,
            "approach_tracking_tolerance": self.APPROACH_TRACKING_TOLERANCE,
            "approach_settle_timeout_steps": self.APPROACH_SETTLE_TIMEOUT_STEPS,
            "ray_alignment_kp": self.RAY_ALIGNMENT_KP,
            "ray_alignment_max_xy_step": self.RAY_ALIGNMENT_MAX_XY_STEP,
            "ray_alignment_xy_tolerance": self.RAY_ALIGNMENT_XY_TOLERANCE,
            "ray_alignment_stable_steps": self.RAY_ALIGNMENT_STABLE_STEPS,
            "grasp_reach_reference": "first_cube_obb_hit_from_offset_ray_origin",
            "green_ray_visual_length": self.GREEN_RAY_VISUAL_LENGTH,
            "green_ray_visual_colors": (
                "active:yellow=miss,green=hit; blue=ray_origin,purple=gripper; "
                "+x=red,-x=orange,+y=green,-y=yellow,+z=blue,-z=cyan"
            ),
        }


def configure_autogen_reference_action(env_cfg) -> None:
    """Adapt the environment action terms to Autogen wrist IK and gripper units."""

    env_cfg.actions.arm_action.class_type = PhaseAwareDifferentialInverseKinematicsAction
    env_cfg.actions.arm_action.body_name = "wrist"
    env_cfg.actions.gripper_action = isaac_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=1.0,
    )
