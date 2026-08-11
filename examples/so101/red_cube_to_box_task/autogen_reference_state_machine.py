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

import random

import isaaclab.envs.mdp as isaac_mdp
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
    GRASP_DURATION_STEPS = 45
    GRASP_SETTLE_STEPS = 21
    RELEASE_DURATION_STEPS = 180

    GRIPPER_OPEN_POSITION = 1.74533
    GRIPPER_CLOSED_POSITION = 0.0
    CLOSE_OPENNESS_RANGE = (0.18, 0.235)

    GREEN_RAY_ORIGIN_OFFSET = (0.0, 0.0, -0.04)
    GREEN_RAY_DIRECTION = (-1.0, 0.0, 0.0)

    GROUND_GUARD_HEIGHT_W = 0.01
    MIN_WRIST_HEIGHT_W = 0.03
    MAX_DESCENT_WRIST_XY_ERROR = 0.05

    MAX_STEPS = 2500

    def __init__(self) -> None:
        self._arm_action_term = None
        self._rng: random.Random | None = None
        self._wrist_body_index: int | None = None
        self.reset()

    def setup(self, env) -> None:
        """Resolve the adapted wrist IK action and reproduce Autogen damping."""

        env.scene["robot"].write_joint_damping_to_sim(damping=10.0)
        self._arm_action_term = resolve_action_term(env.action_manager, "arm_action")
        if not isinstance(self._arm_action_term, PhaseAwareDifferentialInverseKinematicsAction):
            raise RuntimeError("autogen_reference requires PhaseAwareDifferentialInverseKinematicsAction")
        # simple_state_machine.py explicitly disables its optional posture
        # correction before descending. Its Lula call constrains wrist_link
        # position only; retain that actual runtime behavior here.
        self._arm_action_term.set_position_only(enabled=True)

        body_names = list(env.scene["robot"].data.body_names)
        self._wrist_body_index = body_names.index("wrist")
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
        self._initial_cube_z_w: torch.Tensor | None = None
        self._posture_target: torch.Tensor | None = None
        self._wrist_position_w: torch.Tensor | None = None
        self._descent_wrist_xy_error: torch.Tensor | None = None
        self.green_ray_hit = False
        self.green_ray_origin_w: torch.Tensor | None = None
        self.green_ray_direction_w: torch.Tensor | None = None
        self.green_ray_hit_distance: torch.Tensor | None = None
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

        self._update_state(env)

        assert self._command_pos_b is not None
        robot = env.scene["robot"]
        wrist_quat_w = robot.data.body_quat_w[:, self._wrist_body_index]
        target_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), wrist_quat_w)
        gripper = torch.full(
            (env.num_envs, 1),
            self._gripper_command,
            device=env.device,
            dtype=self._command_pos_b.dtype,
        )
        return torch.cat((self._command_pos_b, target_quat_b, gripper), dim=-1)

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
                self._transition("descend")
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
                self._transition("grasp")
            elif bool((self._descent_wrist_xy_error > self.MAX_DESCENT_WRIST_XY_ERROR).any().item()):
                self._fail("actual wrist XY drifted more than 50 mm during Autogen descent")
            elif bool((wrist_pos_w[:, 2] < self.MIN_WRIST_HEIGHT_W).any().item()):
                self._fail("actual wrist reached the Autogen descent safety height")
            elif self._state_step > self.MAX_DESCEND_STEPS:
                self._fail("descent timed out before the Autogen green ray intersected the cube")
            else:
                assert self._command_pos_b is not None
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
            if self._state_step > self.GRASP_SETTLE_STEPS:
                self._transition("lift")
        elif self._state == "lift":
            if self._state_step % self.GRASP_CHECK_INTERVAL == 0:
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

    def _green_ray_intersects_cube(self, env) -> bool:
        """Evaluate the original infinite green-ray versus cube OBB test."""

        ee_frame = env.scene["ee_frame"]
        cube = env.scene["cube"]
        # The source Autogen implementation constructs this ray from
        # ``gripper_frame_link``.  In this LeIsaac scene that frame is the
        # first FrameTransformer target; the articulation body named
        # ``gripper`` is a different frame and changes the ray geometry.
        gripper_pos_w = ee_frame.data.target_pos_w[:, 0]
        gripper_quat_w = ee_frame.data.target_quat_w[:, 0]
        jaw_pos_w = ee_frame.data.target_pos_w[:, 1]
        offset = torch.tensor(self.GREEN_RAY_ORIGIN_OFFSET, device=env.device).repeat(env.num_envs, 1)
        direction = torch.tensor(self.GREEN_RAY_DIRECTION, device=env.device).repeat(env.num_envs, 1)
        origin_w = gripper_pos_w + quat_apply(gripper_quat_w, offset)
        direction_w = quat_apply(gripper_quat_w, direction)
        self.gripper_frame_position_w = gripper_pos_w.detach().clone()
        self.jaw_detection_position_w = jaw_pos_w.detach().clone()
        self.green_ray_origin_w = origin_w.detach().clone()
        self.green_ray_direction_w = direction_w.detach().clone()
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
        hit = (~parallel_outside) & (t_far >= torch.clamp(t_near, min=0.0))
        nearest_forward_hit = torch.clamp(t_near, min=0.0)
        self.green_ray_hit_distance = torch.where(
            hit,
            nearest_forward_hit,
            torch.full_like(nearest_forward_hit, torch.nan),
        ).detach()
        return bool(hit.all().item())

    @staticmethod
    def _object_grasped(env) -> bool:
        robot = env.scene["robot"]
        jaw_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 1]
        cube_pos_w = env.scene["cube"].data.root_pos_w
        distance = torch.linalg.vector_norm(cube_pos_w - jaw_pos_w, dim=-1)
        grasped = (distance < 0.02) & (robot.data.joint_pos[:, -1] < 0.26)
        return bool(grasped.all().item())

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
    def servo_parameters(self) -> dict[str, object]:
        return {
            "source": "bundled_so101_autogen_simple_state_machine",
            "control_frame": "wrist",
            "ik_adapter": "position_only_wrist_xyz,source_posture_correction_disabled",
            "grasp_trigger": "original_green_ray_cube_obb",
            "green_ray_frame": "ee_frame.target[0]:gripper_frame_link_equivalent",
            "grasp_confirmation_frame": "ee_frame.target[1]:jaw_detection_frame",
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
