"""Deterministic MuJoCo RedCubeToBox environment for the Menagerie PiPER."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from examples.piper import contract


@dataclass(frozen=True)
class ResetInfo:
    seed: int
    cube_position_m: np.ndarray
    target_position_m: np.ndarray


@dataclass(frozen=True)
class StepResult:
    observation: contract.Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required for PiPER simulation. Install the piper dependency group or run "
            "`python -m pip install 'mujoco>=2.3.4,<4'`."
        ) from exc
    return mujoco


def resolve_model_dir(model_dir: Path | None = None) -> Path:
    """Locate the external Menagerie agilex_piper directory without vendoring meshes."""

    candidates: list[Path] = []
    if model_dir is not None:
        candidates.append(Path(model_dir))
    if value := os.environ.get("MUJOCO_MENAGERIE_PATH"):
        candidates.extend([Path(value) / "agilex_piper", Path(value)])
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / "third_party" / "mujoco_menagerie" / "agilex_piper",
            repo_root.parent / "mujoco_menagerie" / "agilex_piper",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "piper.xml").is_file() and (resolved / "assets").is_dir():
            return resolved
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find MuJoCo Menagerie agilex_piper. Set MUJOCO_MENAGERIE_PATH to the Menagerie checkout. "
        f"Searched:\n  - {searched}"
    )


def _add_task_elements(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(contract.PHYSICS_TIMESTEP_S))

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="checker", type="2d", builtin="checker", width="256", height="256")
    ET.SubElement(asset, "material", name="table_material", texture="checker", texrepeat="6 6")
    ET.SubElement(asset, "material", name="target_material", rgba="0.1 0.35 0.9 1")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Menagerie piper.xml has no worldbody")
    link6 = worldbody.find(".//body[@name='link6']")
    if link6 is None:
        raise ValueError("Menagerie piper.xml has no link6 body")
    ET.SubElement(
        link6,
        "site",
        name="gripper_site",
        pos="0 0 0.09",
        size="0.008",
        rgba="0 1 0 0.7",
    )
    for finger_name in ("link7", "link8"):
        finger = worldbody.find(f".//body[@name='{finger_name}']")
        if finger is None:
            raise ValueError(f"Menagerie piper.xml has no {finger_name} body")
        for geom in finger.findall("geom"):
            if geom.get("class") == "collision":
                geom.set("friction", "1.5 0.01 0.001")
                geom.set("condim", "3")
    gripper_actuator = root.find("actuator/position[@name='gripper']")
    if gripper_actuator is None:
        raise ValueError("Menagerie piper.xml has no gripper actuator")
    gripper_actuator.set("kp", "160")
    gripper_actuator.set("kv", "8")
    gripper_actuator.set("forcerange", "-60 60")

    ET.SubElement(worldbody, "light", name="task_light", pos="0.2 -0.2 1.1", directional="true")
    ET.SubElement(
        worldbody,
        "camera",
        name="base_camera",
        pos="0.72 -0.72 0.58",
        mode="targetbody",
        target="camera_target",
        fovy="48",
    )
    ET.SubElement(worldbody, "body", name="camera_target", pos="0.32 0 0.10")
    ET.SubElement(
        worldbody,
        "geom",
        name="table",
        type="box",
        pos="0.30 0 -0.025",
        size="0.36 0.30 0.025",
        material="table_material",
        friction="1.2 0.01 0.001",
    )

    cube = ET.SubElement(worldbody, "body", name="red_cube", pos="0.36 -0.12 0.026")
    ET.SubElement(cube, "freejoint", name="red_cube_freejoint")
    ET.SubElement(
        cube,
        "geom",
        name="red_cube_geom",
        type="box",
        size="0.025 0.025 0.025",
        mass="0.025",
        rgba="0.85 0.05 0.05 1",
        friction="1.5 0.01 0.001",
    )

    target = ET.SubElement(worldbody, "body", name="target_box", mocap="true", pos="0.36 0.12 0.006")
    ET.SubElement(
        target,
        "geom",
        name="target_floor",
        type="box",
        pos="0 0 0",
        size="0.055 0.055 0.006",
        material="target_material",
        contype="1",
        conaffinity="1",
    )
    wall_specs = (
        ("target_wall_x_neg", "-0.061 0 0.025", "0.006 0.061 0.025"),
        ("target_wall_x_pos", "0.061 0 0.025", "0.006 0.061 0.025"),
        ("target_wall_y_neg", "0 -0.061 0.025", "0.055 0.006 0.025"),
        ("target_wall_y_pos", "0 0.061 0.025", "0.055 0.006 0.025"),
    )
    for name, position, size in wall_specs:
        ET.SubElement(
            target,
            "geom",
            name=name,
            type="box",
            pos=position,
            size=size,
            material="target_material",
        )
    ET.SubElement(target, "site", name="target_center", pos="0 0 0.027", size="0.006", rgba="0 1 1 0.8")

    key = root.find("keyframe/key[@name='home']")
    if key is not None:
        original = key.get("qpos", "")
        key.set("qpos", f"{original} 0.36 -0.12 0.026 1 0 0 0".strip())


def build_task_xml(model_dir: Path) -> tuple[str, dict[str, bytes]]:
    """Build the task MJCF and in-memory mesh asset dictionary."""

    model_dir = resolve_model_dir(model_dir)
    root = ET.parse(model_dir / "piper.xml").getroot()
    root.set("model", "piper_red_cube_to_box")
    _add_task_elements(root)
    xml = ET.tostring(root, encoding="unicode")
    assets = {
        f"assets/{path.name}": path.read_bytes() for path in sorted((model_dir / "assets").iterdir()) if path.is_file()
    }
    return xml, assets


class PiperRedCubeToBoxEnv:
    """Small Gym-like environment with a strict seven-dimensional action boundary."""

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        render: bool = True,
        maximum_steps: int = 720,
        success_stable_steps: int = 5,
    ) -> None:
        self._mujoco = _import_mujoco()
        self.model_dir = resolve_model_dir(model_dir)
        xml, assets = build_task_xml(self.model_dir)
        self.model = self._mujoco.MjModel.from_xml_string(xml, assets=assets)
        self.data = self._mujoco.MjData(self.model)
        self.maximum_steps = maximum_steps
        self.success_stable_steps = success_stable_steps
        self._renderer = (
            self._mujoco.Renderer(self.model, height=contract.IMAGE_HEIGHT, width=contract.IMAGE_WIDTH)
            if render
            else None
        )
        self._step_count = 0
        self._success_count = 0
        self._last_action = np.concatenate([contract.HOME_ARM_Q_RAD, [1.0]]).astype(np.float32)
        self._rng = np.random.default_rng(0)

        self._arm_joint_ids = np.array(
            [self._name2id(self._mujoco.mjtObj.mjOBJ_JOINT, name) for name in contract.ARM_JOINT_NAMES]
        )
        self._arm_qpos_ids = self.model.jnt_qposadr[self._arm_joint_ids]
        self._arm_dof_ids = self.model.jnt_dofadr[self._arm_joint_ids]
        self._finger_joint_id = self._name2id(self._mujoco.mjtObj.mjOBJ_JOINT, "joint7")
        self._finger_qpos_id = int(self.model.jnt_qposadr[self._finger_joint_id])
        self._other_finger_joint_id = self._name2id(self._mujoco.mjtObj.mjOBJ_JOINT, "joint8")
        self._other_finger_qpos_id = int(self.model.jnt_qposadr[self._other_finger_joint_id])
        cube_joint_id = self._name2id(self._mujoco.mjtObj.mjOBJ_JOINT, "red_cube_freejoint")
        self._cube_qpos_adr = int(self.model.jnt_qposadr[cube_joint_id])
        self._cube_body_id = self._name2id(self._mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self._finger_body_ids = {
            self._name2id(self._mujoco.mjtObj.mjOBJ_BODY, "link7"),
            self._name2id(self._mujoco.mjtObj.mjOBJ_BODY, "link8"),
        }
        self._target_body_id = self._name2id(self._mujoco.mjtObj.mjOBJ_BODY, "target_box")
        self._target_mocap_id = int(self.model.body_mocapid[self._target_body_id])
        self.gripper_site_id = self._name2id(self._mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
        self.target_site_id = self._name2id(self._mujoco.mjtObj.mjOBJ_SITE, "target_center")

    def _name2id(self, object_type, name: str) -> int:
        object_id = int(self._mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise ValueError(f"task model is missing {name}")
        return object_id

    @property
    def arm_qpos_ids(self) -> np.ndarray:
        return self._arm_qpos_ids.copy()

    @property
    def arm_dof_ids(self) -> np.ndarray:
        return self._arm_dof_ids.copy()

    @property
    def mujoco(self):
        """Expose the lazily imported MuJoCo module to kinematics helpers."""

        return self._mujoco

    @property
    def model_sha256(self) -> str:
        return hashlib.sha256((self.model_dir / "piper.xml").read_bytes()).hexdigest()

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def cube_position(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_id].copy()

    @property
    def target_position(self) -> np.ndarray:
        return self.data.site_xpos[self.target_site_id].copy()

    @property
    def gripper_position(self) -> np.ndarray:
        return self.data.site_xpos[self.gripper_site_id].copy()

    @property
    def gripper_rotation(self) -> np.ndarray:
        return self.data.site_xmat[self.gripper_site_id].reshape(3, 3).copy()

    def reset(self, *, seed: int = 0) -> tuple[contract.Observation, ResetInfo]:
        self._rng = np.random.default_rng(seed)
        self._mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[self._arm_qpos_ids] = contract.HOME_ARM_Q_RAD
        self.data.qpos[self._finger_qpos_id] = contract.GRIPPER_JOINT_MAX_M
        self.data.qpos[self._other_finger_qpos_id] = -contract.GRIPPER_JOINT_MAX_M

        cube_position = np.array(
            [self._rng.uniform(0.35, 0.39), self._rng.uniform(-0.13, -0.09), 0.026], dtype=np.float64
        )
        target_position = np.array(
            [self._rng.uniform(0.35, 0.39), self._rng.uniform(0.09, 0.13), 0.006], dtype=np.float64
        )
        cube_slice = slice(self._cube_qpos_adr, self._cube_qpos_adr + 7)
        self.data.qpos[cube_slice] = np.concatenate([cube_position, [1.0, 0.0, 0.0, 0.0]])
        self.data.mocap_pos[self._target_mocap_id] = target_position
        self.data.mocap_quat[self._target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.ctrl[:] = np.concatenate([contract.HOME_ARM_Q_RAD, [contract.GRIPPER_JOINT_MAX_M]])
        self._mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            self._mujoco.mj_step(self.model, self.data)
        self._step_count = 0
        self._success_count = 0
        self._last_action = np.concatenate([contract.HOME_ARM_Q_RAD, [1.0]]).astype(np.float32)
        return self.observe(), ResetInfo(seed, cube_position.copy(), target_position.copy())

    def render(self) -> np.ndarray:
        if self._renderer is None:
            return np.zeros((contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3), dtype=np.uint8)
        self._renderer.update_scene(self.data, camera="base_camera")
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def state(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos[self._arm_qpos_ids],
                [contract.normalize_gripper_joint(float(self.data.qpos[self._finger_qpos_id]))],
            ]
        ).astype(np.float32)

    def observe(self) -> contract.Observation:
        state = contract.validate_state(self.state())
        return contract.Observation(image=self.render(), state=state, timestamp_ns=time.monotonic_ns())

    def _inside_target(self) -> bool:
        relative = self.cube_position - self.target_position
        released = contract.normalize_gripper_joint(float(self.data.qpos[self._finger_qpos_id])) >= 0.90
        return bool(
            abs(relative[0]) <= 0.032
            and abs(relative[1]) <= 0.032
            and abs(relative[2]) <= 0.012
            and released
            and not self.has_two_finger_contact()
        )

    def has_two_finger_contact(self) -> bool:
        """Return whether the cube currently touches collision geoms on both fingers."""

        contacting_fingers: set[int] = set()
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            bodies = {
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            }
            if self._cube_body_id not in bodies:
                continue
            contacting_fingers.update(bodies & self._finger_body_ids)
        return contacting_fingers == self._finger_body_ids

    def step(self, action: np.ndarray, *, rate_limit: bool = True) -> StepResult:
        requested = contract.validate_action(action)
        executed = contract.limit_action_step(self._last_action, requested) if rate_limit else requested
        self.data.ctrl[:6] = executed[:6]
        self.data.ctrl[6] = contract.denormalize_gripper(float(executed[6]))
        for _ in range(contract.PHYSICS_STEPS_PER_CONTROL):
            self._mujoco.mj_step(self.model, self.data)
        self._last_action = executed
        self._step_count += 1

        in_target = self._inside_target()
        self._success_count = self._success_count + 1 if in_target else 0
        success = self._success_count >= self.success_stable_steps
        truncated = self._step_count >= self.maximum_steps and not success
        distance = float(np.linalg.norm(self.cube_position - self.target_position))
        return StepResult(
            observation=self.observe(),
            reward=1.0 if success else -distance,
            terminated=success,
            truncated=truncated,
            info={
                "success": success,
                "cube_in_target": in_target,
                "cube_target_distance_m": distance,
                "executed_action": executed.copy(),
                "step": self._step_count,
            },
        )

    def hold_measured_pose(self) -> contract.Observation:
        """Hold the latest measured pose for one control tick after a policy fault.

        This deliberately bypasses the application soft-limit check used for new
        policy targets: a disturbed measured joint may be inside the Menagerie
        hard limit but just outside our softer command envelope.  The safe action
        in that case is still to hold the fresh measurement, not to continue an
        older queued command or reject the hold itself.
        """

        measured = contract.validate_state(self.state())
        self.data.ctrl[:6] = measured[:6]
        self.data.ctrl[6] = contract.denormalize_gripper(float(measured[6]))
        for _ in range(contract.PHYSICS_STEPS_PER_CONTROL):
            self._mujoco.mj_step(self.model, self.data)
        self._last_action = measured.copy()
        self._step_count += 1
        return self.observe()

    def set_arm_configuration(self, q_rad: np.ndarray) -> None:
        """Set arm qpos for kinematic computations; does not advance dynamics."""

        q_rad = np.asarray(q_rad, dtype=np.float64)
        if q_rad.shape != (6,) or not np.isfinite(q_rad).all():
            raise ValueError("arm configuration must be finite shape (6,)")
        self.data.qpos[self._arm_qpos_ids] = np.clip(q_rad, contract.ARM_Q_MIN_APP_RAD, contract.ARM_Q_MAX_APP_RAD)
        self._mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
