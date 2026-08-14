import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np

isaaclab = ModuleType("isaaclab")
isaaclab.__path__ = []
isaaclab_app = ModuleType("isaaclab.app")
isaaclab_app.AppLauncher = object
sys.modules.setdefault("isaaclab", isaaclab)
sys.modules.setdefault("isaaclab.app", isaaclab_app)

path = Path(__file__).with_name("red_cube_to_box_expert_dataset.py")
spec = importlib.util.spec_from_file_location("red_cube_to_box_expert_dataset", path)
assert spec is not None
assert spec.loader is not None
collector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collector
spec.loader.exec_module(collector)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return FakeTensor(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_pre_step_observation_is_paired_with_post_apply_absolute_target() -> None:
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=FakeTensor([[1, 2, 3, 4, 5, 6]]),
            joint_vel=FakeTensor([[0, 0, 0, 0, 0, 0]]),
            joint_pos_target=FakeTensor([[11, 12, 13, 14, 15, 16]]),
            body_pos_w=FakeTensor([[[0, 0, 0], [1, 2, 3]]]),
            body_quat_w=FakeTensor([[[1, 0, 0, 0], [1, 0, 0, 0]]]),
        )
    )
    cube = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=FakeTensor([[0.1, 0.2, 0.3]]),
            root_quat_w=FakeTensor([[1, 0, 0, 0]]),
        )
    )
    observations = {"policy": {"front": FakeTensor([np.zeros((8, 12, 3), dtype=np.uint8)])}}
    pre_step = collector.snapshot_pre_step(observations, robot, cube, joint_ids=list(range(6)), gripper_body_index=1)

    robot.data.joint_pos = FakeTensor([[7, 8, 9, 10, 11, 12]])
    paired = collector.attach_applied_joint_target(pre_step, robot, joint_ids=list(range(6)), timestamp=0.25)

    np.testing.assert_array_equal(paired["obs/joint_pos"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(paired["actions"], [11, 12, 13, 14, 15, 16])
    assert paired["timestamps"] == 0.25
