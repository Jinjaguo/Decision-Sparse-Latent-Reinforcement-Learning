import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.controller_snapshot import (
    CONTROLLER_FIELDS, ROBOT_BUFFER_FIELDS, capture, deserialize, field_errors, restore, serialize,
)  # noqa: E402


class _Delta:
    def __init__(self, dim):
        self.last = np.arange(dim, dtype=float)
        self.current = np.arange(dim, dtype=float) + 1


class _Ring:
    def __init__(self, dim):
        self.buf = np.arange(10 * dim, dtype=float).reshape(10, dim)
        self.ptr = 3
        self._size = 4


def _environment():
    controller = SimpleNamespace()
    for index, name in enumerate(CONTROLLER_FIELDS):
        setattr(controller, name, None if name == "ori_ref" else (True if name == "new_update" else np.asarray([float(index)])))
    robot = SimpleNamespace(controller=controller, torques=np.asarray([1.0]))
    for name in ROBOT_BUFFER_FIELDS:
        setattr(robot, name, _Ring(2) if name == "recent_ee_vel_buffer" else _Delta(2))
    return SimpleNamespace(robots=[robot], env=SimpleNamespace(timestep=3, cur_time=0.15, done=False))


class ControllerSnapshotTest(unittest.TestCase):
    def test_round_trip_and_serialization(self):
        env = _environment()
        original = capture(env)
        env.robots[0].controller.goal_pos[:] = 99
        env.robots[0].recent_actions.current[:] = 88
        env.env.timestep = 99
        restore(env, original)
        self.assertTrue(all(value == 0.0 for value in field_errors(original, capture(env)).values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.npz"
            serialize(path, original)
            loaded = deserialize(path)
        self.assertTrue(all(value == 0.0 for value in field_errors(original, loaded).values()))


if __name__ == "__main__":
    unittest.main()
