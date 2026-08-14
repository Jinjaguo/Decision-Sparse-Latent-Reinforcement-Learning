import tempfile
import unittest
from pathlib import Path
import sys

import mujoco
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.mujoco_snapshot import capture, deserialize, restore, serialize  # noqa: E402


class _RawSim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><joint name='slider' type='slide'/><geom size='.1'/></body></worldbody><actuator><motor joint='slider'/></actuator></mujoco>")
        self.data = mujoco.MjData(self.model)


class MujocoSnapshotTest(unittest.TestCase):
    def test_native_round_trip_and_serialization(self):
        sim = _RawSim()
        sim.data.qpos[0] = 0.3
        sim.data.qvel[0] = -0.2
        sim.data.ctrl[0] = 0.4
        sim.data.qacc_warmstart[0] = 0.5
        mujoco.mj_forward(sim.model, sim.data)
        original = capture(sim, "integration")
        sim.data.qpos[0] = 1.0
        sim.data.ctrl[0] = -1.0
        restore(sim, original)
        restored = capture(sim, "integration")
        self.assertEqual(0.0, float(np.linalg.norm(original.values - restored.values)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            serialize(path, original)
            loaded = deserialize(path)
        self.assertEqual(original.kind, loaded.kind)
        self.assertTrue(np.array_equal(original.values, loaded.values))


if __name__ == "__main__":
    unittest.main()
