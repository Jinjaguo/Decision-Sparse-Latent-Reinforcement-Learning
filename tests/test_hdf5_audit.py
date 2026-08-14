from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.utils.hdf5_audit import audit_hdf5  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is not installed")
class Hdf5AuditTest(unittest.TestCase):
    def test_discovers_episode_schema_and_semantic_fields(self) -> None:
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["env"] = "test"
                episode = handle.create_group("data/demo_0")
                episode.create_dataset("states", data=np.zeros((4, 9)))
                episode.create_dataset("actions", data=np.zeros((3, 7)))
                obs = episode.create_group("obs")
                obs.create_dataset("joint_states", data=np.zeros((3, 7)))
            audit = audit_hdf5(path)
            self.assertEqual(audit["episode_detection"]["episode_count"], 1)
            self.assertEqual(audit["episode_detection"]["episodes"][0]["states_length"], 4)
            self.assertTrue(audit["presence"]["actions"])
            self.assertTrue(audit["presence"]["joint_states"])
            self.assertFalse(audit["presence"]["gripper_states"])


if __name__ == "__main__":
    unittest.main()
