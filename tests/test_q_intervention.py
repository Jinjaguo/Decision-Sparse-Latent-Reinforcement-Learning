import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.interventions.q_intervention import (  # noqa: E402
    apply_arm_q, non_arm_integration_linf, scaled_joint_delta,
)


class QInterventionTest(unittest.TestCase):
    def test_mask_preserves_non_arm_components(self):
        before = {"mjSTATE_QPOS": np.arange(5, dtype=float), "mjSTATE_CTRL": np.asarray([2.0])}
        after = {name: value.copy() for name, value in before.items()}
        after["mjSTATE_QPOS"][[1, 3]] += np.asarray([0.1, -0.1])
        errors = non_arm_integration_linf(before, after, [1, 3])
        self.assertEqual(0.0, max(errors.values()))

    def test_scaled_delta_and_limit_check(self):
        delta = scaled_joint_delta(np.asarray([3.0, 4.0]), np.asarray([2.0, 2.0]), 0.005)
        self.assertTrue(np.allclose(delta, [0.006, 0.008]))
        data = SimpleNamespace(qpos=np.zeros(3))
        result = apply_arm_q(data, [0, 2], delta, np.asarray([-1.0, -1.0]), np.asarray([1.0, 1.0]))
        self.assertTrue(np.allclose(result, delta))
        self.assertEqual(0.0, data.qpos[1])


if __name__ == "__main__":
    unittest.main()
