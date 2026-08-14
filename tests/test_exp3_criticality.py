import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.metrics.criticality import (  # noqa: E402
    aggregate_interventions, concentration_metrics, quaternion_geodesic,
    rotation_geodesic,
)


class Exp3CriticalityTest(unittest.TestCase):
    def test_known_single_spike_12_point_curve(self):
        result = concentration_metrics([12.0] + [0.0] * 11)
        self.assertEqual(1.0, result["top10_mass"])
        self.assertEqual(1.0, result["top20_mass"])
        self.assertAlmostEqual(11.0 / 12.0, result["gini"])
        self.assertEqual(0.0, result["normalized_entropy"])

    def test_uniform_12_point_curve(self):
        result = concentration_metrics([1.0] * 12)
        self.assertAlmostEqual(2.0 / 12.0, result["top10_mass"])
        self.assertAlmostEqual(3.0 / 12.0, result["top20_mass"])
        self.assertAlmostEqual(4.0 / 12.0, result["top30_mass"])
        self.assertAlmostEqual(0.0, result["gini"])
        self.assertAlmostEqual(1.0, result["normalized_entropy"])

    def test_all_zero_convention(self):
        result = concentration_metrics([0.0] * 12)
        self.assertTrue(result["all_zero"])
        self.assertEqual(0.0, result["top20_mass"])
        self.assertEqual(0.0, result["gini"])
        self.assertEqual(1.0, result["normalized_entropy"])

    def test_orientation_distances(self):
        identity = np.eye(3)
        half_turn = np.diag([-1.0, -1.0, 1.0])
        self.assertEqual(0.0, rotation_geodesic(identity, identity))
        self.assertAlmostEqual(np.pi, rotation_geodesic(identity, half_turn))
        self.assertEqual(0.0, quaternion_geodesic([1, 0, 0, 0], [-1, 0, 0, 0]))

    def test_eight_way_aggregation(self):
        result = aggregate_interventions(range(8))
        self.assertEqual(3.5, result["median"])
        with self.assertRaises(ValueError):
            aggregate_interventions(range(7))


if __name__ == "__main__":
    unittest.main()

