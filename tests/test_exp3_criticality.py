import sys
import unittest
from pathlib import Path
import json

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.metrics.criticality import (  # noqa: E402
    aggregate_interventions, concentration_metrics, quaternion_geodesic,
    rotation_geodesic,
)
from scripts.exp3.analyze_criticality import bh_adjust  # noqa: E402


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
        drifted = identity.copy()
        drifted[0, 0] = np.nextafter(1.0, 0.0)
        self.assertEqual(0.0, rotation_geodesic(drifted, drifted.copy()))
        self.assertEqual(0.0, quaternion_geodesic([0.999999999999, 0, 0, 0], [0.999999999999, 0, 0, 0]))

    def test_eight_way_aggregation(self):
        result = aggregate_interventions(range(8))
        self.assertEqual(3.5, result["median"])
        with self.assertRaises(ValueError):
            aggregate_interventions(range(7))

    def test_frozen_pcg64_directions_are_exactly_reproducible(self):
        path = REPOSITORY_ROOT / "experiments/exp3_time_indexed_q_criticality/manifests/direction_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(432, manifest["direction_count"])
        self.assertEqual(864, manifest["signed_intervention_count"])
        for row in manifest["directions"]:
            seed = np.random.SeedSequence(manifest["master_seed_uint64"], spawn_key=tuple(row["spawn_key"]))
            raw = np.random.Generator(np.random.PCG64(seed)).standard_normal(7)
            unit = raw / np.linalg.norm(raw)
            self.assertTrue(np.array_equal(raw, np.asarray(row["raw_direction"])))
            self.assertTrue(np.array_equal(unit, np.asarray(row["unit_direction"])))
            delta = np.asarray(row["unsigned_delta_q"])
            self.assertTrue(np.allclose(delta, 0.005 * np.asarray([5.7946, 3.5256, 5.7946, 3.002, 5.7946, 3.77, 5.7946]) * unit))

    def test_benjamini_hochberg_known_values(self):
        self.assertTrue(np.allclose(bh_adjust([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04]))


if __name__ == "__main__":
    unittest.main()
