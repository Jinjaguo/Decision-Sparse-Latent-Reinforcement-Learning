from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_sparse_rl.metrics.exp14 import diverse_topk, object_frame_target, ridge_inverse, unit_vector, waypoint_chunk


def test_unit_vector_handles_zero_and_direction():
    assert np.allclose(unit_vector([0, 0, 0]), 0)
    assert np.allclose(unit_vector([3, 0, 0]), [1, 0, 0])


def test_waypoint_chunk_integrates_requested_translation():
    ref = np.zeros((10, 7)); ref[:, 6] = -1
    out = waypoint_chunk(ref, [.05, -.025, 0], taper=True)
    assert np.allclose(out[:, :3].sum(0) * .05, [.05, -.025, 0])
    assert np.all(out[:, 6] == -1)


def test_waypoint_rejects_bad_shape():
    with pytest.raises(ValueError): waypoint_chunk(np.zeros((10, 6)), [0, 0, 0])


def test_object_frame_target_uses_robust_relative_center():
    assert np.allclose(object_frame_target([1, 2, 3], [[1, 0, 0], [3, 0, 0], [2, 0, 0]]), [3, 2, 3])


def test_ridge_inverse_recovers_forward_direction():
    x = np.vstack([np.eye(6), -np.eye(6)])
    y = x[:, :3] * 2
    action = ridge_inverse(x, y, [1, 0, 0], l2=1e-6)
    assert action[0] > .49 and np.linalg.norm(action[1:]) < 1e-8


def test_diverse_topk_avoids_duplicate_before_fallback():
    x = np.asarray([[0, 0], [0, .01], [1, 0], [0, 1]], float)
    assert diverse_topk(x, [4, 3, 2, 1], 3, .1) == [0, 2, 3]
