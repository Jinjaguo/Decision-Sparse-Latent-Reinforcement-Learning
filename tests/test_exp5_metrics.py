import numpy as np
import pytest

from decision_sparse_rl.metrics.exp5 import (
    bh_fdr, central_operator, mahalanobis_cost, monotone_match, operator_geometry,
    projector, projector_similarity, robust_scales, select_prototype_branches,
    shrinkage_covariance,
)


def test_scaling_floor_and_covariance():
    x = np.array([[1., 2.], [1., 4.], [1., 6.]])
    center, scale = robust_scales(x, [0.5, 0.1])
    np.testing.assert_allclose(center, [1., 4.]); assert scale[0] == 0.5
    covariance = shrinkage_covariance((x - center) / scale)
    assert np.all(np.linalg.eigvalsh(covariance) > 0)


def test_mahalanobis_and_monotone_known_answer():
    x = np.arange(4, dtype=float)[:, None]
    cost = mahalanobis_cost(x, x, np.eye(1))
    np.testing.assert_allclose(np.diag(cost), 0)
    path = monotone_match(cost, 0.25)["path"]
    np.testing.assert_array_equal(path, np.column_stack([np.arange(4), np.arange(4)]))


def test_prototype_selection_is_unique_and_deterministic():
    x = np.arange(8, dtype=float)[:, None]; p = np.array([[1.1], [1.2], [6.9]])
    selected = select_prototype_branches(x, p)
    assert selected.tolist() == [1, 2, 7]


def test_operator_and_projector_exact_cases():
    identity = np.eye(7); j = central_operator(identity, -identity, 1.0)
    np.testing.assert_allclose(j, identity)
    geometry = operator_geometry(j); assert geometry["effective_rank"] == pytest.approx(7)
    p1 = projector(np.eye(7), 1); assert projector_similarity(p1, p1, 1) == pytest.approx(1)
    p2 = projector(np.roll(np.eye(7), 1, axis=0), 1); assert projector_similarity(p1, p2, 1) == pytest.approx(0)


def test_bh_fdr_known_answer():
    np.testing.assert_allclose(bh_fdr([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
