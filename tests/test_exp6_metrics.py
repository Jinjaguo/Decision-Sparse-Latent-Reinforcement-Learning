import numpy as np

from decision_sparse_rl.metrics.exp6 import (
    antithetic_asymmetry,
    first_contact_divergence,
    projector,
    projector_similarity,
    relative_discrepancy,
    repeatability_max_abs,
    signal_to_floor,
    trust_region_passes,
)


def test_identical_projectors_are_one_and_orthogonal_rank_one_is_zero():
    eye = np.eye(3)
    p0 = projector(eye[:, [0]], 1)
    p1 = projector(eye[:, [1]], 1)
    assert projector_similarity(p0, p0, 1) == 1.0
    assert np.isclose(projector_similarity(p0, p1, 1), 0.0)


def test_frozen_relative_and_antithetic_metrics():
    assert np.isclose(relative_discrepancy(2.0, 3.0), 0.4)
    assert antithetic_asymmetry(np.array([1.0, 2.0]), np.array([-1.0, -2.0])) == 0.0
    assert signal_to_floor(1e-8, 0.0) == 1e4


def test_repeatability_and_contact_divergence():
    assert repeatability_max_abs([np.array([1.0, 2.0]), np.array([1.0, 2.0])]) == 0.0
    assert first_contact_divergence([{"a"}, {"a"}], [{"a"}, {"b"}]) == 2
    assert first_contact_divergence([set()], [set()]) is None


def test_trust_region_threshold_boundaries_are_inclusive():
    passed, failed = trust_region_passes(0.80, 0.75, 0.20, 0.25, 0.35, 100.0)
    assert passed and failed == []
    passed, failed = trust_region_passes(0.79, 0.75, 0.20, 0.25, 0.35, 100.0)
    assert not passed and failed == ["top1_similarity"]
