import numpy as np

from decision_sparse_rl.metrics.exp6 import (
    adjacent_radius_pairs,
    antithetic_asymmetry,
    first_contact_divergence,
    heldout_relative_error,
    projector,
    projector_similarity,
    relative_discrepancy,
    repeatability_max_abs,
    signal_to_floor,
    trust_region_passes,
    zero_floor_upper,
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


def test_radius_ordering_and_adjacent_pairing():
    assert adjacent_radius_pairs([0.0003125, 0.000625, 0.00125]) == [(0.0003125, 0.000625), (0.000625, 0.00125)]
    with np.testing.assert_raises(ValueError):
        adjacent_radius_pairs([0.00125, 0.000625])


def test_zero_floor_and_heldout_prediction():
    assert zero_floor_upper([0.0, 1e-13, 0.0]) == 1e-13
    assert heldout_relative_error(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_known_convergent_and_nonconvergent_operator_projectors():
    eye = np.eye(7)
    stable_a = projector(eye[:, :2], 2)
    stable_b = projector((2.0 * eye)[:, :2], 2)
    changed = projector(eye[:, 2:4], 2)
    assert projector_similarity(stable_a, stable_b, 2) == 1.0
    assert projector_similarity(stable_a, changed, 2) == 0.0


def test_unresolved_trust_region_exposes_all_failed_criteria():
    passed, failed = trust_region_passes(0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
    assert not passed
    assert set(failed) == {"top1_similarity", "top2_similarity", "spectral_discrepancy", "sign_asymmetry", "heldout_vector_error", "signal_to_floor"}


def test_contact_sequence_length_mismatch_is_rejected():
    with np.testing.assert_raises(ValueError):
        first_contact_divergence([set()], [set(), set()])
