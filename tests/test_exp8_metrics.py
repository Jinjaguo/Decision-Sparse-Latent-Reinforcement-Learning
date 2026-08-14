import numpy as np
import pytest

from decision_sparse_rl.metrics.exp8 import (
    assert_demo_isolation,
    binary_operating_metrics,
    contact_age_sequence,
    expected_calibration_error,
    permutation_invariant_pool,
    project_contact_motion,
    ridge_predict,
    select_risk_threshold,
    tangent_basis,
    top1_projector_similarity,
    upper_tail,
)


def test_tangent_gauge_is_deterministic_and_right_handed():
    frames = [np.stack(tangent_basis([0.2, -0.3, 0.9])) for _ in range(3)]
    assert np.array_equal(frames[0], frames[1])
    assert np.allclose(frames[0] @ frames[0].T, np.eye(3))
    assert np.linalg.det(frames[0]) == pytest.approx(1.0)


def test_tangent_gauge_rejects_singular_normal():
    with pytest.raises(ValueError):
        tangent_basis([0.0, 0.0, 0.0])


def test_action_projection_known_answer():
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    assert np.allclose(project_contact_motion(jacobian, [0.5, -0.25], np.eye(3)), [0.5, -0.5, 0.0])


def test_pair_pool_is_permutation_invariant():
    rows = [[1.0, 4.0], [3.0, 2.0]]
    assert np.array_equal(permutation_invariant_pool(rows), permutation_invariant_pool(rows[::-1]))


def test_contact_age_resets_when_identity_disappears():
    ages = contact_age_sequence([{"a"}, {"a", "b"}, {"b"}, set(), {"a"}])
    assert ages == [{"a": 1}, {"a": 2, "b": 1}, {"b": 2}, {}, {"a": 1}]


def test_ridge_is_deterministic_and_predicts_linear_case():
    x = np.arange(5.0)[:, None]
    y = 2.0 + 3.0 * x
    first = ridge_predict(x, y, np.array([[5.0]]), 1e-12)
    second = ridge_predict(x, y, np.array([[5.0]]), 1e-12)
    assert np.array_equal(first, second)
    assert first[0, 0] == pytest.approx(17.0)


def test_projector_similarity_ignores_sign():
    assert top1_projector_similarity([1, 0], [-1, 0]) == pytest.approx(1.0)
    assert top1_projector_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_crossfit_demo_isolation():
    assert_demo_isolation(["a", "b"], ["c"])
    with pytest.raises(ValueError):
        assert_demo_isolation(["a", "b"], ["b"])


def test_risk_threshold_uses_sensitivity_constraint():
    y = [1, 1, 1, 1, 0, 0]
    p = [0.9, 0.8, 0.7, 0.6, 0.55, 0.1]
    threshold = select_risk_threshold(y, p, 0.75)
    metrics = binary_operating_metrics(y, p, threshold)
    assert metrics["sensitivity"] >= 0.75
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["false_safe_rate"] == pytest.approx(0.0)


def test_calibration_and_tail_metrics_known_answers():
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    tail = upper_tail(np.arange(101.0))
    assert tail["median"] == 50.0
    assert tail["p90"] == 90.0
    assert tail["p95"] == 95.0
