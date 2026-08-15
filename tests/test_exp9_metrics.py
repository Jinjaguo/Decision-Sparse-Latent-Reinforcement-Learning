import numpy as np

from decision_sparse_rl.metrics.exp9 import (
    binary_metrics,
    contact_set_events,
    energy_score,
    lifted_features,
    interval_coverage,
    mixture_moments,
    padded_future,
    padded_window,
    select_specificity_threshold,
    validate_demo_fold_isolation,
)


def test_history_padding_and_mask():
    x = np.arange(12, dtype=float).reshape(4, 3)
    out, mask = padded_window(x, 1, 5)
    assert out.shape == (5, 3)
    assert np.array_equal(mask, [0, 0, 0, 1, 1])
    assert np.array_equal(out[-2:], x[:2])


def test_future_padding_and_mask():
    x = np.arange(12, dtype=float).reshape(4, 3)
    out, mask = padded_future(x, 3, 3)
    assert np.array_equal(mask, [1, 0, 0])
    assert np.array_equal(out[0], x[3])
    assert np.all(out[1:] == 0)


def test_contact_events_known_answer():
    additions, removals = contact_set_events(["a", "b"], ["b", "c"])
    assert additions == ["c"]
    assert removals == ["a"]


def test_energy_score_point_mass_is_error_norm():
    samples = np.array([[[1.0, 2.0]]] * 3).reshape(1, 3, 2)
    score = energy_score(samples, np.array([[1.0, 4.0]]))
    assert np.allclose(score, [2.0])


def test_threshold_respects_sensitivity():
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2]
    t = select_specificity_threshold(y, p, 0.75)
    metrics = binary_metrics(y, p, t)
    assert metrics["sensitivity"] >= 0.75
    assert metrics["specificity"] >= 0.75


def test_lifted_dictionary_deterministic():
    x = np.arange(30, dtype=float).reshape(3, 10)
    a = lifted_features(x, 6)
    b = lifted_features(x.copy(), 6)
    assert np.array_equal(a, b)
    assert a.shape[0] == 3 and a.shape[1] > x.shape[1]


def test_mixture_weights_and_moments_known_answer():
    w = np.array([[0.25, 0.75]])
    means = np.array([[[0.0], [2.0]]])
    stds = np.ones_like(means)
    mean, std = mixture_moments(w, means, stds)
    assert np.allclose(mean, [[1.5]])
    assert np.allclose(std, [[np.sqrt(1.75)]])


def test_invalid_mixture_rejected():
    with np.testing.assert_raises(ValueError):
        mixture_moments(np.array([[0.2, 0.2]]), np.zeros((1, 2, 1)), np.ones((1, 2, 1)))


def test_interval_coverage_known_answer():
    assert interval_coverage([0, 1, 2, 3], [0, 0, 0, 4], [0, 1, 1, 5]) == 0.5


def test_demo_fold_isolation():
    assert validate_demo_fold_isolation([("a", "d0")], [("a", "d1")])
    assert not validate_demo_fold_isolation([("a", "d0")], [("a", "d0")])
