import numpy as np

from decision_sparse_rl.metrics.exp4 import basis_rms, direction_pair_metrics, first_crossing_interpolate, gram_spectrum, orthonormal_audit


def test_direction_pair_metrics_and_rms():
    assert direction_pair_metrics(3.0, 1.0) == {"symmetric": 2.0, "asymmetry": 0.5 - 1.25e-16}
    assert basis_rms([2.0] * 7) == 2.0


def test_first_crossing_preserves_nonmonotone_temporal_identity():
    values, source = first_crossing_interpolate([0.0, 0.8, 0.4, 1.0], [0.0, 8.0, 4.0, 10.0], [0.5, 0.9])
    np.testing.assert_allclose(values, [5.0, 9.0])
    np.testing.assert_array_equal(source, [0, 2])


def test_first_crossing_duplicate_and_nearest_tie_use_earlier_index():
    values, source = first_crossing_interpolate([0.2, 0.2], [1.0, 2.0], [0.2, 0.5])
    np.testing.assert_allclose(values, [1.0, 1.0])
    np.testing.assert_array_equal(source, [0, 0])


def test_gram_spectrum_known_diagonal_operator():
    columns = np.diag(np.arange(1.0, 8.0)) * 0.02
    result = gram_spectrum(columns, 0.01)
    np.testing.assert_allclose(result["singular_values"], np.arange(7.0, 0.0, -1.0))


def test_orthonormal_audit():
    result = orthonormal_audit(np.eye(7))
    assert result["passed"]
    assert result["maximum_orthogonality_error"] == 0.0
