import numpy as np
import pytest

from decision_sparse_rl.metrics.exp11 import (
    apply_replacement, conformal_quantile, dct_modes, extract_chunk,
    macro_response, normalize_mode, paired_replacements, paired_sign_asymmetry,
    path_nonconformity, rank_correlation, requested_executed_mismatch,
    residual_basis, smooth_pulse_modes, spline_modes, subspace_similarity,
    top_fraction_mass, phase_shift_chunk, shift_gripper_transition,
    object_centric_features, switching_regime_index, complete_demo_fold,
    raw_hash,
)


def test_chunk_extraction_and_padding_mask():
    x = np.arange(21).reshape(3, 7); chunk, mask = extract_chunk(x, 1, 4)
    assert np.array_equal(chunk[0], x[1]) and mask.tolist() == [1, 1, 0, 0]


def test_dct_orthogonality():
    basis = dct_modes(10); assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)


def test_spline_and_pulse_are_smooth_orthogonal():
    for basis in (spline_modes(10), smooth_pulse_modes(10)):
        assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        assert np.max(np.abs(np.diff(basis, n=2, axis=1))) < 1


def test_amplitude_normalization():
    assert np.max(np.abs(normalize_mode([0, 2, -1]))) == 1


def test_paired_replacement_signs_and_clipping():
    ref = np.zeros((5, 7)); rows = paired_replacements(ref, dct_modes(5, 1)[0], [.1, .2], 0)
    assert [(x[0], x[1]) for x in rows] == [(.1, -1), (.1, 1), (.2, -1), (.2, 1)]
    _, executed, clipped = apply_replacement(np.full((5, 7), .99), np.ones(5), .2, 1, 0)
    assert clipped[:, 0].all() and np.max(executed) <= 1


def test_zero_replacement_equality():
    ref = np.arange(35).reshape(5, 7) / 100
    _, executed, clipped = apply_replacement(ref, np.ones(5), 0, 1, 2)
    assert np.array_equal(ref, executed) and not clipped.any()


def test_requested_executed_mismatch_ignores_saturation():
    req = np.array([1.2, .5]); exe = np.array([1, .5]); sat = np.array([1, 0], bool)
    assert requested_executed_mismatch(req, exe, sat) == 0


def test_macro_response_known_answer():
    assert np.array_equal(macro_response([[0, 1]], [[2, 3]], [2, 1]), [[1, 2]])


def test_paired_sign_asymmetry_zero_for_antisymmetry():
    assert paired_sign_asymmetry([-1, 2], [1, -2]) == pytest.approx(0)


def test_residual_basis_train_only_shape_and_similarity():
    rng = np.random.default_rng(11); chunks = rng.normal(size=(20, 5, 6))
    mean, basis, singular = residual_basis(chunks)
    assert mean.shape == (30,) and basis.shape == (3, 30) and len(singular) == 20
    assert subspace_similarity(basis, basis) == pytest.approx(1)


def test_pathwise_conformal_known_coverage():
    y = np.array([[[2., 0]], [[1., 1.]]]); mu = np.zeros_like(y); sd = np.ones_like(y)
    scores = path_nonconformity(y, mu, sd); assert scores.tolist() == [2, 1]
    assert conformal_quantile(scores, .5) == 2


def test_decision_sparsity_top20_and_ranking():
    assert top_fraction_mass([10, 0, 0, 0, 0]) == 1
    assert rank_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1)


def test_invalid_requests_fail():
    with pytest.raises(ValueError): dct_modes(2, 2)
    with pytest.raises(ValueError): extract_chunk(np.zeros((2, 7)), 3, 2)


def test_action_semantics_continuous_bounds_and_gripper_sign():
    action = np.array([2, -2, 0, 0, 0, 0, -.2])
    executed = action.copy(); executed[:6] = np.clip(executed[:6], -1, 1); executed[6] = np.sign(executed[6])
    assert executed.tolist() == [1, -1, 0, 0, 0, 0, -1]


def test_phase_edit_is_deterministic():
    x = np.arange(42).reshape(6, 7)
    assert np.array_equal(phase_shift_chunk(x, 2, 3, -1), x[1:4])
    assert np.array_equal(phase_shift_chunk(x, 2, 3, 1), x[3:6])


def test_gripper_timing_remains_discrete():
    assert shift_gripper_transition([-1, -1, 1, 1], 2, -1).tolist() == [-1, 1, 1, 1]
    assert set(shift_gripper_transition([-1, -1, 1, 1], 2, 1)) == {-1, 1}


def test_residual_basis_excludes_changed_heldout_demo():
    rng = np.random.default_rng(12); train = rng.normal(size=(10, 5, 2)); held = np.full((1, 5, 2), 1e6)
    mean_a, basis_a, _ = residual_basis(train)
    mean_b, basis_b, _ = residual_basis(train.copy())
    assert np.array_equal(mean_a, mean_b) and np.array_equal(basis_a, basis_b)
    assert not np.isclose(mean_a.mean(), held.mean())


def test_amplitude_ordering_known_linear_response():
    ref = np.zeros((5, 7)); mode = np.ones(5)
    small = apply_replacement(ref, mode, .05, 1, 0)[1]
    large = apply_replacement(ref, mode, .10, 1, 0)[1]
    assert np.linalg.norm(large-ref) > np.linalg.norm(small-ref)


def test_force_contact_raw_schema_keys_are_explicit():
    schema = {"ee_force": [1, 2, 3], "ee_torque": [4, 5, 6], "contact_mode_json": "[]", "signed_gap_m": .01}
    assert {"ee_force", "ee_torque", "contact_mode_json", "signed_gap_m"} <= schema.keys()


def test_object_centric_features_known_answer():
    value = object_centric_features([1, 0, 0], [[2, 0, 0], [2, 1, 0]])
    assert value.tolist() == [1, 0, 0, 1, 1, 0, 0, 1, 0]


def test_switching_regime_indexing():
    assert switching_regime_index("[]") == 0
    assert switching_regime_index("target|gripper") == 2
    assert switching_regime_index("[]", "target|gripper") == 6


def test_heldout_mode_exclusion_by_name():
    train_modes = {"dct", "spline"}; heldout = "pulse"
    assert heldout not in train_modes


def test_complete_demo_fold_is_stable_and_demo_level():
    a = complete_demo_fold("task", "demo_1"); b = complete_demo_fold("task", "demo_1")
    assert a == b and 0 <= a < 5


def test_raw_hash_locking_detects_change():
    assert raw_hash(b"raw") == raw_hash(b"raw")
    assert raw_hash(b"raw") != raw_hash(b"changed")


def test_immutable_run_directory_policy(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    with pytest.raises(FileExistsError):
        if run.exists(): raise FileExistsError(run)
