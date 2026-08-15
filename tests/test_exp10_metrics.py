import numpy as np
import pytest

from decision_sparse_rl.metrics.exp10 import (
    action_summary,
    frozen_phase_sequence,
    history_summary,
    interval_coverage,
    one_hot,
    regime_from_pairs,
    sequence_edit_rate,
)


def test_phase_schema_is_monotone_and_predicate_terminal():
    phase = frozen_phase_sequence([0, .2, .3, .31, .7, .9], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1])
    assert np.all(np.diff(phase) >= 0)
    assert phase[-1] == 6 and phase[3] >= 2


def test_phase_schema_rejects_bad_shapes():
    with pytest.raises(ValueError):
        frozen_phase_sequence([0, 1], [True], [False, True])


def test_summaries_have_fixed_shapes_and_masks():
    a = np.arange(21).reshape(3, 7)
    assert action_summary(a, [1, 1, 0]).shape == (36,)
    h = np.arange(12).reshape(3, 4)
    assert history_summary(h, [0, 1, 1]).shape == (17,)


def test_regime_vocabulary_and_edit_rate():
    assert regime_from_pairs([], []) == 0
    assert regime_from_pairs([], ["object|gripper_finger"]) == 2
    assert regime_from_pairs(["object|gripper_finger"], []) == 6
    assert sequence_edit_rate([0, 2, 2], [0, 3, 2]) == pytest.approx(1 / 3)


def test_one_hot_and_coverage():
    assert one_hot([0, 6]).shape == (2, 7)
    assert interval_coverage(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2))) == 1.0
