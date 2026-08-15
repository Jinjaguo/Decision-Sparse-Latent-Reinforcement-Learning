from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_sparse_rl.metrics.exp13 import (
    authorize_family, bounded_candidate, fit_progress_direction, nearest_library,
    shift_chunk, temporal_warp,
)


def test_temporal_warp_preserves_exact_gripper_signs():
    chunk = np.zeros((5, 7)); chunk[:, 0] = np.arange(5); chunk[:, 6] = [-1, -1, 1, 1, 1]
    out = temporal_warp(chunk, .5)
    assert set(out[:, 6]) <= {-1, 1}
    assert np.all(np.diff(out[:, 0]) >= 0)


def test_shift_chunk_clips_only_indexes():
    actions = np.arange(35).reshape(5, 7)
    assert np.array_equal(shift_chunk(actions, 1, 3, -2)[0], actions[0])
    assert np.array_equal(shift_chunk(actions, 1, 3, 2)[-1], actions[4])


def test_progress_direction_recovers_informative_channel():
    rng = np.random.default_rng(13); x = rng.normal(size=(50, 6)); y = 3 * x[:, 2] + .01 * rng.normal(size=50)
    direction = fit_progress_direction(x, y)
    assert np.argmax(np.abs(direction)) == 2 and direction[2] > 0


def test_nearest_library_excludes_target_demo():
    contexts = np.asarray([[0, 0], [1, 1], [2, 2]], float)
    selected = nearest_library([.1, .1], contexts, [True, False, False], 1)
    assert selected.tolist() == [1]


def test_bounded_candidate_and_authorization():
    ref = np.zeros((2, 7)); desired = np.ones((2, 7)) * 2; desired[:, 6] = -.3
    delta, executed = bounded_candidate(ref, desired)
    assert np.max(executed[:, :6]) == 1 and np.all(executed[:, 6] == -1)
    assert authorize_family(.05, .9, .1, .03)
    assert not authorize_family(.11, .9, .1, .03)

