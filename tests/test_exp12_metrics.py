from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_sparse_rl.metrics.exp12 import (
    abstaining_choice,
    catastrophic_selection,
    group_candidates,
    nominal_improvement_opportunity,
    oracle_best_indices,
    pairwise_accuracy,
    pairwise_preferences,
    regret,
    require_absent,
    sha256_file,
    top1_accuracy,
    validate_demo_isolation,
)


def test_same_state_grouping_and_candidate_isolation():
    rows = [
        {"branch_id": "a", "candidate": 0},
        {"branch_id": "b", "candidate": 1},
        {"branch_id": "a", "candidate": 2},
    ]
    groups = group_candidates(rows)
    assert [x["candidate"] for x in groups["a"]] == [0, 2]
    assert [x["candidate"] for x in groups["b"]] == [1]


def test_pairwise_and_setwise_labels_respect_ties():
    quality = [1.0, 1.03, 0.2]
    labels = pairwise_preferences(quality, tie_tolerance=0.05)
    assert labels[0][2] == 0
    assert labels[1][2] == 1
    assert set(oracle_best_indices(quality, 0.05)) == {0, 1}


def test_regret_top1_and_pairwise_accuracy():
    quality = [0.0, 1.0, 0.5]
    assert regret(quality, 2) == 0.5
    assert top1_accuracy(quality, 1) == 1.0
    assert top1_accuracy(quality, 2) == 0.0
    assert pairwise_accuracy(quality, [0.1, 0.9, 0.4]) == 1.0


def test_catastrophic_selection_requires_safe_alternative():
    assert catastrophic_selection([False, True], 1)
    assert not catastrophic_selection([True, True], 1)


def test_nominal_opportunity_and_abstention():
    opportunity, gap = nominal_improvement_opportunity([0.4, 0.8, 0.5], 0)
    assert opportunity and np.isclose(gap, 0.4)
    assert abstaining_choice([0.2, 0.9], [0.1, 0.8], 0, 0.5) == (0, True)
    assert abstaining_choice([0.2, 0.9], [0.1, 0.2], 0, 0.5) == (1, False)


def test_whole_demo_split_isolation():
    assert validate_demo_isolation([{"demo_key": "a", "fold": 0}, {"demo_key": "b", "fold": 1}])
    assert not validate_demo_isolation([{"demo_key": "a", "fold": 0}, {"demo_key": "a", "fold": 1}])


def test_prediction_write_once_and_hash_lock(tmp_path):
    target = tmp_path / "prediction.lock"
    require_absent(target)
    target.write_bytes(b"first")
    first = sha256_file(target)
    try:
        require_absent(target)
        assert False, "existing prediction lock was accepted"
    except FileExistsError:
        pass
    target.write_bytes(b"second")
    assert sha256_file(target) != first
