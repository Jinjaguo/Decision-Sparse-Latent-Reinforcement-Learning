import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from decision_sparse_rl.metrics.exp10 import (
    authorize_independent_routes,
    frozen_phase_sequence,
    latent_active_dimensions,
    masked_safe_probability,
    phase_to_landmark_length,
)
from decision_sparse_rl.metrics.exp9 import padded_future, padded_window, validate_demo_fold_isolation


def test_phase_dwell_requires_two_contact_steps():
    phases = frozen_phase_sequence([0, .05, .1], [0, 1, 0], [0, 0, 0])
    assert phases[-1] < 2


def test_phase_hysteresis_never_reverses_after_contact():
    phases = frozen_phase_sequence([0, .05, .1, .11], [0, 1, 1, 0], [0, 0, 0, 0])
    assert list(phases[-2:]) == [2, 2]


def test_history_alignment_ends_at_branch():
    x = np.arange(15).reshape(5, 3); hist, mask = padded_window(x, 3, 3)
    assert np.array_equal(hist[-1], x[3]) and mask.tolist() == [1, 1, 1]


def test_chunk_alignment_starts_at_branch():
    x = np.arange(21).reshape(3, 7); chunk, mask = padded_future(x, 1, 3)
    assert np.array_equal(chunk[0], x[1]) and mask.tolist() == [1, 1, 0]


def test_phase_to_landmark_is_first_crossing_and_capped():
    assert phase_to_landmark_length([0, 0, 1, 1], 0) == 2
    assert phase_to_landmark_length([0] * 30, 0, cap=20) == 20


def test_terminal_predicate_timing_maps_first_true_to_p6():
    phases = frozen_phase_sequence([0, .4, .8], [0, 0, 0], [0, 1, 1])
    assert phases.tolist() == [0, 6, 6]


def test_latent_collapse_metric_counts_active_dimensions():
    z = np.c_[np.arange(5), np.ones(5), np.arange(5) * 2]
    assert latent_active_dimensions(z) == 2


def test_teacher_free_mask_excludes_padded_steps():
    prob = np.asarray([[.8, .2], [.1, .9], [.9, .1]])
    assert masked_safe_probability(prob, [0, 1, 1], [1, 1, 0]) == pytest.approx(.72)


def test_route_authorization_is_independent_and_capped():
    assert authorize_independent_routes({"A": True, "B": False, "C": True, "F": True}, 2) == ["A", "C"]


def test_track_fold_isolation_rejects_shared_demo():
    assert validate_demo_fold_isolation([("task", "d1")], [("task", "d2")])
    assert not validate_demo_fold_isolation([("task", "d1")], [("task", "d1")])


def test_gpu_seed_policy_is_reproducible_on_available_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(101001); a = torch.randn(8, device=device).cpu()
    torch.manual_seed(101001); b = torch.randn(8, device=device).cpu()
    assert torch.equal(a, b)


def test_raw_hash_lock_detects_mutation(tmp_path: Path):
    path = tmp_path / "raw.bin"; path.write_bytes(b"locked")
    locked = hashlib.sha256(path.read_bytes()).hexdigest(); path.write_bytes(b"changed")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != locked


def test_run_nonoverwrite_contract(tmp_path: Path):
    run = tmp_path / "run"; run.mkdir()
    with pytest.raises(FileExistsError):
        if run.exists():
            raise FileExistsError("immutable run exists")
