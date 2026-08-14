import json
from pathlib import Path

import numpy as np
import pytest

from decision_sparse_rl.metrics.criticality import concentration_metrics
from scripts.exp4.analyze_criticality import equivalence_passes, verify_raw_hashes
from scripts.exp4.freeze_protocol import canonical_basis


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPOSITORY_ROOT / "experiments/exp4_replicated_progress_criticality/manifests"


def test_two_spike_concentration_curve():
    result = concentration_metrics([1.0, 1.0] + [0.0] * 10)
    assert result["top10_mass"] == 1.0
    assert result["top20_mass"] == 1.0
    assert result["normalized_entropy"] == pytest.approx(np.log(2) / np.log(12))


def test_all_frozen_exp4_bases_regenerate_and_pass_limits():
    manifest = json.loads((MANIFEST_ROOT / "direction_basis_manifest.json").read_text())
    assert manifest["basis_count"] == 252
    assert manifest["direction_count"] == 2016
    assert manifest["signed_intervention_count"] == 4032
    for record in manifest["bases"]:
        seed = np.random.SeedSequence(manifest["master_seed_uint64"], spawn_key=tuple(record["spawn_key"]))
        basis, audit, _ = canonical_basis(seed)
        np.testing.assert_array_equal(basis, np.asarray(record["basis_matrix_columns"]))
        assert audit["passed"]
        assert record["maximum_orthogonality_error"] <= 1e-12
    assert all(row["both_signs_within_joint_limits"] for row in manifest["directions"])


def test_progress_manifest_has_all_reference_channels_and_milestones():
    manifest = json.loads((MANIFEST_ROOT / "task_demo_manifest.json").read_text())
    assert manifest["task_demo_count"] == 21
    for row in manifest["tasks"]:
        progress = row["progress"]
        assert len(progress["raw"]) == row["trajectory_length"]
        assert len(progress["clipped"]) == row["trajectory_length"]
        assert np.all(np.isfinite(progress["raw"]))
        if row["task"] == "put_the_bowl_on_the_plate":
            assert set(progress["milestones"]) == {"reach_index", "lift_2cm_index", "transport_5cm_index", "exact_on_first_true_index"}
            values = [progress["milestones"][name] for name in ("reach_index", "lift_2cm_index", "transport_5cm_index", "exact_on_first_true_index")]
            assert values == sorted(values)
        else:
            assert progress["derivation"]["successful_direction_sign"] in (-1, 1)


def test_gpu_equivalence_gate_pass_and_fallback_behavior():
    tolerances = {"scalar_atol": 1e-12, "matrix_atol": 1e-11, "spectrum_atol": 1e-10}
    comparisons = {"aggregation_s_rms_abs": 1e-15, "concentration_max_abs": 1e-15, "spearman_rank_inputs_exact": True, "bootstrap_max_abs": 0.0, "gram_max_abs": 1e-14, "singular_values_max_abs": 1e-13, "interpolation_max_abs": 0.0, "interpolation_sources_exact": True}
    assert equivalence_passes(comparisons, tolerances)
    failed = dict(comparisons); failed["bootstrap_max_abs"] = 1e-4
    assert not equivalence_passes(failed, tolerances)


def test_raw_hash_verification_detects_mutation(tmp_path):
    payload = tmp_path / "raw.bin"; payload.write_bytes(b"frozen")
    import hashlib
    lock = {"schema_version": 1, "sha256": {"raw.bin": hashlib.sha256(b"frozen").hexdigest()}}
    (tmp_path / "raw_artifact_hashes.json").write_text(json.dumps(lock))
    assert verify_raw_hashes(tmp_path) == lock["sha256"]
    payload.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_raw_hashes(tmp_path)
