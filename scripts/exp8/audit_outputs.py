#!/usr/bin/env python
"""Audit completeness, raw hashes, and classification of the final EXP8 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

ARTIFACTS = [
    "reference_contact_frames.parquet", "contact_frame_support_audit.parquet", "zero_controls.parquet",
    "interventions.parquet", "per_step_effects.parquet", "horizon_operators.parquet", "operator_matrices.parquet",
    "baseline_predictions.parquet", "contact_frame_predictions.parquet", "crossdemo_subspace_reuse.parquet",
    "heldout_vector_predictions.parquet", "ablation_results.parquet", "horizon_locality.parquet",
    "risk_predictions.parquet", "risk_metrics.json", "gpu_audit.json", "gpu_cpu_equivalence.json",
    "scientific_decision.json", "failure_examples.json", "raw_hash_manifest.json",
]
PLOTS = [
    "contact_frame_feature_support.png", "contact_normal_distribution.png", "signed_gap_vs_contact_age.png",
    "action_projection_distribution.png", "baseline_vs_contactframe_top1.png", "crossdemo_top1_by_task.png",
    "heldout_vector_error_distribution.png", "heldout_vector_error_p90_by_task.png", "ablation_incremental_value.png",
    "horizon_coverage_adjusted_similarity.png", "risk_roc_pr.png", "risk_calibration.png",
    "risk_specificity_sensitivity_tradeoff.png", "false_safe_rate_by_task.png", "contact_identity_generalization.png",
    "gpu_cpu_equivalence.png",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True); args = parser.parse_args()
    run, artifacts = args.run.resolve(), args.run.resolve() / "artifacts"
    missing_artifacts = [name for name in ARTIFACTS if not (artifacts / name).exists()]
    missing_plots = [name for name in PLOTS if not (run / "plots" / name).exists()]
    decision = json.loads((artifacts / "scientific_decision.json").read_text(encoding="utf-8")) if not missing_artifacts else {}
    raw = json.loads((artifacts / "raw_hash_manifest.json").read_text(encoding="utf-8")) if (artifacts / "raw_hash_manifest.json").exists() else {}
    hashes = raw.get("sha256", {})
    raw_hashes_valid = all((artifacts / name).exists() and sha(artifacts / name) == value for name, value in hashes.items() if name in ("zero_controls.parquet", "interventions.parquet", "per_step_effects.parquet"))
    criteria = {
        "all_required_artifacts": not missing_artifacts, "all_required_plots": not missing_plots,
        "raw_hashes_valid": raw_hashes_valid, "exact_360_zero": not missing_artifacts and pq.read_metadata(artifacts / "zero_controls.parquet").num_rows == 360,
        "exact_17280_interventions": not missing_artifacts and pq.read_metadata(artifacts / "interventions.parquet").num_rows == 17280,
        "classification_valid": decision.get("classification") in ["continuous_contact_field_replicates", "contact_geometry_improves_but_tail_risk_remains", "mode_risk_gate_passes_without_operator_reuse", "continuous_geometry_insufficient", "support_or_identifiability_failure", "no_support"],
        "online_and_latent_remain_forbidden": decision.get("online_control_eligible") is False and decision.get("latent_RL_eligible") is False,
    }
    result = {"classification": decision.get("classification"), "required_artifact_count": len(ARTIFACTS), "required_plot_count": len(PLOTS), "missing_artifacts": missing_artifacts, "missing_plots": missing_plots, "criteria": criteria, "hashes": {name: sha(artifacts / name) for name in ARTIFACTS if (artifacts / name).exists()}, "gate": {"passed": all(criteria.values())}}
    (artifacts / "output_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
