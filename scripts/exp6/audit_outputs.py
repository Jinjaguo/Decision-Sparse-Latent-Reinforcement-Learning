#!/usr/bin/env python
"""Audit EXP6 formal artifact completeness, row counts, hashes, and plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

REQUIRED = ["zero_controls.parquet", "zero_repeatability.parquet", "radius_interventions.parquet", "per_step_effects.parquet", "signed_output_vectors.parquet", "radius_operator_summary.parquet", "operator_matrices_by_radius.npz", "adjacent_radius_comparisons.parquet", "trust_region_summary.parquet", "contact_mode_transitions.parquet", "contact_conditioned_convergence.parquet", "heldout_direction_prediction.parquet", "signal_to_floor.parquet", "gpu_audit.json", "gpu_cpu_equivalence.json", "scientific_decision.json", "failure_examples.json", "raw_hash_manifest.json"]
PLOTS = ["signal_to_zero_floor_by_radius.png", "repeatability_by_radius.png", "spectral_norm_vs_radius.png", "top1_similarity_adjacent_radii.png", "top2_similarity_adjacent_radii.png", "spectral_discrepancy_adjacent_radii.png", "sign_asymmetry_vs_radius.png", "heldout_prediction_error_vs_radius.png", "trust_region_radius_distribution.png", "trust_region_by_task.png", "trust_region_by_contact_state.png", "contact_mode_divergence_vs_convergence.png", "contact_mode_divergence_timing.png", "operator_spectrum_vs_radius.png", "demo_level_small_radius_convergence.png", "gpu_cpu_equivalence.png"]


def sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest(); return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--analysis-run", type=Path, required=True); parser.add_argument("--raw-run", type=Path, required=True); args = parser.parse_args(); artifacts = args.analysis_run / "artifacts"; plots = args.analysis_run / "plots"; lock = json.loads((args.raw_run / "artifacts/raw_hash_manifest.json").read_text())
    raw_mapping = {"zero_controls.parquet": "zero_controls.parquet", "interventions.parquet": "radius_interventions.parquet", "per_step_effects.parquet": "per_step_effects.parquet"}; hash_checks = {source: sha(artifacts / target) == expected for source, expected in lock["sha256"].items() if source in raw_mapping for target in [raw_mapping[source]]}
    criteria = {"all_required_artifacts": all((artifacts / name).is_file() for name in REQUIRED), "all_required_plots": all((plots / name).is_file() and (plots / name).stat().st_size > 5000 for name in PLOTS), "formal_interventions_19200": pq.read_metadata(artifacts / "radius_interventions.parquet").num_rows == 19200, "operators_1200": pq.read_metadata(artifacts / "radius_operator_summary.parquet").num_rows == 1200, "adjacent_comparisons_960": pq.read_metadata(artifacts / "adjacent_radius_comparisons.parquet").num_rows == 960, "trust_rows_240": pq.read_metadata(artifacts / "trust_region_summary.parquet").num_rows == 240, "contact_pairs_9600": pq.read_metadata(artifacts / "contact_mode_transitions.parquet").num_rows == 9600, "raw_hashes_unchanged": all(hash_checks.values()) and len(hash_checks) == 3}
    result = {"schema_version": 1, "passed": all(criteria.values()), "criteria": criteria, "hash_checks": hash_checks, "required_artifact_count": len(REQUIRED), "plot_count": len(PLOTS)}; (artifacts / "formal_output_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2)); return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
