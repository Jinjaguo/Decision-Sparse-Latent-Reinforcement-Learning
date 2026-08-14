#!/usr/bin/env python
"""Audit exact EXP7 formal artifact, figure, raw-lock, and decision coverage."""

from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

ARTIFACTS=["reference_contact_geometry.parquet","boundary_margin_calibration.parquet","zero_controls.parquet","interventions.parquet","per_step_effects.parquet","mode_outcomes.parquet","horizon_operator_summary.parquet","operator_matrices.parquet","within_mode_convergence.parquet","boundary_margin_analysis.parquet","horizon_comparison.parquet","heldout_direction_prediction.parquet","mode_conditioned_crossdemo.parquet","mode_predictor_predictions.parquet","mode_predictor_metrics.json","gpu_audit.json","gpu_cpu_equivalence.json","scientific_decision.json","failure_examples.json","raw_hash_manifest.json"]
PLOTS=["signed_gap_repeatability.png","contact_mode_frequency.png","boundary_margin_distribution.png","within_mode_top1_by_radius.png","within_mode_top2_by_radius.png","within_mode_spectral_discrepancy.png","within_mode_sign_asymmetry.png","convergence_by_boundary_margin.png","convergence_by_horizon.png","intent_to_perturb_vs_conditional.png","mode_transition_categories.png","heldout_vector_error_within_mode.png","crossdemo_subspace_time_vs_progress_vs_mode.png","mode_predictor_roc_pr.png","mode_predictor_calibration.png","task_specific_hybrid_summary.png","gpu_cpu_equivalence.png"]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,required=True); args=p.parse_args(); run=args.run.resolve(); missing_art=[x for x in ARTIFACTS if not (run/"artifacts"/x).is_file()]; missing_plot=[x for x in PLOTS if not (run/"plots"/x).is_file()]; decision=json.loads((run/"artifacts/scientific_decision.json").read_text()) if not missing_art else {}; audit={"required_artifact_count":len(ARTIFACTS),"required_plot_count":len(PLOTS),"missing_artifacts":missing_art,"missing_plots":missing_plot,"classification":decision.get("classification"),"hashes":{x:sha(run/"artifacts"/x) for x in ARTIFACTS if (run/"artifacts"/x).is_file()},"gate":{"passed":not missing_art and not missing_plot and bool(decision.get("classification"))}}; (run/"artifacts/output_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2)); return 0 if audit["gate"]["passed"] else 2
if __name__=="__main__": raise SystemExit(main())
