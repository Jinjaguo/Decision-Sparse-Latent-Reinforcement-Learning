"""Audit EXP11 required manifests, artifacts, plots, hashes, and gates."""

import argparse,json,hashlib
from pathlib import Path
import pyarrow.parquet as pq

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--stage0",required=True,type=Path);p.add_argument("--calibration",required=True,type=Path);p.add_argument("--calibration-analysis",required=True,type=Path);p.add_argument("--formal",required=True,type=Path);p.add_argument("--formal-analysis",required=True,type=Path);a=p.parse_args();out=Path("runs")/a.run_id
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    required_stage0=["phase_action_support.parquet","action_residual_spectrum.parquet","basis_stability.parquet","action_bound_audit.parquet","stage0_family_support.json"]
    required_cal=["calibration_zero_controls.parquet","calibration_replacements.parquet","calibration_per_step.parquet","execution_fidelity.parquet","effect_size_summary.parquet","wrench_schema_audit.json","family_authorization.json","calibration_report.md"]
    required_formal=["zero_controls.parquet","replacements.parquet","per_step_response.parquet"]
    required_analysis=["contact_force_trajectory.parquet","raw_hash_manifest.json","linear_predictions.parquet","object_centric_predictions.parquet","switching_predictions.parquet","temporal_predictions.parquet","graph_predictions.parquet","terminal_predictions.parquet","conformal_sets.parquet","route_metrics.parquet","macro_sensitivity.parquet","decision_sparsity_results.parquet","ablation_results.parquet","scientific_decision.json","failure_examples.json"]
    plot_names=["action_residual_spectrum.png","temporal_basis_examples.png","basis_stability_by_phase.png","calibration_effect_distribution.png","effect_vs_amplitude.png","clipping_rate_by_family.png","phase_effect_heatmap_calibration.png","macro_effect_by_phase.png","macro_effect_by_temporal_mode.png","macro_effect_by_amplitude.png","paired_sign_asymmetry.png","trajectory_energy_by_route.png","pathwise_coverage_by_route.png","coverage_width_tradeoff.png","terminal_error_p90.png","predicate_consequence_roc.png","coarse_regime_prediction.png","object_centric_vs_full_state.png","switching_vs_global.png","conformal_vs_uncalibrated.png","crossdemo_macro_rank.png","top20_consequence_mass.png","heldout_temporal_mode_ranking.png","task_specific_summary.png"]
    locations={**{x:a.stage0/"artifacts"/x for x in required_stage0},**{x:a.calibration/"artifacts"/x for x in required_cal},**{x:a.formal/"artifacts"/x for x in required_formal},**{x:a.formal_analysis/"artifacts"/x for x in required_analysis}}
    plot_locations={x:(a.stage0/"plots"/x if x in plot_names[:3] else a.calibration_analysis/"plots"/x if x in plot_names[3:7] else a.formal_analysis/"plots"/x) for x in plot_names}
    checks={f"artifact:{k}":v.exists() and v.stat().st_size>0 for k,v in locations.items()};checks.update({f"plot:{k}":v.exists() and v.stat().st_size>0 for k,v in plot_locations.items()})
    formal_metrics=json.loads((a.formal/"metrics.json").read_text());analysis_metrics=json.loads((a.formal_analysis/"metrics.json").read_text());checks.update({"formal_zero_gate":formal_metrics["zero_gate_passed"],"formal_replacements_1232":formal_metrics["replacement_count"]==1232,"formal_demos_21":formal_metrics["reference_count"]==21,"raw_hash_verified":analysis_metrics["raw_hash_verified"],"plots_exact_24":len(plot_locations)==24,"all_required_nonempty":all(checks.values())})
    row_counts={k:pq.read_metadata(v).num_rows for k,v in locations.items() if v.suffix==".parquet" and v.exists()};hashes={str(v):sha(v) for v in list(locations.values())+list(plot_locations.values()) if v.exists()};payload={"status":"completed","passed":all(checks.values()),"checks":checks,"row_counts":row_counts,"sha256":hashes}
    (out/"metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(payload,indent=2));raise SystemExit(0 if payload["passed"] else 2)
if __name__=="__main__":main()
