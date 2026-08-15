"""Independent read-only audit of completed EXP10 retrospective outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts.exp10.run_stage_a_routes import sha256


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--dataset-run", default="runs/exp10_a0_phase_macro_dataset_r2_20260814"); parser.add_argument("--model-run", default="runs/exp10_a1_a6_multiroute_models_r1_20260814"); parser.add_argument("--endpoint-run", default="runs/exp10_a7_endpoint_route_20260814"); parser.add_argument("--protocol-run", default="runs/exp10_a9_protocol_artifacts_20260814")
    args = parser.parse_args(); out = Path("runs") / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    out.mkdir(parents=True)
    dataset, model, endpoint, protocol = map(Path, (args.dataset_run, args.model_run, args.endpoint_run, args.protocol_run))
    drows = pq.read_table(dataset / "artifacts" / "macro_trajectory_dataset.parquet").num_rows
    tpath = model / "artifacts" / "trajectory_predictions.parquet"; epath = model / "artifacts" / "event_terminal_predictions.parquet"
    trows, erows = pq.read_table(tpath).num_rows, pq.read_table(epath).num_rows
    lock = json.loads((model / "artifacts" / "prediction_lock.json").read_text(encoding="utf-8")); auth = json.loads((model / "artifacts" / "route_authorization.json").read_text(encoding="utf-8")); dgate = json.loads((endpoint / "metrics.json").read_text(encoding="utf-8"))
    required_dataset = ["reference_phase_labels.parquet", "phase_support_audit.json", "target_support_audit.json", "macro_trajectory_dataset.parquet"]
    required_model = ["trajectory_predictions.parquet", "event_terminal_predictions.parquet", "prediction_lock.json", "trajectory_metrics.parquet", "event_terminal_metrics.parquet", "route_authorization.json"]
    required_protocol = ["phase_support_audit.parquet", "phase_transition_matrix.parquet", "macro_chunk_support.parquet", *[f"track{x}_predictions.parquet" for x in "ABCDEF"], "route_metrics.parquet", "route_authorization.json", "ablation_results.parquet", "seed_variance.json", "stageA_report.md"]
    checks = {
        "dataset_rows_17280": drows == 17280,
        "trajectory_prediction_rows": trows == 23 * 2 * 17280,
        "event_prediction_rows": erows == 3 * 17280,
        "trajectory_hash_matches": sha256(tpath) == lock["trajectory_predictions_sha256"],
        "event_hash_matches": sha256(epath) == lock["event_terminal_predictions_sha256"],
        "dataset_artifacts_present": all((dataset / "artifacts" / x).exists() for x in required_dataset),
        "dataset_manifests_six": len(list((dataset / "manifests").glob("*.json"))) == 6,
        "model_artifacts_present": all((model / "artifacts" / x).exists() for x in required_model),
        "plots_exactly_16": len(list((model / "plots").glob("*.png"))) == 16,
        "exact_protocol_artifacts_present": all((protocol / "artifacts" / x).exists() for x in required_protocol),
        "exact_protocol_plots_16": len(list((protocol / "plots").glob("*.png"))) == 16,
        "all_primary_routes_failed": not any(auth["route_gates"].values()),
        "track_d_or_gate_failed": not dgate["track_d_authorized"],
        "new_cohort_not_authorized": not auth["new_cohort_authorized"],
        "no_stage_b_run_declared": len(auth["qualified_routes"]) == 0,
    }
    result = {"status": "completed", "run_id": args.run_id, "checks": checks, "passed": all(checks.values()), "dataset_rows": drows, "trajectory_prediction_rows": trows, "event_prediction_rows": erows, "final_route_gates": {**auth["route_gates"], "D": bool(dgate["track_d_authorized"])}, "stage_b_executed": False, "reason": "no independently qualifying retrospective route"}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "stdout.log").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "stderr.log").write_text("", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
