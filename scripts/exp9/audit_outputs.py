"""Independent completeness and immutability audit for EXP9 Stage A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from decision_sparse_rl.metrics.exp9 import ece, rank_auc


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def average_precision(y, p):
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    return float(np.sum((np.cumsum(ys) / np.arange(1, len(ys) + 1)) * ys) / max(1, ys.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--model-run", required=True)
    parser.add_argument("--output-run", required=True)
    args = parser.parse_args()
    target, model, out = map(Path, (args.target_run, args.model_run, args.output_run))
    if out.exists():
        raise FileExistsError(out)
    artifacts = out / "artifacts"; artifacts.mkdir(parents=True)

    audit = json.loads((target / "artifacts" / "retrospective_target_audit.json").read_text(encoding="utf-8"))
    source_hashes_valid = all(sha256(item["path"]) == item["sha256"] for item in audit["sources"].values())
    lock = json.loads((model / "artifacts" / "prediction_lock.json").read_text(encoding="utf-8"))
    prediction_path = model / "artifacts" / "retrospective_predictions.parquet"
    prediction_hash_valid = sha256(prediction_path) == lock["sha256"]
    table = pq.read_table(prediction_path)
    rows = table.to_pylist()
    keys = {(r["model"], r["intervention_id"], r["horizon"]) for r in rows}
    expected_models = {"baseline_b", "factorized", "lifted_hybrid", "graph_mixture", "temporal_sequence_mixture"}
    expected_plots = {
        "event_prevalence_by_task_horizon.png", "specificity_sensitivity_tradeoff.png",
        "energy_score_by_model_horizon.png", "predictive_interval_coverage.png",
        "stageA_model_selection.png", "seed_variance.png",
    }
    selection = json.loads((model / "artifacts" / "architecture_selection.json").read_text(encoding="utf-8"))
    task_metrics = []
    for model_name in ("graph_mixture", "temporal_sequence_mixture"):
        for task in sorted({r["task"] for r in rows}):
            for horizon in ("1", "3", "5"):
                subset = [r for r in rows if r["model"] == model_name and r["task"] == task and r["horizon"] == horizon]
                y = np.asarray([r["event_label"] for r in subset]); p = np.asarray([r["event_probability"] for r in subset]); pred = np.asarray([p[i] >= subset[i]["threshold"] for i in range(len(subset))])
                tp, tn = np.sum((y == 1) & pred), np.sum((y == 0) & ~pred)
                fp, fn = np.sum((y == 0) & pred), np.sum((y == 1) & ~pred)
                by_demo = {}
                for r in subset: by_demo.setdefault(r["episode"], []).append(r["energy_score"])
                task_metrics.append({"model": model_name, "task": task, "horizon": horizon, "n": len(subset), "auroc": rank_auc(y, p), "auprc": average_precision(y, p), "ece": ece(y, p), "sensitivity": float(tp / max(1, tp + fn)), "specificity": float(tn / max(1, tn + fp)), "false_safe_rate": float(fp / max(1, tn + fp)), "demo_mean_energy_score": float(np.mean([np.mean(v) for v in by_demo.values()])), "coverage_90": float(np.mean([r["coverage_90"] for r in subset]))})
    pq.write_table(pa.Table.from_pylist(task_metrics), artifacts / "retrospective_task_metrics.parquet")

    criteria = {
        "target_gate_passed": audit["gate"]["passed"],
        "source_hashes_valid": source_hashes_valid,
        "prediction_hash_valid": prediction_hash_valid,
        "prediction_rows_exact": table.num_rows == 259200 == lock["rows"],
        "prediction_keys_unique": len(keys) == 259200,
        "model_set_exact": {r["model"] for r in rows} == expected_models,
        "folds_exact": {int(r["fold"]) for r in rows} == set(range(5)),
        "horizons_exact": {r["horizon"] for r in rows} == {"1", "3", "5"},
        "plots_complete_nonempty": all((model / "plots" / p).stat().st_size > 1000 for p in expected_plots),
        "stage_b_not_authorized": selection["new_cohort_authorized"] is False,
        "stage_b_runs_absent": not any(Path("runs").glob("exp9_b*")),
    }
    result = {"status": "completed", "gate": {"passed": all(criteria.values()), "criteria": criteria}, "model_run": str(model), "target_run": str(target), "prediction_sha256": lock["sha256"], "classification": "action_conditioning_insufficient", "stage_b_executed": False, "task_metric_rows": len(task_metrics)}
    (artifacts / "output_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
