"""Complete EXP10 Track-D endpoint OR gate with locked cross-fit predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.exp10.run_stage_a_routes import demo_bootstrap, make_features, ridge, sha256


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--dataset-run", default="runs/exp10_a0_phase_macro_dataset_r2_20260814"); parser.add_argument("--model-run", default="runs/exp10_a1_a6_multiroute_models_r1_20260814")
    args = parser.parse_args(); out = Path("runs") / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"; artifacts.mkdir(parents=True)
    rows = pq.read_table(Path(args.dataset_run) / "artifacts" / "macro_trajectory_dataset.parquet").to_pylist(); xmap = make_features(rows)
    task = np.asarray([r["task"] for r in rows]); episode = np.asarray([r["episode"] for r in rows]); fold = np.asarray([r["fold"] for r in rows]); role = np.asarray([r["direction_role"] for r in rows]); target = [np.asarray(r["endpoint_response"]) for r in rows]
    predictions = []; errors = {"exp9_temporal": np.full(len(rows), np.nan), "D_terminal_direct": np.full(len(rows), np.nan)}
    for current_task in sorted(set(task)):
        ti = np.flatnonzero(task == current_task); y = np.vstack([target[i] for i in ti]); ff, rr = fold[ti], role[ti]
        for current_fold in range(5):
            tr = np.flatnonzero((ff != current_fold) & (rr == "basis")); te = np.flatnonzero(ff == current_fold)
            for model, xkey in (("exp9_temporal", "exp9_temporal"), ("D_terminal_direct", "B_hybrid_phase")):
                _, pred = ridge(xmap[xkey][ti][tr], y[tr], xmap[xkey][ti][te])
                err = np.linalg.norm(pred - y[te], axis=1); errors[model][ti[te]] = err
                for local, gi in enumerate(ti[te]): predictions.append({"model": model, "intervention_id": rows[gi]["intervention_id"], "task": task[gi], "episode": episode[gi], "fold": int(fold[gi]), "endpoint_prediction": pred[local].tolist(), "endpoint_l2": float(err[local])})
    path = artifacts / "endpoint_predictions.parquet"; pq.write_table(pa.Table.from_pylist(predictions), path, compression="zstd")
    lock = {"sha256": sha256(path), "rows": len(predictions), "locked_before_gate": True}; (artifacts / "prediction_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    base, direct = errors["exp9_temporal"], errors["D_terminal_direct"]
    improvement = float((base.mean() - direct.mean()) / base.mean()); delta, ci = demo_bootstrap(direct, base, task, episode, seed=10104)
    event_metrics = pq.read_table(Path(args.model_run) / "artifacts" / "event_terminal_metrics.parquet").to_pylist(); classification = next(r for r in event_metrics if r["model"] == "D_terminal_direct")
    classification_gate = classification["auroc"] >= .80 and classification["auroc_demo_bootstrap_ci"][0] >= .70
    endpoint_gate = improvement >= .15 and ci[0] > 0
    result = {"status": "completed", "run_id": args.run_id, "prediction_lock": lock, "mean_endpoint_l2": {"exp9_temporal": float(base.mean()), "D_terminal_direct": float(direct.mean())}, "p90_endpoint_l2": {"exp9_temporal": float(np.quantile(base, .9)), "D_terminal_direct": float(np.quantile(direct, .9))}, "relative_improvement": improvement, "absolute_demo_improvement": delta, "demo_bootstrap_ci": ci, "classification_gate": bool(classification_gate), "endpoint_gate": bool(endpoint_gate), "track_d_authorized": bool(classification_gate or endpoint_gate)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "config.json").write_text(json.dumps({"command": " ".join(sys.argv), "stage": "EXP10 Track D supplemental frozen OR gate"}, indent=2), encoding="utf-8"); (out / "stdout.log").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "stderr.log").write_text("", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
