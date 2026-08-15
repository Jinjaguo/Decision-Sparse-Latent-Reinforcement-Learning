"""Re-fit EXP10 CVAE members separately to quantify frozen-seed variance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from decision_sparse_rl.metrics.exp9 import energy_score
from scripts.exp10.run_stage_a_routes import SEEDS, cvae_fit_predict, make_features, sha256


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--dataset-run", default="runs/exp10_a0_phase_macro_dataset_r2_20260814")
    args = parser.parse_args(); out = Path("runs") / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"; artifacts.mkdir(parents=True)
    rows = pq.read_table(Path(args.dataset_run) / "artifacts" / "macro_trajectory_dataset.parquet").to_pylist(); x = make_features(rows)["F_cvae16"]
    task = np.asarray([r["task"] for r in rows]); episode = np.asarray([r["episode"] for r in rows]); fold = np.asarray([r["fold"] for r in rows]); role = np.asarray([r["direction_role"] for r in rows]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = []
    for horizon in ("h5", "h10"):
        target = [np.asarray(r[f"response_{horizon}"]) for r in rows]
        for seed in SEEDS:
            scores = np.full(len(rows), np.nan)
            for current_task in sorted(set(task)):
                ti = np.flatnonzero(task == current_task); y = np.vstack([target[i] for i in ti]); ff, rr = fold[ti], role[ti]
                for current_fold in range(5):
                    tr = np.flatnonzero((ff != current_fold) & (rr == "basis")); te = np.flatnonzero(ff == current_fold)
                    _, _, samples, _ = cvae_fit_predict(x[ti][tr], y[tr], x[ti][te], seed + current_fold, device)
                    scores[ti[te]] = energy_score(samples, y[te])
            metrics.append({"model": "F_cvae16", "seed": seed, "horizon": horizon.upper(), "mean_energy": float(scores.mean()), "all_rows_scored": bool(np.isfinite(scores).all())})
            print(metrics[-1], flush=True)
    path = artifacts / "seed_metrics.parquet"; pq.write_table(pa.Table.from_pylist(metrics), path, compression="zstd")
    summary = {h: {"mean": float(np.mean([r["mean_energy"] for r in metrics if r["horizon"] == h])), "std": float(np.std([r["mean_energy"] for r in metrics if r["horizon"] == h], ddof=1)), "range": [float(min(r["mean_energy"] for r in metrics if r["horizon"] == h)), float(max(r["mean_energy"] for r in metrics if r["horizon"] == h))]} for h in ("H5", "H10")}
    result = {"status": "completed", "run_id": args.run_id, "device": str(device), "seeds": SEEDS, "seed_variance": summary, "prediction_metric_hash": sha256(path)}
    (artifacts / "seed_variance.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "config.json").write_text(json.dumps({"command": " ".join(sys.argv), "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}, indent=2), encoding="utf-8"); (out / "stdout.log").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "stderr.log").write_text("", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
