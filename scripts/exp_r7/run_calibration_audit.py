"""EXP_R7 calibration and fixed-threshold audit of held-out R5 probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def brier(probability: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probability - labels) ** 2))


def ece(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probability >= lower) & (probability < upper if index < bins - 1 else probability <= upper)
        if np.any(mask):
            total += float(mask.mean()) * abs(float(probability[mask].mean()) - float(labels[mask].mean()))
    return total


def threshold_metrics(probability: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    predicted = probability >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum((~predicted) & (labels == 1)))
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "support": int(labels.sum()),
    }


def audit(rows: list[dict], probability_key: str, label_key: str) -> dict:
    probability = np.asarray([float(row[probability_key]) for row in rows], dtype=np.float64)
    labels = np.asarray([int(row[label_key]) for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "positive_support": int(labels.sum()),
        "brier": brier(probability, labels),
        "ece_10_bins": ece(probability, labels),
        "threshold_0_5": threshold_metrics(probability, labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-run", type=Path, default=Path("runs/exp_r5_s1_leave_one_demo_out_20260816_r2"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    rows = pq.read_table(ROOT / args.input_run / "artifacts/predictions.parquet").to_pylist()
    required = {"factorized_success_probability", "factorized_unsafe_probability", "factorized_score", "scalar_score", "success", "unsafe", "utility"}
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    if missing or len(rows) != 540:
        raise RuntimeError(f"R5 probability artifact is incomplete: missing={missing}, rows={len(rows)}")
    results = {
        "factorized_success": audit(rows, "factorized_success_probability", "success"),
        "factorized_unsafe": audit(rows, "factorized_unsafe_probability", "unsafe"),
        "factorized_utility": audit(rows, "factorized_score", "utility"),
        "scalar_utility": audit(rows, "scalar_score", "utility"),
    }
    by_task = {}
    for task in sorted({row["task"] for row in rows}):
        subset = [row for row in rows if row["task"] == task]
        by_task[task] = {"factorized_unsafe": audit(subset, "factorized_unsafe_probability", "unsafe"), "factorized_utility": audit(subset, "factorized_score", "utility")}
    metrics = {"status": "completed", "experiment": "EXP_R7", "input_rows": len(rows), "fixed_threshold": 0.5, "overall": results, "by_task": by_task}
    dump(artifacts / "protocol.json", {"experiment": "EXP_R7", "input_run": str(args.input_run), "threshold": 0.5, "threshold_selected_from_heldout": False, "untouched_confirmation_opened": False})
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
