"""Audit required EXP12 locks, row counts, hashes, plots, and decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.metrics.exp12 import sha256_file


REQUIRED_ARTIFACTS = (
    "consequence_predictions.parquet", "pairwise_ranking_predictions.parquet",
    "setwise_ranking_predictions.parquet", "richer_future_predictions.parquet",
    "ranking_metrics.json", "tail_metrics.json", "ablation_results.parquet",
    "encoding_results.parquet", "scientific_decision.json", "failure_examples.json",
    "prediction_hash_manifest.json",
)
REQUIRED_PLOTS = (
    "candidate_set_size_distribution.png", "oracle_improvement_opportunity.png",
    "pairwise_accuracy_by_task.png", "top1_accuracy_by_task.png", "regret_distribution.png",
    "catastrophic_selection_rate.png", "specialist_ablation.png", "encoding_ablation.png",
    "loss_ablation.png", "combination_ablation.png", "uncertainty_abstention_curve.png",
    "consequence_vs_richer_future.png", "failure_examples_summary.png",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--analysis-run", type=Path, default=Path("runs/exp12_a1_ranking_models_r1_20260815"))
    parser.add_argument("--dataset-run", type=Path, default=Path("runs/exp12_a0_candidate_dataset_r1_20260815"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable audit run exists: {out}")
    out.mkdir(parents=True)
    analysis = ROOT / args.analysis_run
    dataset = ROOT / args.dataset_run
    checks = []
    for name in REQUIRED_ARTIFACTS:
        path = analysis / "artifacts" / name
        checks.append({"check": f"artifact:{name}", "passed": path.is_file() and path.stat().st_size > 0, "size": path.stat().st_size if path.exists() else 0})
    for name in REQUIRED_PLOTS:
        path = analysis / "plots" / name
        checks.append({"check": f"plot:{name}", "passed": path.is_file() and path.stat().st_size > 2000, "size": path.stat().st_size if path.exists() else 0})
    lock = json.loads((analysis / "artifacts/prediction_hash_manifest.json").read_text())
    for name, digest in lock.items():
        path = analysis / "artifacts" / name
        checks.append({"check": f"hash:{name}", "passed": path.is_file() and sha256_file(path) == digest})
    row_expectations = {
        "consequence_predictions.parquet": 1036,
        "setwise_ranking_predictions.parquet": 84 * 18,
        "richer_future_predictions.parquet": 1036,
    }
    for name, expected in row_expectations.items():
        actual = pq.read_metadata(analysis / "artifacts" / name).num_rows
        checks.append({"check": f"rows:{name}", "passed": actual == expected, "actual": actual, "expected": expected})
    candidate_rows = pq.read_metadata(dataset / "artifacts/candidate_dataset.parquet").num_rows
    opportunity_rows = pq.read_metadata(dataset / "artifacts/candidate_opportunity.parquet").num_rows
    checks.extend([
        {"check": "candidate_dataset_rows", "passed": candidate_rows == 1316, "actual": candidate_rows},
        {"check": "candidate_groups", "passed": opportunity_rows == 84, "actual": opportunity_rows},
        {"check": "prediction_lock_verified_by_analysis", "passed": bool(json.loads((analysis / "metrics.json").read_text())["prediction_hashes_verified"])},
        {"check": "scientific_decision_has_axes", "passed": len(json.loads((analysis / "artifacts/scientific_decision.json").read_text())["axes"]) == 8},
    ])
    metrics = {"status": "completed", "source_analysis": analysis.name, "source_dataset": dataset.name, "passed": all(x["passed"] for x in checks), "checks": checks}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
