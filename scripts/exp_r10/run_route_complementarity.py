"""EXP_R10 routewise complementarity and held-out route-selection audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def grouped_top1(rows: list[dict], score_key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["branch_id"]].append(row)
    selected = []
    for group in groups.values():
        selected.append(sorted(group, key=lambda row: (-float(row[score_key]), row["route"]))[0])
    return {
        "branch_count": len(selected),
        "utility_rate": float(np.mean([row["utility"] for row in selected])),
        "unsafe_rate": float(np.mean([row["unsafe"] for row in selected])),
        "route_frequency": {route: sum(row["route"] == route for row in selected) for route in sorted({row["route"] for row in selected})},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--r3-run", type=Path, default=Path("runs/exp_r3_s1_instrumented_candidate_matrix_20260816_r4"))
    parser.add_argument("--r6-run", type=Path, default=Path("runs/exp_r6_s1_factorized_ablations_20260816_r1"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    summaries = pq.read_table(ROOT / args.r3_run / "artifacts/candidate_summaries.parquet").to_pylist()
    predictions = pq.read_table(ROOT / args.r6_run / "artifacts/predictions.parquet").to_pylist()
    full = [row for row in predictions if row["variant"] == "factorized_full"]
    no_route = [row for row in predictions if row["variant"] == "factorized_no_route"]
    if len(summaries) != 540 or len(full) != 540 or len(no_route) != 540:
        raise RuntimeError("incomplete R3/R6 artifacts")
    profiles = {}
    for route in sorted({row["route"] for row in summaries}):
        subset = [row for row in summaries if row["route"] == route]
        profiles[route] = {
            "count": len(subset),
            "success_rate": float(np.mean([row["success"] for row in subset])),
            "unsafe_rate": float(np.mean([row["safety_stop"] for row in subset])),
            "utility_rate": float(np.mean([int(row["success"] and not row["safety_stop"]) for row in subset])),
            "by_task": {},
        }
        for task in sorted({row["task"] for row in subset}):
            task_rows = [row for row in subset if row["task"] == task]
            profiles[route]["by_task"][task] = {
                "count": len(task_rows),
                "success_rate": float(np.mean([row["success"] for row in task_rows])),
                "unsafe_rate": float(np.mean([row["safety_stop"] for row in task_rows])),
                "utility_rate": float(np.mean([int(row["success"] and not row["safety_stop"]) for row in task_rows])),
            }
    full_top1 = grouped_top1(full, "score")
    no_route_top1 = grouped_top1(no_route, "score")
    metrics = {"status": "completed", "experiment": "EXP_R10", "route_profiles": profiles, "heldout_selection": {"factorized_full": full_top1, "factorized_no_route": no_route_top1}}
    dump(artifacts / "protocol.json", {"experiment": "EXP_R10", "r3_run": str(args.r3_run), "r6_run": str(args.r6_run), "uses_heldout_outcomes_only_for_audit": True, "route_prior_fitted": False})
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
