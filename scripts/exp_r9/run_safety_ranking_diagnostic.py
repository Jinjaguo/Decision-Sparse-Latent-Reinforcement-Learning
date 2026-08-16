"""EXP_R9 unsafe-candidate ranking diagnostic using held-out R5/R6 scores."""

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


def average_precision(rows: list[dict], score_key: str) -> float | None:
    ordered = sorted(rows, key=lambda row: (-float(row[score_key]), row["route"]))
    positives = sum(int(row["unsafe"]) for row in ordered)
    if positives == 0:
        return None
    seen = 0
    total = 0.0
    for index, row in enumerate(ordered, start=1):
        if row["unsafe"]:
            seen += 1
            total += seen / index
    return total / positives


def rank_stats(rows: list[dict], score_key: str) -> dict:
    ordered = sorted(rows, key=lambda row: (-float(row[score_key]), row["route"]))
    unsafe_positions = [index + 1 for index, row in enumerate(ordered) if row["unsafe"]]
    if not unsafe_positions:
        return {"unsafe_support": 0, "average_precision": None, "mean_unsafe_rank": None, "top1_unsafe_hit": None}
    return {
        "unsafe_support": len(unsafe_positions),
        "average_precision": average_precision(rows, score_key),
        "mean_unsafe_rank": float(np.mean(unsafe_positions)),
        "top1_unsafe_hit": int(bool(ordered[0]["unsafe"])),
    }


def aggregate(rows: list[dict], score_key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["branch_id"]].append(row)
    stats = [rank_stats(group, score_key) for group in groups.values()]
    supported = [item for item in stats if item["unsafe_support"]]
    return {
        "branch_count": len(groups),
        "unsafe_branch_support": len(supported),
        "mean_unsafe_average_precision": float(np.mean([x["average_precision"] for x in supported])) if supported else None,
        "mean_unsafe_rank": float(np.mean([x["mean_unsafe_rank"] for x in supported])) if supported else None,
        "top1_unsafe_hit_rate": float(np.mean([x["top1_unsafe_hit"] for x in supported])) if supported else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--r5-run", type=Path, default=Path("runs/exp_r5_s1_leave_one_demo_out_20260816_r2"))
    parser.add_argument("--r6-run", type=Path, default=Path("runs/exp_r6_s1_factorized_ablations_20260816_r1"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    r5 = pq.read_table(ROOT / args.r5_run / "artifacts/predictions.parquet").to_pylist()
    r6 = pq.read_table(ROOT / args.r6_run / "artifacts/predictions.parquet").to_pylist()
    factorized = [{**row, "unsafe_score": row["factorized_unsafe_probability"]} for row in r5]
    safety = [row for row in r6 if row["variant"] == "safety_only"]
    if len(factorized) != 540 or len(safety) != 540:
        raise RuntimeError("incomplete held-out probability artifacts")
    for left, right in zip(sorted(factorized, key=lambda x: (x["branch_id"], x["route"])), sorted(safety, key=lambda x: (x["branch_id"], x["route"]))):
        if (left["branch_id"], left["route"], left["unsafe"]) != (right["branch_id"], right["route"], right["unsafe"]):
            raise RuntimeError("R5/R6 unsafe label alignment mismatch")
    for row in safety:
        row["unsafe_score"] = 1.0 - float(row["score"])
    metrics = {
        "status": "completed",
        "experiment": "EXP_R9",
        "factorized_unsafe_head": aggregate(factorized, "unsafe_score"),
        "safety_only_head": aggregate(safety, "unsafe_score"),
        "by_task": {},
    }
    for task in sorted({row["task"] for row in factorized}):
        metrics["by_task"][task] = {
            "factorized_unsafe_head": aggregate([row for row in factorized if row["task"] == task], "unsafe_score"),
            "safety_only_head": aggregate([row for row in safety if row["task"] == task], "unsafe_score"),
        }
    dump(artifacts / "protocol.json", {"experiment": "EXP_R9", "r5_run": str(args.r5_run), "r6_run": str(args.r6_run), "uses_heldout_outcomes_only_for_audit": True, "threshold_tuning": False})
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
