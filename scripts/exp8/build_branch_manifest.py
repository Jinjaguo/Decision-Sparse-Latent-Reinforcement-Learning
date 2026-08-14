#!/usr/bin/env python
"""Build 12 deterministic outcome-blind EXP8 branches per demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _quantile_bins(values, bins=4):
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return np.searchsorted(edges[1:-1], values, side="right")


def choose(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda row: int(row["action_index"]))
    gap_bin = _quantile_bins([row["minimum_signed_gap_m"] for row in rows])
    progress_bin = _quantile_bins([row["physical_progress_clipped"] for row in rows])
    enriched = []
    for index, row in enumerate(rows):
        enriched.append({**row, "signed_gap_quantile": int(gap_bin[index]), "progress_quantile": int(progress_bin[index])})
    targets = np.linspace(0, len(rows) - 1, 12)
    strata = {}
    for row in enriched:
        key = (
            row["exact_mode_json"], row["signed_gap_quantile"], row["gripper_state"], bool(row["predicate"]),
            min(int(row["active_pair_count"]), 3), row["progress_quantile"],
        )
        strata.setdefault(key, []).append(row)
    chosen, used = [], set()
    for key, members in sorted(strata.items(), key=lambda item: (len(item[1]), str(item[0]))):
        if len(chosen) >= 12:
            break
        candidate = min(members, key=lambda row: (min(abs(row["action_index"] - target) for target in targets), row["action_index"]))
        if candidate["action_index"] not in used:
            chosen.append(candidate)
            used.add(candidate["action_index"])
    for target in targets:
        if len(chosen) >= 12:
            break
        candidate = min((row for row in enriched if row["action_index"] not in used), key=lambda row: (abs(row["action_index"] - target), row["action_index"]))
        chosen.append(candidate)
        used.add(candidate["action_index"])
    return sorted(chosen, key=lambda row: row["action_index"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = pq.read_table(args.contact_frames).to_pylist()
    trajectories = []
    for task, episode in sorted({(row["task"], row["episode"]) for row in rows}):
        population = [row for row in rows if row["task"] == task and row["episode"] == episode]
        selected = choose(population)
        branches = []
        for row in selected:
            branches.append({
                "kind": "balanced_time_progress_mode_gap_gripper_predicate_support",
                "action_index": int(row["action_index"]), "branch_time": int(row["action_index"]),
                "normalized_time": float(row["normalized_time"]), "physical_progress_clipped": float(row["physical_progress_clipped"]),
                "reference_contact_mode_json": row["exact_mode_json"], "reference_signed_gap_m": float(row["minimum_signed_gap_m"]),
                "reference_gripper_state": row["gripper_state"], "reference_predicate_state": bool(row["predicate"]),
                "active_pair_count": int(row["active_pair_count"]), "valid_pair_count": int(row["valid_pair_count"]),
            })
        trajectories.append({"task": task, "episode": episode, "trajectory_length": int(population[0]["trajectory_length"]), "branches": branches})
    result = {
        "schema_version": 1,
        "selection_rule": "rare outcome-blind joint strata over exact mode, continuous gap quantile, gripper, predicate, active-pair support and progress; then nearest unused to uniform time targets",
        "nearest_unused_replacement": "deterministic absolute action-index distance, then smaller action index; applied only for frozen joint-limit admissibility",
        "trajectory_count": len(trajectories), "branch_count": sum(len(row["branches"]) for row in trajectories),
        "trajectories": trajectories,
        "gate": {"passed": len(trajectories) == 30 and all(len(row["branches"]) == 12 for row in trajectories)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"trajectories": len(trajectories), "branches": result["branch_count"], "gate": result["gate"]}, indent=2))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
