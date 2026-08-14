#!/usr/bin/env python
"""Freeze EXP2 R3 branch times from an R2 local-reference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.branch_times import QUANTILES, select_branch_times  # noqa: E402
from decision_sparse_rl.logging.run_directory import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/branch_times_reconciliation.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = args.reference_run.resolve()
    source_manifest_path = run / "artifacts/reference_snapshots_manifest.json"
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not source["gate"]["passed"]:
        raise RuntimeError("reference run did not pass R2 gate")
    trajectories = []
    for episode in source["episodes"]:
        directory = run / episode["relative_directory"]
        with np.load(directory / "trajectory_states.npz", allow_pickle=False) as archive:
            actions = np.asarray(archive["actions"], dtype=np.float64)
        boundaries = json.loads((directory / "boundaries.json").read_text(encoding="utf-8"))
        counts = [int(item["contact_count"]) for item in boundaries]
        choices = select_branch_times(actions, counts)
        for choice in choices:
            index = int(choice["action_index"])
            choice["contact_count"] = counts[index]
            choice["gripper_command"] = float(actions[index, -1])
            choice["normalized_time"] = index / max(len(actions) - 1, 1)
        trajectories.append({"task": episode["task"], "suite": episode["suite"], "task_id": episode["task_id"], "episode": episode["episode"], "trajectory_length": len(actions), "branches": choices})
    manifest = {
        "schema_version": 1,
        "source_reference_run_id": source["run_id"],
        "source_reference_manifest": str(source_manifest_path),
        "selection_is_q_outcome_blind": True,
        "quantiles": list(QUANTILES),
        "rounding": "Python round(q * (T - 1)) converted to int",
        "duplicate_rule": "nearest unused valid action index; at equal distance choose the lower index first",
        "gripper_event_rule": "first adjacent pair of nonzero last-action-component signs that differ",
        "contact_event_rule": "first argmax of absolute adjacent raw contact-count change; event boundary is the later index",
        "trajectory_count": len(trajectories),
        "branch_count": sum(len(item["branches"]) for item in trajectories),
        "trajectories": trajectories,
    }
    if manifest["trajectory_count"] != 9 or manifest["branch_count"] != 108:
        raise RuntimeError("frozen branch coverage is incomplete")
    write_json(args.output, manifest)
    print(json.dumps({"output": str(args.output.resolve()), "trajectory_count": 9, "branch_count": 108}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

