#!/usr/bin/env python
"""Audit unused EXP8 demonstrations without running q outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[2]
TASK_FILES = {
    "open_the_middle_drawer_of_the_cabinet": "open_the_middle_drawer_of_the_cabinet_demo.hdf5",
    "put_the_bowl_on_the_plate": "put_the_bowl_on_the_plate_demo.hdf5",
    "turn_on_the_stove": "turn_on_the_stove_demo.hdf5",
}
USED = {
    "open_the_middle_drawer_of_the_cabinet": list(range(31)),
    "put_the_bowl_on_the_plate": list(range(30)),
    "turn_on_the_stove": list(range(30)),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {}
    for task, filename in TASK_FILES.items():
        with h5py.File(ROOT / "data/libero_goal" / filename, "r") as handle:
            available = sorted(int(name.rsplit("_", 1)[-1]) for name in handle["data"])
        unused = [episode for episode in available if episode not in USED[task]]
        tasks[task] = {
            "available": available,
            "excluded_as_previously_used": USED[task],
            "unused_candidates_ascending": unused,
            "candidate_count": len(unused),
        }
    result = {
        "schema_version": 1,
        "experiment": "EXP8",
        "selection_rule": "scan unused demos in ascending order; accept only successful, finite, exact corrected-D references; preserve rejects; stop after 10 qualified per task",
        "tasks": tasks,
        "gate": {"passed": all(row["candidate_count"] >= 10 for row in tasks.values())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gate": result["gate"], "candidate_counts": {k: v["candidate_count"] for k, v in tasks.items()}}, indent=2))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

