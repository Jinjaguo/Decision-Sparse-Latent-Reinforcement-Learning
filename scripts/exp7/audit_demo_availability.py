#!/usr/bin/env python
"""Freeze the EXP7 independent-demo availability audit without running outcomes."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[2]
TASK_FILES = {
    "open_the_middle_drawer_of_the_cabinet": "open_the_middle_drawer_of_the_cabinet_demo.hdf5",
    "put_the_bowl_on_the_plate": "put_the_bowl_on_the_plate_demo.hdf5",
    "turn_on_the_stove": "turn_on_the_stove_demo.hdf5",
}
USED = {
    "open_the_middle_drawer_of_the_cabinet": list(range(21)),
    "put_the_bowl_on_the_plate": list(range(20)),
    "turn_on_the_stove": list(range(20)),
}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    rows = {}
    for task, filename in TASK_FILES.items():
        with h5py.File(ROOT / "data/libero_goal" / filename, "r") as handle:
            available = sorted(int(name.split("_")[-1]) for name in handle["data"])
        unused = [index for index in available if index not in USED[task]]
        rows[task] = {"available": available, "excluded_as_previously_used": USED[task], "unused_candidates_ascending": unused, "candidate_count": len(unused)}
    result = {"schema_version": 1, "independent": True, "selection_rule": "scan unused episode indices in strict ascending order independently by task; accept success+finite+exact corrected-D; preserve failures; stop each task after 10 accepted", "tasks": rows, "gate": {"passed": all(x["candidate_count"] >= 10 for x in rows.values())}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gate": result["gate"], "candidate_counts": {k:v["candidate_count"] for k,v in rows.items()}}, indent=2)); return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
