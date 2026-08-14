#!/usr/bin/env python
"""Build outcome-blind EXP4 branch candidates for the corrected-D regression."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import shlex
import sys
import traceback

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.branch_times import QUANTILES, select_branch_times  # noqa: E402
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = create_run_directory(args.run_root, args.run_id)
    reference = args.reference_run.resolve()
    stdout, stderr = io.StringIO(), io.StringIO()
    config = {"stage": "E4-2_outcome_blind_branch_candidates", "reference_run": reference.name}
    git_state = {"project": git_record(REPOSITORY_ROOT)}
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            source_path = reference / "artifacts/reference_snapshots_manifest.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            if not source.get("gate", {}).get("passed"):
                raise RuntimeError("held-out reference gate did not pass")
            trajectories = []
            for episode in source["episodes"]:
                directory = reference / episode["relative_directory"]
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
                "source_reference_manifest": str(source_path),
                "selection_is_q_outcome_blind": True,
                "quantiles": list(QUANTILES),
                "rounding": "Python round(q * (T - 1)) converted to int",
                "duplicate_rule": "nearest unused valid action index; at equal distance choose the lower index first",
                "gripper_event_rule": "identical to EXP2/EXP3 branch_times.select_branch_times",
                "contact_event_rule": "identical to EXP2/EXP3 branch_times.select_branch_times",
                "trajectory_count": len(trajectories),
                "branch_count": sum(len(x["branches"]) for x in trajectories),
                "trajectories": trajectories,
            }
            criteria = {"exactly_21_trajectories": manifest["trajectory_count"] == 21, "exactly_12_unique_branches_each": all(len({x["action_index"] for x in t["branches"]}) == 12 for t in trajectories), "exactly_252_branches": manifest["branch_count"] == 252}
            gate = {"passed": all(criteria.values()), "criteria": criteria}
            write_json(run_dir / "artifacts/branch_candidates.json", manifest)
            write_json(run_dir / "artifacts/failure_examples.json", [] if gate["passed"] else trajectories)
            metrics = {"run_id": args.run_id, "status": "completed", "gate": gate, "trajectory_count": manifest["trajectory_count"], "branch_count": manifest["branch_count"]}
            print(json.dumps(metrics, sort_keys=True))
        write_run_record(run_dir, config=config, command=shlex.join([sys.executable, *sys.argv]), environment={"python": sys.version, "numpy": np.__version__}, git_state=git_state, stdout=stdout.getvalue(), stderr=stderr.getvalue(), metrics=metrics)
        return 0 if gate["passed"] else 2
    except Exception as exc:
        stderr.write(traceback.format_exc())
        metrics = {"run_id": args.run_id, "status": "failed", "gate": {"passed": False}, "error": repr(exc)}
        write_run_record(run_dir, config=config, command=shlex.join([sys.executable, *sys.argv]), environment={"python": sys.version, "numpy": np.__version__}, git_state=git_state, stdout=stdout.getvalue(), stderr=stderr.getvalue(), metrics=metrics)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
