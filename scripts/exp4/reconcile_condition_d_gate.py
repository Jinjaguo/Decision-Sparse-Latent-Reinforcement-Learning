#!/usr/bin/env python
"""Reconcile EXP2's hard-coded nine-demo label with EXP4's 21-demo D audit."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import traceback

import numpy as np
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    args = parser.parse_args()
    run_dir = create_run_directory(args.run_root, args.run_id)
    source = args.source_run.resolve(); stdout, stderr = io.StringIO(), io.StringIO()
    config = {"stage": "E4-2_condition_d_gate_reconciliation", "source_run": source.name, "reason": "EXP2 summarizer hard-coded exactly nine demos"}
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
            pairs_path = source / "artifacts/zero_twin_pairs.parquet"
            steps_path = source / "artifacts/zero_twin_comparisons.parquet"
            pairs = pq.read_table(pairs_path).to_pylist(); steps = pq.read_table(steps_path).to_pylist()
            demos = {(x["task"], x["episode"]) for x in pairs}
            branches = {(x["task"], x["episode"], int(x["branch_time"])) for x in pairs}
            d = metrics["gate"]["conditions"]["D_INTEGRATION_CONTROLLER_ROBOT"]
            criteria = {
                "source_completed": metrics["status"] == "completed",
                "condition_d_only": set(metrics["gate"]["conditions"]) == {"D_INTEGRATION_CONTROLLER_ROBOT"},
                "exactly_21_demos": len(demos) == 21,
                "exactly_252_branches": len(branches) == 252,
                "exactly_three_pairs_per_branch": len(pairs) == 756 and all(sum(1 for row in pairs if (row["task"], row["episode"], int(row["branch_time"])) == key) == 3 for key in branches),
                "expected_step_count": len(steps) == metrics["step_count"] == 39714,
                "all_finite": bool(d["criteria"]["all_integration_states_finite"]),
                "integration_median_at_most_1e-10": float(d["metrics"]["integration_l2"]["median"]) <= 1e-10,
                "integration_p95_at_most_1e-8": float(d["metrics"]["integration_l2"]["p95"]) <= 1e-8,
                "integration_max_at_most_1e-6": float(d["metrics"]["integration_l2"]["maximum"]) <= 1e-6,
                "terminal_object_pose_p95_at_most_1e-6": float(d["metrics"]["terminal_object_pose_l2"]["p95"]) <= 1e-6,
                "controller_eef_q_qvel_exact": all(float(d["metrics"][name]["maximum"]) == 0.0 for name in ("controller_l2_max_field", "eef_position_l2", "qpos_l2", "qvel_l2")),
                "final_success_agreement": bool(d["criteria"]["final_success_agreement"]),
                "all_strata_pass": bool(d["criteria"]["no_systematic_stratum_spikes"]),
            }
            gate = {"passed": all(criteria.values()), "criteria": criteria}
            result = {"schema_version": 1, "source_run": source.name, "source_metrics_sha256": sha256(source / "metrics.json"), "source_pairs_sha256": sha256(pairs_path), "source_steps_sha256": sha256(steps_path), "gate": gate, "counts": {"demos": len(demos), "branches": len(branches), "pairs": len(pairs), "steps": len(steps)}, "continuous_maxima": {name: float(value["maximum"]) for name, value in d["metrics"].items()}, "note": "Only the obsolete sample-count criterion was replaced; every numeric criterion is read unchanged from immutable source evidence."}
            write_json(run_dir / "artifacts/condition_d_reconciled_gate.json", result)
            write_json(run_dir / "artifacts/failure_examples.json", [] if gate["passed"] else [result])
            out_metrics = {"run_id": args.run_id, "status": "completed", **result}
            print(json.dumps(out_metrics, sort_keys=True))
        write_run_record(run_dir, config=config, command=shlex.join([sys.executable, *sys.argv]), environment={"python": sys.version, "numpy": np.__version__, "pyarrow": pq.__version__ if hasattr(pq, "__version__") else "module"}, git_state={"project": git_record(REPOSITORY_ROOT)}, stdout=stdout.getvalue(), stderr=stderr.getvalue(), metrics=out_metrics)
        return 0 if gate["passed"] else 2
    except Exception as exc:
        stderr.write(traceback.format_exc()); out_metrics = {"run_id": args.run_id, "status": "failed", "gate": {"passed": False}, "error": repr(exc)}
        write_run_record(run_dir, config=config, command=shlex.join([sys.executable, *sys.argv]), environment={"python": sys.version}, git_state={"project": git_record(REPOSITORY_ROOT)}, stdout=stdout.getvalue(), stderr=stderr.getvalue(), metrics=out_metrics)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
