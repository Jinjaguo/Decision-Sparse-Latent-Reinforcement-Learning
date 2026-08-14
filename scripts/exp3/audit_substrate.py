#!/usr/bin/env python
"""Create the EXP3 T0 substrate-audit run without simulator outcomes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pyarrow as pa

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--condition-d-run", type=Path, required=True)
    args = parser.parse_args()
    run_dir = create_run_directory(REPOSITORY_ROOT / "runs", args.run_id)
    reference = args.reference_run.resolve()
    condition_d = args.condition_d_run.resolve()
    files = {
        "runtime_state_schema": REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/runtime_state_schema.json",
        "controller_state_schema": REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/controller_state_schema.json",
        "policy_step_boundary": REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/policy_step_boundary.json",
        "branch_manifest": REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/branch_times_reconciliation.json",
        "reference_manifest": reference / "artifacts/reference_snapshots_manifest.json",
        "condition_d_metrics": condition_d / "metrics.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    hashes = {name: sha256(path) for name, path in files.items() if path.is_file()}
    ref_manifest = json.loads(files["reference_manifest"].read_text(encoding="utf-8"))
    d_metrics = json.loads(files["condition_d_metrics"].read_text(encoding="utf-8"))
    branch = json.loads(files["branch_manifest"].read_text(encoding="utf-8"))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    criteria = {
        "no_missing_inputs": not missing,
        "reference_gate_passed": bool(ref_manifest.get("gate", {}).get("passed")),
        "nine_reference_episodes": len(ref_manifest.get("episodes", [])) == 9,
        "condition_d_gate_passed": bool(d_metrics.get("gate", {}).get("passed")),
        "condition_d_selected": d_metrics.get("gate", {}).get("selected_condition") == "D_INTEGRATION_CONTROLLER_ROBOT",
        "nine_branch_trajectories": len(branch.get("trajectories", [])) == 9,
        "twelve_unique_branches_each": all(len(row.get("branches", [])) == 12 and len({x["action_index"] for x in row["branches"]}) == 12 for row in branch.get("trajectories", [])),
    }
    metrics = {"run_id": args.run_id, "status": "completed", "gate": {"passed": all(criteria.values()), "criteria": criteria}, "hashes": hashes, "gpu_detected": gpu.returncode == 0, "gpu": gpu.stdout.strip(), "missing": missing}
    write_json(run_dir / "artifacts/input_hashes.json", hashes)
    write_run_record(
        run_dir,
        config={"stage": "T0_substrate_audit", "reference_run": str(reference), "condition_d_run": str(condition_d)},
        command=shlex.join([sys.executable, *sys.argv]),
        environment={"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "pyarrow": pa.__version__, "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite"), "gpu": gpu.stdout.strip()},
        git_state={"project": git_record(REPOSITORY_ROOT), "libero": git_record(REPOSITORY_ROOT / "third_party/LIBERO"), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")},
        metrics=metrics,
        stdout=json.dumps(metrics, indent=2, sort_keys=True) + "\n",
    )
    return 0 if metrics["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
