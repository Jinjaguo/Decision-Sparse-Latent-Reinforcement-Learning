#!/usr/bin/env python
"""Audit all HDF5 files named by the frozen EXP1 pilot selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.logging.run_directory import (  # noqa: E402
    create_run_directory,
    write_json,
    write_run_record,
)
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402
from decision_sparse_rl.utils.hdf5_audit import audit_hdf5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "exp1_decision_sparsity" / "manifests" / "selected_tasks_pilot.json",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "exp1_decision_sparsity" / "manifests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = create_run_directory(args.run_root, args.run_id)
    command = shlex.join([sys.executable, *sys.argv])
    config = {
        "run_id": args.run_id,
        "stage": "E2_dataset_schema_audit",
        "selection": str(args.selection.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "manifest_dir": str(args.manifest_dir.resolve()),
    }
    environment = {"python": sys.version, "executable": sys.executable}
    git_state = {"project": git_record(REPOSITORY_ROOT)}
    written_manifests = []
    try:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        audits = []
        for task in selection["tasks"]:
            dataset_path = args.dataset_root / task["demonstration_relative_path"]
            if not dataset_path.is_file():
                raise FileNotFoundError(f"selected dataset does not exist: {dataset_path}")
            output_path = args.manifest_dir / f"dataset_schema_{task['name']}.json"
            if output_path.exists():
                raise FileExistsError(f"refusing to overwrite schema manifest: {output_path}")
            audit = audit_hdf5(dataset_path)
            audit["task"] = task
            write_json(output_path, audit)
            write_json(run_dir / "artifacts" / output_path.name, audit)
            written_manifests.append(str(output_path.resolve()))
            audits.append(audit)
        metrics = {
            "run_id": args.run_id,
            "status": "completed",
            "task_count": len(audits),
            "episode_counts": {
                audit["task"]["name"]: audit["episode_detection"]["episode_count"]
                for audit in audits
            },
            "manifests": written_manifests,
        }
        stdout = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        stderr = ""
    except Exception as exc:
        stderr = traceback.format_exc()
        metrics = {
            "run_id": args.run_id,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "manifests_written_before_failure": written_manifests,
        }
        stdout = ""
    write_run_record(
        run_dir,
        config=config,
        command=command,
        environment=environment,
        git_state=git_state,
        metrics=metrics,
        stdout=stdout,
        stderr=stderr,
    )
    if metrics["status"] == "failed":
        print(stderr, file=sys.stderr, end="")
        print(f"run_dir={run_dir}", file=sys.stderr)
        return 1
    print(stdout, end="")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
