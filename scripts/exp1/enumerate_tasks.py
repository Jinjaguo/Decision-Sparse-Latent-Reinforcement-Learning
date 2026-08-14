#!/usr/bin/env python
"""Enumerate EXP1 tasks from the exact checked-out LIBERO benchmark API."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import shlex
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.libero_source import (  # noqa: E402
    enumerate_registered_benchmarks,
    import_benchmark_from_source,
    write_libero_config,
)
from decision_sparse_rl.logging.run_directory import (  # noqa: E402
    create_run_directory,
    write_json,
    write_run_record,
)
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party" / "LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "experiments"
        / "exp1_decision_sparsity"
        / "manifests"
        / "tasks.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite task manifest: {args.manifest}")
    run_dir = create_run_directory(args.run_root, args.run_id)
    config_directory = run_dir / "artifacts" / "libero_config"
    config_file = write_libero_config(config_directory, args.libero_root, args.dataset_root)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        benchmark = import_benchmark_from_source(args.libero_root, config_directory)
        suites, suite_errors = enumerate_registered_benchmarks(benchmark)
    task_count = sum(suite["task_count"] for suite in suites.values())
    manifest = {
        "source": {
            "benchmark_module": str(Path(benchmark.__file__).resolve()),
            "libero_git": git_record(args.libero_root),
            "libero_config": str(config_file.resolve()),
        },
        "suite_count": len(suites),
        "task_count": task_count,
        "suites": suites,
        "registered_suite_errors": suite_errors,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.manifest, manifest)
    write_json(run_dir / "artifacts" / "tasks.json", manifest)
    summary = {
        "run_id": args.run_id,
        "suite_count": len(suites),
        "task_count": task_count,
        "registered_suite_error_count": len(suite_errors),
        "manifest": str(args.manifest.resolve()),
    }
    stdout = captured.getvalue() + json.dumps(summary, indent=2, sort_keys=True) + "\n"
    project_git = git_record(REPOSITORY_ROOT)
    libero_git = git_record(args.libero_root)
    write_run_record(
        run_dir,
        config={
            "run_id": args.run_id,
            "stage": "E1_task_enumeration",
            "libero_root": str(args.libero_root.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "manifest": str(args.manifest.resolve()),
        },
        command=shlex.join([sys.executable, *sys.argv]),
        environment={"python": sys.version, "executable": sys.executable},
        git_state={"project": project_git, "libero": libero_git},
        metrics=summary,
        stdout=stdout,
        stderr=json.dumps(suite_errors, indent=2, sort_keys=True) + "\n" if suite_errors else "",
    )
    print(stdout, end="")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
