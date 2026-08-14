#!/usr/bin/env python
"""Create an immutable EXP1 Stage E0 audit for the active Python environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.logging.run_directory import (  # noqa: E402
    create_run_directory,
    write_run_record,
)
from decision_sparse_rl.utils.environment_audit import build_environment_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party" / "LIBERO")
    parser.add_argument(
        "--robosuite-source-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party" / "robosuite-src",
    )
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_environment_audit(
        project_root=REPOSITORY_ROOT,
        libero_root=args.libero_root,
        robosuite_source_root=args.robosuite_source_root,
        dataset_root=args.dataset_root,
    )
    run_dir = create_run_directory(args.run_root, args.run_id)
    command = shlex.join([sys.executable, *sys.argv])
    config = {
        "run_id": args.run_id,
        "stage": "E0_environment_audit",
        "libero_root": str(args.libero_root.resolve()),
        "robosuite_source_root": str(args.robosuite_source_root.resolve()),
        "dataset_root": str(args.dataset_root.resolve()) if args.dataset_root else None,
    }
    summary = {
        "run_id": args.run_id,
        "robosuite_matches_pin": audit["requirements"]["robosuite_matches_pin"],
        "dataset_root_selected": audit["dataset_root"]["selected"],
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    write_run_record(
        run_dir,
        config=config,
        command=command,
        environment=audit,
        git_state={
            key: audit[key]
            for key in ("project_git", "libero_git", "robosuite_source_git")
        },
        metrics=summary,
        stdout=rendered,
    )
    print(rendered, end="")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
