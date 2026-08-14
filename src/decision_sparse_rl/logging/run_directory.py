"""Immutable on-disk run records required by the project protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_RUN_FILES = (
    "config_resolved.yaml",
    "command.txt",
    "environment.txt",
    "git_state.txt",
    "stdout.log",
    "stderr.log",
    "metrics.json",
)


def create_run_directory(run_root: Path, run_id: str) -> Path:
    """Create a new run directory, refusing to reuse any prior run ID."""

    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError(f"run_id must be one non-empty path component: {run_id!r}")
    run_dir = run_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    return run_dir


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_run_record(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    command: str,
    environment: Mapping[str, Any],
    git_state: Mapping[str, Any],
    metrics: Mapping[str, Any],
    stdout: str = "",
    stderr: str = "",
) -> None:
    """Populate all mandatory files in a newly created run directory."""

    write_json(run_dir / "config_resolved.yaml", dict(config))
    (run_dir / "command.txt").write_text(command.rstrip() + "\n", encoding="utf-8")
    write_json(run_dir / "environment.txt", dict(environment))
    write_json(run_dir / "git_state.txt", dict(git_state))
    write_json(run_dir / "metrics.json", dict(metrics))
    (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
