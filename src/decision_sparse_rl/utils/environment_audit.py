"""Read-only environment and source provenance inspection for EXP1 Stage E0."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, Optional


AUDITED_DISTRIBUTIONS = (
    "libero",
    "robosuite",
    "mujoco",
    "mujoco-py",
    "numpy",
    "h5py",
    "torch",
)


def distribution_record(name: str) -> Dict[str, Any]:
    """Return package version and installed location without importing it."""

    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None, "location": None}
    return {
        "installed": True,
        "version": dist.version,
        "location": str(Path(dist.locate_file("")).resolve()),
    }


def parse_pinned_requirement(requirements_file: Path, package: str) -> Optional[str]:
    """Read an exact ``name==version`` pin from the checked-out requirements file."""

    pattern = re.compile(rf"^\s*{re.escape(package)}\s*==\s*([^\s#]+)", re.IGNORECASE)
    for line in requirements_file.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def git_record(repository: Path) -> Dict[str, Any]:
    """Capture SHA and dirty status while scoping Git's ownership exception locally."""

    absolute = repository.resolve()

    def run(*args: str) -> str:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={absolute}",
                "-C",
                str(absolute),
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        status = run("status", "--short")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        return {
            "path": str(absolute),
            "sha": sha,
            "branch": branch,
            "dirty": bool(status),
            "status": status.splitlines(),
            "error": None,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "path": str(absolute),
            "sha": None,
            "branch": None,
            "dirty": None,
            "status": [],
            "error": str(exc),
        }


def command_output(command: Iterable[str]) -> Dict[str, Any]:
    """Run a read-only system query and preserve its exit status and output."""

    try:
        result = subprocess.run(
            list(command), check=False, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def resolve_dataset_root(
    explicit_root: Optional[Path], project_root: Path, libero_root: Path
) -> Dict[str, Any]:
    """Resolve dataset provenance without importing LIBERO or creating user config."""

    candidates = []
    if explicit_root is not None:
        candidates.append(("command_line", explicit_root.expanduser()))
    env_root = os.environ.get("LIBERO_DATASET_ROOT")
    if env_root:
        candidates.append(("LIBERO_DATASET_ROOT", Path(env_root).expanduser()))
    candidates.extend(
        [
            ("project_data", project_root / "data"),
            ("checked_out_source_default", libero_root / "libero" / "datasets"),
        ]
    )
    records = [
        {"source": source, "path": str(path.resolve()), "exists": path.exists()}
        for source, path in candidates
    ]
    selected = next((record for record in records if record["exists"]), None)
    return {"selected": selected, "candidates": records}


def build_environment_audit(
    project_root: Path, libero_root: Path, robosuite_source_root: Path,
    dataset_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the complete Stage E0 record for the active Python interpreter."""

    requirements_file = libero_root / "requirements.txt"
    expected_robosuite = parse_pinned_requirement(requirements_file, "robosuite")
    packages = {name: distribution_record(name) for name in AUDITED_DISTRIBUTIONS}
    installed_robosuite = packages["robosuite"]["version"]
    return {
        "project_git": git_record(project_root),
        "libero_git": git_record(libero_root),
        "robosuite_source_git": git_record(robosuite_source_root),
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_prefix": str(Path(sys.prefix).resolve()),
            "operating_system": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "requirements": {
            "file": str(requirements_file.resolve()),
            "expected_robosuite": expected_robosuite,
            "robosuite_matches_pin": (
                installed_robosuite == expected_robosuite
                if installed_robosuite is not None and expected_robosuite is not None
                else False
            ),
        },
        "cuda": {
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc": command_output(["nvcc", "--version"]),
        },
        "dataset_root": resolve_dataset_root(dataset_root, project_root, libero_root),
    }
