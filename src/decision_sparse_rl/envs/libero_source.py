"""Verified source import and task enumeration for the checked-out LIBERO revision."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple


def write_libero_config(
    config_directory: Path, libero_root: Path, dataset_root: Path
) -> Path:
    """Write LIBERO's verified path schema without triggering its interactive import."""

    package_root = (libero_root / "libero" / "libero").resolve()
    required_paths = {
        "benchmark_root": package_root,
        "bddl_files": package_root / "bddl_files",
        "init_states": package_root / "init_files",
        "assets": package_root / "assets",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"checked-out LIBERO paths are missing: {missing}")
    config = {key: str(value) for key, value in required_paths.items()}
    config["datasets"] = str(dataset_root.resolve())
    config_directory.mkdir(parents=True, exist_ok=False)
    config_file = config_directory / "config.yaml"
    # JSON is valid YAML and avoids another parser in the setup path.
    config_file.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config_file


def import_benchmark_from_source(libero_root: Path, config_directory: Path) -> Any:
    """Import the benchmark API from the exact checkout and assert its source path."""

    # The checkout uses an implicit outer ``libero`` namespace directory. The
    # repository root, not that namespace directory, must therefore be on sys.path.
    import_root = libero_root.resolve()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_directory.resolve())
    sys.path.insert(0, str(import_root))
    benchmark = importlib.import_module("libero.libero.benchmark")
    module_path = Path(benchmark.__file__).resolve()
    if import_root not in module_path.parents:
        raise RuntimeError(
            f"LIBERO benchmark imported from {module_path}, expected under {import_root}"
        )
    return benchmark


def enumerate_registered_benchmarks(benchmark: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Query all registered benchmark classes and preserve any unusable registrations."""

    suites: Dict[str, Any] = {}
    errors: Dict[str, Any] = {}
    mapping = benchmark.get_benchmark_dict()
    for suite_name in sorted(mapping):
        benchmark_class = mapping[suite_name]
        try:
            suite = benchmark_class()
            tasks = []
            for task_id in range(suite.get_num_tasks()):
                task = suite.get_task(task_id)
                tasks.append(
                    {
                        "suite": suite_name,
                        "task_id": task_id,
                        "name": task.name,
                        "language": task.language,
                        "problem": task.problem,
                        "problem_folder": task.problem_folder,
                        "bddl_file": task.bddl_file,
                        "bddl_file_path": str(Path(suite.get_task_bddl_file_path(task_id)).resolve()),
                        "init_states_file": task.init_states_file,
                        "demonstration_relative_path": suite.get_task_demonstration(task_id),
                    }
                )
            suites[suite_name] = {
                "benchmark_class": benchmark_class.__name__,
                "task_count": len(tasks),
                "tasks": tasks,
            }
        except Exception as exc:
            errors[suite_name] = {
                "benchmark_class": benchmark_class.__name__,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
    return suites, errors
