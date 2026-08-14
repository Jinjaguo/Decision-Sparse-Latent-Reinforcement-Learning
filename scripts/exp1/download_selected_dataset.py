#!/usr/bin/env python
"""Download only the suite implied by the frozen EXP1 pilot selection."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import shlex
import sys
import traceback
from typing import Iterable, TextIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.libero_source import (  # noqa: E402
    import_module_from_source,
    write_libero_config,
)
from decision_sparse_rl.logging.run_directory import (  # noqa: E402
    create_run_directory,
    write_run_record,
)
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


class Tee(io.TextIOBase):
    """Mirror text to the console and an in-memory run log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def selected_suite(selection_file: Path) -> str:
    """Return the one suite shared by every frozen selected task."""

    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    suites = {task["suite"] for task in selection["tasks"]}
    declared = selection["suite_download_scope"]
    if suites != {declared}:
        raise ValueError(
            f"selection suites {sorted(suites)} do not match declared scope {declared!r}"
        )
    return declared


def dataset_inventory(suite_directory: Path) -> Iterable[dict]:
    """Record every downloaded HDF5 file without opening it before Stage E2."""

    for path in sorted(suite_directory.glob("*.hdf5")):
        stat = path.stat()
        yield {
            "path": str(path.resolve()),
            "name": path.name,
            "size_bytes": stat.st_size,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party" / "LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT
        / "experiments"
        / "exp1_decision_sparsity"
        / "manifests"
        / "selected_tasks_pilot.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = create_run_directory(args.run_root, args.run_id)
    config = {
        "run_id": args.run_id,
        "stage": "E2_dataset_download",
        "source": "huggingface",
        "selection": str(args.selection.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "libero_root": str(args.libero_root.resolve()),
    }
    command = shlex.join([sys.executable, *sys.argv])
    environment = {"python": sys.version, "executable": sys.executable}
    git_state = {
        "project": git_record(REPOSITORY_ROOT),
        "libero": git_record(args.libero_root),
    }
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        suite = selected_suite(args.selection)
        suite_directory = args.dataset_root / suite
        if suite_directory.exists():
            raise FileExistsError(f"refusing to overwrite existing suite: {suite_directory}")
        config_directory = run_dir / "artifacts" / "libero_config"
        write_libero_config(config_directory, args.libero_root, args.dataset_root)
        download_utils = import_module_from_source(
            args.libero_root, config_directory, "libero.libero.utils.download_utils"
        )
        with contextlib.redirect_stdout(Tee(sys.stdout, stdout_buffer)), contextlib.redirect_stderr(
            Tee(sys.stderr, stderr_buffer)
        ):
            download_utils.libero_dataset_download(
                datasets=suite,
                download_dir=str(args.dataset_root.resolve()),
                check_overwrite=False,
                use_huggingface=True,
            )
        inventory = list(dataset_inventory(suite_directory))
        metrics = {
            "run_id": args.run_id,
            "status": "completed",
            "suite": suite,
            "hdf5_count": len(inventory),
            "total_bytes": sum(item["size_bytes"] for item in inventory),
            "files": inventory,
        }
        if not inventory:
            raise RuntimeError(f"download completed without HDF5 files in {suite_directory}")
    except Exception as exc:
        error_text = traceback.format_exc()
        stderr_buffer.write(error_text)
        metrics = {
            "run_id": args.run_id,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        write_run_record(
            run_dir,
            config=config,
            command=command,
            environment=environment,
            git_state=git_state,
            metrics=metrics,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
        )
        print(error_text, file=sys.stderr, end="")
        print(f"run_dir={run_dir}", file=sys.stderr)
        return 1
    write_run_record(
        run_dir,
        config=config,
        command=command,
        environment=environment,
        git_state=git_state,
        metrics=metrics,
        stdout=stdout_buffer.getvalue(),
        stderr=stderr_buffer.getvalue(),
    )
    print(json.dumps({key: metrics[key] for key in ("suite", "hdf5_count", "total_bytes")}, indent=2))
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
