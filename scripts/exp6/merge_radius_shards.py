#!/usr/bin/env python
"""Merge immutable EXP6 formal shards and enforce exact frozen coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(paths: list[Path], output: Path) -> None:
    writer = None
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer:
            writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-id", required=True); parser.add_argument("--source-runs", nargs="+", required=True); parser.add_argument("--zero-run", type=Path, required=True); parser.add_argument("--manifest-dir", type=Path, required=True); parser.add_argument("--run-root", type=Path, default=ROOT / "runs"); args = parser.parse_args()
    run = create_run_directory(args.run_root, args.run_id); intervention_paths, effect_paths, source_hashes = [], [], {}
    for name in args.source_runs:
        source = (args.run_root / name).resolve(); metrics = json.loads((source / "metrics.json").read_text())
        if not metrics["gate"]["passed"]:
            raise RuntimeError(f"failed source shard {name}")
        intervention_paths.append(source / "artifacts/interventions.parquet"); effect_paths.append(source / "artifacts/per_step_effects.parquet"); source_hashes[name] = sha(source / "metrics.json")
    artifacts = run / "artifacts"; merge(intervention_paths, artifacts / "interventions.parquet"); merge(effect_paths, artifacts / "per_step_effects.parquet")
    zero = args.zero_run.resolve(); shutil.copy2(zero / "artifacts/zero_controls.parquet", artifacts / "zero_controls.parquet"); shutil.copy2(zero / "artifacts/zero_reference_steps.parquet", artifacts / "zero_reference_steps.parquet"); shutil.copytree(args.manifest_dir.resolve(), artifacts / "frozen_manifests")
    rows = pq.read_table(artifacts / "interventions.parquet").to_pylist(); manifest = json.loads((args.manifest_dir / "direction_basis_manifest.json").read_text()); expected = {(row["task"], row["episode"], int(row["branch_time"]), row["radius_label"], int(row["direction_index"]), sign, 0) for row in manifest["directions"] for sign in (-1, 1)}; actual = {(row["task"], row["episode"], int(row["branch_time"]), row["radius_label"], int(row["direction_index"]), int(row["sign"]), int(row["repeat_index"])) for row in rows}
    failures = [row for row in rows if not row["joint_limit_valid"] or not row["all_states_finite"] or row["non_arm_max_linf"] > 1e-12 or row["q_injection_max_abs_error"] > 1e-15]
    effects_schema = set(pq.read_schema(artifacts / "per_step_effects.parquet").names); criteria = {"exact_19200": len(rows) == 19200, "unique_rows": len(actual) == len(rows), "exact_frozen_coverage": actual == expected, "no_execution_failures": not failures, "all_source_shards": len(args.source_runs) == 10, "exact_contact_pair_fields": {"zero_contact_pairs_json", "perturbed_contact_pairs_json"}.issubset(effects_schema)}
    names = ["zero_controls.parquet", "zero_reference_steps.parquet", "interventions.parquet", "per_step_effects.parquet"]; hashes = {name: sha(artifacts / name) for name in names}; write_json(artifacts / "raw_hash_manifest.json", {"schema_version": 1, "locked_before_analysis": True, "source_metrics_sha256": source_hashes, "sha256": hashes}); write_json(artifacts / "failure_examples.json", failures[:100]); labels = list(json.loads((args.manifest_dir / "radius_manifest.json").read_text())["labels"].keys())
    metrics = {"run_id": args.run_id, "status": "completed", "gate": {"passed": all(criteria.values()), "criteria": criteria}, "intervention_count": len(rows), "per_step_effect_count": pq.read_metadata(artifacts / "per_step_effects.parquet").num_rows, "success_flip_count": sum(row["success_flip"] for row in rows), "maximum_non_arm_linf": max(row["non_arm_max_linf"] for row in rows), "maximum_q_injection_error": max(row["q_injection_max_abs_error"] for row in rows), "by_radius": {label: sum(row["radius_label"] == label for row in rows) for label in labels}, "raw_hashes": hashes}
    write_run_record(run, config={"stage": "EXP6 formal raw merge and lock", "source_runs": args.source_runs}, command=shlex.join([sys.executable, *sys.argv]), environment={"python": sys.version, "pyarrow": pa.__version__}, git_state={"project": git_record(ROOT)}, metrics=metrics, stdout=json.dumps(metrics), stderr=""); print(json.dumps(metrics, indent=2)); return 0 if metrics["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
