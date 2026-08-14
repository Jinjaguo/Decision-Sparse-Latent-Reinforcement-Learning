#!/usr/bin/env python
"""Audit repeated EXP6 calibration executions and decide radius admission."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.metrics.exp6 import repeatability_max_abs, signal_to_floor  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def write_parquet(rows: list[dict], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--zero-run", type=Path, required=True)
    parser.add_argument("--intervention-run", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run = create_run_directory(args.run_root, args.run_id); artifacts = run / "artifacts"
    config = json.loads((ROOT / "experiments/exp6_radius_convergence/configs/exp6.json").read_text())
    zero_metrics = json.loads((args.zero_run / "metrics.json").read_text()); intervention_metrics = json.loads((args.intervention_run / "metrics.json").read_text())
    if not zero_metrics["gate"]["passed"] or not intervention_metrics["gate"]["passed"]:
        raise RuntimeError("calibration source gates must pass")
    zero_rows = pq.read_table(args.zero_run / "artifacts/zero_controls.parquet").to_pylist(); rows = pq.read_table(args.intervention_run / "artifacts/interventions.parquet").to_pylist()
    integration_floor = max(float(row["maximum_integration_l2"]) for row in zero_rows)
    signed_output_floor = 0.0; scalar_floor = 0.0; operator_floor = 0.0
    zero_floor = max(integration_floor, signed_output_floor, scalar_floor, operator_floor)
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["task"], row["episode"], int(row["branch_time"]), row["radius_label"], int(row["direction_index"]), int(row["sign"]))
        grouped.setdefault(key, []).append(row)
    repeat_rows = []; gate_by_radius = {}
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda row: row["repeat_index"])
        scalar_error = repeatability_max_abs([np.asarray([row["primary_remaining_horizon_mean"]]) for row in values])
        vector_error = repeatability_max_abs([np.asarray(row["signed_output_remaining_horizon_mean"]) for row in values])
        repeat_rows.append({"task": key[0], "episode": key[1], "branch_time": key[2], "radius_label": key[3], "direction_index": key[4], "sign": key[5], "repeat_count": len(values), "scalar_max_abs": scalar_error, "vector_max_abs": vector_error, "q_injection_max_abs_error": max(row["q_injection_max_abs_error"] for row in values), "non_arm_max_linf": max(row["non_arm_max_linf"] for row in values), "all_finite": all(row["all_states_finite"] for row in values)})
    # Build each repeated seven-column operator and compare its spectrum/matrix exactly.
    operator_rows = []
    branches = sorted({(row["task"], row["episode"], int(row["branch_time"]), row["radius_label"], float(row["radius_fraction"])) for row in rows})
    for task, episode, branch_time, label, radius in branches:
        operators = []; rankings = []
        for repeat in (0, 1):
            selected = [row for row in rows if row["task"] == task and row["episode"] == episode and int(row["branch_time"]) == branch_time and row["radius_label"] == label and int(row["repeat_index"]) == repeat]
            lookup = {(int(row["direction_index"]), int(row["sign"])): row for row in selected}
            columns = []
            for direction in range(7):
                plus = np.asarray(lookup[(direction, 1)]["signed_output_remaining_horizon_mean"]); minus = np.asarray(lookup[(direction, -1)]["signed_output_remaining_horizon_mean"])
                columns.append((plus - minus) / (2.0 * radius))
            operator = np.stack(columns, axis=1); operators.append(operator)
            rankings.append(np.argsort([row["primary_remaining_horizon_mean"] for row in sorted(selected, key=lambda value: (value["direction_index"], value["sign"]))]).tolist())
        singular0 = np.linalg.svd(operators[0], compute_uv=False); singular1 = np.linalg.svd(operators[1], compute_uv=False)
        antithetic_norms = []
        base = [row for row in rows if row["task"] == task and row["episode"] == episode and int(row["branch_time"]) == branch_time and row["radius_label"] == label and int(row["repeat_index"]) == 0]
        lookup = {(int(row["direction_index"]), int(row["sign"])): row for row in base}
        for direction in range(8):
            plus = np.asarray(lookup[(direction, 1)]["signed_output_remaining_horizon_mean"]); minus = np.asarray(lookup[(direction, -1)]["signed_output_remaining_horizon_mean"])
            antithetic_norms.append(float(np.linalg.norm((plus - minus) / 2.0)))
        operator_rows.append({"task": task, "episode": episode, "branch_time": branch_time, "radius_label": label, "radius_fraction": radius, "operator_repeat_max_abs": float(np.max(np.abs(operators[0] - operators[1]))), "spectrum_repeat_max_abs": float(np.max(np.abs(singular0 - singular1))), "direction_sign_rank_repeatable": rankings[0] == rankings[1], "minimum_antithetic_response_norm": min(antithetic_norms), "signal_to_floor": signal_to_floor(min(antithetic_norms), zero_floor)})
    for label in sorted({row["radius_label"] for row in rows}, key=lambda value: next(row["radius_fraction"] for row in rows if row["radius_label"] == value)):
        rr = [row for row in repeat_rows if row["radius_label"] == label]; oo = [row for row in operator_rows if row["radius_label"] == label]
        criteria = {"two_repeats_complete": all(row["repeat_count"] == 2 for row in rr), "q_injection_precision": max(row["q_injection_max_abs_error"] for row in rr) <= config["q_injection_atol"], "non_arm_preserved": max(row["non_arm_max_linf"] for row in rr) <= config["non_arm_integration_linf_max"], "all_finite": all(row["all_finite"] for row in rr), "scalar_repeatability": max(row["scalar_max_abs"] for row in rr) <= config["repeat_scalar_atol"], "vector_repeatability": max(row["vector_max_abs"] for row in rr) <= config["repeat_vector_atol"], "operator_repeatability": max(max(row["operator_repeat_max_abs"], row["spectrum_repeat_max_abs"]) for row in oo) <= config["repeat_operator_atol"], "rank_repeatability": all(row["direction_sign_rank_repeatable"] for row in oo), "signal_to_floor": min(row["signal_to_floor"] for row in oo) >= config["signal_to_floor_min"]}
        gate_by_radius[label] = {"passed": all(criteria.values()), "criteria": criteria, "minimum_signal_to_floor": min(row["signal_to_floor"] for row in oo), "maximum_scalar_repeat_error": max(row["scalar_max_abs"] for row in rr), "maximum_vector_repeat_error": max(row["vector_max_abs"] for row in rr), "maximum_operator_repeat_error": max(max(row["operator_repeat_max_abs"], row["spectrum_repeat_max_abs"]) for row in oo), "maximum_q_injection_error": max(row["q_injection_max_abs_error"] for row in rr)}
    primary = gate_by_radius["r000625"]["passed"]; optional = gate_by_radius["r0003125"]["passed"]
    decision = {"schema_version": 1, "primary_smallest_resolvable": primary, "optional_radius_admitted": optional, "zero_floor": {"integration_state": integration_floor, "signed_physical_output": signed_output_floor, "scalar_effect": scalar_floor, "operator_spectral": operator_floor, "ratio_resolution_constant": config["resolution_constant"]}, "radius_gates": gate_by_radius, "formal_action": "freeze formal manifests with optional radius" if primary and optional else "freeze primary-only formal manifests" if primary else "stop formal EXP6"}
    write_parquet(repeat_rows, artifacts / "zero_repeatability.parquet"); write_parquet(operator_rows, artifacts / "calibration_operator_repeatability.parquet"); write_json(artifacts / "resolution_gate.json", decision); write_json(artifacts / "zero_floor.json", decision["zero_floor"]); write_json(artifacts / "failure_examples.json", [])
    metrics = {"run_id": args.run_id, "status": "completed", "gate": {"passed": primary}, "optional_radius_admitted": optional, "repeat_rows": len(repeat_rows), "operator_rows": len(operator_rows), "zero_floor": decision["zero_floor"], "radius_gates": gate_by_radius}
    write_run_record(run, config={"stage": "EXP6 numerical resolution audit", "zero_run": args.zero_run.name, "intervention_run": args.intervention_run.name}, command=shlex.join([sys.executable, *sys.argv]), environment={"python": sys.version, "numpy": np.__version__}, git_state={"project": git_record(ROOT)}, metrics=metrics, stdout=json.dumps(metrics), stderr="")
    print(json.dumps(metrics, indent=2)); return 0 if primary else 2


if __name__ == "__main__":
    raise SystemExit(main())
