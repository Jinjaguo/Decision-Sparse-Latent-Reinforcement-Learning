#!/usr/bin/env python
"""Assemble frozen-horizon EXP8 operators from locked formal raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from decision_sparse_rl.metrics.exp7 import antithetic_asymmetry, relative_error, transition_category

HORIZONS = (1, 3, 5, "remaining")


def write(rows, path):
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    raw, manifests, out = args.raw_run.resolve(), args.manifest_dir.resolve(), args.output_run.resolve()
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    lock = json.loads((raw / "artifacts/raw_data_lock_manifest.json").read_text(encoding="utf-8"))
    for name, expected in lock["sha256"].items():
        if sha(raw / "artifacts" / name) != expected:
            raise RuntimeError(f"raw hash mismatch: {name}")
    interventions = pq.read_table(raw / "artifacts/interventions.parquet").to_pandas()
    steps = pq.read_table(raw / "artifacts/per_step_effects.parquet").to_pandas()
    directions = json.loads((manifests / "direction_basis_manifest.json").read_text(encoding="utf-8"))["directions"]
    direction_lookup = {(row["task"], row["episode"], int(row["branch_time"]), float(row["radius_fraction"]), int(row["direction_index"])): row for row in directions}
    step_lookup = {key: group.sort_values("continuation_offset") for key, group in steps.groupby("intervention_id")}
    horizon_rows = []
    for row in interventions.to_dict("records"):
        group = step_lookup[row["intervention_id"]]
        for horizon in HORIZONS:
            take = group if horizon == "remaining" else group.head(int(horizon))
            vector = np.mean(np.vstack(take["signed_normalized_physical_output_vector"]), axis=0)
            preserved = bool(np.all(take["zero_contact_mode_json"] == take["perturbed_contact_mode_json"]))
            terminal = take.iloc[-1]
            horizon_rows.append({
                **{key: row[key] for key in ("task", "episode", "branch_time", "radius_fraction", "radius_label", "direction_index", "direction_role", "sign", "intervention_id")},
                "horizon": str(horizon), "signed_output_vector": vector.tolist(), "response_norm": float(np.linalg.norm(vector)),
                "mode_preserved_through_horizon": preserved, "reference_mode_json": terminal["zero_contact_mode_json"],
                "perturbed_mode_json": terminal["perturbed_contact_mode_json"], "reference_signed_gap_m": float(take.iloc[0]["zero_signed_gap_m"]),
                "reference_normal_velocity_mps": float(take.iloc[0]["zero_normal_relative_velocity_mps"]),
            })
    hdf = pd.DataFrame(horizon_rows)
    pair_keys = ["task", "episode", "branch_time", "radius_fraction", "radius_label", "direction_index", "direction_role", "horizon"]
    outcomes = []
    for key, group in hdf.groupby(pair_keys):
        plus, minus = group[group.sign == 1].iloc[0], group[group.sign == -1].iloc[0]
        reference, plus_mode, minus_mode = tuple(json.loads(plus.reference_mode_json)), tuple(json.loads(plus.perturbed_mode_json)), tuple(json.loads(minus.perturbed_mode_json))
        outcomes.append({**dict(zip(pair_keys, key)), "transition_category": transition_category(reference, plus_mode, minus_mode), "plus_preserved": bool(plus.mode_preserved_through_horizon), "minus_preserved": bool(minus.mode_preserved_through_horizon), "both_signs_preserved": bool(plus.mode_preserved_through_horizon and minus.mode_preserved_through_horizon), "reference_mode_json": plus.reference_mode_json, "plus_mode_json": plus.perturbed_mode_json, "minus_mode_json": minus.perturbed_mode_json})
    outcome_df = pd.DataFrame(outcomes)
    hdf = hdf.merge(outcome_df[pair_keys + ["transition_category", "both_signs_preserved"]], on=pair_keys)
    operators, matrices = [], []
    operator_keys = ["task", "episode", "branch_time", "radius_fraction", "radius_label", "horizon"]
    for key, group in hdf[hdf.direction_role == "basis"].groupby(operator_keys):
        columns, asymmetry, preservation, basis_vectors = [], [], [], []
        for direction_index in range(7):
            direction_group = group[group.direction_index == direction_index]
            plus, minus = direction_group[direction_group.sign == 1].iloc[0], direction_group[direction_group.sign == -1].iloc[0]
            plus_vector, minus_vector = np.asarray(plus.signed_output_vector), np.asarray(minus.signed_output_vector)
            columns.append((plus_vector - minus_vector) / (2 * float(key[3])))
            asymmetry.append(antithetic_asymmetry(plus_vector, minus_vector))
            preservation.append(bool(plus.mode_preserved_through_horizon and minus.mode_preserved_through_horizon))
            direction = direction_lookup[(key[0], key[1], int(key[2]), float(key[3]), direction_index)]
            basis_vectors.append(np.asarray(direction["unit_direction_scaled_coordinates"]))
        basis_operator = np.column_stack(columns)
        basis_matrix = np.column_stack(basis_vectors)
        canonical_operator = basis_operator @ basis_matrix.T
        singular = np.linalg.svd(canonical_operator, compute_uv=False)
        gram = canonical_operator.T @ canonical_operator
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        record = {**dict(zip(operator_keys, key)), "intent_to_perturb": True, "all_basis_both_signs_preserved": all(preservation), "preserved_direction_fraction": float(np.mean(preservation)), "spectral_norm": float(singular[0]), "leading_share": float(singular[0] ** 2 / np.sum(singular ** 2)), "median_sign_asymmetry": float(np.median(asymmetry))}
        operators.append(record)
        matrices.append({**dict(zip(operator_keys, key)), "operator_matrix": canonical_operator.tolist(), "gram_matrix": gram.tolist(), "top1_vector": eigenvectors[:, -1].tolist(), "top1_projector": np.outer(eigenvectors[:, -1], eigenvectors[:, -1]).reshape(-1).tolist()})
    matrix_lookup = {(row["task"], row["episode"], row["branch_time"], row["radius_fraction"], row["horizon"]): np.asarray(row["operator_matrix"]) for row in matrices}
    heldout = []
    for _, plus in hdf[(hdf.direction_role == "heldout_random") & (hdf.sign == 1)].iterrows():
        minus = hdf[(hdf.task == plus.task) & (hdf.episode == plus.episode) & (hdf.branch_time == plus.branch_time) & (hdf.radius_fraction == plus.radius_fraction) & (hdf.horizon == plus.horizon) & (hdf.direction_index == 7) & (hdf.sign == -1)].iloc[0]
        actual = (np.asarray(plus.signed_output_vector) - np.asarray(minus.signed_output_vector)) / (2 * plus.radius_fraction)
        direction = np.asarray(direction_lookup[(plus.task, plus.episode, int(plus.branch_time), float(plus.radius_fraction), 7)]["unit_direction_scaled_coordinates"])
        local = matrix_lookup[(plus.task, plus.episode, plus.branch_time, plus.radius_fraction, plus.horizon)] @ direction
        heldout.append({"task": plus.task, "episode": plus.episode, "branch_time": plus.branch_time, "radius_fraction": plus.radius_fraction, "horizon": plus.horizon, "both_signs_preserved": bool(plus.mode_preserved_through_horizon and minus.mode_preserved_through_horizon), "unit_direction": direction.tolist(), "actual_vector": actual.tolist(), "local_operator_prediction": local.tolist(), "local_vector_relative_error": relative_error(local, actual), "actual_norm": float(np.linalg.norm(actual)), "local_predicted_norm": float(np.linalg.norm(local))})
    write(horizon_rows, artifacts / "horizon_response_rows.parquet")
    write(outcomes, artifacts / "mode_outcomes.parquet")
    write(operators, artifacts / "horizon_operators.parquet")
    write(matrices, artifacts / "operator_matrices.parquet")
    write(heldout, artifacts / "heldout_local_predictions.parquet")
    metrics = {"raw_hashes_verified": True, "horizon_response_rows": len(horizon_rows), "mode_outcomes": len(outcomes), "operator_rows": len(operators), "heldout_rows": len(heldout), "gate": {"passed": len(operators) == 360 * 3 * 4 and len(heldout) == 360 * 3 * 4}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
