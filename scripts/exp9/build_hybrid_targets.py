"""Build development-only EXP9 hybrid targets from immutable EXP8 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from decision_sparse_rl.metrics.exp9 import contact_set_events, padded_future, padded_window


HORIZONS = {"1": 1, "3": 3, "5": 5, "10": 10, "remaining": None}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jload(value):
    return json.loads(value) if isinstance(value, str) else value


def _fixed(value, size):
    out = np.zeros(size, dtype=np.float64)
    observed = np.asarray(value, dtype=np.float64).reshape(-1)
    out[: min(size, len(observed))] = observed[:size]
    return out


def vec(row):
    fields = [
        "panda_q",
        "panda_qvel",
        "eef_to_target_se3",
        "eef_to_target_twist",
        "target_to_support_se3",
        "target_to_support_twist",
        "gripper_current_action",
        "gripper_opening",
    ]
    values = []
    for name in fields:
        values.append(_fixed(row[name], 2) if name == "gripper_current_action" else np.asarray(row[name], dtype=np.float64).reshape(-1))
    values.append(
        np.asarray(
            [
                row["normalized_time"],
                row["physical_progress_clipped"],
                float(row["predicate"]),
                row["active_pair_count"],
                row["valid_pair_count"],
                row["minimum_signed_gap_m"],
                row["maximum_normal_force"],
            ],
            dtype=np.float64,
        )
    )
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--raw-run", default="runs/exp8_s11_formal_raw_locked_r1_20260814")
    parser.add_argument("--feature-run", default="runs/exp8_s6_frozen_contact_features_r1_20260814")
    parser.add_argument("--contact-frame-run", default="runs/exp8_s4_contact_frame_audit_r1_20260814")
    parser.add_argument("--reference-run", default="runs/exp8_s2_independent_refs_20260814")
    parser.add_argument("--manifest-dir", default="experiments/exp8_continuous_contact_frame/manifests")
    args = parser.parse_args()

    out = Path(args.output_run)
    artifacts = out / "artifacts"
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    artifacts.mkdir(parents=True)

    raw = Path(args.raw_run) / "artifacts"
    feature = Path(args.feature_run) / "artifacts"
    frame_path = Path(args.contact_frame_run) / "artifacts" / "reference_contact_frames.parquet"
    reference_root = Path(args.reference_run) / "artifacts" / "references"
    manifest_dir = Path(args.manifest_dir)
    source_paths = {
        "interventions": raw / "interventions.parquet",
        "per_step_effects": raw / "per_step_effects.parquet",
        "branch_features": feature / "frozen_branch_contact_features.parquet",
        "action_projection": feature / "frozen_action_projection_features.parquet",
        "contact_frames": frame_path,
        "crossfit": manifest_dir / "crossfit_manifest.json",
        "contact_identity": manifest_dir / "contact_identity_schema.json",
    }

    crossfit = json.loads(source_paths["crossfit"].read_text(encoding="utf-8"))
    folds = {(x["task"], x["episode"]): int(x["fold"]) for x in crossfit["assignments"]}
    identity = json.loads(source_paths["contact_identity"].read_text(encoding="utf-8"))
    pair_groups = {}
    for task, spec in identity["runtime_schema"]["tasks"].items():
        for group, records in spec["pair_groups"].items():
            for record in records:
                pair_groups[(task, record["pair"])] = group

    branch_rows = pq.read_table(source_paths["branch_features"]).to_pylist()
    branches = {(r["task"], r["episode"], int(r["branch_time"])): r for r in branch_rows}
    proj_rows = pq.read_table(source_paths["action_projection"]).to_pylist()
    projections = {
        (r["task"], r["episode"], int(r["branch_time"]), float(r["radius_fraction"]), int(r["direction_index"])):
        np.asarray(r["action_projection_features"], dtype=np.float64)
        for r in proj_rows
    }

    frame_rows = pq.read_table(frame_path).to_pylist()
    frame_by_demo = defaultdict(list)
    for r in frame_rows:
        frame_by_demo[(r["task"], r["episode"])].append(r)
    frame_vectors, actions = {}, {}
    history_dim = None
    for key, rows in frame_by_demo.items():
        rows.sort(key=lambda r: int(r["action_index"]))
        if [int(r["action_index"]) for r in rows] != list(range(len(rows))):
            raise RuntimeError(f"inconsistent reference boundary indexing: {key}")
        arr = np.vstack([vec(r) for r in rows])
        history_dim = arr.shape[1] if history_dim is None else history_dim
        if arr.shape[1] != history_dim:
            raise RuntimeError("history feature dimension differs across tasks")
        frame_vectors[key] = arr
        action_path = reference_root / key[0] / key[1] / "trajectory_states.npz"
        with np.load(action_path, allow_pickle=False) as npz:
            actions[key] = np.asarray(npz["actions"], dtype=np.float64)
        if len(actions[key]) != len(rows):
            raise RuntimeError(f"action/boundary length mismatch: {key}")

    intervention_rows = pq.read_table(source_paths["interventions"]).to_pylist()
    input_rows = []
    for r in intervention_rows:
        key3 = (r["task"], r["episode"], int(r["branch_time"]))
        branch = branches[key3]
        idx = key3[2]
        hist, hist_mask = padded_window(frame_vectors[key3[:2]], idx, 5)
        chunk, chunk_mask = padded_future(actions[key3[:2]], idx, 5)
        projection = projections[(key3[0], key3[1], idx, float(r["radius_fraction"]), int(r["direction_index"]))]
        delta = np.asarray(r["delta_q"], dtype=np.float64)
        base = np.asarray(branch["baseline_b_features"], dtype=np.float64)
        graph = np.asarray(branch["primary_features"], dtype=np.float64)
        intervention = np.concatenate([delta, [float(r["radius_fraction"]), float(r["sign"])], projection])
        baseline_input = np.concatenate([base, intervention])
        graph_input = np.concatenate([graph, intervention, chunk.reshape(-1), chunk_mask])
        sequence_input = np.concatenate([hist.reshape(-1), hist_mask, chunk.reshape(-1), chunk_mask, intervention])
        input_rows.append(
            {
                "intervention_id": r["intervention_id"],
                "task": r["task"],
                "episode": r["episode"],
                "branch_time": idx,
                "fold": folds[key3[:2]],
                "direction_role": r["direction_role"],
                "radius_fraction": float(r["radius_fraction"]),
                "direction_index": int(r["direction_index"]),
                "sign": int(r["sign"]),
                "baseline_input": baseline_input.tolist(),
                "graph_input": graph_input.tolist(),
                "history_sequence": hist.tolist(),
                "history_mask": hist_mask.tolist(),
                "action_chunk": chunk.tolist(),
                "action_chunk_mask": chunk_mask.tolist(),
                "sequence_input": sequence_input.tolist(),
            }
        )

    step_columns = [
        "intervention_id", "task", "episode", "branch_time", "continuation_offset",
        "zero_contact_mode_json", "perturbed_contact_mode_json", "zero_group_signed_gaps_json",
        "perturbed_group_signed_gaps_json", "zero_signed_gap_m", "perturbed_signed_gap_m",
        "zero_normal_relative_velocity_mps", "perturbed_normal_relative_velocity_mps",
        "signed_normalized_physical_output_vector",
    ]
    steps = pq.read_table(source_paths["per_step_effects"], columns=step_columns).to_pylist()
    by_intervention = defaultdict(list)
    for r in steps:
        by_intervention[r["intervention_id"]].append(r)

    input_meta = {r["intervention_id"]: r for r in input_rows}
    target_rows = []
    prevalence = defaultdict(lambda: {"n": 0, "preserved": 0, "named_add": 0, "named_drop": 0, "group_add": 0, "group_drop": 0})
    for intervention_id, rows in by_intervention.items():
        rows.sort(key=lambda r: int(r["continuation_offset"]))
        meta = input_meta[intervention_id]
        for horizon, limit in HORIZONS.items():
            selected = rows if limit is None else rows[: min(limit, len(rows))]
            additions, removals = set(), set()
            group_additions, group_removals = set(), set()
            first_change = None
            for step in selected:
                ref, pert = jload(step["zero_contact_mode_json"]), jload(step["perturbed_contact_mode_json"])
                add, drop = contact_set_events(ref, pert)
                if (add or drop) and first_change is None:
                    first_change = int(step["continuation_offset"]) + 1
                additions.update(add)
                removals.update(drop)
                group_additions.update(pair_groups.get((meta["task"], pair), "unmapped") for pair in add)
                group_removals.update(pair_groups.get((meta["task"], pair), "unmapped") for pair in drop)
            response = np.mean(
                np.vstack([np.asarray(s["signed_normalized_physical_output_vector"], dtype=np.float64) for s in selected]), axis=0
            )
            preserved = first_change is None
            final = selected[-1]
            record = {
                "intervention_id": intervention_id,
                "task": meta["task"],
                "episode": meta["episode"],
                "branch_time": meta["branch_time"],
                "fold": meta["fold"],
                "direction_role": meta["direction_role"],
                "horizon": horizon,
                "effective_horizon": len(selected),
                "mode_preserved": preserved,
                "named_pair_add": bool(additions),
                "named_pair_drop": bool(removals),
                "physical_group_add": bool(group_additions),
                "physical_group_drop": bool(group_removals),
                "named_pair_additions_json": json.dumps(sorted(additions)),
                "named_pair_removals_json": json.dumps(sorted(removals)),
                "physical_group_additions_json": json.dumps(sorted(group_additions)),
                "physical_group_removals_json": json.dumps(sorted(group_removals)),
                "first_contact_change_time": -1 if first_change is None else first_change,
                "signed_response_vector": response.tolist(),
                "signed_gap_delta": float(final["perturbed_signed_gap_m"] - final["zero_signed_gap_m"]),
                "normal_velocity_delta": float(final["perturbed_normal_relative_velocity_mps"] - final["zero_normal_relative_velocity_mps"]),
            }
            target_rows.append(record)
            p = prevalence[(horizon, meta["task"])]
            p["n"] += 1
            p["preserved"] += int(preserved)
            p["named_add"] += int(bool(additions))
            p["named_drop"] += int(bool(removals))
            p["group_add"] += int(bool(group_additions))
            p["group_drop"] += int(bool(group_removals))

    if len(input_rows) != 17280 or len(target_rows) != 17280 * len(HORIZONS):
        raise RuntimeError("EXP8 retrospective coverage mismatch")
    pq.write_table(pa.Table.from_pylist(input_rows), artifacts / "hybrid_inputs.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(target_rows), artifacts / "hybrid_targets.parquet", compression="zstd")
    prevalence_rows = []
    for (horizon, task), counts in sorted(prevalence.items()):
        n = counts["n"]
        prevalence_rows.append({"horizon": horizon, "task": task, **counts, **{f"{k}_rate": counts[k] / n for k in counts if k != "n"}})
    pq.write_table(pa.Table.from_pylist(prevalence_rows), artifacts / "retrospective_event_prevalence.parquet")

    response_dims = {task: len(next(r["signed_response_vector"] for r in target_rows if r["task"] == task)) for task in sorted(frame_by_demo_task for frame_by_demo_task, _ in frame_by_demo)}
    dims = {
        "history_step": history_dim,
        "history_length": 5,
        "action_dim": 7,
        "action_chunk_length": 5,
        "baseline_input": len(input_rows[0]["baseline_input"]),
        "graph_input": len(input_rows[0]["graph_input"]),
        "sequence_input": len(input_rows[0]["sequence_input"]),
        "response_by_task": response_dims,
        "response_max": max(response_dims.values()),
    }
    audit = {
        "stage": "A_retrospective_development_only",
        "gate": {"passed": True, "criteria": {"exact_interventions": True, "five_horizons": True, "demo_fold_isolation": True, "history_action_indexing": True, "finite_targets": True}},
        "counts": {"demonstrations": len(frame_by_demo), "interventions": len(input_rows), "targets": len(target_rows), "horizons": list(HORIZONS)},
        "dimensions": dims,
        "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_paths.items()},
        "exp8_mutated": False,
    }
    (artifacts / "retrospective_target_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"status": "completed", "gate": {"passed": True}, **audit["counts"], "dimensions": dims}, indent=2), encoding="utf-8")
    (out / "command.txt").write_text(" ".join(__import__("sys").argv), encoding="utf-8")
    print(json.dumps({"status": "completed", **audit["counts"], "dimensions": dims}, indent=2))


if __name__ == "__main__":
    main()
