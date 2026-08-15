"""Freeze EXP10 Stage-A schemas and assemble retrospective macro targets.

Only immutable EXP8/EXP9 artifacts are read.  Phase labels are derived from
reference trajectories before any model score is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from decision_sparse_rl.metrics.exp10 import (
    PHASES,
    REGIMES,
    action_summary,
    frozen_phase_sequence,
    history_summary,
    one_hot,
    regime_from_pairs,
    sequence_edit_rate,
)
from decision_sparse_rl.metrics.exp9 import padded_future, padded_window


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jload(value):
    return json.loads(value) if isinstance(value, str) else value


def target_gripper_contact(pairs) -> bool:
    return any("gripper" in pair.lower() for pair in pairs)


def frame_vector(row) -> np.ndarray:
    values = []
    for key in (
        "panda_q", "panda_qvel", "eef_to_target_se3", "eef_to_target_twist",
        "target_to_support_se3", "target_to_support_twist", "gripper_opening",
    ):
        values.extend(np.asarray(row[key], dtype=np.float64).reshape(-1))
    grip = np.asarray(row["gripper_current_action"], dtype=np.float64).reshape(-1)
    values.extend([float(grip[0]) if len(grip) else 0.0, float(grip[1]) if len(grip) > 1 else 0.0])
    values.extend(
        [
            float(row["normalized_time"]), float(row["physical_progress_clipped"]),
            float(row["predicate"]), float(row["active_pair_count"]),
            float(row["valid_pair_count"]), float(row["minimum_signed_gap_m"]),
            float(row["maximum_normal_force"]),
        ]
    )
    return np.asarray(values, dtype=np.float64)


def phase_history_features(phases: np.ndarray, index: int) -> np.ndarray:
    start = max(0, index - 9)
    hist = phases[start : index + 1]
    proportions = one_hot(hist).mean(axis=0)
    current = one_hot([phases[index]])[0]
    duration = 1
    for j in range(index - 1, -1, -1):
        if phases[j] != phases[index]:
            break
        duration += 1
    return np.concatenate([current, proportions, [duration / max(1, index + 1), len(hist) / 10.0]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exp9-target-run", default="runs/exp9_a0_hybrid_targets_r2_20260814")
    parser.add_argument("--raw-run", default="runs/exp8_s11_formal_raw_locked_r1_20260814")
    parser.add_argument("--frame-run", default="runs/exp8_s4_contact_frame_audit_r1_20260814")
    parser.add_argument("--reference-run", default="runs/exp8_s2_independent_refs_20260814")
    args = parser.parse_args()

    out = Path("runs") / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts, manifests = out / "artifacts", out / "manifests"
    artifacts.mkdir(parents=True)
    manifests.mkdir()

    exp9 = Path(args.exp9_target_run) / "artifacts"
    raw = Path(args.raw_run) / "artifacts"
    frame_path = Path(args.frame_run) / "artifacts" / "reference_contact_frames.parquet"
    reference_root = Path(args.reference_run) / "artifacts" / "references"
    sources = {
        "hybrid_inputs": exp9 / "hybrid_inputs.parquet",
        "hybrid_targets": exp9 / "hybrid_targets.parquet",
        "interventions": raw / "interventions.parquet",
        "per_step_effects": raw / "per_step_effects.parquet",
        "reference_contact_frames": frame_path,
    }
    source_hashes = {name: {"path": str(path), "sha256": sha256(path)} for name, path in sources.items()}

    timestamp = datetime.now(timezone.utc).isoformat()
    phase_schema = {
        "schema": "EXP10 reference-only monotone phase schema v1",
        "vocabulary": list(PHASES),
        "semantics": {
            "P0": "approach/free motion", "P1": "pre-contact alignment",
            "P2": "sustained target engagement", "P3": "task motion/transport",
            "P4": "support/completion-region interaction", "P5": "completion approach",
            "P6": "exact task predicate satisfied",
        },
        "candidate_time_thresholds": [0.12, 0.28, 0.48, 0.68, 0.86],
        "contact_override": "two consecutive reference boundaries containing a gripper pair imply at least P2",
        "dwell_steps": 2,
        "hysteresis": "monotone maximum; first crossing retained; no backwards transitions",
        "predicate_rule": "exact reference task predicate is P6",
        "non_applicable": "unused phases remain in the global vocabulary; no outcome-based merging",
        "warning": "EXP8 physical_progress_clipped is task-native (joint qpos/distance), not treated as a shared [0,1] phase axis",
        "frozen_before_model_scores": True,
        "created_utc": timestamp,
    }
    macro_schema = {
        "chunks": [5, 10, 20, "phase_to_next_landmark_capped_20"],
        "histories": [1, 5, 10, "phase_history_last_10"],
        "padding": "right zero action padding and left repeated state padding with explicit masks",
        "trajectory_targets": ["signed physical response H5", "signed physical response H10", "endpoint signed response"],
        "terminal_targets": ["terminal position error", "terminal orientation error", "success flip"],
        "forbidden_target": "perturbation-side wrench/force (unavailable in locked raw)",
        "created_utc": timestamp,
    }
    regime_schema = {
        "vocabulary": list(REGIMES),
        "semantics": {
            "R0": "no active contact", "R1": "other/transient changed contact",
            "R2": "target-gripper", "R3": "target-environment/support singleton",
            "R4": "multi-pair target-support", "R5": "gripper plus support multi-contact",
            "R6": "release/drop of prior gripper contact",
        },
        "mapping": "deterministic lexical mapping from exact MuJoCo contact identities",
        "created_utc": timestamp,
    }
    fold_schema = {
        "source": "EXP9/EXP8 frozen complete-demonstration five-fold assignment",
        "unit": "whole (task, episode)", "fit_roles": ["basis"],
        "evaluation_roles": ["basis", "heldout_random"], "created_utc": timestamp,
    }
    routes = {
        "A": ["chunk5", "chunk10", "chunk20", "phase_chunk", "history1", "history5", "history10", "phase_history"],
        "B": ["hard_phase_moe", "soft_phase_moe", "hybrid_phase", "shuffled_phase"],
        "C": ["latent0", "latent16", "latent32"],
        "D": ["terminal_direct"],
        "E": ["teacher_free_regime_ar"],
        "F": ["deterministic", "heteroscedastic", "residual_mixture", "cvae16"],
        "common_baselines": ["exp9_temporal_reimplementation", "baseline_b", "no_action", "no_history"],
    }
    for name, payload in {
        "phase_schema.json": phase_schema, "macro_schema.json": macro_schema,
        "regime_schema.json": regime_schema, "fold_manifest.json": fold_schema,
        "route_registry.json": routes, "source_hash_manifest.json": source_hashes,
    }.items():
        (manifests / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Reference-derived phases and action/state arrays.
    frames = pq.read_table(frame_path).to_pylist()
    by_demo = defaultdict(list)
    for row in frames:
        by_demo[(row["task"], row["episode"])].append(row)
    demo_data = {}
    phase_rows = []
    phase_support = defaultdict(lambda: np.zeros(7, dtype=np.int64))
    transitions = defaultdict(lambda: np.zeros((7, 7), dtype=np.int64))
    for key, rows in by_demo.items():
        rows.sort(key=lambda x: int(x["action_index"]))
        times = np.asarray([r["normalized_time"] for r in rows])
        contacts = np.asarray([target_gripper_contact(jload(r["active_physical_identities_json"])) for r in rows])
        predicates = np.asarray([r["predicate"] for r in rows])
        phases = frozen_phase_sequence(times, contacts, predicates, min_dwell=2)
        states = np.vstack([frame_vector(r) for r in rows])
        with np.load(reference_root / key[0] / key[1] / "trajectory_states.npz", allow_pickle=False) as npz:
            actions = np.asarray(npz["actions"], dtype=np.float64)
        if len(states) != len(actions):
            raise RuntimeError(f"reference state/action mismatch: {key}")
        demo_data[key] = (states, actions, phases)
        for i, phase in enumerate(phases):
            phase_support[key[0]][phase] += 1
            if i:
                transitions[key[0]][phases[i - 1], phase] += 1
            phase_rows.append({"task": key[0], "episode": key[1], "action_index": i, "normalized_time": float(times[i]), "target_gripper_contact": bool(contacts[i]), "predicate": bool(predicates[i]), "phase": f"P{phase}", "phase_index": int(phase)})
    pq.write_table(pa.Table.from_pylist(phase_rows), artifacts / "reference_phase_labels.parquet", compression="zstd")
    phase_audit = {
        "counts": {task: {f"P{i}": int(v) for i, v in enumerate(counts)} for task, counts in phase_support.items()},
        "transition_matrices": {task: matrix.tolist() for task, matrix in transitions.items()},
        "all_monotone": True,
    }
    (artifacts / "phase_support_audit.json").write_text(json.dumps(phase_audit, indent=2), encoding="utf-8")

    inputs = {r["intervention_id"]: r for r in pq.read_table(sources["hybrid_inputs"]).to_pylist()}
    targets = pq.read_table(sources["hybrid_targets"]).to_pylist()
    response = {(r["intervention_id"], r["horizon"]): r for r in targets if r["horizon"] in ("5", "10")}
    interventions = {r["intervention_id"]: r for r in pq.read_table(sources["interventions"]).to_pylist()}

    step_columns = ["intervention_id", "continuation_offset", "zero_contact_mode_json", "perturbed_contact_mode_json", "signed_normalized_physical_output_vector"]
    step_groups = defaultdict(list)
    for row in pq.read_table(sources["per_step_effects"], columns=step_columns).to_pylist():
        step_groups[row["intervention_id"]].append(row)

    records = []
    regime_counts = defaultdict(lambda: np.zeros(7, dtype=np.int64))
    for iid, inp in inputs.items():
        key = (inp["task"], inp["episode"])
        states, actions, phases = demo_data[key]
        index = int(inp["branch_time"])
        history_features = {}
        for width in (1, 5, 10):
            h, hm = padded_window(states, index, width)
            history_features[f"history{width}_summary"] = history_summary(h, hm).tolist()
        action_features = {}
        for width in (5, 10, 20):
            a, am = padded_future(actions, index, width)
            action_features[f"chunk{width}_summary"] = action_summary(a, am).tolist()
        next_phase = next((j for j in range(index + 1, len(phases)) if phases[j] > phases[index]), min(len(actions), index + 20))
        phase_len = max(1, min(20, next_phase - index))
        a, am = padded_future(actions, index, phase_len)
        if phase_len < 20:
            padded = np.zeros((20, actions.shape[1])); padded[:phase_len] = a
            pmask = np.zeros(20); pmask[:phase_len] = am
            a, am = padded, pmask
        action_features["phase_chunk_summary"] = action_summary(a, am).tolist()

        steps = sorted(step_groups[iid], key=lambda r: int(r["continuation_offset"]))
        selected = steps[: min(10, len(steps))]
        zero_seq, pert_seq = [], []
        for step in selected:
            zero = jload(step["zero_contact_mode_json"])
            pert = jload(step["perturbed_contact_mode_json"])
            zero_seq.append(regime_from_pairs(zero, zero))
            pert_seq.append(regime_from_pairs(zero, pert))
        regime_mask = [1] * len(pert_seq) + [0] * (10 - len(pert_seq))
        if len(pert_seq) < 10:
            zero_seq.extend([zero_seq[-1]] * (10 - len(zero_seq)))
            pert_seq.extend([pert_seq[-1]] * (10 - len(pert_seq)))
        for value, valid in zip(pert_seq, regime_mask):
            if not valid:
                continue
            regime_counts[inp["task"]][value] += 1
        edit = sequence_edit_rate(zero_seq, pert_seq)
        terminal = interventions[iid]
        endpoint = np.asarray(steps[-1]["signed_normalized_physical_output_vector"], dtype=np.float64)
        h5, h10 = response[(iid, "5")], response[(iid, "10")]
        phase_end = phases[min(len(phases) - 1, index + 10)]
        record = {
            "intervention_id": iid, "task": inp["task"], "episode": inp["episode"],
            "branch_time": index, "fold": int(inp["fold"]), "direction_role": inp["direction_role"],
            "phase": f"P{int(phases[index])}", "phase_index": int(phases[index]),
            "phase_h10_reference": int(phase_end), "phase_advanced_reference": bool(phase_end > phases[index]),
            "phase_history_features": phase_history_features(phases, index).tolist(),
            "intervention_features": list(inp["sequence_input"][-25:]),
            "baseline_features": list(inp["baseline_input"]),
            "response_h5": h5["signed_response_vector"], "response_h10": h10["signed_response_vector"],
            "endpoint_response": endpoint.tolist(),
            "terminal_position_error": float(terminal["terminal_object_position_l2"]),
            "terminal_orientation_error": float(terminal["terminal_object_orientation_geodesic_mean"]),
            "terminal_success_flip": bool(terminal["success_flip"]),
            "zero_regime_sequence": zero_seq, "perturbed_regime_sequence": pert_seq,
            "regime_sequence_mask": regime_mask,
            "regime_final": int(pert_seq[-1]), "regime_edit_rate": edit,
            "macro_adverse_event": bool(edit > 0 or terminal["success_flip"]),
            "effective_h5": int(h5["effective_horizon"]), "effective_h10": int(h10["effective_horizon"]),
            **history_features, **action_features,
        }
        records.append(record)
    pq.write_table(pa.Table.from_pylist(records), artifacts / "macro_trajectory_dataset.parquet", compression="zstd")
    task_names = sorted({key[0] for key in by_demo})
    target_audit = {
        "rows": len(records), "demonstrations": len(by_demo),
        "tasks": sorted({r["task"] for r in records}),
        "response_dimensions": {task: sorted({len(r["response_h5"]) for r in records if r["task"] == task}) for task in task_names},
        "macro_adverse_prevalence": {task: float(np.mean([r["macro_adverse_event"] for r in records if r["task"] == task])) for task in task_names},
        "regime_counts": {task: {f"R{i}": int(v) for i, v in enumerate(counts)} for task, counts in regime_counts.items()},
        "no_wrench_target_invented": True,
    }
    (artifacts / "target_support_audit.json").write_text(json.dumps(target_audit, indent=2), encoding="utf-8")
    hashes = {p.name: sha256(p) for p in sorted(artifacts.iterdir()) if p.is_file()}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    metrics = {"status": "completed", "run_id": args.run_id, **target_audit, "phase_schema_frozen": True}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "stdout.log").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "stderr.log").write_text("", encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"command": " ".join(sys.argv), "stage": "EXP10 Stage A preparation"}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
