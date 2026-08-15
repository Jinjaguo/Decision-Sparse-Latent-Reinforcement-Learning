"""Materialize the immutable EXP12 same-state candidate-ranking dataset."""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.metrics.exp10 import action_summary
from decision_sparse_rl.metrics.exp12 import nominal_improvement_opportunity, sha256_file
from scripts.exp11.run_replacement_stage import modified_chunk


TASKS = (
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "turn_on_the_stove",
)
TASK_BODIES = {
    TASKS[0]: ("wooden_cabinet_1_cabinet_middle",),
    TASKS[1]: ("akita_black_bowl_1_main", "plate_1_main"),
    TASKS[2]: ("flat_stove_1_button",),
}
FAMILY_ORDER = ("nominal", "I-A_analytic", "I-B_residual", "I-C_phase_edit")
BASIS_ORDER = ("nominal", "dct", "spline", "pulse", "residual_svd", "reference_index_shift")


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_pq(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty artifact {path.name}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def one_hot(index: int, count: int) -> np.ndarray:
    out = np.zeros(count, dtype=np.float64)
    out[index] = 1.0
    return out


def quat_distance(left, right) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))


def object_arrays(boundary: dict, task: str) -> tuple[np.ndarray, np.ndarray]:
    names = list(boundary["body_names"])
    pos = np.asarray([boundary["body_positions"][names.index(x)] for x in TASK_BODIES[task]], dtype=np.float64)
    quat = np.asarray([boundary["body_quaternions"][names.index(x)] for x in TASK_BODIES[task]], dtype=np.float64)
    return pos, quat


def padded_objects(position: np.ndarray, quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((2, 3), dtype=np.float64)
    quat = np.zeros((2, 4), dtype=np.float64)
    pos[: len(position)] = position
    quat[: len(quaternion)] = quaternion
    return pos, quat


def contact_features(boundary: dict, task: str) -> np.ndarray:
    names = [f"{x['geom1_name']}|{x['geom2_name']}".lower() for x in boundary["contact_pairs"]]
    joined = " ".join(names)
    tokens = {
        TASKS[0]: ("wooden_cabinet", "cabinet_middle"),
        TASKS[1]: ("akita_black_bowl", "plate"),
        TASKS[2]: ("flat_stove", "button"),
    }[task]
    flags = [
        any("gripper" in x or "finger" in x for x in names),
        any(any(token in x for token in tokens) for x in names),
        any(any(token in x for token in ("plate", "table", "cabinet", "stove")) for x in names),
        all(token in joined for token in tokens),
    ]
    return np.asarray([min(float(boundary["contact_count"]) / 100.0, 2.0), *flags], dtype=np.float64)


def base_context(boundaries: list[dict], task: str, t: int) -> dict[str, np.ndarray]:
    boundary = boundaries[t]
    pos, quat = object_arrays(boundary, task)
    ppos, pquat = padded_objects(pos, quat)
    eef = np.asarray(boundary["eef_position"], dtype=np.float64)
    orientation = np.asarray(boundary["eef_orientation_matrix"], dtype=np.float64).reshape(-1)
    arm = np.asarray(boundary["panda_arm_q"], dtype=np.float64)
    gripper = np.asarray(boundary["gripper_state"], dtype=np.float64)
    relative = ppos - eef
    common = np.r_[one_hot(TASKS.index(task), len(TASKS)), one_hot(int(boundary.get("progress_channels", {}).get("exact_task_predicate", False)), 2), t / max(1, len(boundaries) - 1)]
    physical = np.r_[common, arm, gripper, eef, orientation]
    obj = np.r_[common, eef, orientation, ppos.reshape(-1), pquat.reshape(-1), relative.reshape(-1)]
    contact = np.r_[obj, contact_features(boundary, task)]
    history = []
    for index in range(t - 4, t + 1):
        b = boundaries[max(0, index)]
        hp, hq = object_arrays(b, task)
        hp, _ = padded_objects(hp, hq)
        history.extend(np.r_[b["eef_position"], b["panda_arm_q"], b["gripper_state"], hp.reshape(-1)])
    return {"physical": physical, "object": obj, "contact": contact, "history": np.r_[contact, history]}


def action_features(chunk: np.ndarray) -> np.ndarray:
    mask = np.ones(len(chunk), dtype=np.float64)
    return np.r_[action_summary(chunk, mask), chunk.reshape(-1)]


def future_vector(step_rows: list[dict], horizon: int = 20) -> np.ndarray:
    rows = sorted(step_rows, key=lambda x: x["continuation_offset"])
    if not rows:
        raise ValueError("missing candidate future")
    vectors = []
    for offset in range(horizon):
        row = rows[min(offset, len(rows) - 1)]
        pos, quat = padded_objects(np.asarray(row["task_object_positions"]), np.asarray(row["task_object_quaternions"]))
        mode = str(row["contact_mode_json"]).lower()
        flags = [mode not in ("", "[]", "()"), "gripper" in mode, "plate" in mode or "table" in mode, bool(row["regime_changed"])]
        vectors.extend(np.r_[row["eef_position"], row["eef_orientation"], pos.reshape(-1), quat.reshape(-1), float(row["task_predicate"]), flags])
    return np.asarray(vectors, dtype=np.float64)


def reference_future(boundaries: list[dict], terminal_pos: np.ndarray, terminal_quat: np.ndarray, task: str, t: int, horizon: int = 20) -> np.ndarray:
    vectors = []
    for offset in range(1, horizon + 1):
        index = t + offset
        boundary = boundaries[min(index, len(boundaries) - 1)]
        pos, quat = object_arrays(boundary, task)
        if index >= len(boundaries):
            names = list(boundaries[0]["body_names"])
            pos = np.asarray([terminal_pos[names.index(x)] for x in TASK_BODIES[task]])
            quat = np.asarray([terminal_quat[names.index(x)] for x in TASK_BODIES[task]])
        pos, quat = padded_objects(pos, quat)
        mode_names = [f"{x['geom1_name']}|{x['geom2_name']}".lower() for x in boundary["contact_pairs"]]
        joined = " ".join(mode_names)
        flags = [bool(mode_names), "gripper" in joined, "plate" in joined or "table" in joined, False]
        vectors.extend(np.r_[boundary["eef_position"], np.asarray(boundary["eef_orientation_matrix"]).reshape(-1), pos.reshape(-1), quat.reshape(-1), float(boundary["task_success"]), flags])
    return np.asarray(vectors, dtype=np.float64)


def motion_quality(task: str, initial_pos, initial_quat, terminal_pos, terminal_quat, reference_pos, reference_quat) -> float:
    initial_pos, terminal_pos, reference_pos = map(lambda x: np.asarray(x, dtype=np.float64), (initial_pos, terminal_pos, reference_pos))
    if task == TASKS[0]:
        direction = reference_pos[0] - initial_pos[0]
        denom = max(float(np.dot(direction, direction)), 1e-12)
        value = float(np.dot(terminal_pos[0] - initial_pos[0], direction) / denom)
    elif task == TASKS[2]:
        denom = max(quat_distance(initial_quat[0], reference_quat[0]), 1e-12)
        value = quat_distance(initial_quat[0], terminal_quat[0]) / denom
    else:
        initial = max(float(np.linalg.norm(initial_pos[0, :2] - initial_pos[1, :2])), 1e-6)
        current = float(np.linalg.norm(terminal_pos[0, :2] - terminal_pos[1, :2]))
        value = 1.0 - current / initial
    return float(np.clip(value, -2.0, 2.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-run", type=Path, default=Path("runs/exp11_s2_formal_replacements_r1_20260814"))
    parser.add_argument("--reference-run", type=Path, default=Path("runs/exp11_s2_formal_refs_20260814"))
    parser.add_argument("--fidelity-run", type=Path, default=Path("runs/exp11_s2_formal_fidelity_reaudit_r1_20260814"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts, manifests = out / "artifacts", out / "manifests"
    artifacts.mkdir(parents=True)
    manifests.mkdir()

    formal = ROOT / args.formal_run
    reference = ROOT / args.reference_run
    sources = [
        formal / "artifacts/replacements.parquet",
        formal / "artifacts/per_step_response.parquet",
        formal / "manifests/replacement_plan.json",
        formal / "manifests/formal_branch_manifest.json",
        reference / "artifacts/reference_snapshots_manifest.json",
        ROOT / args.fidelity_run / "metrics.json",
        ROOT / "experiments/exp12_action_consequence_ranking/configs/exp12.json",
        ROOT / "experiments/exp12_action_consequence_ranking/manifests/ranking_target_schema.json",
    ]
    raw_hash = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
    dump(manifests / "source_hash_manifest.json", raw_hash)

    summaries = pq.read_table(sources[0]).to_pylist()
    step_rows = pq.read_table(sources[1]).to_pylist()
    specs = {x["intervention_id"]: x for x in json.loads(sources[2].read_text())}
    branches = {x["branch_id"]: x for x in json.loads(sources[3].read_text())}
    references = json.loads(sources[4].read_text())["episodes"]
    fidelity = json.loads(sources[5].read_text())["rows"]
    valid_pair = {(x["task"], x["family"]): bool(x["execution_valid"]) for x in fidelity}
    ref_lookup = {(x["task"], x["episode"]): x for x in references}
    by_step: dict[str, list[dict]] = defaultdict(list)
    for row in step_rows:
        by_step[row["intervention_id"]].append(row)

    cache = {}
    dataset = []
    branch_nominal = {}
    for summary in summaries:
        task, episode, branch_id = summary["task"], summary["episode"], summary["branch_id"]
        key = (task, episode)
        if key not in cache:
            record = ref_lookup[key]
            directory = reference / record["relative_directory"]
            boundaries = json.loads((directory / "boundaries.json").read_text())
            with np.load(directory / "trajectory_states.npz", allow_pickle=False) as z:
                cache[key] = {
                    "boundaries": boundaries,
                    "actions": np.asarray(z["actions"], dtype=np.float64),
                    "terminal_pos": np.asarray(z["terminal_body_positions"], dtype=np.float64),
                    "terminal_quat": np.asarray(z["terminal_body_quaternions"], dtype=np.float64),
                }
        item = cache[key]
        boundaries, actions = item["boundaries"], item["actions"]
        branch, spec = branches[branch_id], specs[summary["intervention_id"]]
        t = int(branch["branch_time"])
        context = base_context(boundaries, task, t)
        ref, _, executed = modified_chunk(actions, branch, spec)
        action = action_features(executed)
        steps = by_step[summary["intervention_id"]]
        terminal = max(steps, key=lambda x: x["continuation_offset"])
        initial_pos, initial_quat = object_arrays(boundaries[0], task)
        names = list(boundaries[0]["body_names"])
        reference_pos = np.asarray([item["terminal_pos"][names.index(x)] for x in TASK_BODIES[task]])
        reference_quat = np.asarray([item["terminal_quat"][names.index(x)] for x in TASK_BODIES[task]])
        terminal_pos = np.asarray(terminal["task_object_positions"], dtype=np.float64)
        terminal_quat = np.asarray(terminal["task_object_quaternions"], dtype=np.float64)
        motion = motion_quality(task, initial_pos, initial_quat, terminal_pos, terminal_quat, reference_pos, reference_quat)
        contact = 1.0 - float(summary["regime_change_fraction_h20"])
        outcome = float(summary["terminal_perturbed_success"])
        composite = 4.0 * outcome + contact + float(np.clip(motion, -1, 1.5))
        metadata = np.r_[
            one_hot(TASKS.index(task), len(TASKS)),
            one_hot(int(summary["phase"][1:]), 7),
            one_hot(FAMILY_ORDER.index(summary["family"]), len(FAMILY_ORDER)),
            one_hot(BASIS_ORDER.index(summary["basis_family"]), len(BASIS_ORDER)),
            float(summary["amplitude"]), float(summary["sign"]), t / max(1, len(actions) - 1),
        ]
        dataset.append({
            "candidate_id": summary["intervention_id"], "branch_id": branch_id, "task": task,
            "episode": episode, "demo_key": f"{task}|{episode}", "phase": summary["phase"],
            "family": summary["family"], "basis_family": summary["basis_family"], "is_nominal": False,
            "primary_valid": bool(valid_pair.get((task, summary["family"]), False) and summary["all_states_finite"]),
            "clipped_chunk": bool(summary["clipped_chunk"]), "outcome_quality": outcome,
            "contact_quality": contact, "motion_quality": motion, "composite_quality": composite,
            "catastrophic": bool(not outcome or contact < .5 or motion < 0),
            "macro_effect_h10": float(summary["macro_effect_h10"]),
            "terminal_object_position_l2": float(summary["terminal_object_position_l2"]),
            "x_e1": np.r_[metadata, context["physical"], action].tolist(),
            "x_e2": np.r_[metadata, context["object"], action].tolist(),
            "x_e3": np.r_[metadata, context["contact"], action].tolist(),
            "x_e4": np.r_[metadata, context["history"], action].tolist(),
            "future_rich20": future_vector(steps).tolist(),
        })
        branch_nominal[branch_id] = (task, episode, t, context, metadata, initial_pos, initial_quat, reference_pos, reference_quat)

    for branch_id, (task, episode, t, context, metadata, initial_pos, initial_quat, reference_pos, reference_quat) in branch_nominal.items():
        item = cache[(task, episode)]
        chunk = item["actions"][t : t + 10]
        motion = motion_quality(task, initial_pos, initial_quat, reference_pos, reference_quat, reference_pos, reference_quat)
        nominal_meta = metadata.copy()
        family_start = 3 + 7
        nominal_meta[family_start : family_start + len(FAMILY_ORDER)] = one_hot(0, len(FAMILY_ORDER))
        basis_start = family_start + len(FAMILY_ORDER)
        nominal_meta[basis_start : basis_start + len(BASIS_ORDER)] = one_hot(0, len(BASIS_ORDER))
        nominal_meta[-3:] = [0.0, 0.0, t / max(1, len(item["actions"]) - 1)]
        action = action_features(chunk)
        composite = 4.0 + 1.0 + float(np.clip(motion, -1, 1.5))
        dataset.append({
            "candidate_id": f"nominal|{branch_id}", "branch_id": branch_id, "task": task,
            "episode": episode, "demo_key": f"{task}|{episode}", "phase": f"P{int(np.argmax(metadata[3:10]))}",
            "family": "nominal", "basis_family": "nominal", "is_nominal": True,
            "primary_valid": True, "clipped_chunk": False, "outcome_quality": 1.0,
            "contact_quality": 1.0, "motion_quality": motion, "composite_quality": composite,
            "catastrophic": False, "macro_effect_h10": 0.0, "terminal_object_position_l2": 0.0,
            "x_e1": np.r_[nominal_meta, context["physical"], action].tolist(),
            "x_e2": np.r_[nominal_meta, context["object"], action].tolist(),
            "x_e3": np.r_[nominal_meta, context["contact"], action].tolist(),
            "x_e4": np.r_[nominal_meta, context["history"], action].tolist(),
            "future_rich20": reference_future(item["boundaries"], item["terminal_pos"], item["terminal_quat"], task, t).tolist(),
        })

    dataset.sort(key=lambda x: (x["branch_id"], not x["is_nominal"], x["candidate_id"]))
    write_pq(artifacts / "candidate_dataset.parquet", dataset)
    primary = [x for x in dataset if x["primary_valid"]]
    groups = defaultdict(list)
    for row in primary:
        groups[row["branch_id"]].append(row)
    group_rows, opportunity = [], []
    for branch_id, rows in sorted(groups.items()):
        nominal = next(i for i, x in enumerate(rows) if x["is_nominal"])
        quality = [x["composite_quality"] for x in rows]
        available, gap = nominal_improvement_opportunity(quality, nominal, .05)
        group_rows.append({"branch_id": branch_id, "task": rows[0]["task"], "demo_key": rows[0]["demo_key"], "candidate_count": len(rows), "candidate_ids": [x["candidate_id"] for x in rows], "nominal_candidate_id": rows[nominal]["candidate_id"]})
        opportunity.append({"branch_id": branch_id, "task": rows[0]["task"], "demo_key": rows[0]["demo_key"], "candidate_count": len(rows), "nominal_quality": quality[nominal], "oracle_quality": max(quality), "oracle_improvement_gap": gap, "opportunity": available, "oracle_candidate_id": rows[int(np.argmax(quality))]["candidate_id"]})
    write_pq(artifacts / "candidate_opportunity.parquet", opportunity)
    dump(manifests / "candidate_group_manifest.json", {"group_count": len(group_rows), "groups": group_rows, "all_groups_ge_2": all(x["candidate_count"] >= 2 for x in group_rows), "all_groups_ge_3": all(x["candidate_count"] >= 3 for x in group_rows), "intent_to_replace_rows": len(dataset), "primary_rows": len(primary)})
    demo_order = sorted({x["demo_key"] for x in primary})
    crossfit = [{"demo_key": demo, "fold": index % 5} for index, demo in enumerate(demo_order)]
    dump(manifests / "crossfit_manifest.json", {"method": "sorted demo round-robin, outcome blind", "folds": 5, "assignments": crossfit})
    dump(manifests / "feature_dimensions.json", {key: len(primary[0][key]) for key in ("x_e1", "x_e2", "x_e3", "x_e4", "future_rich20")})
    metrics = {
        "status": "completed", "run_id": args.run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_rows": len(dataset), "primary_rows": len(primary), "groups": len(groups),
        "candidate_set_sizes": {str(k): sum(len(v) == k for v in groups.values()) for k in sorted({len(v) for v in groups.values()})},
        "opportunity_groups": sum(x["opportunity"] for x in opportunity),
        "opportunity_by_task": {task: sum(x["opportunity"] for x in opportunity if x["task"] == task) for task in TASKS},
        "raw_hash_verified": all(sha256_file(ROOT / path) == digest for path, digest in raw_hash.items()),
    }
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
