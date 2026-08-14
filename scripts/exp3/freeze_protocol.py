#!/usr/bin/env python
"""Freeze all outcome-blind EXP3 manifests after the T1 restore gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from decision_sparse_rl.logging.run_directory import write_json  # noqa: E402


OBJECT_CHANNELS = {
    "open_the_middle_drawer_of_the_cabinet": {
        "bodies": ["wooden_cabinet_1_cabinet_middle"],
        "bddl_goal": ["open", "wooden_cabinet_1_middle_region"],
        "predicate_source": "Open.__call__ -> SiteObjectState.is_open -> WoodenCabinet.is_open",
    },
    "turn_on_the_stove": {
        "bodies": ["flat_stove_1_button"],
        "bddl_goal": ["turnon", "flat_stove_1"],
        "predicate_source": "TurnOn.__call__ -> ObjectState.turn_on -> FlatStove.turn_on",
    },
    "put_the_bowl_on_the_plate": {
        "bodies": ["akita_black_bowl_1_main", "plate_1_main"],
        "bddl_goal": ["on", "akita_black_bowl_1", "plate_1"],
        "predicate_source": "On.__call__ -> plate ObjectState.check_ontop(bowl)",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()


def common(name: str, timestamp: str, project_sha: str, sources: Dict[str, Any]) -> Dict[str, Any]:
    return {"schema_version": 1, "manifest_name": name, "frozen_at_utc": timestamp,
            "project_git_sha": project_sha, "source_runs": sources["runs"],
            "source_file_sha256": sources["hashes"], "outcome_blind": True}


def body_pose(boundary: Dict[str, Any], names: List[str]) -> tuple:
    indexes = []
    for name in names:
        if name not in boundary["body_names"]:
            raise RuntimeError(f"audited body {name!r} absent")
        indexes.append(boundary["body_names"].index(name))
    positions = np.asarray(boundary["body_positions"], dtype=np.float64)[indexes]
    quaternions = np.asarray(boundary["body_quaternions"], dtype=np.float64)[indexes]
    return positions, quaternions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "experiments/exp3_time_indexed_q_criticality/manifests")
    args = parser.parse_args()
    t1 = args.t1_run.resolve(); reference = args.reference_run.resolve(); output = args.output.resolve()
    t1_metrics = json.loads((t1 / "metrics.json").read_text(encoding="utf-8"))
    if not t1_metrics.get("gate", {}).get("passed") or t1_metrics["gate"].get("selected_condition") != "D_INTEGRATION_CONTROLLER_ROBOT":
        raise RuntimeError("cannot freeze EXP3: supplied T1 corrected-D gate did not pass")
    output.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now(timezone.utc).isoformat()
    project_sha = current_sha()
    branch_path = REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/branch_times_reconciliation.json"
    ref_manifest_path = reference / "artifacts/reference_snapshots_manifest.json"
    limits_path = REPOSITORY_ROOT / "runs/exp2_r5_q_smoke_20260814T012633/artifacts/joint_limit_manifest.json"
    config_path = REPOSITORY_ROOT / "experiments/exp3_time_indexed_q_criticality/configs/exp3.json"
    sources = {"runs": {"reference": reference.name, "t1_condition_d": t1.name, "joint_limit_audit": "exp2_r5_q_smoke_20260814T012633"},
               "hashes": {"branch_manifest": sha256(branch_path), "reference_manifest": sha256(ref_manifest_path), "t1_metrics": sha256(t1 / "metrics.json"), "joint_limits": sha256(limits_path), "exp3_config": sha256(config_path)}}
    branch_source = json.loads(branch_path.read_text(encoding="utf-8"))
    reference_manifest = json.loads(ref_manifest_path.read_text(encoding="utf-8"))
    limits_list = json.loads(limits_path.read_text(encoding="utf-8"))
    limits = {x["task"]: x for x in limits_list}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    refs = {(x["task"], x["episode"]): x for x in reference_manifest["episodes"]}

    task_demo = {**common("task_demo_manifest", timestamp, project_sha, sources), "tasks": []}
    branch_out = {**common("branch_manifest", timestamp, project_sha, sources),
                  "selection_rule": "verbatim frozen EXP2 outcome-blind branch set", "branch_count": 108,
                  "trajectories": branch_source["trajectories"]}
    all_q, all_q_velocity, all_eef = [], [], []
    object_positions: Dict[str, List[np.ndarray]] = {task: [] for task in OBJECT_CHANNELS}
    boundary_cache: Dict[tuple, List[Dict[str, Any]]] = {}
    for trajectory in branch_source["trajectories"]:
        key = (trajectory["task"], trajectory["episode"])
        ref = refs[key]; ref_dir = reference / ref["relative_directory"]
        boundary_path = ref_dir / "boundaries.json"
        boundaries = json.loads(boundary_path.read_text(encoding="utf-8")); boundary_cache[key] = boundaries
        q = np.asarray([b["panda_arm_q"] for b in boundaries], dtype=np.float64)
        eef = np.asarray([b["eef_position"] for b in boundaries], dtype=np.float64)
        all_q.extend(q); all_eef.extend(eef)
        if len(q) > 1: all_q_velocity.extend(np.diff(q, axis=0) * config["control_frequency_hz"])
        for boundary in boundaries:
            pos, _ = body_pose(boundary, OBJECT_CHANNELS[trajectory["task"]]["bodies"])
            object_positions[trajectory["task"]].append(pos)
        task_demo["tasks"].append({"suite": trajectory["suite"], "task": trajectory["task"], "task_id": trajectory["task_id"],
            "episode": trajectory["episode"], "trajectory_length": trajectory["trajectory_length"],
            "reference_relative_directory": ref["relative_directory"], "trajectory_states_sha256": ref["trajectory_states_sha256"],
            "boundaries_sha256": sha256(boundary_path), "branch_count": len(trajectory["branches"])})

    seed_material = f"EXP3_DIRECTION_MANIFEST_V1|{project_sha}|{sources['hashes']['branch_manifest']}"
    master_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
    direction_rows = []
    for ti, trajectory in enumerate(branch_source["trajectories"]):
        joint = limits[trajectory["task"]]; ranges = np.asarray(joint["upper"]) - np.asarray(joint["lower"])
        boundaries = boundary_cache[(trajectory["task"], trajectory["episode"])]
        for bi, branch in enumerate(trajectory["branches"]):
            q0 = np.asarray(boundaries[int(branch["action_index"])]["panda_arm_q"])
            for di in range(config["directions_per_branch"]):
                child_seed = np.random.SeedSequence(master_seed, spawn_key=(ti, bi, di))
                rng = np.random.Generator(np.random.PCG64(child_seed))
                raw = rng.standard_normal(7); unit = raw / np.linalg.norm(raw)
                unsigned_delta = config["joint_range_fraction"] * ranges * unit
                valid = all(np.all(q0 + sign * unsigned_delta >= np.asarray(joint["lower"])) and np.all(q0 + sign * unsigned_delta <= np.asarray(joint["upper"])) for sign in config["signs"])
                if not valid: raise RuntimeError(f"frozen intervention violates limit at {trajectory['task']} {trajectory['episode']} {branch['action_index']}")
                direction_rows.append({"task": trajectory["task"], "episode": trajectory["episode"], "branch_time": branch["action_index"], "direction_index": di,
                    "seed_entropy": master_seed, "spawn_key": [ti, bi, di], "bit_generator": "PCG64",
                    "raw_direction": raw.tolist(), "unit_direction": unit.tolist(), "unsigned_delta_q": unsigned_delta.tolist(), "both_signs_within_joint_limits": valid})
    direction = {**common("direction_manifest", timestamp, project_sha, sources), "seed_material": seed_material,
                 "master_seed_uint64": master_seed, "prior_r5_seed_excluded": 20260814,
                 "generation": "SeedSequence(master_seed, spawn_key=(trajectory_ordinal, branch_ordinal, direction_index)); PCG64.standard_normal(7); L2 normalize",
                 "formula": "delta_q[j] = sign * 0.005 * verified_joint_range[j] * unit_direction[j]",
                 "direction_count": len(direction_rows), "signed_intervention_count": 2 * len(direction_rows), "directions": direction_rows}

    joint_range_norm = float(np.linalg.norm(np.asarray(limits_list[0]["upper"]) - np.asarray(limits_list[0]["lower"])))
    qvel_norms = np.linalg.norm(np.asarray(all_q_velocity), axis=1)
    eef_array = np.asarray(all_eef)
    eef_diagonal = float(np.linalg.norm(np.max(eef_array, axis=0) - np.min(eef_array, axis=0)))
    object_scales = {}
    for task, arrays in object_positions.items():
        arr = np.concatenate(arrays, axis=0)
        object_scales[task] = max(float(np.linalg.norm(np.max(arr, axis=0) - np.min(arr, axis=0))), 0.01)
    normalization = {**common("effect_normalization", timestamp, project_sha, sources),
        "reference_only": True, "continuous_denominators": {
            "arm_q_l2": {"value": joint_range_norm, "derivation": "L2 norm of audited Panda joint ranges"},
            "arm_qvel_l2": {"value": max(float(np.percentile(qvel_norms, 95)), 1.0), "derivation": "P95 reference finite-difference q speed at 20 Hz; 1 rad/s stability floor"},
            "eef_position_l2": {"value": max(eef_diagonal, 0.05), "derivation": "reference EEF workspace bounding-box diagonal; 5 cm stability floor"},
            "eef_orientation_geodesic": {"value": float(np.pi), "derivation": "physical SO(3) maximum"},
            "task_object_position_l2": {"value_by_task": object_scales, "derivation": "reference target-body bounding-box diagonal; 1 cm stability floor"},
            "task_object_orientation_geodesic_mean": {"value": float(np.pi), "derivation": "physical quaternion maximum"}},
        "binary_and_count_channels_not_normalized": ["contact_pair_symmetric_difference_count", "raw_contact_count_difference", "task_predicate_divergence", "terminal_success_flip"],
        "zero_noise_rule": "report raw matched-zero distributions; continuous exact-zero gate is <=1e-12 integration L2"}

    channel_schema = {**common("effect_channel_schema", timestamp, project_sha, sources), "channels": [
        {"name": "arm_q_l2", "role": "primary_component", "shape": 7}, {"name": "arm_qvel_l2", "role": "primary_component", "shape": 7},
        {"name": "eef_position_l2", "role": "primary_component", "shape": 3}, {"name": "eef_orientation_geodesic", "role": "primary_component", "units": "radian"},
        {"name": "task_object_position_l2", "role": "primary_component"}, {"name": "task_object_orientation_geodesic_mean", "role": "primary_component", "units": "radian"},
        {"name": "contact_pair_symmetric_difference_count", "role": "secondary"}, {"name": "raw_contact_count_difference", "role": "secondary"},
        {"name": "task_predicate_divergence", "role": "secondary"}, {"name": "terminal_success_flip", "role": "secondary"},
        {"name": "integration_state_l2", "role": "diagnostic_only"}],
        "task_object_audit": OBJECT_CHANNELS, "arm_state": {"qpos_indexes": limits_list[0]["qpos_indexes"], "joint_names": limits_list[0]["joint_names"]},
        "eef_source": "env.robots[0].eef_site_id -> sim.data.site_xpos/site_xmat", "contact_source": "all active sim.data.contact pairs with geom IDs and names",
        "predicate_source": "env.check_success(), exactly the parsed BDDL conjunction"}

    primary = {**common("primary_metric_spec", timestamp, project_sha, sources),
        "per_step_metric": "equal-weight arithmetic mean of the six separately normalized continuous physical channels",
        "components": ["arm_q_l2", "arm_qvel_l2", "eef_position_l2", "eef_orientation_geodesic", "task_object_position_l2", "task_object_orientation_geodesic_mean"],
        "intervention_summary": "arithmetic mean of per-step metric over every remaining action, including the first post-intervention transition",
        "remaining_horizon_control": "duration-normalized mean (sum divided by continuation length)",
        "branch_primary": "median across exactly 8 interventions (4 directions x 2 signs)",
        "meaningful_effect_threshold": 0.01, "threshold_interpretation": "one percent of the equal-weight normalized physical scale",
        "terminal_secondary": ["task_object_position_l2", "task_object_orientation_geodesic_mean", "task_predicate_divergence", "terminal_success_flip"]}

    events = []
    for trajectory in branch_source["trajectories"]:
        boundaries = boundary_cache[(trajectory["task"], trajectory["episode"])]
        for event_type, kind in (("contact_count_change", "first_maximum_contact_count_change"), ("gripper_sign_change", "first_gripper_command_sign_change")):
            hit = next((b for b in trajectory["branches"] if b["kind"] == kind and b.get("event_valid", True)), None)
            events.append({"task": trajectory["task"], "episode": trajectory["episode"], "event_type": event_type, "present": hit is not None,
                           "action_index": None if hit is None else hit["action_index"], "normalized_time": None if hit is None else hit["normalized_time"], "source": "frozen EXP2 reference branch audit"})
        pred = next((i for i, b in enumerate(boundaries) if b["task_success"]), None)
        events.append({"task": trajectory["task"], "episode": trajectory["episode"], "event_type": "task_predicate_first_true", "present": pred is not None,
                       "action_index": pred, "normalized_time": None if pred is None else pred / (trajectory["trajectory_length"] - 1), "source": "reference boundaries env.check_success"})
    event_manifest = {**common("event_manifest", timestamp, project_sha, sources), "event_window_normalized_radius": 0.075,
        "matching_rule": "a branch is in-window when absolute normalized-time distance to a present event is <=0.075",
        "absent_event_rule": "record absent and exclude that demo-event pair from enrichment; never substitute a fallback",
        "drawer_gripper_fallback_is_not_event": True, "events": events}

    sap = {**common("statistical_analysis_plan", timestamp, project_sha, sources),
        "inference_unit": "demonstration", "branch_aggregation": "median across 8; also report mean/P25/P75/min/max/fraction above 0.01",
        "topk_integer_mapping": {"n": 12, "top10": 2, "top20": 3, "top30": 4, "rule": "ceil(k*n)"},
        "all_zero_convention": {"top_mass": 0, "gini": 0, "normalized_entropy": 1, "flag": "all_zero=true"},
        "uniform_null": {"top10_mass": 2/12, "top20_mass": 3/12, "top30_mass": 4/12, "gini": 0, "normalized_entropy": 1},
        "cross_demo_alignment": "Spearman correlation on the ten frozen temporal-quantile branches ordered by nominal quantile; event branches excluded",
        "within_demo_direction_sign_robustness": "median pairwise Spearman correlation among eight 12-point curves; stable if >=0.5",
        "event_enrichment": "demo-stratified ratio of mean branch primary inside versus outside each frozen event window; permute branch times within demo",
        "outcome_relevance": "Spearman across branches for terminal object effect and predicate-divergence fraction; success flips reported descriptively",
        "hierarchical_bootstrap": "resample 3 tasks, then 3 demos within selected task, then 4 directions and 2 signs within branch; PCG64 seed 830031; 4000 resamples",
        "permutation": "shuffle branch-time labels within demo and direction/sign labels within branch; PCG64 seed 830032; 4000 resamples",
        "leave_one_demo_out": "repeat pooled concentration and temporal-rank summaries omitting each of 9 demonstrations",
        "multiple_comparisons": "primary decision uses preregistered top20 statistic only; secondary p-values use Benjamini-Hochberg FDR 0.05",
        "sensitivity": ["mean rather than median across eight", "top10/top30", "remaining-horizon maximum as secondary", "quantile-only curves"]}

    decision = {**common("scientific_decision_rule", timestamp, project_sha, sources),
        "strong_support": {"all_required": ["demo-median top20_mass >= 0.50", "hierarchical 95% CI lower bound for demo-median top20_mass > 0.25 uniform null", "at least 2 of 3 tasks have median cross-demo Spearman >= 0.50", "at least 6 of 9 demos have direction/sign robustness >= 0.50", "all leave-one-demo-out pooled top20_mass estimates >= 0.45"]},
        "partial_support": {"rule": "not strong, but demo-median top20_mass >=0.40 and hierarchical CI lower >0.25, or replicated rank agreement >=0.50 in exactly one task"},
        "broad_sensitivity": {"rule": "demo-median top20_mass <0.40 and >=80% of branches have >=50% of their eight interventions above 0.01"},
        "saturation_flag": {"rule": ">=95% of interventions exceed 0.01 and demo-median top20_mass <0.40"},
        "no_support": {"rule": "none of strong, partial, broad, or saturation conditions hold"},
        "priority": ["strong_support", "partial_support", "saturation_flag", "broad_sensitivity", "no_support"]}

    manifests = {"task_demo_manifest.json": task_demo, "branch_manifest.json": branch_out, "direction_manifest.json": direction,
        "effect_channel_schema.json": channel_schema, "effect_normalization.json": normalization, "primary_metric_spec.json": primary,
        "event_manifest.json": event_manifest, "statistical_analysis_plan.json": sap, "scientific_decision_rule.json": decision}
    for filename, value in manifests.items(): write_json(output / filename, value)
    digest = {name: sha256(output / name) for name in manifests}
    write_json(output / "manifest_hashes.json", {**common("manifest_hashes", timestamp, project_sha, sources), "manifests": digest})
    print(json.dumps({"output": str(output), "master_seed": master_seed, "directions": len(direction_rows), "interventions": 2 * len(direction_rows), "hashes": digest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
