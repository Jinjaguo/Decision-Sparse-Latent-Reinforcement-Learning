#!/usr/bin/env python
"""Freeze all outcome-blind EXP4 manifests after held-out reference gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from decision_sparse_rl.logging.run_directory import write_json  # noqa: E402
from decision_sparse_rl.metrics.exp4 import orthonormal_audit  # noqa: E402


OBJECT_CHANNELS = {
    "open_the_middle_drawer_of_the_cabinet": {"bodies": ["wooden_cabinet_1_cabinet_middle"], "bddl_goal": ["open", "wooden_cabinet_1_middle_region"], "predicate_source": "Open -> SiteObjectState.is_open -> WoodenCabinet.is_open"},
    "turn_on_the_stove": {"bodies": ["flat_stove_1_button"], "bddl_goal": ["turnon", "flat_stove_1"], "predicate_source": "TurnOn -> ObjectState.turn_on -> FlatStove.turn_on"},
    "put_the_bowl_on_the_plate": {"bodies": ["akita_black_bowl_1_main", "plate_1_main"], "bddl_goal": ["on", "akita_black_bowl_1", "plate_1"], "predicate_source": "On -> plate ObjectState.check_ontop(bowl)"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()


def common(name: str, timestamp: str, project_sha: str, sources: Dict[str, Any]) -> Dict[str, Any]:
    return {"schema_version": 1, "manifest_name": name, "frozen_at_utc": timestamp, "project_git_sha": project_sha, "source_runs": sources["runs"], "source_file_sha256": sources["hashes"], "outcome_blind": True, "outcome_blind_declaration": "No EXP4 q-intervention outcome existed or was read when this manifest was frozen."}


def monotonicity(values: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64); differences = np.diff(x)
    return {"nondecreasing": bool(np.all(differences >= -1e-12)), "decrease_count_gt_1e-6": int(np.sum(differences < -1e-6)), "largest_backtrack": float(max(0.0, -float(np.min(differences)))) if differences.size else 0.0, "minimum": float(np.min(x)), "maximum": float(np.max(x))}


def first_at_least(values: np.ndarray, threshold: float, start: int = 0) -> int:
    indexes = np.flatnonzero(values[start:] >= threshold)
    if not len(indexes):
        raise RuntimeError(f"reference milestone {threshold} absent after {start}")
    return int(start + indexes[0])


def articulation_progress(boundaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    q = np.asarray([x["progress_channels"]["joint_qpos"] for x in boundaries], dtype=np.float64)
    denominator = float(q[-1] - q[0])
    if abs(denominator) < 1e-9:
        raise RuntimeError("successful articulation reference has zero terminal displacement")
    raw = (q - q[0]) / denominator; clipped = np.clip(raw, 0.0, 1.0)
    return {"raw": raw, "clipped": clipped, "features": {"joint_qpos": q}, "milestones": {"start_index": 0, "terminal_index": len(q) - 1}, "derivation": {"start_qpos": float(q[0]), "terminal_qpos": float(q[-1]), "successful_direction_sign": int(np.sign(denominator)), "denominator": denominator}, "monotonicity_raw": monotonicity(raw), "monotonicity_clipped": monotonicity(clipped)}


def bowl_progress(boundaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    channels = [x["progress_channels"] for x in boundaries]
    distance = np.asarray([x["gripper_to_bowl_distance_m"] for x in channels])
    planar = np.asarray([x["bowl_to_plate_planar_distance_m"] for x in channels])
    bowl_z = np.asarray([x["bowl_body_z_m"] for x in channels]); lift = bowl_z - bowl_z[0]
    gap = np.asarray([x["bowl_bottom_minus_plate_top_m"] for x in channels])
    opening = np.asarray([x["gripper_opening_l1"] for x in channels])
    predicate = np.asarray([x["exact_task_predicate"] for x in channels], dtype=bool)
    lift_index = first_at_least(lift, 0.02)
    reach_index = int(np.argmin(distance[: lift_index + 1]))
    transport_index = first_at_least(-planar, -0.05, lift_index)
    predicate_indexes = np.flatnonzero(predicate)
    if not len(predicate_indexes):
        raise RuntimeError("successful bowl reference lacks exact On predicate")
    place_index = int(predicate_indexes[0])
    if not (reach_index <= lift_index <= transport_index <= place_index):
        raise RuntimeError(f"bowl milestones out of order: {reach_index, lift_index, transport_index, place_index}")
    raw = np.empty(len(boundaries), dtype=np.float64)
    reach_den = max(float(distance[0] - distance[reach_index]), 1e-9)
    transport_den = max(float(planar[lift_index] - 0.05), 1e-9)
    closed_opening = float(opening[reach_index]); terminal_opening = float(opening[place_index])
    release_den = terminal_opening - closed_opening
    for i in range(len(raw)):
        if i <= reach_index:
            raw[i] = 0.25 * (distance[0] - distance[i]) / reach_den
        elif i <= lift_index:
            raw[i] = 0.25 + 0.25 * lift[i] / 0.02
        elif i <= transport_index:
            raw[i] = 0.50 + 0.25 * (planar[lift_index] - planar[i]) / transport_den
        else:
            planar_score = (0.05 - planar[i]) / 0.02
            vertical_score = 1.0 - abs(gap[i]) / 0.03
            release_score = 1.0 if abs(release_den) < 1e-9 else (opening[i] - closed_opening) / release_den
            placement = np.mean(np.clip([planar_score, vertical_score, release_score], 0.0, 1.0))
            raw[i] = 1.0 if predicate[i] else 0.75 + 0.25 * min(float(placement), 0.999)
    clipped = np.clip(raw, 0.0, 1.0)
    return {"raw": raw, "clipped": clipped, "features": {"gripper_to_bowl_distance_m": distance, "bowl_lift_m": lift, "bowl_to_plate_planar_distance_m": planar, "bowl_bottom_minus_plate_top_m": gap, "gripper_opening_l1": opening, "exact_on_predicate": predicate.astype(int)}, "milestones": {"reach_index": reach_index, "lift_2cm_index": lift_index, "transport_5cm_index": transport_index, "exact_on_first_true_index": place_index}, "derivation": {"reach": "minimum gripper-bowl distance up to first 2 cm lift", "lift_threshold_m": 0.02, "transport_planar_threshold_m": 0.05, "on_planar_threshold_m": 0.03, "placement_vertical_scale_m": 0.03, "predicate_override": "exact On true -> progress 1"}, "monotonicity_raw": monotonicity(raw), "monotonicity_clipped": monotonicity(clipped)}


def to_jsonable_progress(progress: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(progress)
    result["raw"] = np.asarray(result["raw"]).tolist(); result["clipped"] = np.asarray(result["clipped"]).tolist()
    result["features"] = {name: np.asarray(value).tolist() for name, value in result["features"].items()}
    return result


def canonical_basis(seed: np.random.SeedSequence) -> Tuple[np.ndarray, Dict[str, Any], np.random.Generator]:
    rng = np.random.Generator(np.random.PCG64(seed)); raw = rng.standard_normal((7, 7)); q, r = np.linalg.qr(raw)
    signs = np.where(np.diag(r) < 0, -1.0, 1.0); q = q * signs
    if np.linalg.det(q) < 0: q[:, -1] *= -1
    audit = orthonormal_audit(q)
    if not audit["passed"]: raise RuntimeError(f"basis failed orthogonality: {audit}")
    return q, {"raw_gaussian_matrix": raw.tolist(), **audit}, rng


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--progress-audit-run", type=Path, required=True)
    parser.add_argument("--condition-d-gate-run", type=Path, required=True)
    parser.add_argument("--branch-candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "experiments/exp4_replicated_progress_criticality/manifests")
    args = parser.parse_args()
    reference = args.reference_run.resolve(); audit = args.progress_audit_run.resolve(); d_gate = args.condition_d_gate_run.resolve(); branch_run = args.branch_candidate_run.resolve(); output = args.output.resolve()
    if not json.loads((reference / "metrics.json").read_text())["gate"]["passed"]: raise RuntimeError("reference gate failed")
    if not json.loads((audit / "metrics.json").read_text())["gate"]["passed"]: raise RuntimeError("progress audit failed")
    if not json.loads((d_gate / "metrics.json").read_text())["gate"]["passed"]: raise RuntimeError("corrected-D reconciled gate failed")
    if not json.loads((branch_run / "metrics.json").read_text())["gate"]["passed"]: raise RuntimeError("branch candidate gate failed")
    output.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now(timezone.utc).isoformat(); project_sha = current_sha()
    ref_manifest_path = reference / "artifacts/reference_snapshots_manifest.json"; branch_path = branch_run / "artifacts/branch_candidates.json"; audit_path = audit / "artifacts/progress_runtime_audit.json"; d_path = d_gate / "artifacts/condition_d_reconciled_gate.json"
    limits_path = REPOSITORY_ROOT / "runs/exp2_r5_q_smoke_20260814T012633/artifacts/joint_limit_manifest.json"; config_path = REPOSITORY_ROOT / "experiments/exp4_replicated_progress_criticality/configs/exp4.json"; exp3_norm_path = REPOSITORY_ROOT / "experiments/exp3_time_indexed_q_criticality/manifests/effect_normalization.json"
    sources = {"runs": {"reference": reference.name, "progress_audit": audit.name, "condition_d_gate": d_gate.name, "branch_candidates": branch_run.name, "joint_limit_audit": "exp2_r5_q_smoke_20260814T012633", "legacy_exp3": "exp3_t6_full_20260814T022200"}, "hashes": {"reference_manifest": sha256(ref_manifest_path), "branch_candidates": sha256(branch_path), "progress_runtime_audit": sha256(audit_path), "condition_d_gate": sha256(d_path), "joint_limits": sha256(limits_path), "exp4_config": sha256(config_path), "exp3_normalization": sha256(exp3_norm_path)}}
    ref_manifest = json.loads(ref_manifest_path.read_text()); branch_source = json.loads(branch_path.read_text()); audit_data = json.loads(audit_path.read_text()); config = json.loads(config_path.read_text()); limits_list = json.loads(limits_path.read_text()); limits = {x["task"]: x for x in limits_list}; refs = {(x["task"], x["episode"]): x for x in ref_manifest["episodes"]}
    boundary_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}; progress_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    task_demo_rows = []
    for trajectory in branch_source["trajectories"]:
        key = (trajectory["task"], trajectory["episode"]); ref = refs[key]; directory = reference / ref["relative_directory"]; boundary_path = directory / "boundaries.json"; boundaries = json.loads(boundary_path.read_text()); boundary_cache[key] = boundaries
        progress = bowl_progress(boundaries) if trajectory["task"] == "put_the_bowl_on_the_plate" else articulation_progress(boundaries); progress_cache[key] = progress
        task_demo_rows.append({"suite": trajectory["suite"], "task": trajectory["task"], "task_id": trajectory["task_id"], "episode": trajectory["episode"], "trajectory_length": trajectory["trajectory_length"], "reference_relative_directory": ref["relative_directory"], "trajectory_states_sha256": ref["trajectory_states_sha256"], "boundaries_sha256": sha256(boundary_path), "branch_count": len(trajectory["branches"]), "reference_success": ref["success"], "progress": to_jsonable_progress(progress)})
    task_demo = {**common("task_demo_manifest", timestamp, project_sha, sources), "cohort": "held-out demos 3-9 only", "task_demo_count": len(task_demo_rows), "tasks": task_demo_rows}
    reference_validation = {**common("reference_validation_manifest", timestamp, project_sha, sources), "expected_count": 21, "reference_gate": ref_manifest["gate"], "condition_d_gate": json.loads(d_path.read_text())["gate"], "policy_boundary": "corrected EXP2 pre-policy Condition D including PandaGripper.current_action", "no_substitutions": True}
    branch_trajectories = []
    for trajectory in branch_source["trajectories"]:
        progress = progress_cache[(trajectory["task"], trajectory["episode"])]
        branches = []
        for branch in trajectory["branches"]:
            index = int(branch["action_index"]); branches.append({**branch, "physical_progress_raw": float(progress["raw"][index]), "physical_progress_clipped": float(progress["clipped"][index]), "progress_monotonicity_violation_before_branch": bool(np.any(np.diff(progress["clipped"][: index + 1]) < -1e-6))})
        branch_trajectories.append({**trajectory, "branches": branches})
    branch_manifest = {**common("branch_manifest", timestamp, project_sha, sources), "selection_rule": "same ten EXP2 normalized-time quantiles plus frozen contact and gripper audit slots", "branch_count": 252, "trajectories": branch_trajectories}
    progress_schema = {**common("progress_channel_schema", timestamp, project_sha, sources), "runtime_identifier_audit": audit_data["records"], "drawer": {"raw": "(q(t)-q(0))/(q(T)-q(0))", "joint": "wooden_cabinet_1_middle_level", "predicate": "q < -0.14"}, "stove": {"raw": "(q(t)-q(0))/(q(T)-q(0))", "joint": "flat_stove_1_button", "predicate": "q >= 0.5"}, "bowl": {"features": ["gripper_to_bowl_distance_m", "bowl_lift_m", "bowl_to_plate_planar_distance_m", "bowl_bottom_minus_plate_top_m", "gripper_opening_l1", "exact_on_predicate"], "phase_order": ["reach", "grasp_lift", "transport", "place_release"], "thresholds": {"lift_m": 0.02, "transport_planar_m": 0.05, "on_planar_m": 0.03, "vertical_scale_m": 0.03}, "geometry": "runtime union AABB of active collision primitives"}, "stored_variants": ["raw_unclipped", "clipped_0_1", "monotonicity_audit"], "reference_only": True}
    alignment = {**common("progress_alignment_spec", timestamp, project_sha, sources), "normalized_time_baseline": "ten temporal_quantile branches ordered by frozen nominal quantile; audit slots excluded", "physical_progress_grid": np.linspace(0.0, 1.0, 21).tolist(), "interpolation": "piecewise linear along original temporal branch order using first crossing; never sort progress", "nonmonotone_rule": "preserve temporal identity and use first adjacent crossing", "duplicate_rule": "first temporal occurrence", "missing_crossing_rule": "nearest observed progress; equal-distance tie chooses earlier time", "alignment_comparison": "paired task/demo-pair Spearman and Kendall differences with task/demo-cluster bootstrap and within-task paired permutation"}
    seed_material = f"EXP4_DIRECTION_BASIS_V1|{project_sha}|{sources['hashes']['branch_candidates']}"; master_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little"); direction_rows = []; basis_records = []
    for ti, trajectory in enumerate(branch_trajectories):
        joint = limits[trajectory["task"]]; ranges = np.asarray(joint["upper"]) - np.asarray(joint["lower"]); lower = np.asarray(joint["lower"]); upper = np.asarray(joint["upper"]); boundaries = boundary_cache[(trajectory["task"], trajectory["episode"])]
        for bi, branch in enumerate(trajectory["branches"]):
            seed = np.random.SeedSequence(master_seed, spawn_key=(ti, bi)); basis, basis_audit, rng = canonical_basis(seed); order = rng.permutation(7); random_raw = rng.standard_normal(7); random_direction = random_raw / np.linalg.norm(random_raw); q0 = np.asarray(boundaries[int(branch["action_index"])]["panda_arm_q"])
            basis_records.append({"task": trajectory["task"], "episode": trajectory["episode"], "branch_time": branch["action_index"], "spawn_key": [ti, bi], "basis_matrix_columns": basis.tolist(), "execution_order_basis_indices": order.tolist(), "random_raw_direction": random_raw.tolist(), "random_unit_direction": random_direction.tolist(), **basis_audit})
            sequence = [(int(logical), basis[:, int(logical)], "basis", int(position)) for position, logical in enumerate(order)] + [(7, random_direction, "heldout_random", 7)]
            for direction_index, vector, role, execution_position in sequence:
                unsigned = config["joint_range_fraction"] * ranges * vector; valid = all(np.all(q0 + sign * unsigned >= lower) and np.all(q0 + sign * unsigned <= upper) for sign in config["signs"])
                if not valid: raise RuntimeError(f"joint limit failure before outcomes: {trajectory['task']} {trajectory['episode']} {branch['action_index']} d{direction_index}")
                direction_rows.append({"task": trajectory["task"], "episode": trajectory["episode"], "branch_time": branch["action_index"], "direction_index": direction_index, "direction_role": role, "execution_position": execution_position, "seed_entropy": master_seed, "spawn_key": [ti, bi], "unit_direction_scaled_coordinates": vector.tolist(), "unsigned_delta_q": unsigned.tolist(), "both_signs_within_joint_limits": True})
    directions = {**common("direction_basis_manifest", timestamp, project_sha, sources), "seed_material": seed_material, "master_seed_uint64": master_seed, "different_from_exp3_seed": master_seed != 4223443001079176503, "bit_generator": "PCG64", "basis_generation": "PCG64 Gaussian 7x7 -> QR -> positive R diagonal -> determinant +1", "coordinate_system": "joint-range-scaled Panda arm q", "epsilon_fraction": config["joint_range_fraction"], "basis_count": len(basis_records), "direction_count": len(direction_rows), "signed_intervention_count": 2 * len(direction_rows), "bases": basis_records, "directions": direction_rows}
    normalization = json.loads(exp3_norm_path.read_text()); normalization = {**common("effect_normalization", timestamp, project_sha, sources), "inheritance": "numerical denominator values copied exactly from frozen EXP3 for direct comparability", "source_exp3_manifest_sha256": sha256(exp3_norm_path), "reference_only": True, "continuous_denominators": normalization["continuous_denominators"], "operator_signed_vector": {"arm_q": "elementwise divide by audited joint ranges", "arm_qvel": "divide by EXP3 arm_qvel_l2 scalar", "eef_position": "divide by EXP3 eef_position scalar", "eef_orientation_rotvec": "divide by pi", "task_object_position": "divide by EXP3 task-specific object-position scalar", "task_object_orientation_rotvec": "divide by pi"}}
    channels = {**common("effect_channel_schema", timestamp, project_sha, sources), "channels": [{"name": name, "role": "EXP3_primary_component"} for name in ["arm_q_l2", "arm_qvel_l2", "eef_position_l2", "eef_orientation_geodesic", "task_object_position_l2", "task_object_orientation_geodesic_mean"]] + [{"name": name, "role": "secondary_discrete"} for name in ["contact_pair_symmetric_difference_count", "raw_contact_count_difference", "task_predicate_divergence", "terminal_success_flip"]] + [{"name": "integration_state_l2", "role": "diagnostic_only"}, {"name": "signed_normalized_physical_output_vector", "role": "finite_difference_operator"}], "task_object_audit": OBJECT_CHANNELS, "signed_vector_order": ["arm_q[7]", "arm_qvel[7]", "eef_position[3]", "eef_orientation_rotvec[3]", "task_object_position[3*body_count]", "task_object_orientation_rotvec[3*body_count]"]}
    primary = {**common("primary_metric_spec", timestamp, project_sha, sources), "per_intervention_effect": "exact EXP3 equal-weight six-component per-step score averaged over every remaining step", "direction_pair": {"S_j": "(E_plus + E_minus)/2", "A_j": "abs(E_plus-E_minus)/(E_plus+E_minus+1e-15)", "numerical_epsilon": 1e-15}, "branch_primary": "S_RMS = sqrt(mean(S_j^2)) over exactly seven orthonormal basis directions", "heldout_random_excluded_from_primary": True, "historical_exp3_metric": "median across all 16 EXP4 direction/sign intervention effects, reported only as a comparator", "remaining_horizon": ["mean primary", "remaining steps", "normalized remaining horizon", "fixed 20-step diagnostic window"], "meaningful_effect_threshold": 0.01}
    operator = {**common("operator_analysis_spec", timestamp, project_sha, sources), "validity": "signed continuous normalized physical vector only; integration state excluded", "column": "mean_remaining_horizon(y_plus - y_minus)/(2*0.005) for each of seven basis directions", "jacobian_coordinates": "joint-range-scaled q input", "gram": "G=J^T J", "spectral_outputs": ["eigenvalues", "singular_values", "spectral_norm", "frobenius_norm", "effective_rank", "condition_number_nonzero", "top_eigenvalue_share"], "discrete_channels": "contact/predicate analyzed separately and never inserted into J", "invalid_rule": "if signed-vector shape/finite/antithetic coverage fails, mark invalid and omit operator plot; never fabricate"}
    events = []
    for trajectory in branch_trajectories:
        key = (trajectory["task"], trajectory["episode"]); boundaries = boundary_cache[key]; progress = progress_cache[key]
        for event_type, kind in (("contact_count_change", "first_maximum_contact_count_change"), ("gripper_sign_change", "first_gripper_command_sign_change")):
            hit = next((b for b in trajectory["branches"] if b["kind"] == kind and b.get("event_valid", True)), None); events.append({"task": key[0], "episode": key[1], "event_type": event_type, "present": hit is not None, "action_index": None if hit is None else hit["action_index"], "normalized_time": None if hit is None else hit["normalized_time"], "physical_progress": None if hit is None else hit["physical_progress_clipped"]})
        pred = next((i for i, b in enumerate(boundaries) if b["task_success"]), None); events.append({"task": key[0], "episode": key[1], "event_type": "task_predicate_first_true", "present": pred is not None, "action_index": pred, "normalized_time": None if pred is None else pred / (len(boundaries) - 1), "physical_progress": None if pred is None else float(progress["clipped"][pred])})
    event_manifest = {**common("event_manifest", timestamp, project_sha, sources), "time_window_radius": 0.075, "progress_window_radius": 0.075, "events": events, "adjustment": "remaining horizon + early/middle/late phase + demo fixed effects; within-demo permutation; BH FDR 0.05", "secondary_only": True}
    sap = {**common("statistical_analysis_plan", timestamp, project_sha, sources), "inference_unit": "demonstration", "primary_cohort": "held-out demos 3-9; EXP3 demos 0-2 are legacy comparison only", "topk_integer_mapping": {"n": 12, "top10": 2, "top20": 3, "top30": 4, "rule": "ceil(k*n), exact EXP3 convention"}, "all_zero_convention": {"top_mass": 0, "gini": 0, "normalized_entropy": 1, "flag": True}, "uniform_null": {"top10_mass": 2/12, "top20_mass": 3/12, "top30_mass": 4/12}, "replication": ["Spearman", "Kendall tau-b", "ICC(2,1)", "top-k Jaccard"], "alignment_test": "paired demo-pair delta rho (progress minus time), task/demo cluster bootstrap and within-task sign permutation", "bootstrap": {"seed": 940031, "resamples": config["bootstrap_resamples"], "hierarchy": "task -> demo -> seven basis directions -> paired signs"}, "permutation": {"seed": 940032, "resamples": config["permutation_resamples"]}, "variance_decomposition": "hierarchical method-of-moments ANOVA for task/demo/progress-bin/direction/sign/residual if mixed-effects unavailable", "lodo": True, "leave_one_task_out": True, "multiple_comparisons": "BH FDR 0.05 within alignment, event, component, and task-specific secondary families"}
    gpu = {**common("gpu_analysis_spec", timestamp, project_sha, sources), "required_device": "cuda:0 RTX 4090", "simulator": "CPU only", "formal_gpu_operations": ["direction tensor aggregation", "Gram matrices and eigenspectra", "progress interpolation", "bootstrap", "permutation"], "dtype": "float64", "equivalence_calibration": ["S_RMS aggregation", "top-k mass", "Gini", "normalized entropy", "Spearman rank inputs", "bootstrap samples", "Gram matrices", "singular values"], "tolerances": {"exact_rank_indices": True, "scalar_atol": 1e-12, "scalar_rtol": 1e-10, "matrix_atol": 1e-11, "spectrum_atol": 1e-10}, "failure_rule": "stop formal GPU path, diagnose, and retain CPU as truth; record every fallback"}
    decision = {**common("scientific_decision_rule", timestamp, project_sha, sources), "strong_replicated_progress_aligned_sparsity": {"all_required": ["heldout demo median top20_mass >= 0.50", "hierarchical 95% CI lower > 0.25", "progress alignment median paired rho improvement >= 0.15", "at least 2 of 3 tasks progress median rho >= 0.50", "at least 70% of 21 demos direction robustness >= 0.50", "LODO minimum top20_mass >= 0.45", "heldout random direction versus basis aggregate median rho >= 0.50"]}, "classification_priority": ["strong_replicated_progress_aligned_sparsity", "replicated_nonuniformity_without_aligned_sparse_times", "task_specific_progress_alignment", "direction_instability", "uniform_or_broad_sensitivity", "no_support"], "interpretation_constraints": {"no_sparse_policy_training_in_exp4": True, "no_event_trigger_claim_without_adjusted_FDR_replication": True, "no_universal_time_claim_from_task_specific_result": True}}
    manifests = {"task_demo_manifest.json": task_demo, "reference_validation_manifest.json": reference_validation, "branch_manifest.json": branch_manifest, "progress_channel_schema.json": progress_schema, "progress_alignment_spec.json": alignment, "direction_basis_manifest.json": directions, "effect_channel_schema.json": channels, "effect_normalization.json": normalization, "primary_metric_spec.json": primary, "operator_analysis_spec.json": operator, "event_manifest.json": event_manifest, "statistical_analysis_plan.json": sap, "gpu_analysis_spec.json": gpu, "scientific_decision_rule.json": decision}
    for filename, value in manifests.items(): write_json(output / filename, value)
    hashes = {name: sha256(output / name) for name in manifests}; write_json(output / "manifest_hashes.json", {**common("manifest_hashes", timestamp, project_sha, sources), "manifests": hashes})
    print(json.dumps({"output": str(output), "project_sha": project_sha, "master_seed": master_seed, "manifests": len(manifests), "directions": len(direction_rows), "interventions": 2 * len(direction_rows), "hashes": hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
