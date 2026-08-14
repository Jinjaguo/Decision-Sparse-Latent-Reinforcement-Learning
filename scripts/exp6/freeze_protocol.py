#!/usr/bin/env python
"""Freeze calibration or formal EXP6 manifests before observing that phase's outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402
from scripts.exp4.freeze_protocol import articulation_progress, bowl_progress, canonical_basis  # noqa: E402

TASKS = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate", "turn_on_the_stove"]
LABELS = {0.0003125: "r0003125", 0.000625: "r000625", 0.00125: "r00125", 0.0025: "r0025", 0.005: "r005"}
CONTACT_TOKEN = {TASKS[0]: "wooden_cabinet_1", TASKS[1]: "akita_black_bowl_1", TASKS[2]: "flat_stove_1"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common(kind: str, timestamp: str, project: str, sources: dict) -> dict:
    return {"schema_version": 1, "manifest_type": kind, "frozen_at_utc": timestamp, "project_sha": project, "sources": sources, "outcome_blind": True, "formal_q_outcomes_observed": False}


def calibration_trajectories(full: list[dict], calibration: list[dict]) -> list[dict]:
    chosen = {(row["task"], row["episode"], int(row["branch_time"])) for row in calibration}
    result = []
    for trajectory in full:
        branches = [branch for branch in trajectory["branches"] if (trajectory["task"], trajectory["episode"], int(branch["action_index"])) in chosen]
        if branches:
            result.append({**trajectory, "branches": branches})
    return result


def exact_pairs(boundary: dict) -> list[str]:
    return sorted({"|".join(sorted((str(x["geom1_name"]), str(x["geom2_name"])))) for x in boundary["contact_pairs"]})


def replacement_metadata(boundaries: list[dict], task: str, index: int, original: dict) -> dict:
    boundary = boundaries[index]; command = float(boundary["progress_channels"]["gripper_command"])
    grip = "negative" if command < -0.5 else "positive" if command > 0.5 else "neutral"; token = CONTACT_TOKEN[task]
    contact = any(((token in pair.split("|", 1)[0] and "gripper0_" in pair.split("|", 1)[1]) or (token in pair.split("|", 1)[1] and "gripper0_" in pair.split("|", 1)[0])) for pair in exact_pairs(boundary))
    predicate = bool(boundary["progress_channels"]["exact_task_predicate"]); progress = bowl_progress(boundaries)["clipped"] if task == TASKS[1] else articulation_progress(boundaries)["clipped"]
    return {**original, "action_index": index, "branch_time": index, "normalized_time": index / max(len(boundaries) - 1, 1), "physical_progress_raw": float(progress[index]), "physical_progress_clipped": float(progress[index]), "reference_contact_state": contact, "reference_gripper_state": grip, "reference_predicate_state": predicate, "reference_contact_pairs": exact_pairs(boundary), "reference_stratum": f"contact={int(contact)}|gripper={grip}|predicate={int(predicate)}", "selection_distance": abs(index - int(original["action_index"])) / max(len(boundaries) - 1, 1), "deterministic_replacement_reason": f"nearest unused reference boundary satisfying all frozen EXP6 joint-limit checks; replaced action {original['action_index']}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("calibration", "formal"), required=True)
    parser.add_argument("--branch-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--calibration-audit-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run = create_run_directory(args.run_root, args.run_id); out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"manifest directory already exists: {out}")
    out.mkdir(parents=True)
    timestamp = datetime.now(timezone.utc).isoformat(); project = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config_path = ROOT / "experiments/exp6_radius_convergence/configs/exp6.json"; config = json.loads(config_path.read_text())
    branch_path = args.branch_run.resolve() / "artifacts/exp6_branch_candidates.json"; calibration_path = args.branch_run.resolve() / "artifacts/calibration_branch_candidates.json"
    candidate = json.loads(branch_path.read_text()); calibration = json.loads(calibration_path.read_text())
    reference = args.reference_run.resolve(); reference_manifest_path = reference / "artifacts/reference_snapshots_manifest.json"; reference_manifest = json.loads(reference_manifest_path.read_text())
    optional_admitted = False; calibration_decision = None
    if args.phase == "formal":
        if args.calibration_audit_run is None:
            raise ValueError("formal freeze requires --calibration-audit-run")
        decision_path = args.calibration_audit_run.resolve() / "artifacts/resolution_gate.json"; calibration_decision = json.loads(decision_path.read_text())
        if not calibration_decision["primary_smallest_resolvable"]:
            raise RuntimeError("0.000625 calibration failed; formal EXP6 must stop")
        optional_admitted = bool(calibration_decision["optional_radius_admitted"])
    radii = ([config["optional_radius"]] if optional_admitted or args.phase == "calibration" else []) + config["primary_radii"]
    trajectories = candidate["trajectories"] if args.phase == "formal" else calibration_trajectories(candidate["trajectories"], calibration["branches"])
    sources = {"runs": {"reference": reference.name, "branch_subset": args.branch_run.resolve().name, "calibration_audit": None if args.calibration_audit_run is None else args.calibration_audit_run.resolve().name}, "hashes": {"reference_manifest": sha(reference_manifest_path), "branch_candidates": sha(branch_path), "calibration_candidates": sha(calibration_path), "config": sha(config_path)}}
    manifests = {}
    manifests["exp6_cohort_manifest.json"] = {**common("exp6_cohort_manifest", timestamp, project, sources), "trajectory_count": 30, "cohort": [{"task": row["task"], "episode": row["episode"]} for row in candidate["trajectories"]], "drawer_demo17_preserved_as_failed_qualification": True}
    branch_manifest = {**common("exp6_branch_manifest", timestamp, project, sources), "phase": args.phase, "branches_per_demo": 8 if args.phase == "formal" else None, "trajectory_count": len(trajectories), "branch_count": sum(len(row["branches"]) for row in trajectories), "selection_rule": candidate["selection_rule"], "trajectories": trajectories}
    manifests["exp6_branch_manifest.json"] = branch_manifest; manifests["branch_manifest.json"] = {**branch_manifest, "manifest_type": "branch_manifest"}
    limits_list = json.loads((ROOT / "runs/exp2_r5_q_smoke_20260814T012633/artifacts/joint_limit_manifest.json").read_text()); limits = {row["task"]: row for row in limits_list}
    refs = {(row["task"], row["episode"]): row for row in reference_manifest["episodes"]}
    seed_material = f"EXP6_DIRECTIONS_V1|{args.phase}|{project}|{sha(branch_path)}"; master_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
    directions, heldout = [], []
    for trajectory_index, trajectory in enumerate(trajectories):
        boundaries = json.loads((reference / refs[(trajectory["task"], trajectory["episode"])]["relative_directory"] / "boundaries.json").read_text()); limit = limits[trajectory["task"]]; span = np.asarray(limit["upper"]) - np.asarray(limit["lower"])
        used = {int(branch["action_index"]) for branch in trajectory["branches"]}
        for branch_index, branch in enumerate(trajectory["branches"]):
            basis, _, rng = canonical_basis(np.random.SeedSequence(master_seed, spawn_key=(trajectory_index, branch_index))); random = rng.standard_normal(7); random /= np.linalg.norm(random); q = np.asarray(boundaries[branch["action_index"]]["panda_arm_q"])
            vectors = [basis[:, i] for i in range(7)] + [random]
            def admissible(candidate_q: np.ndarray) -> bool:
                return all(np.all(candidate_q + sign * float(radius) * span * vector >= limit["lower"]) and np.all(candidate_q + sign * float(radius) * span * vector <= limit["upper"]) for radius in radii for vector in vectors for sign in (-1, 1))
            if not admissible(q):
                original_index = int(branch["action_index"]); replacement = None
                for candidate_index in sorted((index for index in range(len(boundaries)) if index not in used), key=lambda index: (abs(index - original_index), index)):
                    candidate_q = np.asarray(boundaries[candidate_index]["panda_arm_q"])
                    if admissible(candidate_q):
                        replacement = replacement_metadata(boundaries, trajectory["task"], candidate_index, branch); q = candidate_q; break
                if replacement is None:
                    raise RuntimeError(f"no joint-limit-valid replacement at {trajectory['task']}/{trajectory['episode']}/{original_index}")
                used.remove(original_index); used.add(int(replacement["action_index"])); trajectory["branches"][branch_index] = replacement; branch = replacement
            for radius_index, radius in enumerate(radii):
                for direction_index, vector, role in [(i, basis[:, i], "basis") for i in range(7)] + [(7, random, "heldout_random")]:
                    delta = float(radius) * span * vector
                    if not all(np.all(q + sign * delta >= limit["lower"]) and np.all(q + sign * delta <= limit["upper"]) for sign in (-1, 1)):
                        raise RuntimeError(f"joint-limit reconciliation bug at {trajectory['task']}/{trajectory['episode']}/{branch['action_index']}")
                    row = {"task": trajectory["task"], "episode": trajectory["episode"], "branch_time": branch["action_index"], "radius_fraction": radius, "radius_label": LABELS[float(radius)], "direction_index": direction_index, "direction_role": role, "execution_position": radius_index * 8 + direction_index, "unit_direction_scaled_coordinates": vector.tolist(), "unsigned_delta_q": delta.tolist(), "both_signs_within_joint_limits": True}
                    directions.append(row)
                    if role == "heldout_random": heldout.append(row)
    manifests["radius_manifest.json"] = {**common("radius_manifest", timestamp, project, sources), "phase": args.phase, "primary_radii": config["primary_radii"], "optional_radius": config["optional_radius"], "optional_radius_admitted": optional_admitted, "active_radii": radii, "labels": {LABELS[float(radius)]: radius for radius in radii}, "expected_signed_interventions": 2 * len(directions) * (2 if args.phase == "calibration" else 1)}
    manifests["direction_basis_manifest.json"] = {**common("direction_basis_manifest", timestamp, project, sources), "seed_material": seed_material, "master_seed_uint64": master_seed, "bit_generator": "PCG64", "directions": directions, "direction_rows": len(directions), "repeats_per_signed_intervention": 2 if args.phase == "calibration" else 1}
    manifests["heldout_direction_manifest.json"] = {**common("heldout_direction_manifest", timestamp, project, sources), "role": "eighth direction excluded from fitted seven-column operator", "directions": heldout}
    manifests["numerical_calibration_manifest.json"] = {**common("numerical_calibration_manifest", timestamp, project, sources), "calibration_branches": calibration["branches"], "zero_repetitions": 4, "signed_intervention_repetitions": 2, "gates": {"non_arm_integration_linf_max": 1e-12, "q_injection_atol": config["q_injection_atol"], "all_finite": True, "repeat_scalar_atol": config["repeat_scalar_atol"], "repeat_vector_atol": config["repeat_vector_atol"], "repeat_operator_atol": config["repeat_operator_atol"], "direction_sign_rank_deterministic": True, "signal_to_floor_min": 100.0}, "formal_calibration_decision": calibration_decision}
    manifests["zero_floor_spec.json"] = {**common("zero_floor_spec", timestamp, project, sources), "floors": ["integration-state", "signed-physical-output", "scalar-effect", "operator-spectral"], "upper_bound": "maximum repeated-zero discrepancy", "exact_zero_preserved": True, "ratio_resolution_constant": config["resolution_constant"]}
    exp4 = ROOT / "experiments/exp4_replicated_progress_criticality/manifests"
    for name in ("effect_channel_schema.json", "effect_normalization.json"):
        manifests[name] = {**json.loads((exp4 / name).read_text()), **common(name[:-5], timestamp, project, sources)}
    manifests["signed_output_vector_spec.json"] = {**common("signed_output_vector_spec", timestamp, project, sources), "inherited_exactly_from": "EXP5", "order": ["arm_q[7]", "arm_qvel[7]", "eef_position[3]", "eef_orientation_rotvec[3]", "task_object_position[3*n]", "task_object_orientation_rotvec[3*n]"], "aggregation": "duration-normalized remaining-horizon mean"}
    manifests["primary_metric_spec.json"] = {**common("primary_metric_spec", timestamp, project, sources), "branch_scalar": "RMS of antithetic basis scalar effects", "meaningful_effect_threshold": config["meaningful_effect_threshold"]}
    manifests["operator_metric_spec.json"] = {**common("operator_metric_spec", timestamp, project, sources), "operator": "J_r[:,j]=(y_plus-y_minus)/(2r)", "gram": "J_r.T@J_r", "outputs": ["spectral_norm", "frobenius_norm", "leading_eigenvalue_share", "effective_rank", "top1_projector", "top2_projector"], "projector_similarity": "1-||Pa-Pb||F/sqrt(2k)"}
    manifests["asymmetry_metric_spec.json"] = {**common("asymmetry_metric_spec", timestamp, project, sources), "definition": "||response_plus+response_minus||/(||response_plus||+||response_minus||+1e-12)", "threshold": 0.25}
    manifests["trust_region_spec.json"] = {**common("trust_region_spec", timestamp, project, sources), "largest_adjacent_interval_all_required": {"top1_similarity_min": 0.80, "top2_similarity_min": 0.75, "relative_spectral_discrepancy_max": 0.20, "antithetic_sign_asymmetry_max": 0.25, "heldout_vector_relative_error_max": 0.35, "signal_to_floor_min": 100.0}, "unresolved_rule": "no adjacent pair passes"}
    contact_tokens = {TASKS[0]: ["wooden_cabinet_1", "gripper0_"], TASKS[1]: ["akita_black_bowl_1", "gripper0_"], TASKS[2]: ["flat_stove_1", "gripper0_"]}
    manifests["contact_mode_schema.json"] = {**common("contact_mode_schema", timestamp, project, sources), "exact_named_pairs": True, "task_robot_contact_tokens": contact_tokens, "reference_only_freeze": True, "offsets_one_based": [1, 3, 5, 10]}
    manifests["contact_divergence_spec.json"] = {**common("contact_divergence_spec", timestamp, project, sources), "compare": "exact task-relevant named pair sets between plus and minus continuations", "outputs": ["same_step_1", "same_through_3", "same_through_5", "same_through_10", "first_divergence_step", "exact_pair_set_changes"]}
    manifests["statistical_analysis_plan.json"] = {**common("statistical_analysis_plan", timestamp, project, sources), "independent_unit": "demonstration", "bootstrap": {"resamples": 4000, "seed": 960031, "hierarchy": "task then demonstration"}, "permutation": {"resamples": 4000, "seed": 960032, "unit": "demonstration"}, "multiple_comparisons": "BH FDR 0.05", "H1": "at least 70% demos median smallest-primary top1 >=0.80 and hierarchical CI lower >0.65", "H2": "at least 70% branches smallest-primary spectral discrepancy <=0.20", "H3": "smallest validated demo-median heldout rho >=0.65 and median vector error <=0.35", "H4": "contact-divergent convergence-failure excess CI excludes zero and BH q<0.05"}
    manifests["gpu_analysis_spec.json"] = {**common("gpu_analysis_spec", timestamp, project, sources), "device": "cuda:0 RTX 4090", "dtype": "float64", "simulator": "CPU", "no_automatic_cpu_fallback": True, "equivalence_tolerances": {"scalar_atol": 1e-12, "matrix_atol": 1e-11, "spectrum_atol": 1e-10, "bootstrap_atol": 1e-10}}
    manifests["scientific_decision_rule.json"] = {**common("scientific_decision_rule", timestamp, project, sources), "classification_priority": ["small_radius_local_operator_converges", "contact_mode_conditioned_convergence", "numerical_resolution_prevents_local_limit_test", "nonsmooth_response_persists_below_exp5_radius", "no_support"]}
    for name, value in manifests.items(): write_json(out / name, value)
    hashes = {name: sha(out / name) for name in manifests}; write_json(out / "manifest_hashes.json", {**common("manifest_hashes", timestamp, project, sources), "manifests": hashes})
    shutil.copy2(out / "manifest_hashes.json", run / "artifacts/manifest_hashes.json")
    branch_count = sum(len(row["branches"]) for row in trajectories); expected = 6 if args.phase == "calibration" else 240
    metrics = {"run_id": args.run_id, "status": "completed", "phase": args.phase, "gate": {"passed": branch_count == expected and len(directions) == expected * len(radii) * 8}, "branches": branch_count, "radii": radii, "direction_rows": len(directions), "planned_signed_executions": 2 * len(directions) * (2 if args.phase == "calibration" else 1), "manifest_count": len(manifests)}
    write_run_record(run, config={"stage": f"EXP6 {args.phase} protocol freeze", "output": str(out)}, command=" ".join(sys.argv), environment={"python": sys.version, "numpy": np.__version__}, git_state={"project": git_record(ROOT)}, metrics=metrics, stdout=json.dumps(metrics), stderr="")
    print(json.dumps(metrics, indent=2)); return 0 if metrics["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
