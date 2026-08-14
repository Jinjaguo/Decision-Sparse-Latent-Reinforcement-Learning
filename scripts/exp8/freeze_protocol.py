#!/usr/bin/env python
"""Freeze the complete outcome-blind EXP8 execution and analysis protocol."""

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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.exp4.freeze_protocol import canonical_basis

TASKS = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate", "turn_on_the_stove"]
RADII = [0.0003125, 0.000625, 0.00125]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--branch-input", type=Path, required=True)
    parser.add_argument("--contact-frames", type=Path, required=True)
    parser.add_argument("--contact-audit", type=Path, required=True)
    parser.add_argument("--identity-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    project = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    references = json.loads((args.reference_run / "artifacts/reference_snapshots_manifest.json").read_text(encoding="utf-8"))
    branches = json.loads(args.branch_input.read_text(encoding="utf-8"))
    frame_rows = pq.read_table(args.contact_frames).to_pylist()
    frame_lookup = {(row["task"], row["episode"], int(row["action_index"])): row for row in frame_rows}
    identity = json.loads(args.identity_schema.read_text(encoding="utf-8"))
    contact_audit = json.loads(args.contact_audit.read_text(encoding="utf-8"))
    limits = {row["task"]: row for row in json.loads((ROOT / "runs/exp2_r5_q_smoke_20260814T012633/artifacts/joint_limit_manifest.json").read_text(encoding="utf-8"))}
    sources = {
        "reference_run": args.reference_run.resolve().name,
        "contact_frame_run": args.contact_frames.resolve().parents[1].name,
        "hashes": {
            "reference_manifest": sha(args.reference_run / "artifacts/reference_snapshots_manifest.json"),
            "branch_input": sha(args.branch_input), "contact_frames": sha(args.contact_frames),
            "contact_audit": sha(args.contact_audit), "identity_schema": sha(args.identity_schema),
        },
    }

    def base(kind):
        return {"schema_version": 1, "manifest_type": kind, "frozen_at_utc": now, "project_sha": project, "sources": sources, "seeds": {"master": master_seed}, "outcome_blind": True, "formal_q_outcomes_observed": False}

    master_seed = int.from_bytes(hashlib.sha256(f"EXP8_DIRECTIONS_V1|{project}|{sources['hashes']['branch_input']}".encode()).digest()[:8], "little")
    reference_lookup = {(row["task"], row["episode"]): row for row in references["episodes"]}
    geometry_by_demo = {}
    for row in frame_rows:
        geometry_by_demo.setdefault((row["task"], row["episode"]), []).append(row)
    directions, heldout, replacements = [], [], []
    for trajectory_index, trajectory in enumerate(branches["trajectories"]):
        directory = args.reference_run / reference_lookup[(trajectory["task"], trajectory["episode"])] ["relative_directory"]
        boundaries = json.loads((directory / "boundaries.json").read_text(encoding="utf-8"))
        limit = limits[trajectory["task"]]
        lower, upper = np.asarray(limit["lower"]), np.asarray(limit["upper"])
        span = upper - lower
        used = {int(branch["action_index"]) for branch in trajectory["branches"]}
        for branch_index, branch in enumerate(trajectory["branches"]):
            basis, _, rng = canonical_basis(np.random.SeedSequence(master_seed, spawn_key=(trajectory_index, branch_index)))
            random_direction = rng.standard_normal(7)
            random_direction /= np.linalg.norm(random_direction)
            vectors = [basis[:, index] for index in range(7)] + [random_direction]

            def admissible(index):
                q = np.asarray(boundaries[index]["panda_arm_q"])
                return all(np.all(q + sign * radius * span * vector >= lower) and np.all(q + sign * radius * span * vector <= upper) for radius in RADII for vector in vectors for sign in (-1, 1))

            if not admissible(int(branch["action_index"])):
                original = int(branch["action_index"])
                candidates = sorted((row for row in geometry_by_demo[(trajectory["task"], trajectory["episode"])] if int(row["action_index"]) not in used), key=lambda row: (abs(int(row["action_index"]) - original), int(row["action_index"])))
                replacement = next((row for row in candidates if admissible(int(row["action_index"]))), None)
                if replacement is None:
                    raise RuntimeError(f"no joint-limit-valid replacement for {trajectory['task']}/{trajectory['episode']}/{original}")
                used.remove(original)
                used.add(int(replacement["action_index"]))
                branch.update({
                    "action_index": int(replacement["action_index"]), "branch_time": int(replacement["action_index"]),
                    "normalized_time": float(replacement["normalized_time"]), "physical_progress_clipped": float(replacement["physical_progress_clipped"]),
                    "reference_contact_mode_json": replacement["exact_mode_json"], "reference_signed_gap_m": float(replacement["minimum_signed_gap_m"]),
                    "reference_gripper_state": replacement["gripper_state"], "reference_predicate_state": bool(replacement["predicate"]),
                    "active_pair_count": int(replacement["active_pair_count"]), "valid_pair_count": int(replacement["valid_pair_count"]),
                    "replacement_reason": f"nearest unused joint-limit-valid reference boundary replacing {original}",
                })
                replacements.append({"task": trajectory["task"], "episode": trajectory["episode"], "original": original, "replacement": int(replacement["action_index"])})
            for radius_index, radius in enumerate(RADII):
                for direction_index, vector in enumerate(vectors):
                    row = {
                        "task": trajectory["task"], "episode": trajectory["episode"], "branch_time": int(branch["action_index"]),
                        "radius_fraction": radius, "radius_label": f"r{radius:.7f}".rstrip("0"),
                        "direction_index": direction_index, "direction_role": "basis" if direction_index < 7 else "heldout_random",
                        "execution_position": radius_index * 8 + direction_index,
                        "unit_direction_scaled_coordinates": vector.tolist(), "unsigned_delta_q": (radius * span * vector).tolist(),
                        "both_signs_within_joint_limits": True,
                    }
                    directions.append(row)
                    if direction_index == 7:
                        heldout.append(row)
    manifests = {}
    manifests["exp8_cohort_manifest.json"] = {**base("exp8_cohort_manifest"), "independent": True, "trajectory_count": 30, "cohort": [{"task": row["task"], "episode": row["episode"]} for row in references["episodes"]], "selection_rule": references["selection_rule"]}
    manifests["exp8_branch_manifest.json"] = {**base("exp8_branch_manifest"), **branches, "branch_count": 360, "branches_per_demo": 12, "joint_limit_replacements": replacements}
    manifests["contact_frame_schema.json"] = {**base("contact_frame_schema"), "pair_features": ["physical_group", "nearest_point_a", "nearest_point_b", "relative_displacement", "signed_gap_m", "normal", "tangent1", "tangent2", "relative_velocity_contact", "contact_age_boundaries", "contact_force_torque_contact", "relative_point_jacobian_arm"], "global_features": ["EEF-to-target SE3/twist", "target-to-support SE3/twist", "Panda q/qvel", "gripper opening/current_action", "task progress"], "variable_size_representation": "permutation-invariant mean/min/max/RMS pooling with physical-group one-hot", "no_arbitrary_geom_id_model_input": True, "audit": contact_audit}
    manifests["contact_identity_schema.json"] = {**base("contact_identity_schema"), "identity_encoding": "physical group in primary; exact named pair retained for audit only", "physical_groups": ["target_gripper", "target_environment", "gripper_environment", "task_object_environment"], "runtime_schema": identity}
    manifests["tangent_gauge_spec.json"] = {**base("tangent_gauge_spec"), "normal_orientation": "geom1 to geom2; active mjContact frame is flipped when runtime geom order opposes frozen pair", "reference_axis": "world axis with smallest absolute dot to normal; ties x then y then z", "tangent1": "normalized projection of selected positive world axis", "tangent2": "normal cross tangent1", "singular_rule": "pair marked invalid; never infer from body centers", "repeatability_atol": 1e-12}
    manifests["signed_gap_schema.json"] = {**base("signed_gap_schema"), "api": "mujoco.mj_geomDistance", "query_ladder_m": [0.05, 0.2, 1.0, 5.0], "active_gap": "mjContact.dist", "nearest_points": "fromto segment; active points reconstructed from midpoint, oriented normal and signed gap", "convex_long_range_limitation_disclosed": True, "no_body_center_substitution": True}
    manifests["contact_force_schema.json"] = {**base("contact_force_schema"), "api": "mujoco.mj_contactForce", "semantics": "force:torque in native contact frame, transformed to deterministic frame", "impulse_claimed": False, "normalization": "training-fold robust median absolute scale with 1e-12 floor", "always_run_no_force_ablation": True, "official_documentation": contact_audit["official_sources"]["contact_force"]}
    manifests["contact_age_spec.json"] = {**base("contact_age_spec"), "unit": "consecutive reference policy boundaries", "identity": "exact frozen named physical pair", "reset": "zero immediately when identity absent; reappearance restarts at one", "outcome_independent": True}
    manifests["action_projection_spec.json"] = {**base("action_projection_spec"), "formula": "delta_x_contact=(J_point_B-J_point_A)[:, panda_dof] delta_q", "api": "mujoco.mj_jac at audited nearest/contact points", "components": ["normal", "tangent1", "tangent2", "null_norm"], "no_geom_center_jacobian": True}
    manifests["radius_manifest.json"] = {**base("radius_manifest"), "radii": RADII, "expected_signed_interventions": 17280}
    manifests["direction_basis_manifest.json"] = {**base("direction_basis_manifest"), "master_seed_uint64": master_seed, "directions": directions, "direction_rows": len(directions)}
    manifests["heldout_direction_manifest.json"] = {**base("heldout_direction_manifest"), "role": "eighth random direction excluded from seven-column operator fit", "directions": heldout}
    manifests["horizon_spec.json"] = {**base("horizon_spec"), "horizons": [1, 3, 5, "remaining"], "primary": 1, "report_populations": ["intent_to_perturb", "both_signs_exact_mode_preserved"], "coverage_adjusted_similarity": "paired mode preservation rate times conditional top1 similarity"}
    manifests["signed_output_vector_spec.json"] = {**base("signed_output_vector_spec"), "inherited_exactly_from": "EXP7", "order": ["arm_q[7]", "arm_qvel[7]", "eef_position[3]", "eef_orientation_rotvec[3]", "task_object_position[3*n]", "task_object_orientation_rotvec[3*n]"]}
    manifests["baseline_model_specs.json"] = {**base("baseline_model_specs"), "A": "EXP7 exact mode plus 1 mm margin nearest cross-demo operator", "time": "nearest normalized time", "progress": "nearest scalar physical progress", "B": "ridge on EXP5 reference-only physical state plus continuous signed gap and normal relative velocity", "training_only_scaling": True}
    manifests["primary_model_spec.json"] = {**base("primary_model_spec"), "model": "deterministic permutation-invariant contact-pair pooling plus RBF kernel ridge", "inputs": "continuous contact-frame pairs, action-Jacobian summary, EEF/object relative state and physical-group identity", "outputs": ["flattened H1 operator", "top1 projector", "heldout signed vector", "scalar sensitivity"], "large_neural_network": False}
    manifests["hyperparameter_grid.json"] = {**base("hyperparameter_grid"), "ridge_alpha": [1e-6, 1e-4, 1e-2, 1.0, 100.0], "rbf_bandwidth": [0.25, 0.5, 1.0, 2.0, 4.0], "inner_selection": "mean demo-level top1 similarity", "tie_rule": "larger alpha, then larger bandwidth", "scaling_floor": 1e-12}
    fold_rows = []
    for task in TASKS:
        demos = sorted(row["episode"] for row in references["episodes"] if row["task"] == task)
        fold_rows.extend({"task": task, "episode": episode, "fold": index % 5} for index, episode in enumerate(demos))
    manifests["crossfit_manifest.json"] = {**base("crossfit_manifest"), "unit": "demonstration", "outer_folds": 5, "assignments": fold_rows, "inner_validation": "leave-one-training-demo-out within task", "same_demo_train_test_forbidden": True}
    ablations = ["exact_mode_margin_only", "EXP5_physical_state", "gap_normal_velocity_only", "remove_nearest_points", "remove_normal_tangent_frame", "remove_relative_velocity", "remove_contact_age", "remove_force", "remove_action_projection", "remove_EEF_object_relative_pose", "exact_geom_id_vs_physical_group", "contact_count_diagnostic"]
    manifests["ablation_manifest.json"] = {**base("ablation_manifest"), "ablations": ablations, "all_reported": True, "primary_model_unchanged_after_outcomes": True}
    risk_features = ["task", "radius", "sign", "direction role", "continuous gap", "contact-frame velocity", "contact age", "force", "action projection", "EEF/object relative state", "gripper", "progress"]
    manifests["risk_feature_schema.json"] = {**base("risk_feature_schema"), "features": risk_features, "target": "next-step exact mode preservation", "forbidden": ["perturbed future response", "terminal success", "test-fold threshold labels"]}
    manifests["risk_crossfit_manifest.json"] = {**base("risk_crossfit_manifest"), "assignments": fold_rows, "model": "training-standardized ridge logistic regression", "hyperparameters": [1e-4, 1e-2, 1.0, 100.0], "nested_training_only_selection": True}
    manifests["risk_threshold_rule.json"] = {**base("risk_threshold_rule"), "selection": "maximize training-fold specificity subject to sensitivity >=0.85; tie by sensitivity then highest threshold", "test_fold_never_used": True, "requirements": {"AUROC_CI_lower": 0.75, "ECE_max": 0.05, "specificity_min": 0.70, "sensitivity_min": 0.85}}
    manifests["statistical_analysis_plan.json"] = {**base("statistical_analysis_plan"), "independent_unit": "demonstration", "bootstrap": {"resamples": 4000, "seed": 980031, "cluster": "demo stratified by task"}, "permutation": {"resamples": 4000, "seed": 980032, "unit": "demo"}, "multiple_comparisons": "BH FDR 0.05", "H1": {"mean_improvement_min": 0.15, "CI_lower_gt": 0, "all_tasks_positive": True, "task_CI_positive_required": 2}, "H2": {"demo_median_rho_min": 0.70, "demo_median_vector_error_max": 0.35, "p90_vector_error_max": 0.60}, "H3": {"improvement_CI_lower_gt": 0, "BH_q_max": 0.05}, "H4": "all horizons, intent and conditional, plus coverage-adjusted", "H5": manifests["risk_threshold_rule.json"]["requirements"]}
    manifests["gpu_analysis_spec.json"] = {**base("gpu_analysis_spec"), "device": "cuda:0 RTX 4090", "dtype": "float64", "simulator": "CPU", "audit_components": ["contact-frame coordinates", "normalization", "kernel matrices", "ridge predictions", "Gram/projector", "heldout vectors", "bootstrap/permutation", "risk probabilities", "calibration"], "tolerances": {"atol": 1e-10, "rtol": 1e-8}, "no_relaxation": True, "failure_policy": "formal analysis CPU-only"}
    priority = ["continuous_contact_field_replicates", "contact_geometry_improves_but_tail_risk_remains", "mode_risk_gate_passes_without_operator_reuse", "continuous_geometry_insufficient", "support_or_identifiability_failure", "no_support"]
    manifests["scientific_decision_rule.json"] = {**base("scientific_decision_rule"), "classification_priority": priority, "classification_1_requires": ["H1", "H2", "H3"], "scheduler_eligibility_additionally_requires": "H5", "online_control_forbidden": True, "latent_RL_forbidden": True}
    for name, payload in manifests.items():
        (out / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("effect_channel_schema.json", "effect_normalization.json", "primary_metric_spec.json"):
        shutil.copy2(ROOT / "experiments/exp7_contact_mode_conditioned/manifests" / name, out / name)
    config = {"experiment": "EXP8", "signs": [-1, 1], "directions_per_branch": 24, "expected_branches": 360, "non_arm_integration_linf_max": 1e-12, "q_injection_atol": 1e-15, "zero_median_max": 1e-12, "zero_p95_max": 1e-12, "zero_maximum_max": 1e-12, "zero_terminal_object_pose_p95_max": 1e-12, "meaningful_effect_threshold": 0.01, "radii": RADII}
    configs = out.parent / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "exp8.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    required = list(manifests)
    hashes = {name: sha(out / name) for name in required}
    hash_manifest = {**base("manifest_hashes"), "required_manifest_count": len(required), "manifests": hashes, "direction_hash": hashes["direction_basis_manifest.json"], "radius_hash": hashes["radius_manifest.json"], "model_config_hash": hashes["primary_model_spec.json"], "crossfit_hash": hashes["crossfit_manifest.json"], "risk_threshold_hash": hashes["risk_threshold_rule.json"]}
    (out / "manifest_hashes.json").write_text(json.dumps(hash_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {"required_manifests": len(required), "branches": 360, "direction_rows": len(directions), "planned_signed_interventions": 2 * len(directions), "joint_limit_replacements": len(replacements), "gate": {"passed": len(required) == 25 and len(directions) == 8640 and 2 * len(directions) == 17280}}
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
