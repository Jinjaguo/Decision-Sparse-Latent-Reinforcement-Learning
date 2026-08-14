#!/usr/bin/env python
"""Build frozen permutation-invariant EXP8 branch and direction features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from decision_sparse_rl.metrics.exp8 import permutation_invariant_pool, project_contact_motion

GROUPS = ["target_gripper", "target_environment", "gripper_environment", "task_object_environment"]


def pool(values):
    return permutation_invariant_pool(values).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-frames", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    (run / "artifacts").mkdir(parents=True, exist_ok=False)
    frames = {(row["task"], row["episode"], int(row["action_index"])): row for row in pq.read_table(args.contact_frames).to_pylist()}
    branches = json.loads((args.manifest_dir / "exp8_branch_manifest.json").read_text(encoding="utf-8"))
    directions = json.loads((args.manifest_dir / "direction_basis_manifest.json").read_text(encoding="utf-8"))["directions"]
    direction_lookup = {}
    for row in directions:
        direction_lookup.setdefault((row["task"], row["episode"], int(row["branch_time"])), []).append(row)
    output, direction_output, support = [], [], []
    for trajectory in branches["trajectories"]:
        for branch in trajectory["branches"]:
            key = (trajectory["task"], trajectory["episode"], int(branch["action_index"]))
            row = frames[key]
            pairs = [pair for pair in json.loads(row["pair_features_json"]) if pair["geometry_valid"]]
            if not pairs:
                raise RuntimeError(f"no valid pairs at frozen branch {key}")
            identity = [[float(pair["physical_group"] == group) for group in GROUPS] + [float(pair["active"])] for pair in pairs]
            nearest = [pair["nearest_point_a"] + pair["nearest_point_b"] + pair["relative_displacement"] for pair in pairs]
            frame = [pair["normal"] + pair["tangent1"] + pair["tangent2"] for pair in pairs]
            gap = [[pair["signed_gap_m"]] for pair in pairs]
            velocity = [pair["relative_velocity_contact"] for pair in pairs]
            age = [[pair["contact_age_boundaries"]] for pair in pairs]
            force = [pair["contact_force_torque_contact"] for pair in pairs]
            jacobian = [np.asarray(pair["relative_point_jacobian_arm"]).reshape(-1).tolist() for pair in pairs]
            group_features = {
                "physical_group": pool(identity), "nearest_points": pool(nearest), "normal_tangent_frame": pool(frame),
                "signed_gap": pool(gap), "relative_velocity": pool(velocity), "contact_age": pool(age),
                "force": pool(force), "action_projection": pool(jacobian),
                "eef_object_relative_pose": row["eef_to_target_se3"] + row["eef_to_target_twist"] + row["target_to_support_se3"] + row["target_to_support_twist"],
                "physical_state": row["panda_q"] + row["panda_qvel"] + row["gripper_opening"] + row["gripper_current_action"] + [row["physical_progress_clipped"]],
            }
            normal_velocities = [pair["relative_velocity_contact"][0] for pair in pairs]
            baseline_b = group_features["physical_state"] + group_features["eef_object_relative_pose"] + [min(pair["signed_gap_m"] for pair in pairs), float(np.mean(normal_velocities))]
            primary_order = ["physical_group", "nearest_points", "normal_tangent_frame", "signed_gap", "relative_velocity", "contact_age", "force", "action_projection", "eef_object_relative_pose", "physical_state"]
            primary = np.concatenate([np.asarray(group_features[name], dtype=np.float64) for name in primary_order]).tolist()
            output.append({
                "task": key[0], "episode": key[1], "branch_time": key[2], "normalized_time": row["normalized_time"],
                "physical_progress_clipped": row["physical_progress_clipped"], "exact_mode_json": row["exact_mode_json"],
                "margin_class": "active_or_penetrating" if row["active_pair_count"] > 0 or row["minimum_signed_gap_m"] <= 0 else "near" if row["minimum_signed_gap_m"] < 1e-3 else "far",
                "active_pair_count": row["active_pair_count"], "valid_pair_count": row["valid_pair_count"],
                "feature_groups_json": json.dumps(group_features, sort_keys=True), "baseline_b_features": baseline_b, "primary_features": primary,
            })
            for direction in direction_lookup[key]:
                delta = np.asarray(direction["unsigned_delta_q"], dtype=np.float64)
                projected = []
                for pair in pairs:
                    basis = np.stack((pair["normal"], pair["tangent1"], pair["tangent2"]))
                    contact = project_contact_motion(np.asarray(pair["relative_point_jacobian_arm"]), delta, basis)
                    jacobian = np.asarray(pair["relative_point_jacobian_arm"])
                    null = float(np.sqrt(max(float(delta @ delta - (np.linalg.pinv(jacobian) @ (jacobian @ delta)) @ (np.linalg.pinv(jacobian) @ (jacobian @ delta))), 0.0)))
                    projected.append(contact.tolist() + [null])
                direction_output.append({"task": key[0], "episode": key[1], "branch_time": key[2], "radius_fraction": direction["radius_fraction"], "direction_index": direction["direction_index"], "direction_role": direction["direction_role"], "action_projection_features": pool(projected)})
    for task in sorted({row["task"] for row in output}):
        task_rows = [row for row in output if row["task"] == task]
        original = [frames[(row["task"], row["episode"], row["branch_time"])] for row in task_rows]
        pairs = [pair for row in original for pair in json.loads(row["pair_features_json"]) if pair["geometry_valid"]]
        normals = np.asarray([pair["normal"] for pair in pairs])
        ages = [pair["contact_age_boundaries"] for pair in pairs if pair["active"]]
        support.append({
            "task": task, "branch_count": len(task_rows), "active_branch_count": sum(row["active_pair_count"] > 0 for row in task_rows),
            "free_space_branch_count": sum(row["active_pair_count"] == 0 for row in task_rows), "near_gap_branch_count": sum(row["margin_class"] == "near" for row in task_rows),
            "free_space_valid_pair_count": sum(not pair["active"] for pair in pairs),
            "unique_exact_modes": len({row["exact_mode_json"] for row in task_rows}), "unique_physical_groups": len({pair["physical_group"] for pair in pairs}),
            "normal_component_range": np.ptp(normals, axis=0).tolist(), "maximum_contact_age": max(ages or [0]),
            "force_positive_pairs": sum(abs(pair["contact_force_torque_contact"][0]) > 0 for pair in pairs),
        })
    pq.write_table(pa.Table.from_pylist(output), run / "artifacts/frozen_branch_contact_features.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(direction_output), run / "artifacts/frozen_action_projection_features.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(support), run / "artifacts/contact_frame_support_audit.parquet", compression="zstd")
    criteria = {"exact_360_branches": len(output) == 360, "exact_8640_direction_rows": len(direction_output) == 8640, "all_tasks_active_support": all(row["active_branch_count"] > 0 for row in support), "all_tasks_free_space_nearest_pair_support": all(row["free_space_valid_pair_count"] > 0 for row in support), "all_tasks_multiple_normals": all(max(row["normal_component_range"]) > 0.1 for row in support), "all_tasks_contact_age_variation": all(row["maximum_contact_age"] > 1 for row in support)}
    metrics = {"branch_count": len(output), "direction_feature_count": len(direction_output), "support": support, "gate": {"passed": all(criteria.values()), "criteria": criteria}}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
