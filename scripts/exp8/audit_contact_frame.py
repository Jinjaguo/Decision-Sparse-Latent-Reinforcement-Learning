#!/usr/bin/env python
"""Restore every EXP8 reference boundary and audit continuous contact-frame fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.envs import controller_snapshot, mujoco_snapshot
from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record
from decision_sparse_rl.envs.mujoco_snapshot import native_model_data
from scripts.exp8.contact_frame import measure_contact_frame

TARGET_SUPPORT = {
    "open_the_middle_drawer_of_the_cabinet": ("wooden_cabinet_1_cabinet_middle", "wooden_cabinet_1_base"),
    "put_the_bowl_on_the_plate": ("akita_black_bowl_1_main", "plate_1_main"),
    "turn_on_the_stove": ("flat_stove_1_button", "flat_stove_1_base"),
}


def _relative_pose(position_a, rotation_a, position_b, rotation_b):
    rotation = np.asarray(rotation_a).reshape(3, 3).T @ np.asarray(rotation_b).reshape(3, 3)
    translation = np.asarray(rotation_a).reshape(3, 3).T @ (np.asarray(position_b) - np.asarray(position_a))
    return np.concatenate((translation, rotation.reshape(-1))).tolist()


def _object_twist(model, data, objtype, objid):
    value = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, objtype, int(objid), value, 0)
    return value


def _relative_twist(reference_rotation, twist_a, twist_b):
    rotation = np.asarray(reference_rotation).reshape(3, 3).T
    delta = np.asarray(twist_b) - np.asarray(twist_a)
    return np.concatenate((rotation @ delta[:3], rotation @ delta[3:])).tolist()


def _flatten_pairs(payload):
    rows = json.loads(payload["pair_features_json"])
    result = []
    for row in rows:
        result.extend(row["nearest_point_a"] + row["nearest_point_b"] + row["normal"] + row["tangent1"] + row["tangent2"])
        result.extend(row["relative_velocity_contact"] + row["contact_force_torque_contact"])
    return np.asarray(result, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--identity-schema", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    (run / "artifacts").mkdir(parents=True, exist_ok=False)
    reference = args.reference_run.resolve()
    manifest = json.loads((reference / "artifacts/reference_snapshots_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads(args.identity_schema.read_text(encoding="utf-8"))
    selection, tasks = load_selection(ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json", ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
    selected = {row["name"]: row for row in selection["tasks"]}
    wrapper, robosuite_root, assets_root = bootstrap_runtime(ROOT / "third_party/LIBERO", ROOT / "data", run / "artifacts/libero_config")
    rows, repeats, runtime_audit = [], [], []
    for record in manifest["episodes"]:
        task_meta = selected[record["task"]]
        source = task_source_record(tasks, task_meta["suite"], task_meta["task_id"])
        env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
        load_episode(env, dataset_path=ROOT / "data" / task_meta["demonstration_relative_path"], episode_index=int(record["episode"].rsplit("_", 1)[-1]), robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
        model, data = native_model_data(env.sim)
        robot = env.robots[0]
        arm_qpos = [int(value) for value in robot._ref_joint_pos_indexes]
        arm_dof = [int(value) for value in robot._ref_joint_vel_indexes]
        if len(arm_qpos) != 7 or len(arm_dof) != 7:
            raise RuntimeError("Panda arm index audit failed")
        target_name, support_name = TARGET_SUPPORT[record["task"]]
        target_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name))
        support_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, support_name))
        if target_id < 0 or support_id < 0:
            raise RuntimeError(f"target/support body lookup failed: {record['task']}")
        eef_id = int(robot.eef_site_id)
        directory = reference / record["relative_directory"]
        boundaries = json.loads((directory / "boundaries.json").read_text(encoding="utf-8"))
        with np.load(directory / "trajectory_states.npz", allow_pickle=False) as archive:
            integrations = np.asarray(archive["integration"])
        prior_active: dict[str, int] = {}
        repeat_indexes = set(np.linspace(0, len(boundaries) - 1, 4, dtype=int))
        for index, boundary in enumerate(boundaries):
            snapshot = mujoco_snapshot.MujocoSnapshot("integration", mujoco_snapshot.state_spec("integration"), integrations[index].copy())
            controller = controller_snapshot.deserialize(directory / f"controller_{index:04d}.npz")
            mujoco_snapshot.restore(env.sim, snapshot, forward=True)
            controller_snapshot.restore(env, controller)
            payload = measure_contact_frame(env, schema, record["task"], arm_dof)
            pair_rows = json.loads(payload["pair_features_json"])
            active_now = {row["pair"] for row in pair_rows if row["active"]}
            next_age = {identity: prior_active.get(identity, 0) + 1 for identity in active_now}
            for row in pair_rows:
                row["contact_age_boundaries"] = next_age.get(row["pair"], 0)
            prior_active = next_age
            payload["pair_features_json"] = json.dumps(pair_rows, sort_keys=True)
            eef_position = np.asarray(data.site_xpos[eef_id]).copy()
            eef_rotation = np.asarray(data.site_xmat[eef_id]).reshape(3, 3).copy()
            target_position, target_rotation = np.asarray(data.xpos[target_id]).copy(), np.asarray(data.xmat[target_id]).reshape(3, 3).copy()
            support_position, support_rotation = np.asarray(data.xpos[support_id]).copy(), np.asarray(data.xmat[support_id]).reshape(3, 3).copy()
            eef_twist = _object_twist(model, data, mujoco.mjtObj.mjOBJ_SITE, eef_id)
            target_twist = _object_twist(model, data, mujoco.mjtObj.mjOBJ_BODY, target_id)
            support_twist = _object_twist(model, data, mujoco.mjtObj.mjOBJ_BODY, support_id)
            command = float(boundary["progress_channels"]["gripper_command"])
            row = {
                "task": record["task"], "episode": record["episode"], "action_index": index,
                "trajectory_length": len(boundaries), "normalized_time": index / max(len(boundaries) - 1, 1),
                "physical_progress_clipped": float(boundary["progress_channels"].get("joint_qpos", boundary["progress_channels"].get("bowl_to_plate_planar_distance_m", index / max(len(boundaries) - 1, 1)))),
                "predicate": bool(boundary["progress_channels"]["exact_task_predicate"]),
                "gripper_state": "negative" if command < -0.5 else "positive" if command > 0.5 else "neutral",
                "gripper_current_action": np.asarray(robot.gripper.current_action, dtype=np.float64).tolist(),
                "gripper_opening": np.asarray(data.qpos[[int(value) for value in robot._ref_gripper_joint_pos_indexes]], dtype=np.float64).tolist(),
                "panda_q": np.asarray(data.qpos[arm_qpos], dtype=np.float64).tolist(),
                "panda_qvel": np.asarray(data.qvel[arm_dof], dtype=np.float64).tolist(),
                "eef_to_target_se3": _relative_pose(eef_position, eef_rotation, target_position, target_rotation),
                "eef_to_target_twist": _relative_twist(eef_rotation, eef_twist, target_twist),
                "target_to_support_se3": _relative_pose(target_position, target_rotation, support_position, support_rotation),
                "target_to_support_twist": _relative_twist(target_rotation, target_twist, support_twist),
                **payload,
            }
            rows.append(row)
            if index in repeat_indexes:
                values = []
                for _ in range(3):
                    mujoco_snapshot.restore(env.sim, snapshot, forward=True)
                    controller_snapshot.restore(env, controller)
                    values.append(_flatten_pairs(measure_contact_frame(env, schema, record["task"], arm_dof)))
                repeats.append({"task": record["task"], "episode": record["episode"], "action_index": index, "maximum_feature_range": float(np.max(np.ptp(np.stack(values), axis=0)))})
        runtime_audit.append({"task": record["task"], "episode": record["episode"], "target_body": target_name, "support_body": support_name, "arm_qpos_indexes": arm_qpos, "arm_dof_indexes": arm_dof})
        env.close()
        print(json.dumps({"trajectory": f"{record['task']}/{record['episode']}", "boundaries": len(boundaries)}))
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, run / "artifacts/reference_contact_frames.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(repeats), run / "artifacts/contact_frame_repeatability.parquet", compression="zstd")
    active_counts = [row["active_pair_count"] for row in rows]
    valid_counts = [row["valid_pair_count"] for row in rows]
    force_values = [row["maximum_normal_force"] for row in rows]
    audit = {
        "schema_version": 1,
        "mujoco_version": mujoco.__version__,
        "reference_boundary_count": len(rows),
        "repeat_boundary_count": len(repeats),
        "maximum_repeat_feature_range": max(row["maximum_feature_range"] for row in repeats),
        "active_pair_range": [min(active_counts), max(active_counts)],
        "valid_pair_range": [min(valid_counts), max(valid_counts)],
        "positive_force_boundary_count": sum(value > 0 for value in force_values),
        "force_semantics": "mj_contactForce returns force:torque in native contact frame; transformed to deterministic gauge; force, not impulse",
        "official_sources": {
            "contact_force": "https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html#mj-contactforce",
            "contact_frame": "https://mujoco.readthedocs.io/en/3.2.3/APIreference/APItypes.html#mjcontact",
            "point_jacobian": "https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html#mj-jac",
            "surface_distance": "https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html#mj-geomdistance",
        },
        "runtime_audit": runtime_audit,
    }
    criteria = {
        "all_boundaries_finite": all(np.all(np.isfinite(_flatten_pairs(row))) for row in rows),
        "repeatability_atol": audit["maximum_repeat_feature_range"] <= 1e-12,
        "force_available": audit["positive_force_boundary_count"] > 0,
        "all_boundaries_have_valid_pair": min(valid_counts) > 0,
        "three_task_support": len({row["task"] for row in rows}) == 3,
    }
    audit["gate"] = {"passed": all(criteria.values()), "criteria": criteria}
    (run / "artifacts/contact_frame_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
