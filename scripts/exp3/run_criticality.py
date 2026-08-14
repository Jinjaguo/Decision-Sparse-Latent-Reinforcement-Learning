#!/usr/bin/env python
"""Execute EXP3 dry, matched-zero, or full q-criticality continuations."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs import controller_snapshot, mujoco_snapshot  # noqa: E402
from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record  # noqa: E402
from decision_sparse_rl.interventions.q_intervention import apply_arm_q, non_arm_integration_linf  # noqa: E402
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.metrics.criticality import quaternion_geodesic, rotation_geodesic  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("dry", "zero", "full"), required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--zero-run", type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=REPOSITORY_ROOT / "experiments/exp3_time_indexed_q_criticality/manifests")
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party/LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--selection", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json")
    parser.add_argument("--task-manifest", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--max-branches", type=int)
    parser.add_argument("--max-directions", type=int)
    return parser.parse_args()


def restore_d(env: Any, integration: np.ndarray, controller_path: Path) -> None:
    snapshot = mujoco_snapshot.MujocoSnapshot("integration", mujoco_snapshot.state_spec("integration"), integration.copy())
    mujoco_snapshot.restore(env.sim, snapshot)
    controller_snapshot.restore(env, controller_snapshot.deserialize(controller_path))


def contacts(env: Any) -> List[str]:
    result = set()
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        pair = [f"{int(contact.geom1)}:{env.sim.model.geom_id2name(int(contact.geom1))}", f"{int(contact.geom2)}:{env.sim.model.geom_id2name(int(contact.geom2))}"]
        result.add("|".join(sorted(pair)))
    return sorted(result)


def observe(env: Any, body_ids: List[int]) -> Dict[str, Any]:
    robot = env.robots[0]
    arm_q = [int(x) for x in robot._ref_joint_pos_indexes]
    arm_v = [int(x) for x in robot._ref_joint_vel_indexes]
    eef = int(robot.eef_site_id)
    return {
        "integration": mujoco_snapshot.capture(env.sim, "integration").values,
        "q": np.asarray(env.sim.data.qpos[arm_q]).copy(),
        "qvel": np.asarray(env.sim.data.qvel[arm_v]).copy(),
        "eef_position": np.asarray(env.sim.data.site_xpos[eef]).copy(),
        "eef_orientation": np.asarray(env.sim.data.site_xmat[eef]).copy(),
        "object_positions": np.asarray(env.sim.data.body_xpos[body_ids]).copy(),
        "object_quaternions": np.asarray(env.sim.data.body_xquat[body_ids]).copy(),
        "contacts": contacts(env), "contact_count": int(env.sim.data.ncon),
        "predicate": bool(env.check_success()),
    }


def rollout(env: Any, actions: np.ndarray, branch: int, body_ids: List[int]) -> List[Dict[str, Any]]:
    result = []
    for action in actions[branch:]:
        env.step(action)
        result.append(observe(env, body_ids))
    return result


def identity(trajectory: Dict[str, Any], branch: Dict[str, Any]) -> Dict[str, Any]:
    return {"task": trajectory["task"], "episode": trajectory["episode"], "branch_time": int(branch["action_index"]),
            "branch_normalized_time": float(branch["normalized_time"]), "branch_kind": branch["kind"],
            "remaining_horizon": int(trajectory["trajectory_length"] - branch["action_index"])}


def zero_step_record(base: Dict[str, Any], offset: int, observation: Dict[str, Any]) -> Dict[str, Any]:
    return {**base, "continuation_offset": offset, "absolute_action_index": base["branch_time"] + offset,
            "integration": observation["integration"].tolist(), "arm_q": observation["q"].tolist(), "arm_qvel": observation["qvel"].tolist(),
            "eef_position": observation["eef_position"].tolist(), "eef_orientation": observation["eef_orientation"].tolist(),
            "task_object_positions": observation["object_positions"].tolist(), "task_object_quaternions": observation["object_quaternions"].tolist(),
            "contact_pairs_json": json.dumps(observation["contacts"]), "raw_contact_count": observation["contact_count"], "task_predicate": observation["predicate"]}


def compare_zero(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:
    return {"integration_l2": float(np.linalg.norm(right["integration"] - left["integration"])),
            "arm_q_l2": float(np.linalg.norm(right["q"] - left["q"])), "arm_qvel_l2": float(np.linalg.norm(right["qvel"] - left["qvel"])),
            "eef_position_l2": float(np.linalg.norm(right["eef_position"] - left["eef_position"])),
            "eef_orientation_geodesic": rotation_geodesic(left["eef_orientation"], right["eef_orientation"]),
            "task_object_position_l2": float(np.linalg.norm(right["object_positions"] - left["object_positions"])),
            "task_object_orientation_geodesic_mean": float(np.mean([quaternion_geodesic(a, b) for a, b in zip(left["object_quaternions"], right["object_quaternions"])])),
            "contact_pair_symmetric_difference_count": len(set(left["contacts"]) ^ set(right["contacts"])),
            "raw_contact_count_difference": abs(right["contact_count"] - left["contact_count"]),
            "task_predicate_divergence": bool(right["predicate"] != left["predicate"])}


def reference_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"integration": np.asarray(row["integration"]), "q": np.asarray(row["arm_q"]), "qvel": np.asarray(row["arm_qvel"]),
            "eef_position": np.asarray(row["eef_position"]), "eef_orientation": np.asarray(row["eef_orientation"]),
            "object_positions": np.asarray(row["task_object_positions"]), "object_quaternions": np.asarray(row["task_object_quaternions"]),
            "contacts": json.loads(row["contact_pairs_json"]), "contact_count": int(row["raw_contact_count"]), "predicate": bool(row["task_predicate"])}


def effect_row(base: Dict[str, Any], offset: int, zero: Dict[str, Any], perturbed: Dict[str, Any], normalization: Dict[str, Any]) -> Dict[str, Any]:
    raw = compare_zero(zero, perturbed)
    den = normalization["continuous_denominators"]
    components = {
        "arm_q": raw["arm_q_l2"] / den["arm_q_l2"]["value"],
        "arm_qvel": raw["arm_qvel_l2"] / den["arm_qvel_l2"]["value"],
        "eef_position": raw["eef_position_l2"] / den["eef_position_l2"]["value"],
        "eef_orientation": raw["eef_orientation_geodesic"] / den["eef_orientation_geodesic"]["value"],
        "task_object_position": raw["task_object_position_l2"] / den["task_object_position_l2"]["value_by_task"][base["task"]],
        "task_object_orientation": raw["task_object_orientation_geodesic_mean"] / den["task_object_orientation_geodesic_mean"]["value"],
    }
    return {**base, "continuation_offset": offset, "absolute_action_index": base["branch_time"] + offset, **raw,
            **{f"normalized_{key}": value for key, value in components.items()}, "primary_step_effect": float(np.mean(list(components.values()))),
            "zero_task_predicate": zero["predicate"], "perturbed_task_predicate": perturbed["predicate"],
            "all_states_finite": bool(np.all(np.isfinite(perturbed["integration"])))}


def write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def main() -> int:
    args = parse_args()
    if args.mode == "full" and args.zero_run is None: raise ValueError("full mode requires --zero-run")
    run_dir = create_run_directory(args.run_root, args.run_id)
    captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
    manifest_dir = args.manifest_dir.resolve(); reference_run = args.reference_run.resolve()
    config = json.loads((REPOSITORY_ROOT / "experiments/exp3_time_indexed_q_criticality/configs/exp3.json").read_text())
    branch_manifest = json.loads((manifest_dir / "branch_manifest.json").read_text())
    directions = json.loads((manifest_dir / "direction_manifest.json").read_text())
    normalization = json.loads((manifest_dir / "effect_normalization.json").read_text())
    channel_schema = json.loads((manifest_dir / "effect_channel_schema.json").read_text())
    primary_spec = json.loads((manifest_dir / "primary_metric_spec.json").read_text())
    joint_path = REPOSITORY_ROOT / "runs/exp2_r5_q_smoke_20260814T012633/artifacts/joint_limit_manifest.json"
    joint_limits = {x["task"]: x for x in json.loads(joint_path.read_text())}
    run_config = {"run_id": args.run_id, "stage": f"EXP3_{args.mode}", "mode": args.mode, "reference_run": str(reference_run), "zero_run": None if args.zero_run is None else str(args.zero_run.resolve()),
                  "manifest_dir": str(manifest_dir), "max_trajectories": args.max_trajectories, "max_branches": args.max_branches, "max_directions": args.max_directions, "fixed_config": config}
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "pyarrow": pa.__version__, "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite"), "compute_note": "MuJoCo CPU; GPU permitted but not used by validated simulator path"}
    git_state = {"project": git_record(REPOSITORY_ROOT), "libero": git_record(args.libero_root), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")}
    env = None
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            reference_manifest = json.loads((reference_run / "artifacts/reference_snapshots_manifest.json").read_text())
            if not reference_manifest["gate"]["passed"]: raise RuntimeError("reference gate is not passing")
            selection, task_manifest = load_selection(args.selection, args.task_manifest)
            selected_by_name = {x["name"]: x for x in selection["tasks"]}
            refs = {(x["task"], x["episode"]): x for x in reference_manifest["episodes"]}
            trajectories = branch_manifest["trajectories"][:args.max_trajectories]
            wrapper, robosuite_root, assets_root = bootstrap_runtime(args.libero_root, args.dataset_root, run_dir / "artifacts/libero_config")
            zero_summaries: List[Dict[str, Any]] = []; zero_steps: List[Dict[str, Any]] = []
            intervention_summaries: List[Dict[str, Any]] = []; per_steps: List[Dict[str, Any]] = []; failures: List[Dict[str, Any]] = []
            zero_lookup = {}
            if args.mode == "full":
                zero_metrics = json.loads((args.zero_run / "metrics.json").read_text())
                if not zero_metrics.get("gate", {}).get("passed"): raise RuntimeError("supplied zero-control run failed")
                for row in pq.read_table(args.zero_run / "artifacts/zero_reference_steps.parquet").to_pylist():
                    zero_lookup[(row["task"], row["episode"], row["branch_time"], row["continuation_offset"])] = reference_from_row(row)
                shutil.copy2(args.zero_run / "artifacts/zero_controls.parquet", run_dir / "artifacts/zero_controls.parquet")
            direction_lookup: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
            for row in directions["directions"]:
                direction_lookup.setdefault((row["task"], row["episode"], row["branch_time"]), []).append(row)
            for trajectory_index, trajectory in enumerate(trajectories):
                task = selected_by_name[trajectory["task"]]; source = task_source_record(task_manifest, task["suite"], task["task_id"])
                env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                episode_index = int(trajectory["episode"].split("_")[-1])
                load_episode(env, dataset_path=args.dataset_root / task["demonstration_relative_path"], episode_index=episode_index, robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                ref = refs[(trajectory["task"], trajectory["episode"])]; ref_dir = reference_run / ref["relative_directory"]
                with np.load(ref_dir / "trajectory_states.npz", allow_pickle=False) as archive:
                    actions = np.asarray(archive["actions"], dtype=np.float64); integrations = np.asarray(archive["integration"], dtype=np.float64)
                body_names = channel_schema["task_object_audit"][trajectory["task"]]["bodies"]
                body_ids = [int(env.sim.model.body_name2id(name)) for name in body_names]
                branches = trajectory["branches"][:args.max_branches]
                for branch in branches:
                    base = identity(trajectory, branch); branch_time = base["branch_time"]; controller_path = ref_dir / f"controller_{branch_time:04d}.npz"
                    if args.mode in ("zero", "dry"):
                        restore_d(env, integrations[branch_time], controller_path); twin_a = rollout(env, actions, branch_time, body_ids)
                        restore_d(env, integrations[branch_time], controller_path); twin_b = rollout(env, actions, branch_time, body_ids)
                        differences = [compare_zero(a, b) for a, b in zip(twin_a, twin_b)]
                        for offset, observation in enumerate(twin_a): zero_steps.append(zero_step_record(base, offset, observation))
                        summary = {**base, "zero_continuation_ids": [f"{trajectory['task']}|{trajectory['episode']}|{branch_time}|A", f"{trajectory['task']}|{trajectory['episode']}|{branch_time}|B"],
                            "maximum_integration_l2": max(x["integration_l2"] for x in differences), "maximum_arm_q_l2": max(x["arm_q_l2"] for x in differences),
                            "maximum_arm_qvel_l2": max(x["arm_qvel_l2"] for x in differences), "maximum_eef_position_l2": max(x["eef_position_l2"] for x in differences),
                            "maximum_eef_orientation_geodesic": max(x["eef_orientation_geodesic"] for x in differences),
                            "maximum_task_object_position_l2": max(x["task_object_position_l2"] for x in differences),
                            "maximum_task_object_orientation_geodesic_mean": max(x["task_object_orientation_geodesic_mean"] for x in differences),
                            "maximum_contact_pair_symmetric_difference_count": max(x["contact_pair_symmetric_difference_count"] for x in differences),
                            "maximum_raw_contact_count_difference": max(x["raw_contact_count_difference"] for x in differences),
                            "any_task_predicate_divergence": any(x["task_predicate_divergence"] for x in differences),
                            "terminal_success_a": twin_a[-1]["predicate"], "terminal_success_b": twin_b[-1]["predicate"],
                            "all_states_finite": all(np.all(np.isfinite(x["integration"])) for x in twin_a + twin_b)}
                        zero_summaries.append(summary)
                    if args.mode in ("full", "dry"):
                        rows = direction_lookup[(trajectory["task"], trajectory["episode"], branch_time)][:args.max_directions]
                        limits = joint_limits[trajectory["task"]]; lower = np.asarray(limits["lower"]); upper = np.asarray(limits["upper"]); indexes = limits["qpos_indexes"]
                        for direction in rows:
                            for sign in config["signs"]:
                                signed = int(sign) * np.asarray(direction["unsigned_delta_q"], dtype=np.float64)
                                intervention_id = f"{trajectory['task']}|{trajectory['episode']}|{branch_time}|d{direction['direction_index']}|s{int(sign):+d}"
                                ibase = {**base, "intervention_id": intervention_id, "direction_index": int(direction["direction_index"]), "sign": int(sign)}
                                restore_d(env, integrations[branch_time], controller_path)
                                before = mujoco_snapshot.capture_atomic_components(env.sim)
                                q_before = np.asarray(env.sim.data.qpos[indexes]).copy()
                                q_after = apply_arm_q(env.sim.data, indexes, signed, lower, upper)
                                after = mujoco_snapshot.capture_atomic_components(env.sim)
                                preservation = non_arm_integration_linf(before, after, indexes)
                                perturb = rollout(env, actions, branch_time, body_ids)
                                step_effects = []
                                for offset, observation in enumerate(perturb):
                                    if args.mode == "full": zero = zero_lookup[(trajectory["task"], trajectory["episode"], branch_time, offset)]
                                    else: zero = twin_a[offset]
                                    row = effect_row(ibase, offset, zero, observation, normalization); per_steps.append(row); step_effects.append(row)
                                primary_values = [x["primary_step_effect"] for x in step_effects]
                                summary = {**ibase, "epsilon_fraction": config["joint_range_fraction"], "raw_direction": direction["raw_direction"], "unit_direction": direction["unit_direction"],
                                    "delta_q": signed.tolist(), "q_before": q_before.tolist(), "q_after": q_after.tolist(), "joint_limit_valid": bool(np.all(q_after >= lower) and np.all(q_after <= upper)),
                                    "non_arm_component_linf_json": json.dumps(preservation, sort_keys=True), "non_arm_max_linf": max(preservation.values()),
                                    "matched_zero_run_id": args.run_id if args.mode == "dry" else args.zero_run.name, "matched_zero_branch_key": f"{trajectory['task']}|{trajectory['episode']}|{branch_time}",
                                    "continuation_length": len(step_effects), "primary_remaining_horizon_mean": float(np.mean(primary_values)), "primary_remaining_horizon_maximum": float(np.max(primary_values)),
                                    "fraction_steps_above_threshold": float(np.mean(np.asarray(primary_values) > primary_spec["meaningful_effect_threshold"])),
                                    "terminal_object_position_l2": step_effects[-1]["task_object_position_l2"], "terminal_object_orientation_geodesic_mean": step_effects[-1]["task_object_orientation_geodesic_mean"],
                                    "predicate_divergence_fraction": float(np.mean([x["task_predicate_divergence"] for x in step_effects])),
                                    "terminal_zero_success": step_effects[-1]["zero_task_predicate"], "terminal_perturbed_success": step_effects[-1]["perturbed_task_predicate"],
                                    "success_flip": bool(step_effects[-1]["zero_task_predicate"] != step_effects[-1]["perturbed_task_predicate"]), "all_states_finite": all(x["all_states_finite"] for x in step_effects)}
                                intervention_summaries.append(summary)
                                if not summary["all_states_finite"] or not summary["joint_limit_valid"] or summary["non_arm_max_linf"] > config["non_arm_integration_linf_max"]: failures.append(summary)
                print(json.dumps({"mode": args.mode, "trajectory": f"{trajectory['task']}/{trajectory['episode']}", "branches": len(branches), "zero_branches_total": len(zero_summaries), "interventions_total": len(intervention_summaries)}, sort_keys=True))
                env.close(); env = None
            artifacts = run_dir / "artifacts"
            if args.mode in ("zero", "dry"):
                write_parquet(zero_summaries, artifacts / "zero_controls.parquet"); write_parquet(zero_steps, artifacts / "zero_reference_steps.parquet")
            if args.mode in ("full", "dry"):
                write_parquet(intervention_summaries, artifacts / "interventions.parquet"); write_parquet(per_steps, artifacts / "per_step_effects.parquet")
            write_json(artifacts / "failure_examples.json", failures[:100])
            copied = artifacts / "frozen_manifests"; shutil.copytree(manifest_dir, copied)
            expected_branches = (sum(len(t["branches"][:args.max_branches]) for t in trajectories))
            expected_interventions = expected_branches * min(config["directions_per_branch"], args.max_directions or config["directions_per_branch"]) * len(config["signs"])
            if args.mode == "zero":
                criteria = {"all_expected_branches": len(zero_summaries) == expected_branches, "two_zero_continuations_each": all(len(x["zero_continuation_ids"]) == 2 for x in zero_summaries),
                    "integration_exact_gate": all(x["maximum_integration_l2"] <= config["zero_integration_l2_max"] for x in zero_summaries),
                    "physical_channels_exact": all(max(x["maximum_arm_q_l2"], x["maximum_arm_qvel_l2"], x["maximum_eef_position_l2"], x["maximum_eef_orientation_geodesic"], x["maximum_task_object_position_l2"], x["maximum_task_object_orientation_geodesic_mean"]) <= 1e-12 for x in zero_summaries),
                    "contact_and_predicate_exact": all(x["maximum_contact_pair_symmetric_difference_count"] == 0 and x["maximum_raw_contact_count_difference"] == 0 and not x["any_task_predicate_divergence"] for x in zero_summaries),
                    "success_agreement": all(x["terminal_success_a"] == x["terminal_success_b"] for x in zero_summaries), "all_finite": all(x["all_states_finite"] for x in zero_summaries)}
            elif args.mode == "full":
                criteria = {"all_expected_interventions": len(intervention_summaries) == expected_interventions, "all_108_branches": len({(x["task"], x["episode"], x["branch_time"]) for x in intervention_summaries}) == config["expected_branches"],
                    "four_directions_two_signs_each": all(sum(1 for x in intervention_summaries if (x["task"], x["episode"], x["branch_time"]) == key) == 8 for key in {(x["task"], x["episode"], x["branch_time"]) for x in intervention_summaries}),
                    "only_arm_q_changed": all(x["non_arm_max_linf"] <= config["non_arm_integration_linf_max"] for x in intervention_summaries), "joint_limits_valid": all(x["joint_limit_valid"] for x in intervention_summaries),
                    "all_finite": all(x["all_states_finite"] for x in intervention_summaries), "no_failures": not failures}
            else:
                criteria = {"dry_zero_completed": len(zero_summaries) > 0, "dry_interventions_completed": len(intervention_summaries) > 0, "no_failures": not failures}
            metrics = {"run_id": args.run_id, "status": "completed", "mode": args.mode, "gate": {"passed": all(criteria.values()), "criteria": criteria},
                       "zero_branch_count": len(zero_summaries), "zero_reference_step_count": len(zero_steps), "intervention_count": len(intervention_summaries), "per_step_effect_count": len(per_steps),
                       "maximum_non_arm_linf": max([x["non_arm_max_linf"] for x in intervention_summaries] or [0.0]), "success_flip_count": sum(x["success_flip"] for x in intervention_summaries)}
        write_run_record(run_dir, config=run_config, command=shlex.join([sys.executable, *sys.argv]), environment=environment, git_state=git_state, metrics=metrics, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue())
        return 0 if metrics["gate"]["passed"] else 2
    except Exception as exc:
        captured_stderr.write(traceback.format_exc())
        metrics = {"run_id": args.run_id, "status": "failed", "mode": args.mode, "gate": {"passed": False}, "error": repr(exc)}
        write_run_record(run_dir, config=run_config, command=shlex.join([sys.executable, *sys.argv]), environment=environment, git_state=git_state, metrics=metrics, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue())
        return 1
    finally:
        if env is not None: env.close()


if __name__ == "__main__":
    raise SystemExit(main())
