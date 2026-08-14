#!/usr/bin/env python
"""Execute EXP4 dry, matched-zero, or full direction-resolved continuations."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.exp3 import run_criticality as exp3  # noqa: E402
from decision_sparse_rl.envs import mujoco_snapshot  # noqa: E402
from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record  # noqa: E402
from decision_sparse_rl.interventions.q_intervention import apply_arm_q, non_arm_integration_linf  # noqa: E402
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("dry", "zero", "full"), required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--zero-run", type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=REPOSITORY_ROOT / "experiments/exp4_replicated_progress_criticality/manifests")
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party/LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--selection", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json")
    parser.add_argument("--task-manifest", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--max-branches", type=int)
    parser.add_argument("--max-directions", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def signed_rotvec_matrix(zero: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    left = np.asarray(zero, dtype=np.float64).reshape(3, 3); right = np.asarray(perturbed, dtype=np.float64).reshape(3, 3)
    if np.array_equal(left, right): return np.zeros(3, dtype=np.float64)
    return Rotation.from_matrix(right @ left.T).as_rotvec()


def signed_rotvec_quaternion(zero_wxyz: np.ndarray, perturbed_wxyz: np.ndarray) -> np.ndarray:
    left = np.asarray(zero_wxyz, dtype=np.float64)[[1, 2, 3, 0]]; right = np.asarray(perturbed_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    if np.array_equal(left, right) or np.array_equal(left, -right): return np.zeros(3, dtype=np.float64)
    return (Rotation.from_quat(right) * Rotation.from_quat(left).inv()).as_rotvec()


def signed_output_vector(zero: Dict[str, Any], perturbed: Dict[str, Any], normalization: Dict[str, Any], task: str, joint_ranges: np.ndarray) -> np.ndarray:
    den = normalization["continuous_denominators"]
    parts = [
        (perturbed["q"] - zero["q"]) / joint_ranges,
        (perturbed["qvel"] - zero["qvel"]) / den["arm_qvel_l2"]["value"],
        (perturbed["eef_position"] - zero["eef_position"]) / den["eef_position_l2"]["value"],
        signed_rotvec_matrix(zero["eef_orientation"], perturbed["eef_orientation"]) / np.pi,
        ((perturbed["object_positions"] - zero["object_positions"]) / den["task_object_position_l2"]["value_by_task"][task]).reshape(-1),
    ]
    parts.append(np.concatenate([signed_rotvec_quaternion(a, b) / np.pi for a, b in zip(zero["object_quaternions"], perturbed["object_quaternions"])]))
    result = np.concatenate(parts)
    if np.any(~np.isfinite(result)): raise RuntimeError("non-finite signed physical output vector")
    return result


def write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows: raise RuntimeError(f"refusing empty Parquet shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def merge_shards(paths: List[Path], output: Path) -> None:
    writer = None
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None: writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None: writer.close()


def main() -> int:
    args = parse_args()
    if args.mode == "full" and args.zero_run is None: raise ValueError("full mode requires --zero-run")
    run_dir = create_run_directory(args.run_root, args.run_id); stdout, stderr = io.StringIO(), io.StringIO(); started = time.perf_counter()
    manifest_dir = args.manifest_dir.resolve(); reference = args.reference_run.resolve()
    config = json.loads((REPOSITORY_ROOT / "experiments/exp4_replicated_progress_criticality/configs/exp4.json").read_text())
    branch_manifest = json.loads((manifest_dir / "branch_manifest.json").read_text()); directions = json.loads((manifest_dir / "direction_basis_manifest.json").read_text()); normalization = json.loads((manifest_dir / "effect_normalization.json").read_text()); channel_schema = json.loads((manifest_dir / "effect_channel_schema.json").read_text()); primary_spec = json.loads((manifest_dir / "primary_metric_spec.json").read_text())
    limits_path = REPOSITORY_ROOT / "runs/exp2_r5_q_smoke_20260814T012633/artifacts/joint_limit_manifest.json"; joint_limits = {x["task"]: x for x in json.loads(limits_path.read_text())}
    run_config = {"run_id": args.run_id, "stage": f"EXP4_{args.mode}", "mode": args.mode, "reference_run": reference.name, "zero_run": None if args.zero_run is None else args.zero_run.resolve().name, "manifest_dir": str(manifest_dir), "max_trajectories": args.max_trajectories, "max_branches": args.max_branches, "max_directions": args.max_directions, "fixed_config": config, "sharding": "one immutable task/demo shard written before merge"}
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "pyarrow": pa.__version__, "scipy": importlib.metadata.version("scipy"), "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite"), "compute_note": "MuJoCo CPU; RTX 4090 reserved for post-processing after equivalence audit"}
    git_state = {"project": git_record(REPOSITORY_ROOT), "libero": git_record(args.libero_root), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")}
    env = None
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            ref_manifest = json.loads((reference / "artifacts/reference_snapshots_manifest.json").read_text());
            if not ref_manifest["gate"]["passed"]: raise RuntimeError("reference gate failed")
            selection, task_manifest = load_selection(args.selection, args.task_manifest); selected = {x["name"]: x for x in selection["tasks"]}; refs = {(x["task"], x["episode"]): x for x in ref_manifest["episodes"]}; trajectories = branch_manifest["trajectories"][:args.max_trajectories]
            wrapper, robosuite_root, assets_root = bootstrap_runtime(args.libero_root, args.dataset_root, run_dir / "artifacts/libero_config")
            zero_lookup: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
            if args.mode == "full":
                zero_metrics = json.loads((args.zero_run / "metrics.json").read_text());
                if not zero_metrics.get("gate", {}).get("passed"): raise RuntimeError("supplied zero gate failed")
                for row in pq.read_table(args.zero_run / "artifacts/zero_reference_steps.parquet").to_pylist(): zero_lookup[(row["task"], row["episode"], row["branch_time"], row["continuation_offset"])] = exp3.reference_from_row(row)
                shutil.copy2(args.zero_run / "artifacts/zero_controls.parquet", run_dir / "artifacts/zero_controls.parquet")
            direction_lookup: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
            for row in directions["directions"]: direction_lookup.setdefault((row["task"], row["episode"], int(row["branch_time"])), []).append(row)
            for rows in direction_lookup.values(): rows.sort(key=lambda x: x["execution_position"])
            zero_shards: List[Path] = []; zero_step_shards: List[Path] = []; intervention_shards: List[Path] = []; effect_shards: List[Path] = []; all_failures = []; all_zero_values = []; terminal_zero_values = []; intervention_count = 0; step_count = 0; zero_branch_count = 0; zero_step_count = 0; success_flips = 0; global_nonarm = 0.0
            for trajectory in trajectories:
                task = selected[trajectory["task"]]; source = task_source_record(task_manifest, task["suite"], task["task_id"]); env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"]))); episode_index = int(trajectory["episode"].split("_")[-1]); load_episode(env, dataset_path=args.dataset_root / task["demonstration_relative_path"], episode_index=episode_index, robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                ref = refs[(trajectory["task"], trajectory["episode"])]; ref_dir = reference / ref["relative_directory"]
                with np.load(ref_dir / "trajectory_states.npz", allow_pickle=False) as archive: actions = np.asarray(archive["actions"], dtype=np.float64); integrations = np.asarray(archive["integration"], dtype=np.float64)
                body_ids = [int(env.sim.model.body_name2id(name)) for name in channel_schema["task_object_audit"][trajectory["task"]]["bodies"]]; branches = trajectory["branches"][:args.max_branches]; limits = joint_limits[trajectory["task"]]; lower = np.asarray(limits["lower"]); upper = np.asarray(limits["upper"]); indexes = limits["qpos_indexes"]; joint_ranges = upper - lower
                zsum: List[Dict[str, Any]] = []; zsteps: List[Dict[str, Any]] = []; isum: List[Dict[str, Any]] = []; esteps: List[Dict[str, Any]] = []; failures: List[Dict[str, Any]] = []
                for branch in branches:
                    base = exp3.identity(trajectory, branch); base["physical_progress_raw"] = branch["physical_progress_raw"]; base["physical_progress_clipped"] = branch["physical_progress_clipped"]; branch_time = base["branch_time"]; controller = ref_dir / f"controller_{branch_time:04d}.npz"
                    if args.mode in ("zero", "dry"):
                        exp3.restore_d(env, integrations[branch_time], controller); twin_a = exp3.rollout(env, actions, branch_time, body_ids); exp3.restore_d(env, integrations[branch_time], controller); twin_b = exp3.rollout(env, actions, branch_time, body_ids); differences = [exp3.compare_zero(a, b) for a, b in zip(twin_a, twin_b)]
                        for offset, observation in enumerate(twin_a): zsteps.append(exp3.zero_step_record(base, offset, observation))
                        terminal_object = float(differences[-1]["task_object_position_l2"] + differences[-1]["task_object_orientation_geodesic_mean"])
                        zsum.append({**base, "zero_continuation_ids": [f"{trajectory['task']}|{trajectory['episode']}|{branch_time}|A", f"{trajectory['task']}|{trajectory['episode']}|{branch_time}|B"], "maximum_integration_l2": max(x["integration_l2"] for x in differences), "median_integration_l2": float(np.median([x["integration_l2"] for x in differences])), "p95_integration_l2": float(np.percentile([x["integration_l2"] for x in differences], 95)), "maximum_arm_q_l2": max(x["arm_q_l2"] for x in differences), "maximum_arm_qvel_l2": max(x["arm_qvel_l2"] for x in differences), "maximum_eef_position_l2": max(x["eef_position_l2"] for x in differences), "maximum_eef_orientation_geodesic": max(x["eef_orientation_geodesic"] for x in differences), "maximum_task_object_position_l2": max(x["task_object_position_l2"] for x in differences), "maximum_task_object_orientation_geodesic_mean": max(x["task_object_orientation_geodesic_mean"] for x in differences), "maximum_contact_pair_symmetric_difference_count": max(x["contact_pair_symmetric_difference_count"] for x in differences), "maximum_raw_contact_count_difference": max(x["raw_contact_count_difference"] for x in differences), "any_task_predicate_divergence": any(x["task_predicate_divergence"] for x in differences), "terminal_object_pose_l2": terminal_object, "terminal_success_a": twin_a[-1]["predicate"], "terminal_success_b": twin_b[-1]["predicate"], "all_states_finite": all(np.all(np.isfinite(x["integration"])) for x in twin_a + twin_b)})
                        all_zero_values.extend(x["integration_l2"] for x in differences); terminal_zero_values.append(terminal_object)
                    if args.mode in ("full", "dry"):
                        rows = direction_lookup[(trajectory["task"], trajectory["episode"], branch_time)][:args.max_directions]
                        for direction in rows:
                            for sign in config["signs"]:
                                signed = int(sign) * np.asarray(direction["unsigned_delta_q"], dtype=np.float64); intervention_id = f"{trajectory['task']}|{trajectory['episode']}|{branch_time}|d{direction['direction_index']}|s{int(sign):+d}"; ibase = {**base, "intervention_id": intervention_id, "direction_index": int(direction["direction_index"]), "direction_role": direction["direction_role"], "execution_position": int(direction["execution_position"]), "sign": int(sign)}
                                exp3.restore_d(env, integrations[branch_time], controller); before = mujoco_snapshot.capture_atomic_components(env.sim); q_before = np.asarray(env.sim.data.qpos[indexes]).copy(); q_after = apply_arm_q(env.sim.data, indexes, signed, lower, upper); after = mujoco_snapshot.capture_atomic_components(env.sim); preservation = non_arm_integration_linf(before, after, indexes); perturb = exp3.rollout(env, actions, branch_time, body_ids); effects = []
                                for offset, observation in enumerate(perturb):
                                    zero = zero_lookup[(trajectory["task"], trajectory["episode"], branch_time, offset)] if args.mode == "full" else twin_a[offset]; row = exp3.effect_row(ibase, offset, zero, observation, normalization); vector = signed_output_vector(zero, observation, normalization, trajectory["task"], joint_ranges); row["signed_normalized_physical_output_vector"] = vector.tolist(); esteps.append(row); effects.append(row)
                                primary_values = [x["primary_step_effect"] for x in effects]; fixed = primary_values[: min(20, len(primary_values))]; signed_mean = np.mean(np.asarray([x["signed_normalized_physical_output_vector"] for x in effects]), axis=0)
                                summary = {**ibase, "epsilon_fraction": config["joint_range_fraction"], "unit_direction_scaled_coordinates": direction["unit_direction_scaled_coordinates"], "delta_q": signed.tolist(), "q_before": q_before.tolist(), "q_after": q_after.tolist(), "joint_limit_valid": bool(np.all(q_after >= lower) and np.all(q_after <= upper)), "non_arm_component_linf_json": json.dumps(preservation, sort_keys=True), "non_arm_max_linf": max(preservation.values()), "matched_zero_run_id": args.run_id if args.mode == "dry" else args.zero_run.name, "continuation_length": len(effects), "remaining_horizon_normalized": len(effects) / trajectory["trajectory_length"], "primary_remaining_horizon_mean": float(np.mean(primary_values)), "primary_remaining_horizon_maximum": float(np.max(primary_values)), "primary_fixed_20_step_mean": float(np.mean(fixed)), "fraction_steps_above_threshold": float(np.mean(np.asarray(primary_values) > primary_spec["meaningful_effect_threshold"])), "signed_output_remaining_horizon_mean": signed_mean.tolist(), "terminal_object_position_l2": effects[-1]["task_object_position_l2"], "terminal_object_orientation_geodesic_mean": effects[-1]["task_object_orientation_geodesic_mean"], "predicate_divergence_fraction": float(np.mean([x["task_predicate_divergence"] for x in effects])), "terminal_zero_success": effects[-1]["zero_task_predicate"], "terminal_perturbed_success": effects[-1]["perturbed_task_predicate"], "success_flip": bool(effects[-1]["zero_task_predicate"] != effects[-1]["perturbed_task_predicate"]), "all_states_finite": all(x["all_states_finite"] for x in effects)}; isum.append(summary)
                                if not summary["all_states_finite"] or not summary["joint_limit_valid"] or summary["non_arm_max_linf"] > config["non_arm_integration_linf_max"]: failures.append(summary)
                shard = run_dir / "artifacts/shards" / trajectory["task"] / trajectory["episode"]
                if zsum: zp = shard / "zero_controls.parquet"; zsp = shard / "zero_reference_steps.parquet"; write_parquet(zsum, zp); write_parquet(zsteps, zsp); zero_shards.append(zp); zero_step_shards.append(zsp)
                if isum: ip = shard / "interventions.parquet"; ep = shard / "per_step_effects.parquet"; write_parquet(isum, ip); write_parquet(esteps, ep); intervention_shards.append(ip); effect_shards.append(ep)
                write_json(shard / "failure_examples.json", failures); all_failures.extend(failures); zero_branch_count += len(zsum); zero_step_count += len(zsteps); intervention_count += len(isum); step_count += len(esteps); success_flips += sum(x["success_flip"] for x in isum); global_nonarm = max(global_nonarm, max([x["non_arm_max_linf"] for x in isum] or [0.0])); print(json.dumps({"trajectory": f"{trajectory['task']}/{trajectory['episode']}", "zero_branches": len(zsum), "interventions": len(isum), "effect_steps": len(esteps)}, sort_keys=True)); env.close(); env = None
            artifacts = run_dir / "artifacts"
            if zero_shards: merge_shards(zero_shards, artifacts / "zero_controls.parquet"); merge_shards(zero_step_shards, artifacts / "zero_reference_steps.parquet")
            if intervention_shards: merge_shards(intervention_shards, artifacts / "interventions.parquet"); merge_shards(effect_shards, artifacts / "per_step_effects.parquet")
            write_json(artifacts / "failure_examples.json", all_failures[:100]); shutil.copytree(manifest_dir, artifacts / "frozen_manifests")
            expected_branches = sum(len(t["branches"][:args.max_branches]) for t in trajectories); direction_count = min(8, args.max_directions or 8); expected_interventions = expected_branches * direction_count * 2
            if args.mode == "zero":
                stats = {"median": float(np.median(all_zero_values)), "p95": float(np.percentile(all_zero_values, 95)), "maximum": float(np.max(all_zero_values)), "terminal_object_pose_p95": float(np.percentile(terminal_zero_values, 95))}
                criteria = {"all_expected_branches": zero_branch_count == expected_branches, "two_zero_continuations_each": all(len(x["zero_continuation_ids"]) == 2 for path in zero_shards for x in pq.read_table(path).to_pylist()), "median_gate": stats["median"] <= config["zero_median_max"], "p95_gate": stats["p95"] <= config["zero_p95_max"], "maximum_gate": stats["maximum"] <= config["zero_maximum_max"], "terminal_object_pose_p95_gate": stats["terminal_object_pose_p95"] <= config["zero_terminal_object_pose_p95_max"], "expected_exact_zero": stats["maximum"] == 0.0, "no_failures": not all_failures}
            elif args.mode == "full":
                keys = set(); coverage = {}
                for path in intervention_shards:
                    for x in pq.read_table(path).to_pylist(): key = (x["task"], x["episode"], x["branch_time"]); keys.add(key); coverage.setdefault(key, set()).add((x["direction_index"], x["sign"]))
                criteria = {"all_expected_interventions": intervention_count == expected_interventions, "all_expected_branches": len(keys) == expected_branches, "all_directions_two_signs_each": all(len(x) == direction_count * 2 for x in coverage.values()), "only_arm_q_changed": global_nonarm <= config["non_arm_integration_linf_max"], "joint_limits_valid": not all_failures, "all_finite": not all_failures, "no_branch_removed": len(keys) == expected_branches}
                stats = {}
            else:
                criteria = {"dry_zero_completed": zero_branch_count > 0, "dry_interventions_completed": intervention_count > 0, "no_failures": not all_failures}; stats = {}
            raw_names = [name for name in ("zero_controls.parquet", "zero_reference_steps.parquet", "interventions.parquet", "per_step_effects.parquet") if (artifacts / name).exists()]; raw_hashes = {name: sha256(artifacts / name) for name in raw_names}; write_json(artifacts / "raw_artifact_hashes.json", {"schema_version": 1, "locked_before_analysis": True, "sha256": raw_hashes})
            metrics = {"run_id": args.run_id, "status": "completed", "mode": args.mode, "gate": {"passed": all(criteria.values()), "criteria": criteria}, "zero_statistics": stats, "zero_branch_count": zero_branch_count, "zero_reference_step_count": zero_step_count, "intervention_count": intervention_count, "per_step_effect_count": step_count, "maximum_non_arm_linf": global_nonarm, "success_flip_count": success_flips, "raw_artifact_hashes": raw_hashes, "wall_time_seconds": time.perf_counter() - started}
        write_run_record(run_dir, config=run_config, command=shlex.join([sys.executable, *sys.argv]), environment=environment, git_state=git_state, metrics=metrics, stdout=stdout.getvalue(), stderr=stderr.getvalue())
        return 0 if metrics["gate"]["passed"] else 2
    except Exception as exc:
        stderr.write(traceback.format_exc()); metrics = {"run_id": args.run_id, "status": "failed", "mode": args.mode, "gate": {"passed": False}, "error": repr(exc), "wall_time_seconds": time.perf_counter() - started}
        write_run_record(run_dir, config=run_config, command=shlex.join([sys.executable, *sys.argv]), environment=environment, git_state=git_state, metrics=metrics, stdout=stdout.getvalue(), stderr=stderr.getvalue())
        return 1
    finally:
        if env is not None: env.close()


if __name__ == "__main__":
    raise SystemExit(main())
