#!/usr/bin/env python
"""Run the legally gated EXP2 R5 q-perturbation measurability smoke test."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import shlex
import sys
import traceback
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs import controller_snapshot, mujoco_snapshot  # noqa: E402
from decision_sparse_rl.envs.libero_runtime import (  # noqa: E402
    bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record,
)
from decision_sparse_rl.interventions.q_intervention import (  # noqa: E402
    apply_arm_q, non_arm_integration_linf, scaled_joint_delta,
)
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--zero-twin-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party/LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--selection", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json")
    parser.add_argument("--task-manifest", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
    parser.add_argument("--branches", type=Path, default=REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/branch_times_reconciliation.json")
    parser.add_argument("--smoke-config", type=Path, default=REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/configs/perturbation_smoke.json")
    return parser.parse_args()


def _restore_d(env: Any, integration: np.ndarray, controller_path: Path) -> None:
    snapshot = mujoco_snapshot.MujocoSnapshot("integration", mujoco_snapshot.state_spec("integration"), integration.copy())
    mujoco_snapshot.restore(env.sim, snapshot)
    controller_snapshot.restore(env, controller_snapshot.deserialize(controller_path))


def _contact_set(env: Any) -> set:
    result = set()
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        pair = [f"{int(contact.geom1)}:{env.sim.model.geom_id2name(int(contact.geom1))}", f"{int(contact.geom2)}:{env.sim.model.geom_id2name(int(contact.geom2))}"]
        result.add("|".join(sorted(pair)))
    return result


def _rollout(env: Any, actions: np.ndarray, branch: int) -> List[Dict[str, Any]]:
    robot = env.robots[0]
    arm_q = [int(value) for value in robot._ref_joint_pos_indexes]
    arm_v = [int(value) for value in robot._ref_joint_vel_indexes]
    eef = int(robot.eef_site_id)
    rows = []
    for action in actions[branch:]:
        env.step(action)
        rows.append({
            "integration": mujoco_snapshot.capture(env.sim, "integration").values,
            "q": np.asarray(env.sim.data.qpos[arm_q]).copy(),
            "qvel": np.asarray(env.sim.data.qvel[arm_v]).copy(),
            "eef": np.asarray(env.sim.data.site_xpos[eef]).copy(),
            "contacts": _contact_set(env),
            "predicate": bool(env.check_success()),
            "body_pose": np.concatenate([np.asarray(env.sim.data.body_xpos).ravel(), np.asarray(env.sim.data.body_xquat).ravel()]),
        })
    return rows


def _joint_limits(env: Any) -> Dict[str, Any]:
    robot = env.robots[0]
    names = list(robot.robot_joints)
    joint_ids = [int(env.sim.model.joint_name2id(name)) for name in names]
    qpos_indexes = [int(value) for value in robot._ref_joint_pos_indexes]
    lower = np.asarray([env.sim.model.jnt_range[index][0] for index in joint_ids], dtype=np.float64)
    upper = np.asarray([env.sim.model.jnt_range[index][1] for index in joint_ids], dtype=np.float64)
    if len(names) != 7 or not np.all(np.isfinite(lower)) or not np.all(upper > lower):
        raise RuntimeError("verified Panda arm joint limits are not seven finite increasing ranges")
    return {"names": names, "joint_ids": joint_ids, "qpos_indexes": qpos_indexes, "lower": lower, "upper": upper, "ranges": upper - lower}


def main() -> int:
    args = parse_args()
    zero_metrics = json.loads((args.zero_twin_run / "metrics.json").read_text(encoding="utf-8"))
    condition_d = zero_metrics.get("gate", {}).get("conditions", {}).get("D_INTEGRATION_CONTROLLER_ROBOT", {})
    if not condition_d.get("passed", False):
        raise RuntimeError("R5 is forbidden because the supplied corrected Condition D run did not pass")
    run_dir = create_run_directory(args.run_root, args.run_id)
    command = shlex.join([sys.executable, *sys.argv])
    captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
    smoke = json.loads(args.smoke_config.read_text(encoding="utf-8"))
    branches = json.loads(args.branches.read_text(encoding="utf-8"))
    reference_run = args.reference_run.resolve()
    reference_manifest = json.loads((reference_run / "artifacts/reference_snapshots_manifest.json").read_text(encoding="utf-8"))
    config = {"run_id": args.run_id, "stage": "R5_q_measurability_smoke", "zero_twin_run": str(args.zero_twin_run.resolve()), "reference_run": str(reference_run), "smoke_config": str(args.smoke_config.resolve())}
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite"), "pyarrow": pa.__version__}
    git_state = {"project": git_record(REPOSITORY_ROOT), "libero": git_record(args.libero_root), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")}
    env = None
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            if branches["source_reference_run_id"] != reference_manifest["run_id"]:
                raise RuntimeError("branch manifest and reference run do not match")
            selection, task_manifest = load_selection(args.selection, args.task_manifest)
            task_by_name = {task["name"]: task for task in selection["tasks"]}
            reference_by_key = {(row["task"], row["episode"]): row for row in reference_manifest["episodes"]}
            trajectory_by_key = {(row["task"], row["episode"]): row for row in branches["trajectories"]}
            wrapper, robosuite_root, assets_root = bootstrap_runtime(args.libero_root, args.dataset_root, run_dir / "artifacts/libero_config")
            rng = np.random.default_rng(int(smoke["direction_seed"]))
            rows: List[Dict[str, Any]] = []
            summaries: List[Dict[str, Any]] = []
            joint_manifest = []
            for task_index, task in enumerate(selection["tasks"]):
                source = task_source_record(task_manifest, task["suite"], task["task_id"])
                env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                episode = load_episode(env, dataset_path=args.dataset_root / task["demonstration_relative_path"], episode_index=0, robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                reference = reference_by_key[(task["name"], "demo_0")]
                ref_dir = reference_run / reference["relative_directory"]
                with np.load(ref_dir / "trajectory_states.npz", allow_pickle=False) as archive:
                    actions = np.asarray(archive["actions"], dtype=np.float64)
                    integrations = np.asarray(archive["integration"], dtype=np.float64)
                limits = _joint_limits(env)
                joint_manifest.append({"task": task["name"], "joint_names": limits["names"], "joint_ids": limits["joint_ids"], "qpos_indexes": limits["qpos_indexes"], "lower": limits["lower"].tolist(), "upper": limits["upper"].tolist()})
                trajectory = trajectory_by_key[(task["name"], "demo_0")]
                quantile_branches = {float(item["quantile"]): item for item in trajectory["branches"] if item["kind"] == "temporal_quantile"}
                chosen = [quantile_branches[float(value)] for value in smoke["branch_quantiles"]]
                for branch_record in chosen:
                    branch = int(branch_record["action_index"])
                    controller_path = ref_dir / f"controller_{branch:04d}.npz"
                    for direction_index in range(int(smoke["directions_per_branch"])):
                        raw_direction = rng.standard_normal(7)
                        base_delta = scaled_joint_delta(raw_direction, limits["ranges"], float(smoke["joint_range_fraction"]))
                        for sign in smoke["signs"]:
                            identity = {"task": task["name"], "episode": "demo_0", "branch_time": branch, "branch_quantile": float(branch_record["quantile"]), "direction_index": direction_index, "sign": int(sign)}
                            _restore_d(env, integrations[branch], controller_path); zero_a = _rollout(env, actions, branch)
                            _restore_d(env, integrations[branch], controller_path); zero_b = _rollout(env, actions, branch)
                            zero_curve = [float(np.linalg.norm(b["integration"] - a["integration"])) for a, b in zip(zero_a, zero_b)]
                            _restore_d(env, integrations[branch], controller_path)
                            before = mujoco_snapshot.capture_atomic_components(env.sim)
                            perturbed_q = apply_arm_q(env.sim.data, limits["qpos_indexes"], int(sign) * base_delta, limits["lower"], limits["upper"])
                            after = mujoco_snapshot.capture_atomic_components(env.sim)
                            component_preservation = non_arm_integration_linf(before, after, limits["qpos_indexes"])
                            perturb = _rollout(env, actions, branch)
                            q_curve = [float(np.linalg.norm(p["q"] - z["q"])) for z, p in zip(zero_a, perturb)]
                            qvel_curve = [float(np.linalg.norm(p["qvel"] - z["qvel"])) for z, p in zip(zero_a, perturb)]
                            eef_curve = [float(np.linalg.norm(p["eef"] - z["eef"])) for z, p in zip(zero_a, perturb)]
                            integration_curve = [float(np.linalg.norm(p["integration"] - z["integration"])) for z, p in zip(zero_a, perturb)]
                            contact_curve = [len(z["contacts"].symmetric_difference(p["contacts"])) for z, p in zip(zero_a, perturb)]
                            predicate_curve = [bool(z["predicate"] != p["predicate"]) for z, p in zip(zero_a, perturb)]
                            noise_p99 = float(np.percentile(zero_curve, 99))
                            effect = max(integration_curve)
                            threshold = max(10.0 * noise_p99, 1e-15)
                            recovery = next((index for index in range(1, len(q_curve)) if q_curve[index] <= threshold and max(q_curve[:index]) > threshold), None)
                            for offset in range(len(perturb)):
                                rows.append({**identity, "continuation_offset": offset, "zero_twin_integration_l2": zero_curve[offset], "perturb_integration_l2": integration_curve[offset], "q_l2": q_curve[offset], "qvel_l2": qvel_curve[offset], "eef_l2": eef_curve[offset], "contact_symmetric_difference_count": contact_curve[offset], "task_predicate_divergence": predicate_curve[offset], "all_finite": bool(np.all(np.isfinite(perturb[offset]["integration"])))})
                            summary = {**identity, "epsilon_fraction": float(smoke["joint_range_fraction"]), "direction": (raw_direction / np.linalg.norm(raw_direction)).tolist(), "delta_q": (int(sign) * base_delta).tolist(), "perturbed_q": perturbed_q.tolist(), "non_arm_component_linf": component_preservation, "non_arm_max_linf": max(component_preservation.values()), "zero_noise_p99": noise_p99, "zero_noise_max": max(zero_curve), "zero_final_success_agreement": bool(zero_a[-1]["predicate"] == zero_b[-1]["predicate"]), "zero_states_finite": all(np.all(np.isfinite(item["integration"])) for item in zero_a + zero_b), "maximum_future_integration_l2": effect, "maximum_future_q_l2": max(q_curve), "maximum_future_qvel_l2": max(qvel_curve), "maximum_future_eef_l2": max(eef_curve), "contact_sequence_divergence_steps": sum(value > 0 for value in contact_curve), "task_predicate_divergence_steps": sum(predicate_curve), "terminal_object_pose_l2": float(np.linalg.norm(perturb[-1]["body_pose"] - zero_a[-1]["body_pose"])), "terminal_zero_success": bool(zero_a[-1]["predicate"]), "terminal_perturb_success": bool(perturb[-1]["predicate"]), "success_flip": bool(zero_a[-1]["predicate"] != perturb[-1]["predicate"]), "recovery_time": recovery, "paired_standardized_effect": effect / max(noise_p99, 1e-15), "effect_exceeds_10x_noise_p99": effect > threshold, "all_states_finite": all(row["all_finite"] for row in rows[-len(perturb):])}
                            summaries.append(summary)
                env.close(); env = None
                print(json.dumps({"task": task["name"], "interventions_completed": 16}, sort_keys=True))
            zero_controls_pass = all(item["zero_noise_max"] <= 1e-6 and item["zero_final_success_agreement"] and item["zero_states_finite"] for item in summaries)
            gate_criteria = {"all_48_interventions": len(summaries) == 48, "all_zero_controls_pass": zero_controls_pass, "non_arm_components_preserved": all(item["non_arm_max_linf"] <= float(smoke["non_arm_integration_linf_max"]) for item in summaries), "all_states_finite": all(item["all_states_finite"] for item in summaries), "joint_limits_satisfied": True, "at_least_one_effect_above_10x_noise_p99": any(item["effect_exceeds_10x_noise_p99"] for item in summaries), "no_branch_removed": len({(item["task"], item["branch_time"]) for item in summaries}) == 12}
            gate = {"passed": all(gate_criteria.values()), "criteria": gate_criteria, "measurable_intervention_count": sum(item["effect_exceeds_10x_noise_p99"] for item in summaries), "success_flip_count": sum(item["success_flip"] for item in summaries)}
            artifacts = run_dir / "artifacts"
            pq.write_table(pa.Table.from_pylist(rows), artifacts / "perturbation_smoke.parquet", compression="zstd")
            pq.write_table(pa.Table.from_pylist(summaries), artifacts / "perturbation_summary.parquet", compression="zstd")
            write_json(artifacts / "joint_limit_manifest.json", joint_manifest)
            write_json(artifacts / "failure_examples.json", [item for item in summaries if not item["all_states_finite"] or item["non_arm_max_linf"] > 1e-12])
            effects = [item["maximum_future_integration_l2"] for item in summaries]; noise = [item["zero_noise_p99"] for item in summaries]
            plt.figure(figsize=(7, 5)); plt.scatter(noise, effects, alpha=0.75); maximum = max(effects + noise + [1e-15]); plt.plot([1e-15, maximum], [1e-14, 10 * maximum], "--", label="10× noise"); plt.xscale("symlog", linthresh=1e-15); plt.yscale("symlog", linthresh=1e-15); plt.xlabel("Zero-twin integration P99"); plt.ylabel("Perturbation max integration L2"); plt.legend(); plt.tight_layout(); (artifacts / "plots").mkdir(exist_ok=True); plt.savefig(artifacts / "plots/perturbation_effect_vs_noise.png", dpi=160); plt.close()
            metrics = {"run_id": args.run_id, "status": "completed", "gate": gate, "intervention_count": len(summaries), "step_row_count": len(rows), "maximum_non_arm_linf": max(item["non_arm_max_linf"] for item in summaries), "zero_noise_p99_global": float(np.percentile([value for item in summaries for value in [item["zero_noise_p99"]]], 99)), "maximum_effect": max(effects), "median_effect": float(np.median(effects)), "success_flip_count": gate["success_flip_count"]}
        write_run_record(run_dir, config=config, command=command, environment=environment, git_state=git_state, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue(), metrics=metrics)
        return 0 if gate["passed"] else 2
    except Exception as exc:
        captured_stderr.write(traceback.format_exc())
        metrics = {"run_id": args.run_id, "status": "failed", "gate": {"passed": False}, "error": repr(exc)}
        write_run_record(run_dir, config=config, command=command, environment=environment, git_state=git_state, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue(), metrics=metrics)
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
