#!/usr/bin/env python
"""Run the formal EXP2 R4 zero-perturbation matched-twin gate."""

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
from typing import Any, Dict, List, Sequence, Tuple

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
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


CONDITIONS = (
    ("A_LEGACY", "legacy", False),
    ("B_FULLPHYSICS", "fullphysics", False),
    ("C_INTEGRATION", "integration", False),
    ("D_INTEGRATION_CONTROLLER_ROBOT", "integration", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party/LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--selection", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json")
    parser.add_argument("--task-manifest", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
    parser.add_argument("--branches", type=Path, default=REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/branch_times_reconciliation.json")
    parser.add_argument("--gate-config", type=Path, default=REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/configs/zero_twin_gate.json")
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def _contact_set(env: Any) -> set:
    result = set()
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        names = []
        for geom in (int(contact.geom1), int(contact.geom2)):
            names.append(f"{geom}:{env.sim.model.geom_id2name(geom)}")
        result.add("|".join(sorted(names)))
    return result


def _observe(env: Any, reward: float) -> Dict[str, Any]:
    robot = env.robots[0]
    arm = [int(value) for value in robot._ref_joint_pos_indexes]
    arm_vel = [int(value) for value in robot._ref_joint_vel_indexes]
    gripper = [int(value) for value in robot._ref_gripper_joint_pos_indexes]
    eef = int(robot.eef_site_id)
    return {
        "integration": mujoco_snapshot.capture(env.sim, "integration").values,
        "fullphysics": mujoco_snapshot.capture(env.sim, "fullphysics").values,
        "legacy": mujoco_snapshot.capture(env.sim, "legacy").values,
        "components": mujoco_snapshot.capture_atomic_components(env.sim),
        "controller": controller_snapshot.capture(env),
        "qpos": np.asarray(env.sim.data.qpos[arm]).copy(),
        "qvel": np.asarray(env.sim.data.qvel[arm_vel]).copy(),
        "eef_position": np.asarray(env.sim.data.site_xpos[eef]).copy(),
        "eef_orientation": np.asarray(env.sim.data.site_xmat[eef]).copy(),
        "gripper": np.asarray(env.sim.data.qpos[gripper]).copy(),
        "contacts": _contact_set(env),
        "contact_count": int(env.sim.data.ncon),
        "predicate": bool(env.check_success()),
        "reward": float(reward),
        "body_pose": np.concatenate([np.asarray(env.sim.data.body_xpos).ravel(), np.asarray(env.sim.data.body_xquat).ravel()]),
    }


def _restore_condition(env: Any, kind: str, values: np.ndarray, controller_path: Path, restore_controller: bool) -> None:
    snapshot = mujoco_snapshot.MujocoSnapshot(kind=kind, state_spec=0 if kind == "legacy" else mujoco_snapshot.state_spec(kind), values=np.asarray(values).copy())
    mujoco_snapshot.restore(env.sim, snapshot)
    if restore_controller:
        controller_snapshot.restore(env, controller_snapshot.deserialize(controller_path))


def _run_continuation(env: Any, actions: np.ndarray, branch: int) -> Tuple[List[Dict[str, Any]], bool]:
    rows = []
    for action in actions[branch:]:
        _, reward, _, _ = env.step(action)
        rows.append(_observe(env, reward))
    return rows, bool(env.check_success())


def _first_above(values: Sequence[float], threshold: float) -> Any:
    for index, value in enumerate(values):
        if value > threshold:
            return index
    return None


def _bootstrap_ci(values: np.ndarray, statistic: str, seed: int = 0, resamples: int = 1000) -> List[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for index in range(resamples):
        sample = values[rng.integers(0, values.size, values.size)]
        estimates[index] = np.median(sample) if statistic == "median" else np.percentile(sample, 95)
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
        "median_bootstrap_95ci": _bootstrap_ci(array, "median"),
        "p95_bootstrap_95ci": _bootstrap_ci(array, "p95"),
    }


def _compare_pair(
    twin_a: List[Dict[str, Any]],
    twin_b: List[Dict[str, Any]],
    *,
    identity: Dict[str, Any],
    component_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(twin_a) != len(twin_b):
        raise ValueError("twin continuation lengths differ")
    rows = []
    integration_curve = []
    for offset, (left, right) in enumerate(zip(twin_a, twin_b)):
        differences = {name: right[name] - left[name] for name in ("integration", "fullphysics", "legacy", "qpos", "qvel", "eef_position", "eef_orientation", "gripper", "body_pose")}
        controller_errors = controller_snapshot.field_errors(left["controller"], right["controller"])
        integration_l2 = float(np.linalg.norm(differences["integration"]))
        integration_curve.append(integration_l2)
        contact_difference = left["contacts"].symmetric_difference(right["contacts"])
        row = {
            **identity,
            "continuation_offset": offset,
            "absolute_action_index": int(identity["branch_time"] + offset),
            "integration_l2": integration_l2,
            "integration_linf": float(np.max(np.abs(differences["integration"]))),
            "fullphysics_l2": float(np.linalg.norm(differences["fullphysics"])),
            "legacy_l2": float(np.linalg.norm(differences["legacy"])),
            "qpos_l2": float(np.linalg.norm(differences["qpos"])),
            "qvel_l2": float(np.linalg.norm(differences["qvel"])),
            "controller_l2_max_field": float(max(controller_errors.values(), default=0.0)),
            "robot_buffer_l2_max_field": float(max((value for key, value in controller_errors.items() if key.startswith("buffers.")), default=0.0)),
            "eef_position_l2": float(np.linalg.norm(differences["eef_position"])),
            "eef_orientation_l2": float(np.linalg.norm(differences["eef_orientation"])),
            "gripper_l2": float(np.linalg.norm(differences["gripper"])),
            "contact_count_difference": abs(int(left["contact_count"]) - int(right["contact_count"])),
            "contact_pair_symmetric_difference_count": len(contact_difference),
            "contact_pair_symmetric_difference": json.dumps(sorted(contact_difference)),
            "task_predicate_agreement": bool(left["predicate"] == right["predicate"]),
            "reward_absolute_difference": abs(float(left["reward"]) - float(right["reward"])),
            "all_states_finite": bool(np.all(np.isfinite(left["integration"])) and np.all(np.isfinite(right["integration"]))),
        }
        rows.append(row)
        for component, error in mujoco_snapshot.component_errors(left["components"], right["components"]).items():
            component_rows.append({**identity, "continuation_offset": offset, "absolute_action_index": row["absolute_action_index"], "component": component, **error})
    terminal_object = float(np.linalg.norm(twin_b[-1]["body_pose"] - twin_a[-1]["body_pose"]))
    pair = {
        **identity,
        "continuation_length": len(rows),
        "final_success_a": bool(twin_a[-1]["predicate"]),
        "final_success_b": bool(twin_b[-1]["predicate"]),
        "final_success_agreement": bool(twin_a[-1]["predicate"] == twin_b[-1]["predicate"]),
        "terminal_object_pose_l2": terminal_object,
        "first_nonzero": _first_above(integration_curve, 0.0),
        "first_above_1e-12": _first_above(integration_curve, 1e-12),
        "first_above_1e-10": _first_above(integration_curve, 1e-10),
        "first_above_1e-8": _first_above(integration_curve, 1e-8),
        "first_above_1e-6": _first_above(integration_curve, 1e-6),
        "maximum_integration_l2": max(integration_curve),
    }
    return rows, pair


def _phase(normalized_time: float) -> str:
    return "early" if normalized_time < 1 / 3 else ("middle" if normalized_time < 2 / 3 else "late")


def _summarize(step_rows: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]], gate_config: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = gate_config["thresholds"]
    summaries = {}
    for condition, _, _ in CONDITIONS:
        condition_steps = [row for row in step_rows if row["condition"] == condition]
        condition_pairs = [row for row in pair_rows if row["condition"] == condition]
        integration = [row["integration_l2"] for row in condition_steps]
        terminal = [row["terminal_object_pose_l2"] for row in condition_pairs]
        metric_summary = {}
        for metric in ("integration_l2", "qpos_l2", "qvel_l2", "controller_l2_max_field", "eef_position_l2"):
            metric_summary[metric] = _stats([row[metric] for row in condition_steps])
        metric_summary["terminal_object_pose_l2"] = _stats(terminal)
        by_task = {task: _stats([row["integration_l2"] for row in condition_steps if row["task"] == task]) for task in sorted({row["task"] for row in condition_steps})}
        strata = {}
        for key in ("phase", "branch_contact_event", "task"):
            strata[key] = {}
            for value in sorted({str(row[key]) for row in condition_steps}):
                values = [row["integration_l2"] for row in condition_steps if str(row[key]) == value]
                strata[key][value] = _stats(values)
        no_spikes = all(item["p95"] <= thresholds["integration_state_l2_p95_max"] and item["maximum"] <= thresholds["integration_state_l2_maximum_max"] for groups in strata.values() for item in groups.values())
        criteria = {
            "comparison_count_at_least_minimum": len(condition_pairs) >= gate_config["coverage"]["minimum_comparisons"],
            "all_nine_demonstrations": len({(row["task"], row["episode"]) for row in condition_pairs}) == 9,
            "twelve_branches_each": all(sum(1 for row in condition_pairs if row["task"] == task and row["episode"] == episode) >= 36 for task, episode in {(row["task"], row["episode"]) for row in condition_pairs}),
            "final_success_agreement": all(row["final_success_agreement"] for row in condition_pairs),
            "all_integration_states_finite": all(row["all_states_finite"] for row in condition_steps),
            "integration_median": float(np.median(integration)) <= thresholds["integration_state_l2_median_max"],
            "integration_p95": float(np.percentile(integration, 95)) <= thresholds["integration_state_l2_p95_max"],
            "integration_maximum": float(np.max(integration)) <= thresholds["integration_state_l2_maximum_max"],
            "terminal_object_pose_p95": float(np.percentile(terminal, 95)) <= thresholds["terminal_object_pose_l2_p95_max"],
            "no_systematic_stratum_spikes": no_spikes,
        }
        summaries[condition] = {"passed": all(criteria.values()), "criteria": criteria, "comparison_count": len(condition_pairs), "step_count": len(condition_steps), "metrics": metric_summary, "integration_by_task": by_task, "strata": strata}
    selected = "C_INTEGRATION" if summaries["C_INTEGRATION"]["passed"] else ("D_INTEGRATION_CONTROLLER_ROBOT" if summaries["D_INTEGRATION_CONTROLLER_ROBOT"]["passed"] else None)
    return {"conditions": summaries, "selected_condition": selected, "passed": selected is not None}


def _plots(step_rows: List[Dict[str, Any]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [condition for condition, _, _ in CONDITIONS]
    data = [[row["integration_l2"] for row in step_rows if row["condition"] == condition] for condition in labels]
    plt.figure(figsize=(9, 5)); plt.boxplot(data, labels=labels, showfliers=False); plt.yscale("symlog", linthresh=1e-15); plt.ylabel("Integration-state L2"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(plot_dir / "restore_condition_comparison.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 5))
    for condition in labels:
        grouped: Dict[float, List[float]] = {}
        for row in step_rows:
            if row["condition"] == condition:
                grouped.setdefault(float(row["branch_normalized_time"]), []).append(float(row["integration_l2"]))
        xs = sorted(grouped); plt.plot(xs, [np.percentile(grouped[x], 95) for x in xs], ".", label=condition, alpha=0.7)
    plt.yscale("symlog", linthresh=1e-15); plt.xlabel("Branch normalized time"); plt.ylabel("P95 integration L2"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(plot_dir / "zero_twin_error_by_time.png", dpi=160); plt.close()
    metrics = ("integration_l2", "qpos_l2", "qvel_l2", "controller_l2_max_field", "eef_position_l2")
    x = np.arange(len(metrics)); width = 0.18; plt.figure(figsize=(10, 5))
    for index, condition in enumerate(labels):
        values = [np.percentile([row[metric] for row in step_rows if row["condition"] == condition], 95) for metric in metrics]
        plt.bar(x + index * width, values, width, label=condition)
    plt.yscale("symlog", linthresh=1e-15); plt.xticks(x + 1.5 * width, metrics, rotation=20); plt.ylabel("P95 error"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(plot_dir / "zero_twin_error_by_component.png", dpi=160); plt.close()
    plt.figure(figsize=(9, 5))
    positions = np.arange(len(labels)); width = 0.35
    for offset, event in enumerate((False, True)):
        values = [np.percentile([row["integration_l2"] for row in step_rows if row["condition"] == condition and row["branch_contact_event"] == event], 95) for condition in labels]
        plt.bar(positions + (offset - 0.5) * width, values, width, label=f"contact_event={event}")
    plt.yscale("symlog", linthresh=1e-15); plt.xticks(positions, labels, rotation=20); plt.ylabel("P95 integration L2"); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "contact_stratified_error.png", dpi=160); plt.close()


def main() -> int:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("formal gate requires at least three repeats")
    run_dir = create_run_directory(args.run_root, args.run_id)
    command = shlex.join([sys.executable, *sys.argv])
    captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
    reference_run = args.reference_run.resolve()
    branch_manifest = json.loads(args.branches.read_text(encoding="utf-8"))
    gate_config = json.loads(args.gate_config.read_text(encoding="utf-8"))
    config = {"run_id": args.run_id, "stage": "R4_zero_twins", "reference_run": str(reference_run), "branches": str(args.branches.resolve()), "gate_config": str(args.gate_config.resolve()), "repeats": args.repeats, "conditions": [item[0] for item in CONDITIONS]}
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite"), "pyarrow": pa.__version__}
    git_state = {"project": git_record(REPOSITORY_ROOT), "libero": git_record(args.libero_root), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")}
    env = None
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            reference_manifest = json.loads((reference_run / "artifacts/reference_snapshots_manifest.json").read_text(encoding="utf-8"))
            if not reference_manifest["gate"]["passed"] or branch_manifest["source_reference_run_id"] != reference_manifest["run_id"]:
                raise RuntimeError("branch manifest and passing reference run do not match")
            selection, task_manifest = load_selection(args.selection, args.task_manifest)
            task_by_name = {task["name"]: task for task in selection["tasks"]}
            reference_by_key = {(item["task"], item["episode"]): item for item in reference_manifest["episodes"]}
            wrapper, robosuite_root, assets_root = bootstrap_runtime(args.libero_root, args.dataset_root, run_dir / "artifacts/libero_config")
            step_rows: List[Dict[str, Any]] = []
            pair_rows: List[Dict[str, Any]] = []
            component_rows: List[Dict[str, Any]] = []
            prefix_errors: List[Dict[str, Any]] = []
            for trajectory in branch_manifest["trajectories"]:
                task = task_by_name[trajectory["task"]]
                source = task_source_record(task_manifest, task["suite"], task["task_id"])
                env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                episode_index = int(trajectory["episode"].split("_")[-1])
                reference = reference_by_key[(trajectory["task"], trajectory["episode"])]
                reference_directory = reference_run / reference["relative_directory"]
                with np.load(reference_directory / "trajectory_states.npz", allow_pickle=False) as archive:
                    actions = np.asarray(archive["actions"], dtype=np.float64)
                    reference_states = {kind: np.asarray(archive[kind]).copy() for kind in ("legacy", "fullphysics", "integration")}
                for condition, kind, restores_controller in CONDITIONS:
                    for branch_record in trajectory["branches"]:
                        branch = int(branch_record["action_index"])
                        controller_path = reference_directory / f"controller_{branch:04d}.npz"
                        for repeat in range(args.repeats):
                            episode = load_episode(env, dataset_path=args.dataset_root / task["demonstration_relative_path"], episode_index=episode_index, robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                            for prefix_action in actions[:branch]:
                                env.step(prefix_action)
                            prefix_integration_error = float(np.linalg.norm(mujoco_snapshot.capture(env.sim, "integration").values - reference_states["integration"][branch]))
                            prefix_legacy_error = float(np.linalg.norm(mujoco_snapshot.capture(env.sim, "legacy").values - reference_states["legacy"][branch]))
                            prefix_errors.append({"task": task["name"], "episode": trajectory["episode"], "condition": condition, "branch_time": branch, "repeat": repeat, "integration_l2": prefix_integration_error, "legacy_l2": prefix_legacy_error})
                            if prefix_legacy_error > 1e-10:
                                raise RuntimeError(f"prefix replay did not reproduce local qpos/qvel/time at {task['name']}/{trajectory['episode']}/{branch}: {prefix_legacy_error}")
                            _restore_condition(env, kind, reference_states[kind][branch], controller_path, restores_controller)
                            twin_a, _ = _run_continuation(env, actions, branch)
                            _restore_condition(env, kind, reference_states[kind][branch], controller_path, restores_controller)
                            twin_b, _ = _run_continuation(env, actions, branch)
                            identity = {"condition": condition, "task": task["name"], "episode": trajectory["episode"], "branch_time": branch, "branch_normalized_time": float(branch_record["normalized_time"]), "branch_kind": branch_record["kind"], "branch_contact_event": branch_record["kind"] == "first_maximum_contact_count_change", "phase": _phase(float(branch_record["normalized_time"])), "repeat": repeat}
                            rows, pair = _compare_pair(twin_a, twin_b, identity=identity, component_rows=component_rows)
                            step_rows.extend(rows); pair_rows.append(pair)
                    print(json.dumps({"condition": condition, "task": task["name"], "episode": trajectory["episode"], "pairs_completed": len(trajectory["branches"]) * args.repeats}, sort_keys=True))
                env.close(); env = None
            summary = _summarize(step_rows, pair_rows, gate_config)
            artifacts = run_dir / "artifacts"
            pq.write_table(pa.Table.from_pylist(step_rows), artifacts / "zero_twin_comparisons.parquet", compression="zstd")
            pq.write_table(pa.Table.from_pylist(component_rows), artifacts / "component_errors.parquet", compression="zstd")
            pq.write_table(pa.Table.from_pylist(pair_rows), artifacts / "zero_twin_pairs.parquet", compression="zstd")
            write_json(artifacts / "reference_snapshots_manifest.json", reference_manifest)
            write_json(artifacts / "prefix_replay_errors.json", prefix_errors)
            failures = sorted((row for row in pair_rows if row["maximum_integration_l2"] > 0 or not row["final_success_agreement"]), key=lambda row: row["maximum_integration_l2"], reverse=True)[:100]
            write_json(artifacts / "failure_examples.json", failures)
            _plots(step_rows, artifacts / "plots")
            metrics = {"run_id": args.run_id, "status": "completed", "gate": summary, "pair_count": len(pair_rows), "step_count": len(step_rows), "component_row_count": len(component_rows), "maximum_prefix_replay_integration_l2": max(row["integration_l2"] for row in prefix_errors), "maximum_prefix_replay_legacy_l2": max(row["legacy_l2"] for row in prefix_errors), "r5_legally_reached": bool(summary["passed"])}
            print(json.dumps({"gate": summary, "pair_count": len(pair_rows)}, sort_keys=True))
        write_run_record(run_dir, config=config, command=command, environment=environment, git_state=git_state, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue(), metrics=metrics)
        return 0 if metrics["gate"]["passed"] else 2
    except Exception as exc:
        captured_stderr.write(traceback.format_exc())
        metrics = {"run_id": args.run_id, "status": "failed", "gate": {"passed": False}, "r5_legally_reached": False, "error": repr(exc)}
        write_run_record(run_dir, config=config, command=command, environment=environment, git_state=git_state, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue(), metrics=metrics)
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
