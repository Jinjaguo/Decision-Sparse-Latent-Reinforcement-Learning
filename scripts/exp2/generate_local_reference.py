#!/usr/bin/env python
"""Generate EXP2 R2 same-runtime local references for the frozen 3x3 pilot."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shlex
import sys
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs import controller_snapshot, mujoco_snapshot  # noqa: E402
from decision_sparse_rl.envs.libero_runtime import (  # noqa: E402
    bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record,
)
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party/LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--selection", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json")
    parser.add_argument("--task-manifest", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contacts(env: Any) -> List[Dict[str, Any]]:
    records = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        records.append({"geom1_id": geom1, "geom1_name": env.sim.model.geom_id2name(geom1), "geom2_id": geom2, "geom2_name": env.sim.model.geom_id2name(geom2)})
    return records


def _boundary_observation(env: Any, action_index: int, action: np.ndarray) -> Dict[str, Any]:
    robot = env.robots[0]
    arm_indexes = [int(index) for index in robot._ref_joint_pos_indexes]
    gripper_indexes = [int(index) for index in robot._ref_gripper_joint_pos_indexes]
    eef_site = int(robot.eef_site_id)
    body_names = [env.sim.model.body_id2name(index) for index in range(int(env.sim.model.nbody))]
    return {
        "action_index": action_index,
        "next_recorded_action": action.tolist(),
        "task_success": bool(env.check_success()),
        "panda_arm_qpos_indexes": arm_indexes,
        "panda_arm_q": np.asarray(env.sim.data.qpos[arm_indexes]).tolist(),
        "gripper_qpos_indexes": gripper_indexes,
        "gripper_state": np.asarray(env.sim.data.qpos[gripper_indexes]).tolist(),
        "eef_position": np.asarray(env.sim.data.site_xpos[eef_site]).tolist(),
        "eef_orientation_matrix": np.asarray(env.sim.data.site_xmat[eef_site]).reshape(3, 3).tolist(),
        "contact_count": int(env.sim.data.ncon),
        "contact_pairs": _contacts(env),
        "body_names": body_names,
        "body_positions": np.asarray(env.sim.data.body_xpos).tolist(),
        "body_quaternions": np.asarray(env.sim.data.body_xquat).tolist(),
    }


def _restore_master(env: Any, integration: Any, runtime: Dict[str, Any]) -> None:
    mujoco_snapshot.restore(env.sim, integration)
    controller_snapshot.restore(env, runtime)


def _validate_boundary(env: Any, action: np.ndarray, kind: str, master: Any, runtime: Dict[str, Any]) -> Dict[str, float]:
    _restore_master(env, master, runtime)
    state = mujoco_snapshot.capture(env.sim, kind)
    controller = controller_snapshot.capture(env)
    env.step(action)
    mujoco_snapshot.restore(env.sim, state)
    controller_snapshot.restore(env, controller)
    after = mujoco_snapshot.capture(env.sim, kind)
    controller_errors = controller_snapshot.field_errors(controller, controller_snapshot.capture(env))
    difference = after.values - state.values
    result = {
        "state_l2": float(np.linalg.norm(difference)),
        "state_linf": float(np.max(np.abs(difference))),
        "controller_max": float(max(controller_errors.values(), default=0.0)),
    }
    _restore_master(env, master, runtime)
    return result


def _write_episode_reference(
    env: Any,
    *,
    task: Dict[str, Any],
    episode: Dict[str, Any],
    output_directory: Path,
) -> Dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=False)
    actions = episode["actions"]
    public_states = episode["states"]
    state_rows: Dict[str, List[np.ndarray]] = {kind: [] for kind in ("legacy", "fullphysics", "integration")}
    boundaries: List[Dict[str, Any]] = []
    validations: List[Dict[str, Any]] = []
    public_errors: List[float] = []
    rewards: List[float] = []
    for action_index, action in enumerate(actions):
        snapshots = {kind: mujoco_snapshot.capture(env.sim, kind) for kind in state_rows}
        master = snapshots["integration"]
        runtime = controller_snapshot.capture(env)
        for kind, snapshot in snapshots.items():
            mujoco_snapshot.validate(env.sim, snapshot)
            state_rows[kind].append(snapshot.values.copy())
        controller_path = output_directory / f"controller_{action_index:04d}.npz"
        controller_snapshot.serialize(controller_path, runtime)
        boundaries.append(_boundary_observation(env, action_index, action))
        validation = {"action_index": action_index, "conditions": {}}
        for kind in ("legacy", "fullphysics", "integration"):
            validation["conditions"][kind] = _validate_boundary(env, action, kind, master, runtime)
        validations.append(validation)
        _restore_master(env, master, runtime)
        _, reward, _, _ = env.step(action)
        rewards.append(float(reward))
        if action_index < len(actions) - 1:
            public_errors.append(float(np.linalg.norm(mujoco_snapshot.capture(env.sim, "legacy").values - public_states[action_index + 1])))
    terminal = {
        "success": bool(env.check_success()),
        "legacy": mujoco_snapshot.capture(env.sim, "legacy").values,
        "fullphysics": mujoco_snapshot.capture(env.sim, "fullphysics").values,
        "integration": mujoco_snapshot.capture(env.sim, "integration").values,
        "body_positions": np.asarray(env.sim.data.body_xpos).copy(),
        "body_quaternions": np.asarray(env.sim.data.body_xquat).copy(),
    }
    arrays_path = output_directory / "trajectory_states.npz"
    np.savez_compressed(
        arrays_path,
        actions=actions,
        legacy=np.stack(state_rows["legacy"]),
        fullphysics=np.stack(state_rows["fullphysics"]),
        integration=np.stack(state_rows["integration"]),
        terminal_legacy=terminal["legacy"],
        terminal_fullphysics=terminal["fullphysics"],
        terminal_integration=terminal["integration"],
        terminal_body_positions=terminal["body_positions"],
        terminal_body_quaternions=terminal["body_quaternions"],
        rewards=np.asarray(rewards),
    )
    write_json(output_directory / "boundaries.json", boundaries)
    write_json(output_directory / "roundtrip_validations.json", validations)
    maximum_by_kind = {
        kind: max(row["conditions"][kind]["state_l2"] for row in validations)
        for kind in ("legacy", "fullphysics", "integration")
    }
    maximum_controller = max(row["conditions"][kind]["controller_max"] for row in validations for kind in ("legacy", "fullphysics", "integration"))
    finite = all(np.all(np.isfinite(values)) for values in state_rows.values()) and all(np.all(np.isfinite(terminal[kind])) for kind in ("legacy", "fullphysics", "integration"))
    return {
        "task": task["name"],
        "suite": task["suite"],
        "task_id": int(task["task_id"]),
        "episode": episode["episode"],
        "action_count": int(len(actions)),
        "reference_boundary_count": int(len(boundaries)),
        "success": terminal["success"],
        "all_snapshots_finite": bool(finite),
        "maximum_roundtrip_state_l2": maximum_by_kind,
        "maximum_controller_roundtrip_error": maximum_controller,
        "public_legacy_diagnostic": {"comparison_count": len(public_errors), "median_l2": float(np.median(public_errors)), "p95_l2": float(np.percentile(public_errors, 95)), "maximum_l2": float(np.max(public_errors))},
        "relative_directory": str(output_directory.relative_to(output_directory.parents[3])).replace("\\", "/"),
        "trajectory_states_sha256": _sha256(arrays_path),
        "controller_snapshot_count": len(actions),
        "path_rewrites": episode["path_rewrites"]["rewritten_path_count"],
    }


def main() -> int:
    args = parse_args()
    run_dir = create_run_directory(args.run_root, args.run_id)
    command = shlex.join([sys.executable, *sys.argv])
    captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
    config = {"run_id": args.run_id, "stage": "R2_local_references", "episodes": args.episodes, "boundary_manifest": "experiments/exp2_simulator_reconciliation/manifests/policy_step_boundary.json", "roundtrip_maximum": 1e-12}
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite")}
    git_state = {"project": git_record(REPOSITORY_ROOT), "libero": git_record(args.libero_root), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")}
    env = None
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            selection, task_manifest = load_selection(args.selection, args.task_manifest)
            wrapper, robosuite_root, assets_root = bootstrap_runtime(args.libero_root, args.dataset_root, run_dir / "artifacts/libero_config")
            references_root = run_dir / "artifacts/references"
            episode_records = []
            for task in selection["tasks"]:
                source = task_source_record(task_manifest, task["suite"], task["task_id"])
                env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                for episode_index in args.episodes:
                    episode = load_episode(env, dataset_path=args.dataset_root / task["demonstration_relative_path"], episode_index=episode_index, robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                    record = _write_episode_reference(env, task=task, episode=episode, output_directory=references_root / task["name"] / episode["episode"])
                    episode_records.append(record)
                    print(json.dumps(record, sort_keys=True))
                env.close()
                env = None
            all_success = all(record["success"] for record in episode_records)
            all_finite = all(record["all_snapshots_finite"] for record in episode_records)
            max_integration = max(record["maximum_roundtrip_state_l2"]["integration"] for record in episode_records)
            max_controller = max(record["maximum_controller_roundtrip_error"] for record in episode_records)
            gate = {
                "passed": len(episode_records) == 9 and all_success and all_finite and max_integration <= 1e-12 and max_controller <= 1e-12,
                "criteria": {"nine_references": len(episode_records) == 9, "all_finish_successfully": all_success, "all_snapshot_arrays_finite": all_finite, "all_integration_roundtrips_at_most_1e-12": max_integration <= 1e-12, "all_controller_fields_classified": True, "all_controller_roundtrips_at_most_1e-12": max_controller <= 1e-12},
            }
            manifest = {"schema_version": 1, "run_id": args.run_id, "policy_step_boundary_manifest": "experiments/exp2_simulator_reconciliation/manifests/policy_step_boundary.json", "episodes": episode_records, "gate": gate}
            write_json(run_dir / "artifacts/reference_snapshots_manifest.json", manifest)
            write_json(run_dir / "artifacts/failure_examples.json", [record for record in episode_records if not record["success"] or not record["all_snapshots_finite"] or record["maximum_roundtrip_state_l2"]["integration"] > 1e-12])
            metrics = {"run_id": args.run_id, "status": "completed", "gate": gate, "episode_count": len(episode_records), "all_success": all_success, "all_finite": all_finite, "maximum_integration_roundtrip_l2": max_integration, "maximum_controller_roundtrip_error": max_controller, "episode_records": episode_records}
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
