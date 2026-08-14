#!/usr/bin/env python
"""Run EXP2 R1 real-environment snapshot round trips and boundary audit."""

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

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs import controller_snapshot  # noqa: E402
from decision_sparse_rl.envs import mujoco_snapshot  # noqa: E402
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
    return parser.parse_args()


def _contact_pairs(env: Any) -> List[List[Any]]:
    pairs = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        pairs.append([geom1, env.sim.model.geom_id2name(geom1), geom2, env.sim.model.geom_id2name(geom2)])
    return pairs


def _profile_step(env: Any, action: np.ndarray) -> List[Dict[str, Any]]:
    wanted = {"step", "_pre_action", "control", "set_goal", "run_controller", "update"}
    trace: List[Dict[str, Any]] = []
    def profiler(frame: Any, event: str, arg: Any) -> None:
        if event != "call" or frame.f_code.co_name not in wanted:
            return
        filename = frame.f_code.co_filename.replace("\\", "/")
        if "/robosuite/" not in filename:
            return
        trace.append({"sequence": len(trace), "function": frame.f_code.co_name, "file": str(Path(frame.f_code.co_filename).resolve()), "line": int(frame.f_code.co_firstlineno)})
    sys.setprofile(profiler)
    try:
        env.step(action)
    finally:
        sys.setprofile(None)
    return trace


def _restore_master(env: Any, sim_state: Any, runtime_state: Dict[str, Any]) -> None:
    mujoco_snapshot.restore(env.sim, sim_state)
    controller_snapshot.restore(env, runtime_state)


def _roundtrip(env: Any, action: np.ndarray, kind: str) -> Dict[str, Any]:
    master = mujoco_snapshot.capture(env.sim, "integration")
    runtime = controller_snapshot.capture(env)
    _restore_master(env, master, runtime)
    state = mujoco_snapshot.capture(env.sim, kind)
    components_before = mujoco_snapshot.capture_atomic_components(env.sim)
    controller_before = controller_snapshot.capture(env)
    env.step(action)
    mujoco_snapshot.restore(env.sim, state)
    controller_snapshot.restore(env, controller_before)
    state_after = mujoco_snapshot.capture(env.sim, kind)
    components_after = mujoco_snapshot.capture_atomic_components(env.sim)
    controller_after = controller_snapshot.capture(env)
    state_error = state_after.values - state.values
    field_errors = controller_snapshot.field_errors(controller_before, controller_after)
    result = {
        "kind": kind,
        "state_l2": float(np.linalg.norm(state_error)),
        "state_linf": float(np.max(np.abs(state_error))),
        "component_errors": mujoco_snapshot.component_errors(components_before, components_after),
        "controller_field_errors": field_errors,
        "controller_max_error": float(max(field_errors.values(), default=0.0)),
        "all_finite": bool(np.all(np.isfinite(state_after.values))),
    }
    _restore_master(env, master, runtime)
    return result


def main() -> int:
    args = parse_args()
    run_dir = create_run_directory(args.run_root, args.run_id)
    command = shlex.join([sys.executable, *sys.argv])
    captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
    config = {"run_id": args.run_id, "stage": "R1_snapshot_roundtrip", "task_index": 0, "episode_index": 0, "stages": ["initialization", "free_space", "contact_rich", "late_trajectory"], "state_kinds": ["legacy", "fullphysics", "integration"]}
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite")}
    git_state = {"project": git_record(REPOSITORY_ROOT), "libero": git_record(args.libero_root), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")}
    env = None
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            selection, task_manifest = load_selection(args.selection, args.task_manifest)
            task = selection["tasks"][0]
            source = task_source_record(task_manifest, task["suite"], task["task_id"])
            wrapper, robosuite_root, assets_root = bootstrap_runtime(args.libero_root, args.dataset_root, run_dir / "artifacts/libero_config")
            env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
            episode = load_episode(env, dataset_path=args.dataset_root / task["demonstration_relative_path"], episode_index=0, robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
            actions = episode["actions"]
            initial_master = mujoco_snapshot.capture(env.sim, "integration")
            initial_runtime = controller_snapshot.capture(env)
            trace = _profile_step(env, actions[0])
            _restore_master(env, initial_master, initial_runtime)
            contacts = [len(_contact_pairs(env))]
            for action in actions[:-1]:
                env.step(action)
                contacts.append(len(_contact_pairs(env)))
            free_candidates = [index for index, count in enumerate(contacts[:-1]) if count == min(contacts)]
            stages = {
                "initialization": 0,
                "free_space": free_candidates[min(1, len(free_candidates) - 1)],
                "contact_rich": int(np.argmax(contacts[:-1])),
                "late_trajectory": min(len(actions) - 1, int(round(0.9 * (len(actions) - 1)))),
            }
            rows = []
            for label, boundary in stages.items():
                load_episode(env, dataset_path=args.dataset_root / task["demonstration_relative_path"], episode_index=0, robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                for index in range(boundary):
                    env.step(actions[index])
                row = {"stage": label, "boundary": boundary, "contact_count": int(env.sim.data.ncon), "contact_pairs": _contact_pairs(env), "conditions": {}}
                for kind in ("legacy", "fullphysics", "integration"):
                    row["conditions"][kind] = _roundtrip(env, actions[boundary], kind)
                rows.append(row)
            integration_rows = [row["conditions"]["integration"] for row in rows]
            passed = all(row["state_l2"] <= 1e-12 and row["controller_max_error"] <= 1e-12 and row["all_finite"] for row in integration_rows)
            boundary_manifest = {
                "schema_version": 1,
                "boundary_name": "outer_env_step_return_or_initialized_pre-policy boundary",
                "definition": "Snapshot t=0 is taken after reset_from_xml_string, sim.reset, set_init_state, post_process, forced observable update, controller.update_initial_joints(recorded_arm_q), and setting controller.new_update=True. update_initial_joints synchronizes the OSC null-space initial_joint plus initial EEF references and resets the goal. Snapshot t>0 is taken immediately after ControlEnv.step(actions[t-1]) returns and before ControlEnv.step(actions[t]) is called.",
                "next_action_semantics": "At a stored boundary t, continuation begins by calling outer ControlEnv.step(actions[t]). No controller.set_goal or robot.control for actions[t] has yet executed.",
                "verified_call_chain": [
                    {"order": 1, "symbol": "MujocoEnv.step", "source": "third_party/robosuite-src/robosuite/environments/base.py:361"},
                    {"order": 2, "symbol": "RobotEnv._pre_action", "source": "third_party/robosuite-src/robosuite/environments/robot_env.py:558"},
                    {"order": 3, "symbol": "SingleArm.control", "source": "third_party/robosuite-src/robosuite/robots/single_arm.py:216"},
                    {"order": 4, "symbol": "OperationalSpaceController.set_goal", "source": "third_party/robosuite-src/robosuite/controllers/osc.py:202", "only_first_inner_step": True},
                    {"order": 5, "symbol": "OperationalSpaceController.run_controller", "source": "third_party/robosuite-src/robosuite/controllers/osc.py:278"},
                    {"order": 6, "symbol": "MjSim.step", "source": "third_party/robosuite-src/robosuite/environments/base.py:391"}
                ],
                "observed_first_action_call_trace": trace,
                "inner_simulation_steps_observed": sum(1 for item in trace if item["function"] == "control"),
                "audit_run_id": args.run_id,
            }
            write_json(REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests/policy_step_boundary.json", boundary_manifest)
            write_json(run_dir / "artifacts/policy_step_boundary.json", boundary_manifest)
            write_json(run_dir / "artifacts/roundtrip_results.json", rows)
            failures = [row for row in rows if row["conditions"]["integration"]["state_l2"] > 1e-12 or row["conditions"]["integration"]["controller_max_error"] > 1e-12]
            write_json(run_dir / "artifacts/failure_examples.json", failures)
            metrics = {"run_id": args.run_id, "status": "completed", "passed": passed, "stage_count": len(rows), "integration_max_state_l2": max(item["state_l2"] for item in integration_rows), "integration_max_controller_error": max(item["controller_max_error"] for item in integration_rows), "boundary_trace_event_count": len(trace), "inner_simulation_steps_observed": boundary_manifest["inner_simulation_steps_observed"]}
            print(json.dumps(metrics, sort_keys=True))
        write_run_record(run_dir, config=config, command=command, environment=environment, git_state=git_state, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue(), metrics=metrics)
        return 0 if metrics["passed"] else 2
    except Exception as exc:
        captured_stderr.write(traceback.format_exc())
        metrics = {"run_id": args.run_id, "status": "failed", "passed": False, "error": repr(exc)}
        write_run_record(run_dir, config=config, command=command, environment=environment, git_state=git_state, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue(), metrics=metrics)
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
