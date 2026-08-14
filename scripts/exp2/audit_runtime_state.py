#!/usr/bin/env python
"""Audit the exact runtime state surface used by EXP2 R0."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import shlex
import sys
import traceback
from typing import Any, Dict

import h5py
import mujoco
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.libero_replay import numeric_demo_sort, rewrite_episode_model_paths  # noqa: E402
from decision_sparse_rl.envs.libero_source import (  # noqa: E402
    import_benchmark_from_source,
    import_module_from_source,
    write_libero_config,
)
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


CONTROLLER_FIELDS = {
    "goal_pos": ("causal_policy_goal", True),
    "goal_ori": ("causal_policy_goal", True),
    "relative_ori": ("interpolator_history", True),
    "ori_ref": ("interpolator_history", True),
    "new_update": ("controller_cache_validity", True),
    "torques": ("last_output_diagnostic", True),
    "ee_pos": ("derived_cache", True),
    "ee_ori_mat": ("derived_cache", True),
    "ee_pos_vel": ("derived_cache", True),
    "ee_ori_vel": ("derived_cache", True),
    "joint_pos": ("derived_cache", True),
    "joint_vel": ("derived_cache", True),
    "J_pos": ("derived_cache", True),
    "J_ori": ("derived_cache", True),
    "J_full": ("derived_cache", True),
    "mass_matrix": ("derived_cache", True),
}

ROBOT_FIELDS = {
    "torques": ("last_output_diagnostic", True),
    "recent_qpos": ("robot_buffer", True),
    "recent_actions": ("robot_buffer", True),
    "recent_torques": ("robot_buffer", True),
    "recent_ee_forcetorques": ("robot_buffer", True),
    "recent_ee_pose": ("robot_buffer", True),
    "recent_ee_vel": ("robot_buffer", True),
    "recent_ee_vel_buffer": ("robot_buffer", True),
    "recent_ee_acc": ("robot_buffer", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party" / "LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json",
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json",
    )
    return parser.parse_args()


def _source_reference(obj: Any) -> Dict[str, Any]:
    try:
        path = inspect.getsourcefile(obj)
        _, line = inspect.getsourcelines(obj)
        return {"file": str(Path(path).resolve()) if path else None, "line": line, "symbol": obj.__qualname__}
    except (OSError, TypeError):
        return {"file": None, "line": None, "symbol": getattr(obj, "__qualname__", str(obj))}


def _value(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"type": "NoneType", "shape": None, "dtype": None, "summary": None, "finite": True}
    if isinstance(value, (bool, int, float, str, np.generic)):
        scalar = value.item() if isinstance(value, np.generic) else value
        return {"type": type(value).__name__, "shape": [], "dtype": None, "summary": scalar, "finite": bool(np.isfinite(scalar)) if isinstance(scalar, (int, float)) else True}
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "summary": {
                "minimum": float(np.min(value)) if value.size else None,
                "maximum": float(np.max(value)) if value.size else None,
                "l2": float(np.linalg.norm(value)),
            },
            "finite": bool(np.all(np.isfinite(value))),
        }
    if hasattr(value, "buf"):
        result = _value(np.asarray(value.buf))
        result.update({"type": f"{type(value).__module__}.{type(value).__name__}", "ptr": int(value.ptr), "size": int(value._size)})
        return result
    if hasattr(value, "current") and hasattr(value, "last"):
        current = np.asarray(value.current)
        last = np.asarray(value.last)
        return {
            "type": f"{type(value).__module__}.{type(value).__name__}",
            "shape": list(current.shape),
            "dtype": str(current.dtype),
            "summary": {"current_l2": float(np.linalg.norm(current)), "last_l2": float(np.linalg.norm(last))},
            "finite": bool(np.all(np.isfinite(current)) and np.all(np.isfinite(last))),
        }
    return {"type": f"{type(value).__module__}.{type(value).__name__}", "shape": None, "dtype": None, "summary": repr(value)[:240], "finite": True}


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left == right
    return False


def _capture_fields(obj: Any, fields: Dict[str, Any]) -> Dict[str, Any]:
    return {name: _value(getattr(obj, name, None)) for name in fields}


def _field_schema(path_prefix: str, before: Dict[str, Any], after: Dict[str, Any], reset: Dict[str, Any], definitions: Dict[str, Any], source: Dict[str, Any]) -> list:
    rows = []
    for name, (category, snapshot) in definitions.items():
        item = dict(before[name])
        item.update({
            "object_path": f"{path_prefix}.{name}",
            "category": category,
            "changes_after_one_action": not _equal(before[name], after[name]),
            "changes_after_reset": not _equal(after[name], reset[name]),
            "included_in_condition_d_snapshot": snapshot,
            "control_relevance": "direct or conservative explicit restoration" if snapshot else "not restored",
            "source_reference": source,
        })
        rows.append(item)
    return rows


def _state_flags(model: Any) -> Dict[str, Any]:
    flags = {}
    for name in dir(mujoco.mjtState):
        if not name.startswith("mjSTATE_"):
            continue
        value = int(getattr(mujoco.mjtState, name))
        try:
            size = int(mujoco.mj_stateSize(model, value))
        except Exception as exc:
            size = None
            error = repr(exc)
        else:
            error = None
        flags[name] = {"value": value, "state_size": size, "error": error}
    return flags


def _task_source_record(task_manifest: Dict[str, Any], suite: str, task_id: int) -> Dict[str, Any]:
    for task in task_manifest["suites"][suite]["tasks"]:
        if int(task["task_id"]) == task_id:
            return task
    raise KeyError((suite, task_id))


def _environment_kwargs(bddl_path: Path) -> Dict[str, Any]:
    return {"bddl_file_name": str(bddl_path.resolve()), "robots": ["Panda"], "controller": "OSC_POSE", "initialization_noise": None, "use_camera_obs": False, "has_renderer": False, "has_offscreen_renderer": False, "ignore_done": True, "reward_shaping": True, "control_freq": 20}


def main() -> int:
    args = parse_args()
    run_dir = create_run_directory(args.run_root, args.run_id)
    command = shlex.join([sys.executable, *sys.argv])
    captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
    config = {"run_id": args.run_id, "stage": "R0_runtime_state_audit", "selection": str(args.selection.resolve())}
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "mujoco": mujoco.__version__, "robosuite": importlib.metadata.version("robosuite")}
    git_state = {"project": git_record(REPOSITORY_ROOT), "libero": git_record(args.libero_root), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")}
    env = None
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            selected = json.loads(args.selection.read_text(encoding="utf-8"))
            task_manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
            task = selected["tasks"][0]
            source_task = _task_source_record(task_manifest, task["suite"], int(task["task_id"]))
            dataset_path = args.dataset_root / task["demonstration_relative_path"]
            config_dir = run_dir / "artifacts/libero_config"
            write_libero_config(config_dir, args.libero_root, args.dataset_root)
            import_benchmark_from_source(args.libero_root, config_dir)
            wrapper = import_module_from_source(args.libero_root, config_dir, "libero.libero.envs.env_wrapper")
            import robosuite
            with h5py.File(dataset_path, "r") as handle:
                episode = handle[f"data/{numeric_demo_sort(handle['data'].keys())[0]}"]
                states = np.asarray(episode["states"], dtype=np.float64)
                actions = np.asarray(episode["actions"], dtype=np.float64)
                xml, _ = rewrite_episode_model_paths(episode.attrs["model_file"], robosuite_package_root=Path(robosuite.__file__).resolve().parent, libero_assets_root=args.libero_root / "libero/libero/assets")
            env = wrapper.ControlEnv(**_environment_kwargs(Path(source_task["bddl_file_path"])))
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.set_init_state(states[0])
            robot = env.robots[0]
            controller = robot.controller
            before_controller = _capture_fields(controller, CONTROLLER_FIELDS)
            before_robot = _capture_fields(robot, ROBOT_FIELDS)
            env.step(actions[0])
            after_controller = _capture_fields(controller, CONTROLLER_FIELDS)
            after_robot = _capture_fields(robot, ROBOT_FIELDS)
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.set_init_state(states[0])
            robot_reset = env.robots[0]
            controller_reset = robot_reset.controller
            reset_controller = _capture_fields(controller_reset, CONTROLLER_FIELDS)
            reset_robot = _capture_fields(robot_reset, ROBOT_FIELDS)
            native_model = env.sim.model._model
            flags = _state_flags(native_model)
            dims = {name: int(getattr(env.sim.model, name)) for name in ("nq", "nv", "na", "nu", "nbody", "nmocap", "nuserdata")}
            runtime_manifest = {
                "schema_version": 1,
                "runtime": environment,
                "model_dimensions": dims,
                "legacy_flattened_state_dimension": int(env.sim.get_state().flatten().size),
                "state_flags": flags,
                "fullphysics_state_dimension": flags["mjSTATE_FULLPHYSICS"]["state_size"],
                "integration_state_dimension": flags["mjSTATE_INTEGRATION"]["state_size"],
                "integration_components": [name for name, item in flags.items() if int(item["value"]) != 0 and (int(item["value"]) & int(flags["mjSTATE_INTEGRATION"]["value"])) == int(item["value"])],
                "native_bindings": {"model_type": f"{type(native_model).__module__}.{type(native_model).__name__}", "data_type": f"{type(env.sim.data._data).__module__}.{type(env.sim.data._data).__name__}"},
                "classes": {"environment": f"{type(env.env).__module__}.{type(env.env).__name__}", "robot": f"{type(robot_reset).__module__}.{type(robot_reset).__name__}", "controller": f"{type(controller_reset).__module__}.{type(controller_reset).__name__}", "gripper": f"{type(robot_reset.gripper).__module__}.{type(robot_reset.gripper).__name__}", "interpolator_pos": None if controller_reset.interpolator_pos is None else f"{type(controller_reset.interpolator_pos).__module__}.{type(controller_reset.interpolator_pos).__name__}", "interpolator_ori": None if controller_reset.interpolator_ori is None else f"{type(controller_reset.interpolator_ori).__module__}.{type(controller_reset.interpolator_ori).__name__}"},
                "api_availability": {name: hasattr(mujoco, name) for name in ("mj_stateSize", "mj_getState", "mj_setState", "mj_copyData", "mj_forward")},
            }
            controller_manifest = {
                "schema_version": 1,
                "scope": "Panda SingleArm with fixed-impedance OSC_POSE and no interpolators",
                "source_audit": {
                    "environment_step": _source_reference(type(env.env).step),
                    "robot_control": _source_reference(type(robot_reset).control),
                    "controller_set_goal": _source_reference(type(controller_reset).set_goal),
                    "controller_run": _source_reference(type(controller_reset).run_controller),
                    "controller_update": _source_reference(type(controller_reset).update),
                },
                "controller_fields": _field_schema("env.robots[0].controller", before_controller, after_controller, reset_controller, CONTROLLER_FIELDS, _source_reference(type(controller_reset))),
                "robot_fields": _field_schema("env.robots[0]", before_robot, after_robot, reset_robot, ROBOT_FIELDS, _source_reference(type(robot_reset))),
                "environment_timing_fields": [
                    {"object_path": "env.env.timestep", "included_in_condition_d_snapshot": True, "reason": "used by step horizon/done bookkeeping"},
                    {"object_path": "env.env.cur_time", "included_in_condition_d_snapshot": True, "reason": "advanced by each policy step"},
                    {"object_path": "env.env.done", "included_in_condition_d_snapshot": True, "reason": "episode termination bookkeeping"}
                ],
                "classification_complete": True,
                "classification_note": "All mutable fields read or written by the fixed OSC_POSE control chain and robot policy-step buffers were classified. Static configuration/index fields and recomputable observation caches are intentionally excluded.",
            }
            manifest_dir = REPOSITORY_ROOT / "experiments/exp2_simulator_reconciliation/manifests"
            write_json(manifest_dir / "runtime_state_schema.json", runtime_manifest)
            write_json(manifest_dir / "controller_state_schema.json", controller_manifest)
            write_json(run_dir / "artifacts/runtime_state_schema.json", runtime_manifest)
            write_json(run_dir / "artifacts/controller_state_schema.json", controller_manifest)
            metrics = {"run_id": args.run_id, "status": "completed", "classification_complete": True, "legacy_dimension": runtime_manifest["legacy_flattened_state_dimension"], "fullphysics_dimension": runtime_manifest["fullphysics_state_dimension"], "integration_dimension": runtime_manifest["integration_state_dimension"], "state_flag_count": len(flags)}
            print(json.dumps(metrics, sort_keys=True))
        write_run_record(run_dir, config=config, command=command, environment=environment, git_state=git_state, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue(), metrics=metrics)
        return 0
    except Exception as exc:
        captured_stderr.write(traceback.format_exc())
        metrics = {"run_id": args.run_id, "status": "failed", "error": repr(exc)}
        write_run_record(run_dir, config=config, command=command, environment=environment, git_state=git_state, stdout=captured_stdout.getvalue(), stderr=captured_stderr.getvalue(), metrics=metrics)
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
