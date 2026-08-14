#!/usr/bin/env python
"""Run the EXP1 E3 deterministic demonstration-replay hard gate."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import io
import json
from pathlib import Path
import shlex
import sys
import traceback
from typing import Any, Dict, List

import h5py
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.libero_replay import (  # noqa: E402
    evaluate_replay_gate,
    numeric_demo_sort,
    rewrite_episode_model_paths,
    summarize_replay_rows,
)
from decision_sparse_rl.envs.libero_source import (  # noqa: E402
    import_benchmark_from_source,
    import_module_from_source,
    write_libero_config,
)
from decision_sparse_rl.logging.run_directory import (  # noqa: E402
    create_run_directory,
    write_json,
    write_run_record,
)
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party" / "LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument(
        "--selection",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "exp1_decision_sparsity" / "manifests" / "selected_tasks_pilot.json",
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "exp1_decision_sparsity" / "manifests" / "tasks.json",
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--task-names", nargs="+", default=None)
    return parser.parse_args()


def _task_source_record(task_manifest: Dict[str, Any], suite: str, task_id: int) -> Dict[str, Any]:
    for task in task_manifest["suites"][suite]["tasks"]:
        if int(task["task_id"]) == task_id:
            return task
    raise KeyError(f"task not found in source manifest: {suite}:{task_id}")


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "episode",
        "action_index",
        "recorded_state_index",
        "normalized_time",
        "state_l2_error",
        "time_abs_error",
        "qpos_l2_error",
        "qvel_l2_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _environment_kwargs(bddl_path: Path) -> Dict[str, Any]:
    return {
        "bddl_file_name": str(bddl_path.resolve()),
        "robots": ["Panda"],
        "controller": "OSC_POSE",
        "initialization_noise": None,
        "use_camera_obs": False,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "reward_shaping": True,
        "control_freq": 20,
    }


def main() -> int:
    args = parse_args()
    if len(set(args.episodes)) != len(args.episodes) or any(index < 0 for index in args.episodes):
        raise ValueError("--episodes must contain unique non-negative indices")
    run_dir = create_run_directory(args.run_root, args.run_id)
    command = shlex.join([sys.executable, *sys.argv])
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    task_manifest = json.loads(args.task_manifest.read_text(encoding="utf-8"))
    tasks = selected["tasks"]
    if args.task_names is not None:
        requested = set(args.task_names)
        tasks = [task for task in tasks if task["name"] in requested]
        missing = requested - {task["name"] for task in tasks}
        if missing:
            raise KeyError(f"requested task names are not in frozen selection: {sorted(missing)}")
    config: Dict[str, Any] = {
        "run_id": args.run_id,
        "stage": "E3_deterministic_replay",
        "selection": str(args.selection.resolve()),
        "task_manifest": str(args.task_manifest.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "episodes": args.episodes,
        "task_names": [task["name"] for task in tasks],
        "action_state_alignment": "execute actions[j], compare runtime state to states[j+1], only j < T-1",
        "gate_thresholds": {
            "min_demos_per_task": 3,
            "minimum_success_rate": 0.9,
            "maximum_restore_error": 1e-10,
            "maximum_p95_replay_error": 0.01,
        },
    }
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "robosuite": importlib.metadata.version("robosuite"),
    }
    git_state = {
        "project": git_record(REPOSITORY_ROOT),
        "libero": git_record(args.libero_root),
    }
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    rows: List[Dict[str, Any]] = []
    episode_results: List[Dict[str, Any]] = []
    model_path_records: List[Dict[str, Any]] = []
    env = None
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            config_directory = run_dir / "artifacts" / "libero_config"
            write_libero_config(config_directory, args.libero_root, args.dataset_root)
            import_benchmark_from_source(args.libero_root, config_directory)
            wrapper_module = import_module_from_source(
                args.libero_root, config_directory, "libero.libero.envs.env_wrapper"
            )
            import robosuite

            robosuite_root = Path(robosuite.__file__).resolve().parent
            libero_assets = args.libero_root / "libero" / "libero" / "assets"
            for task in tasks:
                source_task = _task_source_record(task_manifest, task["suite"], int(task["task_id"]))
                bddl_path = Path(source_task["bddl_file_path"])
                dataset_path = args.dataset_root / task["demonstration_relative_path"]
                if not bddl_path.is_file() or not dataset_path.is_file():
                    raise FileNotFoundError(f"missing BDDL or dataset: {bddl_path}, {dataset_path}")
                env = wrapper_module.ControlEnv(**_environment_kwargs(bddl_path))
                with h5py.File(dataset_path, "r") as dataset:
                    demo_names = numeric_demo_sort(dataset["data"].keys())
                    for episode_index in args.episodes:
                        if episode_index >= len(demo_names):
                            raise IndexError(f"episode {episode_index} absent from {dataset_path}")
                        episode_name = demo_names[episode_index]
                        episode = dataset[f"data/{episode_name}"]
                        states = np.asarray(episode["states"][()], dtype=np.float64)
                        actions = np.asarray(episode["actions"][()], dtype=np.float64)
                        if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions):
                            raise ValueError(
                                f"unexpected state/action shapes for {task['name']}/{episode_name}: "
                                f"{states.shape}, {actions.shape}"
                            )
                        model_xml = episode.attrs.get("model_file")
                        if not isinstance(model_xml, str):
                            raise TypeError(f"missing string model_file for {task['name']}/{episode_name}")
                        rewritten_xml, rewrite_record = rewrite_episode_model_paths(
                            model_xml,
                            robosuite_package_root=robosuite_root,
                            libero_assets_root=libero_assets,
                        )
                        model_path_records.append(
                            {"task": task["name"], "episode": episode_name, **rewrite_record}
                        )
                        env.reset_from_xml_string(rewritten_xml)
                        env.sim.reset()
                        env.set_init_state(states[0])
                        first_restore = env.get_sim_state().copy()
                        initial_state_l2_error = float(np.linalg.norm(first_restore - states[0]))
                        env.set_init_state(states[0])
                        repeat_restore = env.get_sim_state().copy()
                        repeat_restore_l2_error = float(np.linalg.norm(repeat_restore - first_restore))
                        nq = int(env.sim.model.nq)
                        nv = int(env.sim.model.nv)
                        if states.shape[1] != 1 + nq + nv:
                            raise ValueError(
                                f"flattened state dimension {states.shape[1]} does not equal "
                                f"1 + nq {nq} + nv {nv}"
                            )
                        for action_index, action in enumerate(actions):
                            env.step(action)
                            if action_index >= len(actions) - 1:
                                continue
                            runtime_state = env.get_sim_state()
                            difference = states[action_index + 1] - runtime_state
                            error = float(np.linalg.norm(difference))
                            rows.append(
                                {
                                    "task": task["name"],
                                    "episode": episode_name,
                                    "action_index": action_index,
                                    "recorded_state_index": action_index + 1,
                                    "normalized_time": action_index / max(len(actions) - 2, 1),
                                    "state_l2_error": error,
                                    "time_abs_error": float(abs(difference[0])),
                                    "qpos_l2_error": float(np.linalg.norm(difference[1 : 1 + nq])),
                                    "qvel_l2_error": float(np.linalg.norm(difference[1 + nq :])),
                                }
                            )
                        result = {
                            "task": task["name"],
                            "episode": episode_name,
                            "trajectory_length": int(len(actions)),
                            "comparison_count": int(len(actions) - 1),
                            "state_dimension": int(states.shape[1]),
                            "model_nq": nq,
                            "model_nv": nv,
                            "action_dimension": int(actions.shape[1]),
                            "initial_state_l2_error": initial_state_l2_error,
                            "repeat_restore_l2_error": repeat_restore_l2_error,
                            "final_success": bool(env.check_success()),
                        }
                        episode_results.append(result)
                        print(json.dumps(result, sort_keys=True))
                env.close()
                env = None

        summary = summarize_replay_rows(rows)
        component_summary = {}
        for component in ("time_abs_error", "qpos_l2_error", "qvel_l2_error"):
            values = np.asarray([row[component] for row in rows], dtype=np.float64)
            component_summary[component] = {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "maximum": float(np.max(values)),
            }
        gate = evaluate_replay_gate(
            summary=summary,
            episode_results=episode_results,
            selected_task_count=len(selected["tasks"]),
        )
        _write_rows(run_dir / "artifacts" / "replay_curves.csv", rows)
        write_json(run_dir / "artifacts" / "episode_results.json", episode_results)
        write_json(run_dir / "artifacts" / "model_path_rewrites.json", model_path_records)
        failures = {
            "unsuccessful_episodes": [result for result in episode_results if not result["final_success"]],
            "largest_state_errors": sorted(rows, key=lambda row: row["state_l2_error"], reverse=True)[:20],
        }
        write_json(run_dir / "artifacts" / "failure_examples.json", failures)
        metrics = {
            "run_id": args.run_id,
            "status": "completed",
            "episode_count": len(episode_results),
            "task_count": len(tasks),
            "summary": summary,
            "component_summary": component_summary,
            "gate": gate,
            "episode_results": episode_results,
        }
        stdout = captured_stdout.getvalue() + json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        stderr = captured_stderr.getvalue()
    except Exception as exc:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        stderr = captured_stderr.getvalue() + traceback.format_exc()
        metrics = {
            "run_id": args.run_id,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "completed_episode_count": len(episode_results),
        }
        if rows:
            _write_rows(run_dir / "artifacts" / "partial_replay_curves.csv", rows)
        if episode_results:
            write_json(run_dir / "artifacts" / "partial_episode_results.json", episode_results)
        stdout = captured_stdout.getvalue()
    write_run_record(
        run_dir,
        config=config,
        command=command,
        environment=environment,
        git_state=git_state,
        metrics=metrics,
        stdout=stdout,
        stderr=stderr,
    )
    if metrics["status"] == "failed":
        print(stderr, file=sys.stderr, end="")
        print(f"run_dir={run_dir}", file=sys.stderr)
        return 1
    print(stdout, end="")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
