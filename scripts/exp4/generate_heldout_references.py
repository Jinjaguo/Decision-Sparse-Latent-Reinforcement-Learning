#!/usr/bin/env python
"""Generate the 21 EXP4 held-out same-runtime references with progress channels."""

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
from typing import Any, Dict, List

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.exp2 import generate_local_reference as exp2_reference  # noqa: E402
from scripts.exp4.audit_runtime_progress import _geom_vertical_half_extent  # noqa: E402
from decision_sparse_rl.envs.libero_runtime import (  # noqa: E402
    bootstrap_runtime,
    environment_kwargs,
    load_episode,
    load_selection,
    task_source_record,
)
from decision_sparse_rl.logging.run_directory import (  # noqa: E402
    create_run_directory,
    write_json,
    write_run_record,
)
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


EPISODES = tuple(range(3, 10))
DRAWER_JOINT = "wooden_cabinet_1_middle_level"
STOVE_JOINT = "flat_stove_1_button"
BOWL = "akita_black_bowl_1"
PLATE = "plate_1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--progress-audit-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party/LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--selection", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json")
    parser.add_argument("--task-manifest", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
    parser.add_argument("--episodes", type=int, nargs="+", default=list(EPISODES))
    return parser.parse_args()


def object_vertical_bounds(env: Any, object_name: str) -> Dict[str, float]:
    body_id = int(env.env.obj_body_id[object_name])
    lower: List[float] = []
    upper: List[float] = []
    for geom_id in range(int(env.sim.model.ngeom)):
        if (
            int(env.sim.model.geom_bodyid[geom_id]) != body_id
            or int(env.sim.model.geom_group[geom_id]) != 0
            or int(env.sim.model.geom_contype[geom_id]) == 0
        ):
            continue
        position = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(env.sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        size = np.asarray(env.sim.model.geom_size[geom_id], dtype=np.float64)
        half = _geom_vertical_half_extent(int(env.sim.model.geom_type[geom_id]), size, rotation)
        lower.append(float(position[2] - half))
        upper.append(float(position[2] + half))
    if not lower:
        raise RuntimeError(f"no audited collision primitives for {object_name}")
    return {"z_min": min(lower), "z_max": max(upper)}


def enhanced_boundary(env: Any, action_index: int, action: np.ndarray, task_name: str) -> Dict[str, Any]:
    base = exp2_reference._boundary_observation_original(env, action_index, action)
    progress: Dict[str, Any] = {
        "gripper_command": float(action[-1]),
        "gripper_opening_l1": float(np.sum(np.abs(np.asarray(base["gripper_state"], dtype=np.float64)))),
        "exact_task_predicate": bool(base["task_success"]),
    }
    if task_name == "open_the_middle_drawer_of_the_cabinet":
        address = env.sim.model.get_joint_qpos_addr(DRAWER_JOINT)
        progress.update({"kind": "drawer", "joint_name": DRAWER_JOINT, "joint_id": int(env.sim.model.joint_name2id(DRAWER_JOINT)), "qpos_address": int(address), "joint_qpos": float(env.sim.data.qpos[address])})
    elif task_name == "turn_on_the_stove":
        address = env.sim.model.get_joint_qpos_addr(STOVE_JOINT)
        progress.update({"kind": "stove", "joint_name": STOVE_JOINT, "joint_id": int(env.sim.model.joint_name2id(STOVE_JOINT)), "qpos_address": int(address), "joint_qpos": float(env.sim.data.qpos[address])})
    else:
        bowl_id = int(env.env.obj_body_id[BOWL]); plate_id = int(env.env.obj_body_id[PLATE])
        bowl_pos = np.asarray(env.sim.data.body_xpos[bowl_id], dtype=np.float64)
        plate_pos = np.asarray(env.sim.data.body_xpos[plate_id], dtype=np.float64)
        eef = np.asarray(base["eef_position"], dtype=np.float64)
        bowl_bounds = object_vertical_bounds(env, BOWL); plate_bounds = object_vertical_bounds(env, PLATE)
        progress.update({
            "kind": "bowl_on_plate",
            "bowl_body_name": env.sim.model.body_id2name(bowl_id),
            "plate_body_name": env.sim.model.body_id2name(plate_id),
            "gripper_to_bowl_distance_m": float(np.linalg.norm(eef - bowl_pos)),
            "bowl_to_plate_planar_distance_m": float(np.linalg.norm(bowl_pos[:2] - plate_pos[:2])),
            "bowl_body_z_m": float(bowl_pos[2]),
            "plate_body_z_m": float(plate_pos[2]),
            "bowl_bottom_z_m": bowl_bounds["z_min"],
            "plate_top_z_m": plate_bounds["z_max"],
            "bowl_bottom_minus_plate_top_m": bowl_bounds["z_min"] - plate_bounds["z_max"],
        })
    base["progress_channels"] = progress
    return base


def main() -> int:
    args = parse_args()
    episodes = tuple(int(x) for x in args.episodes)
    if not episodes or len(set(episodes)) != len(episodes) or min(episodes) < 0:
        raise ValueError("--episodes must contain unique nonnegative indexes")
    run_dir = create_run_directory(args.run_root, args.run_id)
    stdout, stderr = io.StringIO(), io.StringIO()
    audit_run = args.progress_audit_run.resolve()
    audit_metrics = json.loads((audit_run / "metrics.json").read_text(encoding="utf-8"))
    config = {
        "stage": "E4-3_heldout_reference_generation",
        "episodes": list(episodes),
        "progress_audit_run": audit_run.name,
        "policy_boundary": "pre-policy-step, identical to corrected EXP2",
        "roundtrip_maximum": 1e-12,
    }
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "mujoco": importlib.metadata.version("mujoco"), "robosuite": importlib.metadata.version("robosuite")}
    git_state = {"project": git_record(REPOSITORY_ROOT), "libero": git_record(args.libero_root), "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src")}
    env = None
    try:
        if not audit_metrics.get("gate", {}).get("passed"):
            raise RuntimeError("progress runtime audit did not pass")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            selection, task_manifest = load_selection(args.selection, args.task_manifest)
            wrapper, robosuite_root, assets_root = bootstrap_runtime(args.libero_root, args.dataset_root, run_dir / "artifacts/libero_config")
            references_root = run_dir / "artifacts/references"
            records = []
            if not hasattr(exp2_reference, "_boundary_observation_original"):
                exp2_reference._boundary_observation_original = exp2_reference._boundary_observation
            for task in selection["tasks"]:
                source = task_source_record(task_manifest, task["suite"], task["task_id"])
                env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                task_name = task["name"]
                exp2_reference._boundary_observation = lambda target_env, action_index, action, selected=task_name: enhanced_boundary(target_env, action_index, action, selected)
                for episode_index in episodes:
                    episode = load_episode(env, dataset_path=args.dataset_root / task["demonstration_relative_path"], episode_index=episode_index, robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                    record = exp2_reference._write_episode_reference(env, task=task, episode=episode, output_directory=references_root / task_name / episode["episode"])
                    records.append(record)
                    print(json.dumps(record, sort_keys=True))
                env.close(); env = None
            all_success = all(x["success"] for x in records)
            all_finite = all(x["all_snapshots_finite"] for x in records)
            max_integration = max(x["maximum_roundtrip_state_l2"]["integration"] for x in records)
            max_controller = max(x["maximum_controller_roundtrip_error"] for x in records)
            expected_count = 3 * len(episodes)
            exact_episodes = sorted(int(x["episode"].split("_")[-1]) for x in records) == sorted(list(episodes) * 3)
            criteria = {"expected_reference_count": len(records) == expected_count, "exact_requested_demos_per_task": exact_episodes, "all_finish_successfully": all_success, "all_snapshot_arrays_finite": all_finite, "integration_roundtrips_at_most_1e-12": max_integration <= 1e-12, "controller_roundtrips_at_most_1e-12": max_controller <= 1e-12}
            gate = {"passed": all(criteria.values()), "criteria": criteria}
            manifest = {"schema_version": 2, "run_id": args.run_id, "progress_audit_run": audit_run.name, "progress_audit_sha256": sha256(audit_run / "artifacts/progress_runtime_audit.json"), "episodes": records, "gate": gate}
            write_json(run_dir / "artifacts/reference_snapshots_manifest.json", manifest)
            write_json(run_dir / "artifacts/failure_examples.json", [x for x in records if not x["success"] or not x["all_snapshots_finite"]])
            metrics = {"run_id": args.run_id, "status": "completed", "gate": gate, "episode_count": len(records), "all_success": all_success, "all_finite": all_finite, "maximum_integration_roundtrip_l2": max_integration, "maximum_controller_roundtrip_error": max_controller, "episode_records": records}
        write_run_record(run_dir, config=config, command=shlex.join([sys.executable, *sys.argv]), environment=environment, git_state=git_state, stdout=stdout.getvalue(), stderr=stderr.getvalue(), metrics=metrics)
        return 0 if gate["passed"] else 2
    except Exception as exc:
        stderr.write(traceback.format_exc())
        metrics = {"run_id": args.run_id, "status": "failed", "gate": {"passed": False}, "error": repr(exc)}
        write_run_record(run_dir, config=config, command=shlex.join([sys.executable, *sys.argv]), environment=environment, git_state=git_state, stdout=stdout.getvalue(), stderr=stderr.getvalue(), metrics=metrics)
        return 1
    finally:
        exp2_reference._boundary_observation = getattr(exp2_reference, "_boundary_observation_original", exp2_reference._boundary_observation)
        if env is not None: env.close()


if __name__ == "__main__":
    raise SystemExit(main())
