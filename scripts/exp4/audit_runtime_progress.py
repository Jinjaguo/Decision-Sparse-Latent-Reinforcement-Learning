#!/usr/bin/env python
"""Audit EXP4 held-out availability and exact runtime progress identifiers."""

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

import h5py
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

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


HELDOUT_EPISODES = tuple(range(3, 10))
TASK_PROGRESS_TARGETS = {
    "open_the_middle_drawer_of_the_cabinet": {
        "kind": "articulation",
        "site_state": "wooden_cabinet_1_middle_region",
        "object": "wooden_cabinet_1",
        "source_chain": "Open -> SiteObjectState.is_open -> WoodenCabinet.is_open",
    },
    "turn_on_the_stove": {
        "kind": "articulation",
        "object": "flat_stove_1",
        "source_chain": "TurnOn -> ObjectState.turn_on -> FlatStove.turn_on",
    },
    "put_the_bowl_on_the_plate": {
        "kind": "placement",
        "bowl": "akita_black_bowl_1",
        "plate": "plate_1",
        "source_chain": "On -> plate ObjectState.check_ontop(bowl)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--libero-root", type=Path, default=REPOSITORY_ROOT / "third_party/LIBERO")
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--selection", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json")
    parser.add_argument("--task-manifest", type=Path, default=REPOSITORY_ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
    return parser.parse_args()


def joint_record(env: Any, joint_name: str) -> Dict[str, Any]:
    joint_id = int(env.sim.model.joint_name2id(joint_name))
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(qpos_addr, tuple):
        raise RuntimeError(f"progress joint must be scalar: {joint_name} -> {qpos_addr}")
    return {
        "joint_name": joint_name,
        "joint_id": joint_id,
        "qpos_address": int(qpos_addr),
        "joint_type": int(env.sim.model.jnt_type[joint_id]),
        "joint_range": np.asarray(env.sim.model.jnt_range[joint_id], dtype=np.float64).tolist(),
        "initial_qpos": float(env.sim.data.qpos[int(qpos_addr)]),
    }


def _geom_vertical_half_extent(geom_type: int, size: np.ndarray, rotation: np.ndarray) -> float:
    """Exact world-z support radius for primitive MuJoCo collision geoms."""
    if geom_type == 2:  # sphere
        return float(size[0])
    if geom_type == 3:  # capsule: local-z segment plus spherical caps
        return float(abs(rotation[2, 2]) * size[1] + size[0])
    if geom_type == 4:  # ellipsoid
        return float(np.sqrt(np.sum((rotation[2] * size[:3]) ** 2)))
    if geom_type == 5:  # cylinder
        radial = float(np.hypot(rotation[2, 0], rotation[2, 1]) * size[0])
        return radial + float(abs(rotation[2, 2]) * size[1])
    if geom_type == 6:  # box
        return float(np.sum(np.abs(rotation[2]) * size[:3]))
    raise RuntimeError(f"unsupported collision geom type for exact vertical AABB: {geom_type}")


def body_and_extent(env: Any, object_name: str) -> Dict[str, Any]:
    inner = env.env
    body_id = int(inner.obj_body_id[object_name])
    geom_ids = [
        index
        for index in range(int(env.sim.model.ngeom))
        if int(env.sim.model.geom_bodyid[index]) == body_id
        and int(env.sim.model.geom_group[index]) == 0
        and int(env.sim.model.geom_contype[index]) != 0
    ]
    if not geom_ids:
        raise RuntimeError(f"no active collision geoms found for {object_name}")
    geoms = []
    for geom_id in geom_ids:
        geom_type = int(env.sim.model.geom_type[geom_id])
        position = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(env.sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        size = np.asarray(env.sim.model.geom_size[geom_id], dtype=np.float64)
        half_z = _geom_vertical_half_extent(geom_type, size, rotation)
        geoms.append({
            "geom_id": geom_id,
            "geom_name": env.sim.model.geom_id2name(geom_id),
            "geom_type": geom_type,
            "geom_size": size.tolist(),
            "world_position": position.tolist(),
            "world_vertical_half_extent": half_z,
            "world_z_min": float(position[2] - half_z),
            "world_z_max": float(position[2] + half_z),
        })
    return {
        "object_name": object_name,
        "body_name": env.sim.model.body_id2name(body_id),
        "body_id": body_id,
        "collision_geoms": geoms,
        "world_z_min": min(x["world_z_min"] for x in geoms),
        "world_z_max": max(x["world_z_max"] for x in geoms),
        "extent_method": "union AABB of active group-0 collision primitives transformed by runtime geom_xmat",
    }


def inspect_task(env: Any, task_name: str) -> Dict[str, Any]:
    target = TASK_PROGRESS_TARGETS[task_name]
    record: Dict[str, Any] = {"task": task_name, **target}
    if task_name == "open_the_middle_drawer_of_the_cabinet":
        inner = env.env
        site_state = inner.object_sites_dict[target["site_state"]]
        joints = list(site_state.joints)
        record.update({
            "site_parent_name": site_state.parent_name,
            "site_joint_names": joints,
            "resolved_joint": joint_record(env, joints[0]),
            "predicate_threshold": -0.14,
            "predicate_comparison": "qpos < -0.14",
        })
        if len(joints) != 1 or site_state.parent_name != target["object"]:
            raise RuntimeError(f"drawer site did not resolve uniquely: {joints}")
    elif task_name == "turn_on_the_stove":
        joints = list(env.env.get_object(target["object"]).joints)
        record.update({
            "object_joint_names": joints,
            "resolved_joint": joint_record(env, joints[0]),
            "predicate_threshold": 0.5,
            "predicate_comparison": "qpos >= 0.5",
        })
        if len(joints) != 1:
            raise RuntimeError(f"stove object did not resolve uniquely: {joints}")
    else:
        record["bowl_runtime"] = body_and_extent(env, target["bowl"])
        record["plate_runtime"] = body_and_extent(env, target["plate"])
        record["predicate_geometry"] = {
            "vertical": "plate body z <= bowl body z",
            "planar_distance_max_m": 0.03,
            "contact_required": True,
        }
    return record


def main() -> int:
    args = parse_args()
    run_dir = create_run_directory(args.run_root, args.run_id)
    stdout, stderr = io.StringIO(), io.StringIO()
    config = {"stage": "E4-0_E4-1_runtime_progress_audit", "episodes": list(HELDOUT_EPISODES)}
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "mujoco": importlib.metadata.version("mujoco"),
        "robosuite": importlib.metadata.version("robosuite"),
    }
    git_state = {
        "project": git_record(REPOSITORY_ROOT),
        "libero": git_record(args.libero_root),
        "robosuite_source": git_record(REPOSITORY_ROOT / "third_party/robosuite-src"),
    }
    env = None
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            selection, task_manifest = load_selection(args.selection, args.task_manifest)
            wrapper, robosuite_root, assets_root = bootstrap_runtime(
                args.libero_root, args.dataset_root, run_dir / "artifacts/libero_config"
            )
            records: List[Dict[str, Any]] = []
            for task in selection["tasks"]:
                dataset_path = args.dataset_root / task["demonstration_relative_path"]
                with h5py.File(dataset_path, "r") as handle:
                    demo_count = len(handle["data"])
                source = task_source_record(task_manifest, task["suite"], task["task_id"])
                env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                episode = load_episode(
                    env,
                    dataset_path=dataset_path,
                    episode_index=HELDOUT_EPISODES[0],
                    robosuite_package_root=robosuite_root,
                    libero_assets_root=assets_root,
                )
                progress = inspect_task(env, task["name"])
                record = {
                    "task": task["name"],
                    "suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "dataset_relative_path": task["demonstration_relative_path"],
                    "dataset_demo_count": demo_count,
                    "heldout_indices_available": all(index < demo_count for index in HELDOUT_EPISODES),
                    "runtime_probe_episode": episode["episode"],
                    "bddl_file": source["bddl_file_path"],
                    "progress": progress,
                }
                records.append(record)
                print(json.dumps(record, sort_keys=True))
                env.close()
                env = None
            criteria = {
                "three_tasks": len(records) == 3,
                "all_demos_3_through_9_available": all(x["heldout_indices_available"] for x in records),
                "exact_progress_identifier_per_task": len(records) == 3,
                "drawer_scalar_joint": records[0]["progress"]["resolved_joint"]["joint_type"] == 2,
                "stove_scalar_joint": records[1]["progress"]["resolved_joint"]["joint_type"] == 3,
                "bowl_plate_runtime_geometry": "bowl_runtime" in records[2]["progress"],
            }
            gate = {"passed": all(criteria.values()), "criteria": criteria}
            write_json(run_dir / "artifacts/progress_runtime_audit.json", {"schema_version": 1, "records": records, "gate": gate})
            write_json(run_dir / "artifacts/failure_examples.json", [] if gate["passed"] else records)
            metrics = {"run_id": args.run_id, "status": "completed", "gate": gate, "records": records}
        write_run_record(
            run_dir,
            config=config,
            command=shlex.join([sys.executable, *sys.argv]),
            environment=environment,
            git_state=git_state,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            metrics=metrics,
        )
        return 0 if gate["passed"] else 2
    except Exception as exc:
        stderr.write(traceback.format_exc())
        metrics = {"run_id": args.run_id, "status": "failed", "gate": {"passed": False}, "error": repr(exc)}
        write_run_record(
            run_dir,
            config=config,
            command=shlex.join([sys.executable, *sys.argv]),
            environment=environment,
            git_state=git_state,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            metrics=metrics,
        )
        return 1
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
