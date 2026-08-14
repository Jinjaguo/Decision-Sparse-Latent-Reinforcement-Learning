"""Audited MuJoCo surface-distance and exact contact-mode helpers for EXP7."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from decision_sparse_rl.envs.mujoco_snapshot import native_model_data


def geom_name(model: Any, index: int) -> str:
    return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(index)) or f"unnamed_geom_{index}")


def exact_active_pairs(env: Any) -> set[str]:
    model, data = native_model_data(env.sim)
    result: set[str] = set()
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        left, right = int(contact.geom[0]), int(contact.geom[1])
        result.add("|".join(sorted((f"{left}:{geom_name(model, left)}", f"{right}:{geom_name(model, right)}"))))
    return result


def _distance(model: Any, data: Any, left: int, right: int, distmax: float) -> tuple[float, np.ndarray]:
    segment = np.zeros(6, dtype=np.float64)
    value = float(mujoco.mj_geomDistance(model, data, int(left), int(right), float(distmax), segment))
    return value, segment


def _normal_velocity(model: Any, data: Any, left: int, right: int, segment: np.ndarray) -> float:
    axis = np.asarray(segment[3:] - segment[:3], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-15:
        return 0.0
    normal = axis / norm
    jac_left = np.zeros((3, int(model.nv)), dtype=np.float64)
    jac_right = np.zeros((3, int(model.nv)), dtype=np.float64)
    jac_rot = np.zeros_like(jac_left)
    mujoco.mj_jacGeom(model, data, jac_left, jac_rot, int(left))
    mujoco.mj_jacGeom(model, data, jac_right, jac_rot, int(right))
    return float(normal @ ((jac_right - jac_left) @ np.asarray(data.qvel)))


def identify_task(model: Any, schema: dict[str, Any]) -> str:
    names = {geom_name(model, i) for i in range(int(model.ngeom))}
    matches = [task for task, spec in schema["tasks"].items() if set(spec["required_geom_names"]).issubset(names)]
    if len(matches) != 1:
        raise RuntimeError(f"contact schema task match is not unique: {matches}")
    return matches[0]


def measure(env: Any, schema: dict[str, Any], task: str | None = None) -> dict[str, Any]:
    model, data = native_model_data(env.sim)
    task = identify_task(model, schema) if task is None else task
    if task not in schema["tasks"]:
        raise RuntimeError(f"unknown contact-schema task: {task}")
    active = exact_active_pairs(env)
    pair_rows, group_gaps, group_velocities = [], {}, {}
    for group, pairs in schema["tasks"][task]["pair_groups"].items():
        values = []
        for pair in pairs:
            gap, segment = _distance(model, data, pair["geom1_id"], pair["geom2_id"], schema["distance_max_m"])
            velocity = _normal_velocity(model, data, pair["geom1_id"], pair["geom2_id"], segment)
            values.append((gap, velocity, pair))
            pair_rows.append({"group": group, "pair": pair["pair"], "signed_gap_m": gap, "normal_relative_velocity_mps": velocity})
        if values:
            gap, velocity, _ = min(values, key=lambda item: item[0])
            group_gaps[group], group_velocities[group] = float(gap), float(velocity)
    if not group_gaps:
        raise RuntimeError(f"no auditable signed-gap pairs for {task}")
    relevant = set(schema["tasks"][task]["relevant_pairs"])
    mode = tuple(sorted(active & relevant))
    minimum_group = min(group_gaps, key=group_gaps.get)
    return {
        "contact_task": task,
        "contact_mode": mode,
        "contact_mode_json": json.dumps(mode),
        "signed_gap_m": group_gaps[minimum_group],
        "signed_gap_group": minimum_group,
        "normal_relative_velocity_mps": group_velocities[minimum_group],
        "group_signed_gaps_json": json.dumps(group_gaps, sort_keys=True),
        "group_normal_velocities_json": json.dumps(group_velocities, sort_keys=True),
        "pair_geometry_json": json.dumps(pair_rows, sort_keys=True),
    }


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
