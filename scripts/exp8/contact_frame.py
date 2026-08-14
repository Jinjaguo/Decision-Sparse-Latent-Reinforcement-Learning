"""Audited continuous contact-frame measurement for MuJoCo 3.2.3."""

from __future__ import annotations

import json
from typing import Any

import mujoco
import numpy as np

from decision_sparse_rl.envs.mujoco_snapshot import native_model_data
from decision_sparse_rl.metrics.exp8 import tangent_basis
from scripts.exp7.contact_geometry import geom_name


def _key(model: Any, left: int, right: int) -> str:
    return "|".join(sorted((f"{left}:{geom_name(model, left)}", f"{right}:{geom_name(model, right)}")))


def _active(model: Any, data: Any) -> dict[str, tuple[int, Any]]:
    result = {}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        key = _key(model, int(contact.geom[0]), int(contact.geom[1]))
        if key not in result or float(contact.dist) < float(result[key][1].dist):
            result[key] = (index, contact)
    return result


def _point_jacobian(model: Any, data: Any, point: np.ndarray, geom_id: int) -> np.ndarray:
    jacobian = np.zeros((3, int(model.nv)), dtype=np.float64)
    rotational = np.zeros_like(jacobian)
    body_id = int(model.geom_bodyid[int(geom_id)])
    mujoco.mj_jac(model, data, jacobian, rotational, np.asarray(point, dtype=np.float64), body_id)
    return jacobian


def _surface_pair(model: Any, data: Any, spec: dict[str, Any], active: dict[str, tuple[int, Any]], distmax: float) -> dict[str, Any]:
    left, right = int(spec["geom1_id"]), int(spec["geom2_id"])
    key = spec["pair"]
    contact_index = None
    if key in active:
        contact_index, contact = active[key]
        c0, c1 = int(contact.geom[0]), int(contact.geom[1])
        normal = np.asarray(contact.frame[:3], dtype=np.float64).copy()
        if (c0, c1) != (left, right):
            if (c0, c1) != (right, left):
                raise RuntimeError("contact identity disagrees with frozen pair")
            normal = -normal
        gap = float(contact.dist)
        midpoint = np.asarray(contact.pos, dtype=np.float64).copy()
        point_a = midpoint - 0.5 * gap * normal
        point_b = midpoint + 0.5 * gap * normal
        geometry_valid = True
    else:
        geometry_valid = False
        point_a = point_b = np.zeros(3, dtype=np.float64)
        gap = float(distmax)
        used_distmax = float(distmax)
        for query_max in sorted(set((float(distmax), 0.2, 1.0, 5.0))):
            segment = np.zeros(6, dtype=np.float64)
            candidate_gap = float(mujoco.mj_geomDistance(model, data, left, right, query_max, segment))
            candidate_a, candidate_b = segment[:3].copy(), segment[3:].copy()
            displacement = candidate_b - candidate_a
            if np.linalg.norm(displacement) > 1e-12 and candidate_gap < query_max:
                gap, point_a, point_b, used_distmax = candidate_gap, candidate_a, candidate_b, query_max
                geometry_valid = True
                break
        displacement = point_b - point_a
        normal = displacement / np.linalg.norm(displacement) if geometry_valid else np.zeros(3, dtype=np.float64)
    if geometry_valid:
        normal, tangent1, tangent2 = tangent_basis(normal)
        frame = np.stack((normal, tangent1, tangent2))
        relative_jacobian = _point_jacobian(model, data, point_b, right) - _point_jacobian(model, data, point_a, left)
        relative_velocity_world = relative_jacobian @ np.asarray(data.qvel, dtype=np.float64)
        relative_velocity = frame @ relative_velocity_world
    else:
        tangent1 = tangent2 = np.zeros(3, dtype=np.float64)
        frame = np.zeros((3, 3), dtype=np.float64)
        relative_jacobian = np.zeros((3, int(model.nv)), dtype=np.float64)
        relative_velocity = np.zeros(3, dtype=np.float64)
    force = np.zeros(6, dtype=np.float64)
    if contact_index is not None:
        native_force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, int(contact_index), native_force)
        contact_frame = np.asarray(active[key][1].frame, dtype=np.float64).reshape(3, 3)
        world_force = contact_frame.T @ native_force[:3]
        force[:3] = frame @ world_force
        world_torque = contact_frame.T @ native_force[3:]
        force[3:] = frame @ world_torque
    return {
        "physical_group": spec["physical_group"],
        "pair": key,
        "geom1_name": spec["geom1_name"],
        "geom2_name": spec["geom2_name"],
        "active": contact_index is not None,
        "geometry_valid": geometry_valid,
        "distance_query_max_m": 0.0 if contact_index is not None else used_distmax,
        "nearest_point_a": point_a.tolist(),
        "nearest_point_b": point_b.tolist(),
        "relative_displacement": (point_b - point_a).tolist(),
        "signed_gap_m": gap,
        "normal": normal.tolist(),
        "tangent1": tangent1.tolist(),
        "tangent2": tangent2.tolist(),
        "relative_velocity_contact": relative_velocity.tolist(),
        "contact_force_torque_contact": force.tolist(),
        "relative_point_jacobian": relative_jacobian.tolist(),
    }


def measure_contact_frame(env: Any, schema: dict[str, Any], task: str, arm_dof_indices: list[int]) -> dict[str, Any]:
    model, data = native_model_data(env.sim)
    active = _active(model, data)
    pair_specs = []
    for group, pairs in schema["tasks"][task]["pair_groups"].items():
        for pair in pairs:
            pair_specs.append({**pair, "physical_group": group})
    rows = [_surface_pair(model, data, spec, active, float(schema["distance_max_m"])) for spec in pair_specs]
    valid = [row for row in rows if row["geometry_valid"]]
    if not valid:
        raise RuntimeError(f"no valid nearest surface pair for {task}")
    for row in rows:
        jacobian = np.asarray(row.pop("relative_point_jacobian"), dtype=np.float64)
        row["relative_point_jacobian_arm"] = jacobian[:, arm_dof_indices].tolist()
    exact_mode = sorted(key for key in active if key in set(schema["tasks"][task]["relevant_pairs"]))
    return {
        "exact_mode_json": json.dumps(exact_mode),
        "active_physical_identities_json": json.dumps(sorted(row["pair"] for row in rows if row["active"])),
        "pair_features_json": json.dumps(rows, sort_keys=True),
        "valid_pair_count": sum(row["geometry_valid"] for row in rows),
        "active_pair_count": sum(row["active"] for row in rows),
        "minimum_signed_gap_m": min(row["signed_gap_m"] for row in valid),
        "maximum_normal_force": max(abs(row["contact_force_torque_contact"][0]) for row in rows),
    }
