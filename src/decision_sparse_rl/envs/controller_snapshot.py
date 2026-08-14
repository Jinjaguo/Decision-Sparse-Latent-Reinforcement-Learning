"""Explicit snapshots for the audited fixed OSC_POSE controller and robot buffers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


CONTROLLER_FIELDS = (
    "initial_joint", "initial_ee_pos", "initial_ee_ori_mat",
    "goal_pos", "goal_ori", "relative_ori", "ori_ref", "new_update", "torques",
    "ee_pos", "ee_ori_mat", "ee_pos_vel", "ee_ori_vel", "joint_pos", "joint_vel",
    "J_pos", "J_ori", "J_full", "mass_matrix",
)
ROBOT_SCALAR_FIELDS = ("torques",)
ROBOT_BUFFER_FIELDS = (
    "recent_qpos", "recent_actions", "recent_torques", "recent_ee_forcetorques",
    "recent_ee_pose", "recent_ee_vel", "recent_ee_vel_buffer", "recent_ee_acc",
)
ENVIRONMENT_FIELDS = ("timestep", "cur_time", "done")
GRIPPER_FIELDS = ("current_action",)


def _copy_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported explicit snapshot value: {type(value)}")


def _capture_buffer(buffer: Any) -> Dict[str, Any]:
    if hasattr(buffer, "buf"):
        return {"kind": "ring", "buf": np.asarray(buffer.buf).copy(), "ptr": int(buffer.ptr), "size": int(buffer._size)}
    if hasattr(buffer, "last") and hasattr(buffer, "current"):
        return {"kind": "delta", "last": np.asarray(buffer.last).copy(), "current": np.asarray(buffer.current).copy()}
    raise TypeError(f"unsupported audited buffer type: {type(buffer)}")


def capture(env: Any) -> Dict[str, Any]:
    robot = env.robots[0]
    controller = robot.controller
    return {
        "schema_version": 1,
        "controller": {name: _copy_value(getattr(controller, name)) for name in CONTROLLER_FIELDS},
        "robot": {name: _copy_value(getattr(robot, name)) for name in ROBOT_SCALAR_FIELDS},
        "buffers": {name: _capture_buffer(getattr(robot, name)) for name in ROBOT_BUFFER_FIELDS},
        "gripper": {name: _copy_value(getattr(robot.gripper, name)) for name in GRIPPER_FIELDS},
        "environment": {name: _copy_value(getattr(env.env, name)) for name in ENVIRONMENT_FIELDS},
    }


def _restore_buffer(buffer: Any, state: Dict[str, Any]) -> None:
    if state["kind"] == "ring":
        buffer.buf = np.asarray(state["buf"]).copy()
        buffer.ptr = int(state["ptr"])
        buffer._size = int(state["size"])
    elif state["kind"] == "delta":
        buffer.last = np.asarray(state["last"]).copy()
        buffer.current = np.asarray(state["current"]).copy()
    else:
        raise ValueError(f"unknown buffer snapshot kind: {state['kind']!r}")


def restore(env: Any, snapshot: Dict[str, Any]) -> None:
    validate(snapshot)
    robot = env.robots[0]
    controller = robot.controller
    for name, value in snapshot["controller"].items():
        setattr(controller, name, _copy_value(value))
    for name, value in snapshot["robot"].items():
        setattr(robot, name, _copy_value(value))
    for name, state in snapshot["buffers"].items():
        _restore_buffer(getattr(robot, name), state)
    for name, value in snapshot["gripper"].items():
        setattr(robot.gripper, name, _copy_value(value))
    for name, value in snapshot["environment"].items():
        setattr(env.env, name, _copy_value(value))


def validate(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    if int(snapshot.get("schema_version", -1)) != 1:
        raise ValueError("unsupported controller snapshot schema")
    if set(snapshot["controller"]) != set(CONTROLLER_FIELDS):
        raise ValueError("controller fields do not match the audited schema")
    if set(snapshot["robot"]) != set(ROBOT_SCALAR_FIELDS):
        raise ValueError("robot fields do not match the audited schema")
    if set(snapshot["buffers"]) != set(ROBOT_BUFFER_FIELDS):
        raise ValueError("robot buffers do not match the audited schema")
    if set(snapshot["gripper"]) != set(GRIPPER_FIELDS):
        raise ValueError("gripper fields do not match the audited schema")
    for value in _numeric_arrays(snapshot):
        if not np.all(np.isfinite(value)):
            raise ValueError("controller/robot snapshot contains non-finite values")
    return {"schema_version": 1, "all_finite": True, "numeric_array_count": sum(1 for _ in _numeric_arrays(snapshot))}


def _numeric_arrays(snapshot: Dict[str, Any]) -> Iterable[np.ndarray]:
    for section in ("controller", "robot", "gripper", "environment"):
        for value in snapshot[section].values():
            if isinstance(value, (np.ndarray, int, float, bool, np.generic)):
                yield np.asarray(value)
    for state in snapshot["buffers"].values():
        for value in state.values():
            if isinstance(value, (np.ndarray, int, float, bool, np.generic)):
                yield np.asarray(value)


def field_errors(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:
    validate(left)
    validate(right)
    result: Dict[str, float] = {}
    for section in ("controller", "robot", "gripper", "environment"):
        for name in left[section]:
            a, b = left[section][name], right[section][name]
            key = f"{section}.{name}"
            if a is None or b is None:
                result[key] = 0.0 if a is None and b is None else float("inf")
            elif isinstance(a, (str, bool)):
                result[key] = 0.0 if a == b else 1.0
            else:
                result[key] = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    for name in left["buffers"]:
        for key in left["buffers"][name]:
            a, b = left["buffers"][name][key], right["buffers"][name][key]
            result[f"buffers.{name}.{key}"] = 0.0 if isinstance(a, str) and a == b else (1.0 if isinstance(a, str) else float(np.linalg.norm(np.asarray(a) - np.asarray(b))))
    return result


def serialize(path: Path, snapshot: Dict[str, Any]) -> None:
    validate(snapshot)
    arrays: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {"schema_version": 1, "values": {}}
    for section in ("controller", "robot", "gripper", "environment"):
        metadata["values"][section] = {}
        for name, value in snapshot[section].items():
            key = f"{section}__{name}"
            if isinstance(value, np.ndarray):
                arrays[key] = value
                metadata["values"][section][name] = {"storage": "array", "key": key}
            else:
                metadata["values"][section][name] = {"storage": "json", "value": value}
    metadata["values"]["buffers"] = {}
    for name, state in snapshot["buffers"].items():
        metadata["values"]["buffers"][name] = {}
        for field, value in state.items():
            key = f"buffers__{name}__{field}"
            if isinstance(value, np.ndarray):
                arrays[key] = value
                metadata["values"]["buffers"][name][field] = {"storage": "array", "key": key}
            else:
                metadata["values"]["buffers"][name][field] = {"storage": "json", "value": value}
    arrays["__metadata__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(Path(path), **arrays)


def deserialize(path: Path) -> Dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
        result: Dict[str, Any] = {"schema_version": metadata["schema_version"]}
        for section in ("controller", "robot", "gripper", "environment"):
            result[section] = {}
            for name, entry in metadata["values"][section].items():
                result[section][name] = np.asarray(archive[entry["key"]]).copy() if entry["storage"] == "array" else entry["value"]
        result["buffers"] = {}
        for name, state in metadata["values"]["buffers"].items():
            result["buffers"][name] = {}
            for field, entry in state.items():
                result["buffers"][name][field] = np.asarray(archive[entry["key"]]).copy() if entry["storage"] == "array" else entry["value"]
    validate(result)
    return result
