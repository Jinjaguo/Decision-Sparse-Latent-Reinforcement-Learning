"""Explicit MuJoCo state capture, restoration, validation, and serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import mujoco
import numpy as np


KINDS = {
    "fullphysics": "mjSTATE_FULLPHYSICS",
    "integration": "mjSTATE_INTEGRATION",
}


@dataclass(frozen=True)
class MujocoSnapshot:
    kind: str
    state_spec: int
    values: np.ndarray


def native_model_data(sim: Any) -> Tuple[Any, Any]:
    """Return installed MuJoCo model/data from robosuite or a raw holder."""

    model = getattr(sim, "model")
    data = getattr(sim, "data")
    return getattr(model, "_model", model), getattr(data, "_data", data)


def state_spec(kind: str) -> int:
    if kind not in KINDS:
        raise ValueError(f"native state kind must be one of {sorted(KINDS)}, got {kind!r}")
    return int(getattr(mujoco.mjtState, KINDS[kind]))


def capture(sim: Any, kind: str) -> MujocoSnapshot:
    if kind == "legacy":
        values = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()
        return MujocoSnapshot(kind="legacy", state_spec=0, values=values)
    model, data = native_model_data(sim)
    spec = state_spec(kind)
    values = np.empty(int(mujoco.mj_stateSize(model, spec)), dtype=np.float64)
    mujoco.mj_getState(model, data, values, spec)
    return MujocoSnapshot(kind=kind, state_spec=spec, values=values)


def validate(sim: Any, snapshot: MujocoSnapshot) -> Dict[str, Any]:
    if snapshot.values.ndim != 1:
        raise ValueError("snapshot values must be one-dimensional")
    if not np.all(np.isfinite(snapshot.values)):
        raise ValueError("snapshot contains non-finite values")
    if snapshot.kind == "legacy":
        expected = int(np.asarray(sim.get_state().flatten()).size)
    else:
        model, _ = native_model_data(sim)
        expected = int(mujoco.mj_stateSize(model, snapshot.state_spec))
        if snapshot.state_spec != state_spec(snapshot.kind):
            raise ValueError("snapshot state spec does not match kind")
    if snapshot.values.size != expected:
        raise ValueError(f"snapshot has {snapshot.values.size} values; expected {expected}")
    return {"kind": snapshot.kind, "dimension": expected, "all_finite": True}


def restore(sim: Any, snapshot: MujocoSnapshot, *, forward: bool = False) -> None:
    """Restore captured integration inputs.

    The audited robosuite policy-step path begins its next transition with
    ``sim.forward()``. The default therefore does not insert an extra forward at
    the stored pre-policy boundary. Callers that will consume derived fields before
    the next normal environment step must request ``forward=True`` explicitly.
    """
    validate(sim, snapshot)
    if snapshot.kind == "legacy":
        sim.set_state_from_flattened(snapshot.values)
        if forward:
            sim.forward()
        return
    model, data = native_model_data(sim)
    mujoco.mj_setState(model, data, snapshot.values, snapshot.state_spec)
    if forward:
        mujoco.mj_forward(model, data)


def serialize(path: Path, snapshot: MujocoSnapshot) -> None:
    path = Path(path)
    np.savez_compressed(path, kind=np.asarray(snapshot.kind), state_spec=np.int64(snapshot.state_spec), values=snapshot.values)


def deserialize(path: Path) -> MujocoSnapshot:
    with np.load(Path(path), allow_pickle=False) as archive:
        kind = str(archive["kind"].item())
        spec = int(archive["state_spec"].item())
        values = np.asarray(archive["values"], dtype=np.float64).copy()
    return MujocoSnapshot(kind=kind, state_spec=spec, values=values)


def capture_atomic_components(sim: Any) -> Dict[str, np.ndarray]:
    """Capture each atomic component included by mjSTATE_INTEGRATION."""

    model, data = native_model_data(sim)
    integration = state_spec("integration")
    result: Dict[str, np.ndarray] = {}
    for name in dir(mujoco.mjtState):
        if not name.startswith("mjSTATE_"):
            continue
        spec = int(getattr(mujoco.mjtState, name))
        if spec <= 0 or spec & (spec - 1) or spec & integration != spec:
            continue
        values = np.empty(int(mujoco.mj_stateSize(model, spec)), dtype=np.float64)
        mujoco.mj_getState(model, data, values, spec)
        result[name] = values
    return result


def component_errors(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    if before.keys() != after.keys():
        raise ValueError("component sets differ")
    return {
        name: {
            "l2": float(np.linalg.norm(after[name] - before[name])),
            "linf": float(np.max(np.abs(after[name] - before[name]))) if before[name].size else 0.0,
        }
        for name in before
    }
