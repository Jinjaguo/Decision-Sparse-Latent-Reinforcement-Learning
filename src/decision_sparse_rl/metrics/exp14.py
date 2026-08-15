"""Pure task-space proposal helpers for EXP14."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def unit_vector(value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm < 1e-12 else vector / norm


def waypoint_chunk(reference: np.ndarray, displacement: Sequence[float], scale_m: float = 0.05, taper: bool = False) -> np.ndarray:
    """Map a desired total EEF displacement to a bounded open-loop OSC chunk."""
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 7 or len(ref) == 0 or scale_m <= 0:
        raise ValueError("invalid waypoint inputs")
    delta = np.asarray(displacement, dtype=np.float64)
    if delta.shape != (3,):
        raise ValueError("displacement must have shape (3,)")
    weights = np.linspace(1.4, 0.6, len(ref)) if taper else np.ones(len(ref))
    weights /= weights.sum()
    out = ref.copy()
    out[:, :3] = weights[:, None] * delta[None, :] / scale_m
    return out


def object_frame_target(anchor: Sequence[float], relative_waypoints: Sequence[Sequence[float]]) -> np.ndarray:
    relative = np.asarray(relative_waypoints, dtype=np.float64)
    if relative.ndim != 2 or relative.shape[1] != 3 or len(relative) == 0:
        raise ValueError("relative waypoints must be non-empty Nx3")
    return np.asarray(anchor, dtype=np.float64) + np.median(relative, axis=0)


def ridge_inverse(action_summaries: np.ndarray, realized_deltas: np.ndarray, desired_delta: Sequence[float], l2: float = 1.0) -> np.ndarray:
    x, y = np.asarray(action_summaries, float), np.asarray(realized_deltas, float)
    desired = np.asarray(desired_delta, float)
    if x.ndim != 2 or y.shape != (len(x), 3) or x.shape[1] != 6 or desired.shape != (3,) or len(x) < 3:
        raise ValueError("invalid inverse-response arrays")
    forward = np.linalg.solve(x.T @ x + l2 * np.eye(6), x.T @ y)
    return forward @ np.linalg.solve(forward.T @ forward + l2 * np.eye(3), desired)


def diverse_topk(features: np.ndarray, scores: Sequence[float], k: int, minimum_distance: float = 0.05) -> list[int]:
    x, score = np.asarray(features, float), np.asarray(scores, float)
    if x.ndim != 2 or score.shape != (len(x),) or not 0 < k <= len(x):
        raise ValueError("invalid diverse selection")
    chosen: list[int] = []
    for index in np.argsort(-score, kind="stable"):
        if not chosen or min(float(np.linalg.norm(x[index] - x[j])) for j in chosen) >= minimum_distance:
            chosen.append(int(index))
        if len(chosen) == k:
            break
    for index in np.argsort(-score, kind="stable"):
        if int(index) not in chosen:
            chosen.append(int(index))
        if len(chosen) == k:
            break
    return chosen
