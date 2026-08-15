"""Pure multi-source candidate-generation utilities for EXP13."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def temporal_warp(chunk: np.ndarray, scale: float) -> np.ndarray:
    action = np.asarray(chunk, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 7 or scale <= 0:
        raise ValueError("invalid action chunk or warp scale")
    source = np.arange(len(action), dtype=float)
    query = np.clip(np.arange(len(action), dtype=float) * scale, 0, len(action) - 1)
    out = np.zeros_like(action)
    for channel in range(6):
        out[:, channel] = np.interp(query, source, action[:, channel])
    out[:, 6] = np.sign(action[np.rint(query).astype(int), 6])
    return out


def shift_chunk(actions: np.ndarray, start: int, length: int, shift: int) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or length < 1 or not 0 <= start < len(values):
        raise ValueError("invalid shift request")
    indexes = np.clip(np.arange(start, start + length) + int(shift), 0, len(values) - 1)
    return values[indexes].copy()


def fit_progress_direction(action_summaries: np.ndarray, progress_delta: Sequence[float], l2: float = 1.0) -> np.ndarray:
    x, y = np.asarray(action_summaries, float), np.asarray(progress_delta, float)
    if x.ndim != 2 or y.shape != (len(x),) or x.shape[1] != 6 or len(x) < 4:
        raise ValueError("insufficient progress-direction examples")
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-8] = 1
    z = (x - mu) / sd
    coefficient = np.linalg.solve(z.T @ z + np.eye(6) * l2, z.T @ y) / sd
    norm = np.linalg.norm(coefficient)
    return coefficient / max(norm, 1e-12)


def nearest_library(query: Sequence[float], contexts: np.ndarray, excluded: Sequence[bool], count: int = 2) -> np.ndarray:
    q, x, excluded = np.asarray(query, float), np.asarray(contexts, float), np.asarray(excluded, bool)
    if x.ndim != 2 or q.shape != (x.shape[1],) or excluded.shape != (len(x),) or count < 1:
        raise ValueError("invalid library query")
    distance = np.linalg.norm(x - q, axis=1)
    distance[excluded] = np.inf
    valid = np.flatnonzero(np.isfinite(distance))
    if len(valid) < count:
        raise ValueError("insufficient cross-demo library")
    return np.argsort(distance, kind="mergesort")[:count]


def bounded_candidate(reference: np.ndarray, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref, target = np.asarray(reference, float), np.asarray(desired, float)
    if ref.shape != target.shape or ref.ndim != 2 or ref.shape[1] != 7:
        raise ValueError("candidate shapes differ")
    executed = target.copy()
    executed[:, :6] = np.clip(executed[:, :6], -1, 1)
    executed[:, 6] = np.sign(executed[:, 6])
    return target - ref, executed


def authorize_family(clipped_fraction: float, success_rate: float, opportunity_rate: float, diversity: float) -> bool:
    values = (clipped_fraction, success_rate, opportunity_rate, diversity)
    if not all(np.isfinite(values)):
        return False
    return bool(clipped_fraction <= .10 and success_rate >= .80 and opportunity_rate > 0 and diversity >= .02)

