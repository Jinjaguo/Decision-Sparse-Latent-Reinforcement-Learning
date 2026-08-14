"""Outcome-blind metric helpers for EXP3 time-indexed q criticality."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def rotation_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    """Return the SO(3) geodesic angle in radians."""
    a = np.asarray(left, dtype=np.float64).reshape(3, 3)
    b = np.asarray(right, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def quaternion_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    """Return sign-invariant quaternion angular distance in radians."""
    a = np.asarray(left, dtype=np.float64).reshape(4)
    b = np.asarray(right, dtype=np.float64).reshape(4)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0)))


def concentration_metrics(values: Sequence[float]) -> Dict[str, float]:
    """Compute frozen 12-point effect-mass concentration statistics.

    Top-k counts are ceil(k * n): for n=12, top 10/20/30 are 2/3/4.
    An all-zero curve has zero top-k mass and Gini, normalized entropy one,
    and an explicit all_zero flag.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("values must be a non-empty finite nonnegative vector")
    total = float(x.sum())
    if total == 0.0:
        return {"top10_mass": 0.0, "top20_mass": 0.0, "top30_mass": 0.0,
                "gini": 0.0, "normalized_entropy": 1.0, "all_zero": True}
    ordered = np.sort(x)[::-1]
    masses = {}
    for label, fraction in (("top10_mass", 0.10), ("top20_mass", 0.20), ("top30_mass", 0.30)):
        count = int(np.ceil(fraction * x.size))
        masses[label] = float(ordered[:count].sum() / total)
    ascending = ordered[::-1]
    index = np.arange(1, x.size + 1, dtype=np.float64)
    gini = float((2.0 * np.sum(index * ascending) / (x.size * total)) - (x.size + 1.0) / x.size)
    probabilities = x / total
    positive = probabilities[probabilities > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / np.log(x.size)) if x.size > 1 else 0.0
    return {**masses, "gini": gini, "normalized_entropy": entropy, "all_zero": False}


def aggregate_interventions(values: Sequence[float]) -> Dict[str, float]:
    """Frozen branch aggregation across four directions and two signs."""
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (8,) or np.any(~np.isfinite(x)):
        raise ValueError("exactly eight finite intervention values are required")
    return {
        "median": float(np.median(x)), "mean": float(np.mean(x)),
        "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
        "minimum": float(np.min(x)), "maximum": float(np.max(x)),
    }

