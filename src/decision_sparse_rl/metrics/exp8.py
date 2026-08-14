"""Pure metrics and deterministic transforms for EXP8."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def tangent_basis(normal: Sequence[float], floor: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic right-handed frame with the supplied oriented normal."""
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm <= floor:
        raise ValueError("normal is singular")
    n = n / norm
    axes = np.eye(3, dtype=np.float64)
    axis_index = int(np.argmin(np.abs(axes @ n)))
    t1 = axes[axis_index] - float(axes[axis_index] @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    return n, t1, t2


def project_contact_motion(relative_jacobian: np.ndarray, arm_delta_q: Sequence[float], frame: np.ndarray) -> np.ndarray:
    jacobian = np.asarray(relative_jacobian, dtype=np.float64)
    delta = np.asarray(arm_delta_q, dtype=np.float64)
    basis = np.asarray(frame, dtype=np.float64)
    if jacobian.shape != (3, delta.size) or basis.shape != (3, 3):
        raise ValueError("incompatible action-projection shapes")
    return basis @ (jacobian @ delta)


def permutation_invariant_pool(rows: Iterable[Sequence[float]]) -> np.ndarray:
    values = np.asarray(list(rows), dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("pool requires one or more equal-length rows")
    return np.concatenate((values.mean(0), values.min(0), values.max(0), np.sqrt(np.mean(values * values, axis=0))))


def contact_age_sequence(active_identities: Iterable[Iterable[str]]) -> list[dict[str, int]]:
    """Count consecutive reference boundaries for each exact physical identity."""
    previous: dict[str, int] = {}
    result = []
    for identities in active_identities:
        current = {identity: previous.get(identity, 0) + 1 for identity in set(identities)}
        result.append(current)
        previous = current
    return result


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    z = np.asarray(test_x, dtype=np.float64)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1] or x.shape[0] != y.shape[0]:
        raise ValueError("invalid ridge shapes")
    augmented = np.column_stack((np.ones(x.shape[0]), x))
    z_augmented = np.column_stack((np.ones(z.shape[0]), z))
    penalty = np.eye(augmented.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(augmented.T @ augmented + float(alpha) * penalty, augmented.T @ y)
    return z_augmented @ weights


def top1_projector_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an <= 1e-15 or bn <= 1e-15:
        return 0.0
    return float((a @ b / (an * bn)) ** 2)


def select_risk_threshold(y: Sequence[int], probability: Sequence[float], minimum_sensitivity: float = 0.85) -> float:
    """Select on training data only: max specificity subject to sensitivity, then highest threshold."""
    labels = np.asarray(y, dtype=int)
    scores = np.asarray(probability, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    feasible = []
    for threshold in candidates:
        predicted_safe = scores >= threshold
        sensitivity = float(np.mean(predicted_safe[labels == 1])) if np.any(labels == 1) else 0.0
        specificity = float(np.mean(~predicted_safe[labels == 0])) if np.any(labels == 0) else 0.0
        if sensitivity >= minimum_sensitivity:
            feasible.append((specificity, sensitivity, float(threshold)))
    if not feasible:
        return 0.0
    return max(feasible)[2]


def binary_operating_metrics(y: Sequence[int], probability: Sequence[float], threshold: float) -> dict[str, float]:
    labels = np.asarray(y, dtype=int)
    safe = np.asarray(probability, dtype=np.float64) >= float(threshold)
    positive, negative = labels == 1, labels == 0
    return {
        "sensitivity": float(np.mean(safe[positive])) if np.any(positive) else 0.0,
        "specificity": float(np.mean(~safe[negative])) if np.any(negative) else 0.0,
        "ppv": float(np.mean(labels[safe] == 1)) if np.any(safe) else 0.0,
        "npv": float(np.mean(labels[~safe] == 0)) if np.any(~safe) else 0.0,
        "false_safe_rate": float(np.mean(safe[negative])) if np.any(negative) else 0.0,
        "false_block_rate": float(np.mean(~safe[positive])) if np.any(positive) else 0.0,
    }


def expected_calibration_error(y: Sequence[int], probability: Sequence[float], bins: int = 10) -> float:
    labels = np.asarray(y, dtype=np.float64)
    scores = np.asarray(probability, dtype=np.float64)
    ids = np.minimum((scores * bins).astype(int), bins - 1)
    return float(sum(np.mean(ids == index) * abs(np.mean(labels[ids == index]) - np.mean(scores[ids == index])) for index in np.unique(ids)))


def upper_tail(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(array)), "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)), "maximum": float(np.max(array))}


def assert_demo_isolation(train_demo: Sequence[str], test_demo: Sequence[str]) -> None:
    overlap = set(train_demo) & set(test_demo)
    if overlap:
        raise ValueError(f"crossfit demo leakage: {sorted(overlap)}")
