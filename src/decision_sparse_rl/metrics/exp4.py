"""Frozen mathematical helpers for EXP4 direction and progress analysis."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def direction_pair_metrics(positive: float, negative: float, numerical_epsilon: float = 1e-15) -> Dict[str, float]:
    """Return sign-symmetric magnitude S and normalized asymmetry A."""
    ep = float(positive); em = float(negative)
    if not np.isfinite(ep) or not np.isfinite(em) or ep < 0 or em < 0:
        raise ValueError("paired effects must be finite and nonnegative")
    if numerical_epsilon <= 0:
        raise ValueError("numerical_epsilon must be positive")
    return {"symmetric": (ep + em) / 2.0, "asymmetry": abs(ep - em) / (ep + em + numerical_epsilon)}


def basis_rms(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (7,) or np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("exactly seven finite nonnegative basis values are required")
    return float(np.sqrt(np.mean(x * x)))


def first_crossing_interpolate(progress: Sequence[float], values: Sequence[float], grid: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate along temporal order, taking the first crossing of each grid value.

    The progress sequence is never sorted. If no adjacent temporal segment crosses a
    grid point, the closest observed progress is used, with the earlier index winning
    ties. The returned integer array records the earlier source index for auditing.
    """
    p = np.asarray(progress, dtype=np.float64); y = np.asarray(values, dtype=np.float64); g = np.asarray(grid, dtype=np.float64)
    if p.ndim != 1 or y.ndim != 1 or p.size != y.size or p.size == 0:
        raise ValueError("progress and values must be equal non-empty vectors")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(g)):
        raise ValueError("inputs must be finite")
    out = np.empty(g.size, dtype=np.float64); source = np.empty(g.size, dtype=np.int64)
    for gi, target in enumerate(g):
        chosen = None
        for i in range(p.size - 1):
            low, high = min(p[i], p[i + 1]), max(p[i], p[i + 1])
            if low <= target <= high:
                if p[i] == p[i + 1]:
                    out[gi] = y[i]; source[gi] = i
                else:
                    weight = (target - p[i]) / (p[i + 1] - p[i])
                    out[gi] = y[i] + weight * (y[i + 1] - y[i]); source[gi] = i
                chosen = i
                break
        if chosen is None:
            distance = np.abs(p - target)
            index = int(np.flatnonzero(distance == np.min(distance))[0])
            out[gi] = y[index]; source[gi] = index
    return out, source


def gram_spectrum(columns: np.ndarray, epsilon: float) -> Dict[str, np.ndarray]:
    """Compute J, G=J^T J, eigenvalues, and singular values from signed columns."""
    c = np.asarray(columns, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 7 or np.any(~np.isfinite(c)):
        raise ValueError("columns must be a finite output_dim x 7 matrix")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    jacobian = c / (2.0 * epsilon)
    gram = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return {"jacobian": jacobian, "gram": gram, "eigenvalues": eigenvalues, "singular_values": np.sqrt(eigenvalues)}


def orthonormal_audit(matrix: np.ndarray, tolerance: float = 1e-12) -> Dict[str, float]:
    q = np.asarray(matrix, dtype=np.float64)
    if q.shape != (7, 7):
        raise ValueError("basis matrix must be 7x7")
    error = float(np.max(np.abs(q.T @ q - np.eye(7))))
    determinant = float(np.linalg.det(q))
    return {"maximum_orthogonality_error": error, "determinant": determinant, "passed": bool(error <= tolerance and abs(abs(determinant) - 1.0) <= tolerance)}
