"""Reference-only matching and basis-invariant operator metrics for EXP5."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


def robust_scales(values: np.ndarray, floors: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    floor = np.asarray(floors, dtype=np.float64)
    if x.ndim != 2 or floor.shape != (x.shape[1],) or np.any(floor <= 0):
        raise ValueError("invalid values/floors")
    center = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - center), axis=0)
    return center, np.maximum(mad, floor)


def shrinkage_covariance(z: np.ndarray, shrinkage: float = 0.2) -> np.ndarray:
    x = np.asarray(z, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not 0 <= shrinkage <= 1:
        raise ValueError("invalid shrinkage input")
    empirical = np.cov(x, rowvar=False, ddof=1)
    if empirical.ndim == 0:
        empirical = empirical.reshape(1, 1)
    target = np.diag(np.diag(empirical))
    result = (1.0 - shrinkage) * empirical + shrinkage * target
    result += np.eye(result.shape[0]) * 1e-8
    return result


def mahalanobis_cost(a: np.ndarray, b: np.ndarray, precision: np.ndarray) -> np.ndarray:
    left = np.asarray(a, dtype=np.float64); right = np.asarray(b, dtype=np.float64); p = np.asarray(precision, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or p.shape != (left.shape[1], left.shape[1]):
        raise ValueError("incompatible Mahalanobis shapes")
    delta = left[:, None, :] - right[None, :, :]
    squared = np.einsum("...i,ij,...j->...", delta, p, delta)
    return np.sqrt(np.maximum(squared, 0.0))


def monotone_match(cost: np.ndarray, window_fraction: float = 0.25) -> Dict[str, np.ndarray]:
    """DTW path constrained by normalized-time Sakoe-Chiba distance."""
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or not np.all(np.isfinite(c)) or not 0 < window_fraction <= 1:
        raise ValueError("invalid matching input")
    n, m = c.shape; dp = np.full((n, m), np.inf); parent = np.full((n, m, 2), -1, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            if abs(i / max(n - 1, 1) - j / max(m - 1, 1)) > window_fraction:
                continue
            if i == 0 and j == 0:
                dp[i, j] = c[i, j]; continue
            candidates = []
            if i: candidates.append((dp[i - 1, j], i - 1, j))
            if j: candidates.append((dp[i, j - 1], i, j - 1))
            if i and j: candidates.append((dp[i - 1, j - 1], i - 1, j - 1))
            value, pi, pj = min(candidates, key=lambda x: (x[0], x[1] + x[2], x[1], x[2]))
            if np.isfinite(value): dp[i, j] = value + c[i, j]; parent[i, j] = (pi, pj)
    if not np.isfinite(dp[-1, -1]): raise RuntimeError("no feasible monotone path")
    path = []; i, j = n - 1, m - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0: break
        i, j = map(int, parent[i, j])
    path.reverse()
    return {"path": np.asarray(path, dtype=np.int64), "total_cost": np.asarray(dp[-1, -1]), "mean_cost": np.asarray(np.mean([c[i, j] for i, j in path]))}


def select_prototype_branches(descriptors: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    x = np.asarray(descriptors, dtype=np.float64); p = np.asarray(prototypes, dtype=np.float64)
    distances = np.linalg.norm(x[:, None, :] - p[None, :, :], axis=2)
    used = set(); selected = []
    for column in range(p.shape[0]):
        order = np.argsort(distances[:, column], kind="mergesort")
        index = next(int(i) for i in order if int(i) not in used)
        used.add(index); selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def central_operator(positive: np.ndarray, negative: np.ndarray, radius: float) -> np.ndarray:
    plus = np.asarray(positive, dtype=np.float64); minus = np.asarray(negative, dtype=np.float64)
    if plus.shape != minus.shape or plus.ndim != 2 or plus.shape[0] != 7 or radius <= 0:
        raise ValueError("expected seven paired direction vectors")
    return ((plus - minus) / (2.0 * radius)).T


def operator_geometry(operator: np.ndarray) -> Dict[str, np.ndarray | float]:
    j = np.asarray(operator, dtype=np.float64)
    if j.ndim != 2 or j.shape[1] != 7 or np.any(~np.isfinite(j)):
        raise ValueError("operator must be finite output_dim x 7")
    _, singular, vt = np.linalg.svd(j, full_matrices=False)
    squared = singular * singular; total = float(np.sum(squared)); probabilities = squared / total if total else np.zeros_like(squared)
    effective = float(np.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])))) if total else 0.0
    return {"gram": j.T @ j, "singular_values": singular, "right_vectors": vt.T, "spectral_norm": float(singular[0]) if len(singular) else 0.0, "frobenius_norm": float(np.linalg.norm(j)), "leading_share": float(probabilities[0]) if len(probabilities) else 0.0, "effective_rank": effective}


def projector(vectors: np.ndarray, k: int) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or not 1 <= k <= v.shape[1]: raise ValueError("invalid projector rank")
    q, _ = np.linalg.qr(v[:, :k]); return q @ q.T


def projector_similarity(left: np.ndarray, right: np.ndarray, k: int) -> float:
    a = np.asarray(left, dtype=np.float64); b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.shape[0] != a.shape[1]: raise ValueError("projectors must be equal square matrices")
    return float(1.0 - np.linalg.norm(a - b, ord="fro") / np.sqrt(2.0 * k))


def bh_fdr(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=np.float64); order = np.argsort(p); ranked = p[order] * len(p) / np.arange(1, len(p) + 1); ranked = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty_like(p); out[order] = np.minimum(ranked, 1.0); return out
