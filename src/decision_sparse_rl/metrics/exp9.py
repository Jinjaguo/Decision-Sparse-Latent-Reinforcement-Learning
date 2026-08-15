"""Pure metrics and feature utilities for EXP9 retrospective feasibility."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def padded_window(values: np.ndarray, end_index: int, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``length`` rows ending at ``end_index`` (inclusive), left padded.

    Padding repeats the earliest available row while the mask remains zero.  The
    explicit mask prevents a model from interpreting padding as real history.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not 0 <= end_index < len(x) or length <= 0:
        raise ValueError("invalid history window")
    start = max(0, end_index - length + 1)
    observed = x[start : end_index + 1]
    pad = length - len(observed)
    out = np.concatenate([np.repeat(observed[:1], pad, axis=0), observed], axis=0)
    mask = np.concatenate([np.zeros(pad), np.ones(len(observed))]).astype(np.float64)
    return out, mask


def padded_future(values: np.ndarray, start_index: int, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return a right-padded frozen future-action chunk and its validity mask."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not 0 <= start_index <= len(x) or length <= 0:
        raise ValueError("invalid action chunk")
    observed = x[start_index : min(len(x), start_index + length)]
    out = np.zeros((length, x.shape[1]), dtype=np.float64)
    out[: len(observed)] = observed
    mask = np.zeros(length, dtype=np.float64)
    mask[: len(observed)] = 1.0
    return out, mask


def contact_set_events(reference: Iterable[str], perturbed: Iterable[str]):
    """Return sorted named-pair additions and removals."""
    ref, pert = set(reference), set(perturbed)
    return sorted(pert - ref), sorted(ref - pert)


def energy_score(samples: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Multivariate energy score for equally weighted predictive samples.

    ``samples`` has shape [N, S, D] and observations [N, D].
    """
    s = np.asarray(samples, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    if s.ndim != 3 or y.shape != (s.shape[0], s.shape[2]):
        raise ValueError("energy score shape mismatch")
    first = np.linalg.norm(s - y[:, None, :], axis=2).mean(axis=1)
    pair = np.linalg.norm(s[:, :, None, :] - s[:, None, :, :], axis=3).mean(axis=(1, 2))
    return first - 0.5 * pair


def expected_gaussian_energy_score(
    mean: np.ndarray,
    std: np.ndarray,
    observations: np.ndarray,
    seed: int = 0,
    draws: int = 32,
) -> np.ndarray:
    """Deterministic Monte-Carlo energy score for diagonal Gaussians."""
    mu, sigma, y = map(lambda z: np.asarray(z, dtype=np.float64), (mean, std, observations))
    if mu.shape != sigma.shape or mu.shape != y.shape or np.any(sigma <= 0):
        raise ValueError("invalid Gaussian parameters")
    rng = np.random.default_rng(seed)
    samples = mu[:, None, :] + sigma[:, None, :] * rng.standard_normal((len(mu), draws, mu.shape[1]))
    return energy_score(samples, y)


def binary_metrics(y_true: Sequence[int], probability: Sequence[float], threshold: float):
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    return {
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "false_safe_rate": fp / max(1, tn + fp),
        "threshold": float(threshold),
    }


def select_specificity_threshold(y_true: Sequence[int], probability: Sequence[float], min_sensitivity: float = 0.85):
    """Training-only threshold maximizing specificity subject to sensitivity."""
    p = np.asarray(probability, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], p, [1.0 + 1e-12])))
    feasible = []
    for t in candidates:
        m = binary_metrics(y_true, p, float(t))
        if m["sensitivity"] >= min_sensitivity:
            feasible.append(m)
    if not feasible:
        return 0.0
    return max(feasible, key=lambda m: (m["specificity"], m["threshold"]))["threshold"]


def ece(y_true: Sequence[int], probability: Sequence[float], bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if np.any(mask):
            total += np.mean(mask) * abs(float(np.mean(p[mask]) - np.mean(y[mask])))
    return float(total)


def rank_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Tie-aware AUROC without optional sklearn dependency."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def lifted_features(x: np.ndarray, action_start: int, max_quadratic: int = 24) -> np.ndarray:
    """Deterministic compact lifted dictionary with action interactions."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or not 0 < action_start < a.shape[1]:
        raise ValueError("invalid lifted input")
    q = a[:, : min(max_quadratic, action_start)]
    act = a[:, action_start:]
    squares = q * q
    interactions = q[:, : min(8, q.shape[1]), None] * act[:, None, : min(8, act.shape[1])]
    return np.concatenate([a, squares, interactions.reshape(len(a), -1)], axis=1)


def mixture_moments(weights: np.ndarray, means: np.ndarray, stds: np.ndarray):
    """Validate a diagonal Gaussian mixture and return marginal mean/std."""
    w = np.asarray(weights, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    sd = np.asarray(stds, dtype=np.float64)
    if w.ndim != 2 or mu.ndim != 3 or sd.shape != mu.shape or mu.shape[:2] != w.shape:
        raise ValueError("mixture shape mismatch")
    if np.any(w < 0) or not np.allclose(w.sum(axis=1), 1.0, atol=1e-10) or np.any(sd <= 0):
        raise ValueError("invalid mixture parameters")
    mean = np.sum(w[..., None] * mu, axis=1)
    variance = np.sum(w[..., None] * (sd**2 + mu**2), axis=1) - mean**2
    return mean, np.sqrt(np.maximum(variance, 0.0))


def interval_coverage(observations: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y, lo, hi = map(lambda z: np.asarray(z, dtype=np.float64), (observations, lower, upper))
    if y.shape != lo.shape or y.shape != hi.shape or np.any(lo > hi):
        raise ValueError("invalid interval")
    return float(np.mean((y >= lo) & (y <= hi)))


def validate_demo_fold_isolation(train_task_episode, test_task_episode) -> bool:
    """Return true only when no complete demonstration crosses a fold boundary."""
    return set(map(tuple, train_task_episode)).isdisjoint(set(map(tuple, test_task_episode)))
