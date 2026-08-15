"""Pure utilities for EXP10 phase/macro-action retrospective evaluation."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


PHASES = tuple(f"P{i}" for i in range(7))
REGIMES = tuple(f"R{i}" for i in range(7))


def frozen_phase_sequence(
    normalized_time: Sequence[float],
    target_gripper_contact: Sequence[bool],
    predicate: Sequence[bool],
    min_dwell: int = 2,
) -> np.ndarray:
    """Apply the frozen reference-only P0--P6 phase schema.

    Candidate boundaries use time plus physical interaction evidence.  A
    two-step dwell and monotone hysteresis prevent contact chatter from moving a
    trajectory backwards.  P6 is reserved for the exact task predicate.
    """
    t = np.asarray(normalized_time, dtype=np.float64)
    c = np.asarray(target_gripper_contact, dtype=bool)
    p = np.asarray(predicate, dtype=bool)
    if t.ndim != 1 or c.shape != t.shape or p.shape != t.shape or len(t) == 0:
        raise ValueError("phase inputs must be non-empty aligned vectors")
    if min_dwell < 1:
        raise ValueError("min_dwell must be positive")
    contact = np.zeros(len(c), dtype=bool)
    for i in range(len(c)):
        lo = max(0, i - min_dwell + 1)
        contact[i] = i - lo + 1 >= min_dwell and bool(np.all(c[lo : i + 1]))
    raw = np.zeros(len(t), dtype=np.int64)
    raw[t >= 0.12] = 1
    raw[(t >= 0.28) | contact] = 2
    raw[t >= 0.48] = 3
    raw[t >= 0.68] = 4
    raw[t >= 0.86] = 5
    raw[p] = 6
    # First-crossing hysteresis: phases never go backwards.
    return np.maximum.accumulate(raw)


def one_hot(labels: Sequence[int], classes: int = 7) -> np.ndarray:
    x = np.asarray(labels, dtype=np.int64)
    if np.any((x < 0) | (x >= classes)):
        raise ValueError("label outside vocabulary")
    return np.eye(classes, dtype=np.float64)[x]


def action_summary(chunk: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fixed-length macro-action summary: mean/std/first/last/delta + coverage."""
    a = np.asarray(chunk, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if a.ndim != 2 or m.shape != (len(a),):
        raise ValueError("invalid chunk or mask")
    valid = a[m > 0.5]
    if not len(valid):
        valid = np.zeros((1, a.shape[1]), dtype=np.float64)
    return np.concatenate(
        [valid.mean(0), valid.std(0), valid[0], valid[-1], valid[-1] - valid[0], [m.mean()]]
    )


def history_summary(history: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fixed-length state-history summary: last/mean/std/delta + coverage."""
    h = np.asarray(history, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if h.ndim != 2 or m.shape != (len(h),):
        raise ValueError("invalid history or mask")
    valid = h[m > 0.5]
    if not len(valid):
        valid = np.zeros((1, h.shape[1]), dtype=np.float64)
    return np.concatenate(
        [valid[-1], valid.mean(0), valid.std(0), valid[-1] - valid[0], [m.mean()]]
    )


def regime_from_pairs(reference_pairs: Iterable[str], perturbed_pairs: Iterable[str]) -> int:
    """Map exact contact pairs to the frozen coarse R0--R6 vocabulary."""
    ref, pert = set(reference_pairs), set(perturbed_pairs)
    if ref - pert and any("gripper" in x for x in ref - pert):
        return 6  # release/drop of an existing gripper contact
    if not pert:
        return 0
    gripper = any("gripper" in x for x in pert)
    support = any(any(k in x for k in ("plate", "table", "cabinet", "stove")) for x in pert)
    changed = pert != ref
    if gripper and support:
        return 5
    if gripper:
        return 2
    if support and len(pert) > 1:
        return 4
    if support:
        return 3
    return 1 if changed else 0


def sequence_edit_rate(reference: Sequence[int], perturbed: Sequence[int]) -> float:
    """Normalized Levenshtein distance for short coarse-regime sequences."""
    a, b = list(reference), list(perturbed)
    table = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int64)
    table[:, 0] = np.arange(len(a) + 1)
    table[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            table[i, j] = min(
                table[i - 1, j] + 1,
                table[i, j - 1] + 1,
                table[i - 1, j - 1] + int(a[i - 1] != b[j - 1]),
            )
    return float(table[-1, -1] / max(1, len(a), len(b)))


def interval_coverage(y: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.6448536269514722) -> float:
    y, mean, std = map(lambda x: np.asarray(x, dtype=np.float64), (y, mean, std))
    if y.shape != mean.shape or std.shape != y.shape or np.any(std < 0):
        raise ValueError("coverage shape mismatch")
    return float(np.mean(np.abs(y - mean) <= z * std))


def phase_to_landmark_length(phases: Sequence[int], index: int, cap: int = 20) -> int:
    """Frozen chunk length ending at the next strictly later phase, capped."""
    values = np.asarray(phases, dtype=np.int64)
    if values.ndim != 1 or not 0 <= index < len(values) or cap < 1:
        raise ValueError("invalid phase-to-landmark request")
    later = np.flatnonzero(values[index + 1 :] > values[index])
    distance = int(later[0] + 1) if len(later) else len(values) - index
    return max(1, min(cap, distance))


def latent_active_dimensions(latent: np.ndarray, variance_floor: float = 1e-6) -> int:
    z = np.asarray(latent, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("latent must contain at least two rows")
    return int(np.sum(np.var(z, axis=0, ddof=1) > variance_floor))


def authorize_independent_routes(gates: dict[str, bool], maximum: int = 3) -> list[str]:
    """Return up to ``maximum`` routes without imposing an all-or-nothing gate."""
    if maximum < 1 or any(key not in PHASES[:0] + tuple("ABCDEF") for key in gates):
        raise ValueError("invalid route registry")
    return [key for key in "ABCDEF" if gates.get(key, False)][:maximum]


def masked_safe_probability(step_probabilities: np.ndarray, reference_labels: Sequence[int], mask: Sequence[float]) -> float:
    """Probability of matching a reference regime path on valid steps only."""
    probability = np.asarray(step_probabilities, dtype=np.float64)
    labels = np.asarray(reference_labels, dtype=np.int64)
    valid = np.asarray(mask, dtype=np.float64)
    if probability.ndim != 2 or labels.shape != (len(probability),) or valid.shape != labels.shape:
        raise ValueError("invalid rollout probability inputs")
    selected = probability[np.arange(len(labels)), labels]
    return float(np.prod(np.where(valid > .5, selected, 1.0)))
