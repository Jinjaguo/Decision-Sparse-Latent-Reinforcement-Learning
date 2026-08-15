"""Pure grouping, preference, selection, and audit metrics for EXP12."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def group_candidates(rows: Sequence[Mapping], group_key: str = "branch_id") -> dict[str, list[Mapping]]:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        key = str(row[group_key])
        groups[key].append(row)
    return dict(groups)


def pairwise_preferences(
    quality: Sequence[float], tie_tolerance: float = 0.05
) -> list[tuple[int, int, int, float]]:
    values = np.asarray(quality, dtype=np.float64)
    if values.ndim != 1 or tie_tolerance < 0 or not np.all(np.isfinite(values)):
        raise ValueError("invalid pairwise preference input")
    pairs = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            gap = float(values[i] - values[j])
            label = 1 if gap > tie_tolerance else -1 if gap < -tie_tolerance else 0
            pairs.append((i, j, label, abs(gap)))
    return pairs


def oracle_best_indices(quality: Sequence[float], tie_tolerance: float = 0.05) -> np.ndarray:
    values = np.asarray(quality, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or tie_tolerance < 0:
        raise ValueError("invalid oracle set")
    return np.flatnonzero(values >= np.max(values) - tie_tolerance)


def regret(quality: Sequence[float], selected_index: int) -> float:
    values = np.asarray(quality, dtype=np.float64)
    if values.ndim != 1 or not 0 <= selected_index < len(values):
        raise ValueError("invalid selected index")
    return float(np.max(values) - values[selected_index])


def catastrophic_selection(
    catastrophic: Sequence[bool], selected_index: int
) -> bool:
    flags = np.asarray(catastrophic, dtype=bool)
    if flags.ndim != 1 or not 0 <= selected_index < len(flags):
        raise ValueError("invalid catastrophic selection")
    return bool(flags[selected_index] and np.any(~flags))


def nominal_improvement_opportunity(
    quality: Sequence[float], nominal_index: int, minimum_gap: float = 0.05
) -> tuple[bool, float]:
    values = np.asarray(quality, dtype=np.float64)
    if values.ndim != 1 or not 0 <= nominal_index < len(values) or minimum_gap < 0:
        raise ValueError("invalid nominal opportunity")
    gap = float(np.max(values) - values[nominal_index])
    return bool(gap > minimum_gap), gap


def pairwise_accuracy(
    quality: Sequence[float], score: Sequence[float], tie_tolerance: float = 0.05
) -> float:
    q, s = np.asarray(quality, dtype=np.float64), np.asarray(score, dtype=np.float64)
    if q.shape != s.shape or q.ndim != 1:
        raise ValueError("pairwise arrays differ")
    correct = total = 0
    for i, j, label, _ in pairwise_preferences(q, tie_tolerance):
        if label == 0:
            continue
        predicted = 1 if s[i] > s[j] else -1 if s[i] < s[j] else 0
        correct += int(predicted == label)
        total += 1
    return float(correct / total) if total else float("nan")


def top1_accuracy(
    quality: Sequence[float], selected_index: int, tie_tolerance: float = 0.05
) -> float:
    return float(selected_index in set(oracle_best_indices(quality, tie_tolerance).tolist()))


def abstaining_choice(
    score: Sequence[float], uncertainty: Sequence[float], nominal_index: int, threshold: float
) -> tuple[int, bool]:
    scores = np.asarray(score, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if scores.shape != uncertainty.shape or scores.ndim != 1 or threshold < 0:
        raise ValueError("invalid abstention input")
    proposed = int(np.argmax(scores))
    if uncertainty[proposed] > threshold:
        return nominal_index, True
    return proposed, False


def deterministic_choice(keys: Iterable[str], salt: str = "exp12") -> int:
    values = list(keys)
    if not values:
        raise ValueError("empty deterministic choice")
    digest = hashlib.sha256((salt + "|" + "|".join(values)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % len(values)


def validate_demo_isolation(assignments: Sequence[Mapping]) -> bool:
    seen: dict[str, int] = {}
    for row in assignments:
        demo, fold = str(row["demo_key"]), int(row["fold"])
        if demo in seen and seen[demo] != fold:
            return False
        seen[demo] = fold
    return bool(seen)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

