"""Auditable Panda arm-q intervention helpers for EXP2 R5."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def scaled_joint_delta(direction: np.ndarray, joint_ranges: np.ndarray, fraction: float) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    joint_ranges = np.asarray(joint_ranges, dtype=np.float64)
    if direction.shape != joint_ranges.shape or direction.ndim != 1:
        raise ValueError("direction and joint ranges must be same-shape vectors")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("direction must have a finite nonzero norm")
    if fraction <= 0:
        raise ValueError("fraction must be positive")
    return fraction * joint_ranges * (direction / norm)


def apply_arm_q(data: object, qpos_indexes: Sequence[int], delta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    indexes = np.asarray(qpos_indexes, dtype=np.int64)
    before = np.asarray(data.qpos[indexes], dtype=np.float64).copy()
    after = before + np.asarray(delta, dtype=np.float64)
    if np.any(after < lower) or np.any(after > upper):
        raise ValueError("q intervention violates verified runtime joint limits")
    data.qpos[indexes] = after
    return after


def non_arm_integration_linf(
    before: Dict[str, np.ndarray],
    after: Dict[str, np.ndarray],
    arm_qpos_indexes: Sequence[int],
) -> Dict[str, float]:
    if before.keys() != after.keys():
        raise ValueError("integration component sets differ")
    indexes = np.asarray(arm_qpos_indexes, dtype=np.int64)
    result = {}
    for name in before:
        difference = np.asarray(after[name]) - np.asarray(before[name])
        if name == "mjSTATE_QPOS":
            mask = np.ones(difference.size, dtype=bool)
            mask[indexes] = False
            difference = difference[mask]
        result[name] = float(np.max(np.abs(difference))) if difference.size else 0.0
    return result

