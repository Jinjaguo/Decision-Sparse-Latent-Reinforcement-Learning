"""Frozen numerical and convergence metrics for EXP6."""

from __future__ import annotations

from typing import Iterable

import numpy as np


RESOLUTION_CONSTANT = 1e-12


def projector(columns: np.ndarray, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(columns, dtype=np.float64)[:, :rank])
    return q @ q.T


def projector_similarity(left: np.ndarray, right: np.ndarray, rank: int) -> float:
    return float(1.0 - np.linalg.norm(left - right, ord="fro") / np.sqrt(2.0 * rank))


def relative_discrepancy(left: float, right: float) -> float:
    return float(abs(left - right) / (0.5 * (abs(left) + abs(right)) + RESOLUTION_CONSTANT))


def antithetic_asymmetry(plus: np.ndarray, minus: np.ndarray) -> float:
    plus = np.asarray(plus, dtype=np.float64)
    minus = np.asarray(minus, dtype=np.float64)
    return float(np.linalg.norm(plus + minus) / (np.linalg.norm(plus) + np.linalg.norm(minus) + RESOLUTION_CONSTANT))


def signal_to_floor(response_norm: float, measured_floor: float) -> float:
    return float(response_norm / max(float(measured_floor), RESOLUTION_CONSTANT))


def repeatability_max_abs(values: Iterable[np.ndarray]) -> float:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if len(arrays) < 2:
        raise ValueError("repeatability requires at least two observations")
    return float(max(np.max(np.abs(value - arrays[0])) for value in arrays[1:]))


def trust_region_passes(
    top1: float,
    top2: float,
    spectral_discrepancy: float,
    sign_asymmetry: float,
    heldout_vector_error: float,
    signal_floor_ratio: float,
) -> tuple[bool, list[str]]:
    tests = {
        "top1_similarity": top1 >= 0.80,
        "top2_similarity": top2 >= 0.75,
        "spectral_discrepancy": spectral_discrepancy <= 0.20,
        "sign_asymmetry": sign_asymmetry <= 0.25,
        "heldout_vector_error": heldout_vector_error <= 0.35,
        "signal_to_floor": signal_floor_ratio >= 100.0,
    }
    failed = [name for name, passed in tests.items() if not passed]
    return not failed, failed


def first_contact_divergence(plus_sets: list[set[str]], minus_sets: list[set[str]]) -> int | None:
    for offset, (plus, minus) in enumerate(zip(plus_sets, minus_sets), start=1):
        if plus != minus:
            return offset
    return None
