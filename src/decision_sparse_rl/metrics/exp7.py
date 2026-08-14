"""Pure metrics for EXP7 contact-mode-conditioned response analysis."""

from __future__ import annotations

import numpy as np


def margin_class(gap: float, near: float, far: float) -> str:
    value = abs(float(gap))
    if value < near:
        return "ambiguous"
    if value < far:
        return "near_boundary"
    return "interior"


def transition_category(reference: tuple[str, ...], plus: tuple[str, ...], minus: tuple[str, ...]) -> str:
    plus_same, minus_same = plus == reference, minus == reference
    if plus_same and minus_same:
        return "A_both_preserve"
    if not plus_same and not minus_same and plus == minus:
        return "B_same_new_mode"
    if not plus_same and not minus_same and plus != minus:
        return "C_signs_different_modes"
    return "D_one_sign_changes"


def antithetic_asymmetry(plus: np.ndarray, minus: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(plus) + np.asarray(minus))
    denominator = np.linalg.norm(plus) + np.linalg.norm(minus) + 1e-12
    return float(numerator / denominator)


def relative_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(predicted) - np.asarray(actual)) / (np.linalg.norm(actual) + 1e-12))
