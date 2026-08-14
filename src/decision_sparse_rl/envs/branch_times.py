"""Deterministic, outcome-blind EXP2 branch-time selection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


QUANTILES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


def nearest_unused(target: int, used: Iterable[int], length: int) -> int:
    used_set = set(int(value) for value in used)
    if length <= 0:
        raise ValueError("trajectory length must be positive")
    target = min(max(int(target), 0), length - 1)
    for radius in range(length):
        candidates = [target] if radius == 0 else [target - radius, target + radius]
        for candidate in candidates:
            if 0 <= candidate < length and candidate not in used_set:
                return candidate
    raise ValueError("no unused branch time remains")


def first_gripper_sign_change(actions: np.ndarray) -> int:
    commands = np.asarray(actions, dtype=np.float64)[:, -1]
    signs = np.sign(commands)
    for index in range(1, len(signs)):
        if signs[index - 1] != 0 and signs[index] != 0 and signs[index] != signs[index - 1]:
            return index
    raise ValueError("trajectory has no verified nonzero gripper-command sign change")


def first_max_contact_change(contact_counts: Sequence[int]) -> int:
    values = np.asarray(contact_counts, dtype=np.int64)
    if values.size < 2:
        raise ValueError("at least two contact counts are required")
    return int(np.argmax(np.abs(np.diff(values))) + 1)


def select_branch_times(actions: np.ndarray, contact_counts: Sequence[int]) -> List[Dict[str, Any]]:
    length = int(len(actions))
    if len(contact_counts) != length:
        raise ValueError("contact count and action lengths differ")
    selected: List[Dict[str, Any]] = []
    used: List[int] = []
    for quantile in QUANTILES:
        target = int(round(quantile * (length - 1)))
        index = nearest_unused(target, used, length)
        selected.append({"kind": "temporal_quantile", "quantile": quantile, "target": target, "action_index": index, "replacement_offset": index - target})
        used.append(index)
    event_targets = (
        ("first_gripper_command_sign_change", first_gripper_sign_change(actions)),
        ("first_maximum_contact_count_change", first_max_contact_change(contact_counts)),
    )
    for kind, target in event_targets:
        index = nearest_unused(target, used, length)
        selected.append({"kind": kind, "target": target, "action_index": index, "replacement_offset": index - target})
        used.append(index)
    if len(selected) != 12 or len(set(used)) != 12:
        raise AssertionError("branch selection did not produce 12 unique times")
    return selected

