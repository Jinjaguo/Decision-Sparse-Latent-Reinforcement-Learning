"""Pure construction, calibration, and sparsity utilities for EXP11."""

from __future__ import annotations

from typing import Sequence

import hashlib

import numpy as np


def extract_chunk(actions: np.ndarray, start: int, length: int):
    x = np.asarray(actions, dtype=np.float64)
    if x.ndim != 2 or not 0 <= start < len(x) or length < 1:
        raise ValueError("invalid action chunk request")
    out = np.zeros((length, x.shape[1]), dtype=np.float64)
    observed = x[start : min(len(x), start + length)]
    out[: len(observed)] = observed
    mask = np.zeros(length, dtype=np.float64); mask[: len(observed)] = 1.0
    return out, mask


def dct_modes(length: int, count: int = 3) -> np.ndarray:
    if length < 2 or not 1 <= count < length:
        raise ValueError("invalid DCT dimensions")
    t = np.arange(length, dtype=np.float64)
    basis = np.asarray([np.cos(np.pi * (t + .5) * k / length) for k in range(1, count + 1)])
    return basis / np.linalg.norm(basis, axis=1, keepdims=True)


def spline_modes(length: int, count: int = 3) -> np.ndarray:
    """Smooth low-frequency cubic Bernstein modes, orthonormalized."""
    if length < 5 or count != 3:
        raise ValueError("EXP11 spline basis requires length>=5 and three modes")
    t = np.linspace(0, 1, length)
    raw = np.vstack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t)])
    q, _ = np.linalg.qr(raw.T)
    return q[:, :count].T


def smooth_pulse_modes(length: int, count: int = 3) -> np.ndarray:
    if length < 5 or count != 3:
        raise ValueError("EXP11 pulse basis requires length>=5 and three modes")
    t = np.linspace(0, 1, length)
    centers = np.linspace(.25, .75, count)
    raw = np.asarray([np.exp(-.5 * ((t - c) / .22) ** 2) for c in centers])
    q, _ = np.linalg.qr(raw.T)
    return q[:, :count].T


def normalize_mode(mode: np.ndarray) -> np.ndarray:
    x = np.asarray(mode, dtype=np.float64)
    maximum = float(np.max(np.abs(x)))
    if maximum <= 0 or not np.isfinite(maximum):
        raise ValueError("mode is zero or nonfinite")
    return x / maximum


def apply_replacement(reference: np.ndarray, mode: np.ndarray, amplitude: float, sign: int, channel: int):
    ref = np.asarray(reference, dtype=np.float64)
    temporal = normalize_mode(mode)
    if ref.ndim != 2 or temporal.shape != (len(ref),) or channel < 0 or channel >= ref.shape[1] or sign not in (-1, 1) or amplitude < 0:
        raise ValueError("invalid replacement")
    requested = ref.copy(); requested[:, channel] += sign * amplitude * temporal
    executed = np.clip(requested, -1.0, 1.0)
    clipped = np.abs(executed - requested) > 0
    return requested, executed, clipped


def paired_replacements(reference: np.ndarray, mode: np.ndarray, amplitudes: Sequence[float], channel: int):
    rows = []
    for amplitude in amplitudes:
        for sign in (-1, 1):
            requested, executed, clipped = apply_replacement(reference, mode, float(amplitude), sign, channel)
            rows.append((float(amplitude), sign, requested, executed, clipped))
    return rows


def requested_executed_mismatch(requested: np.ndarray, executed: np.ndarray, saturated: np.ndarray) -> float:
    req, exe, sat = np.asarray(requested), np.asarray(executed), np.asarray(saturated, dtype=bool)
    if req.shape != exe.shape or sat.shape != req.shape:
        raise ValueError("mismatch shapes differ")
    valid = ~sat
    return 0.0 if not np.any(valid) else float(np.max(np.abs(req[valid] - exe[valid])))


def macro_response(zero: np.ndarray, replacement: np.ndarray, scales: np.ndarray) -> np.ndarray:
    a, b, s = map(lambda x: np.asarray(x, dtype=np.float64), (zero, replacement, scales))
    if a.shape != b.shape or a.shape[-1] != len(s) or np.any(s <= 0):
        raise ValueError("invalid macro response inputs")
    return (b - a) / s


def paired_sign_asymmetry(negative: np.ndarray, positive: np.ndarray) -> float:
    minus, plus = map(lambda x: np.asarray(x, dtype=np.float64), (negative, positive))
    return float(np.linalg.norm(plus + minus) / (np.linalg.norm(plus) + np.linalg.norm(minus) + 1e-12))


def residual_basis(train_chunks: np.ndarray, rank: int = 3):
    x = np.asarray(train_chunks, dtype=np.float64)
    if x.ndim != 3 or len(x) <= rank:
        raise ValueError("insufficient residual chunks")
    flat = x.reshape(len(x), -1); mean = flat.mean(0); _, singular, vh = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, vh[:rank], singular


def subspace_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a, b = map(lambda x: np.asarray(x, dtype=np.float64), (left, right))
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("subspaces differ")
    singular = np.linalg.svd(a @ b.T, compute_uv=False)
    return float(np.mean(np.clip(singular, 0, 1)))


def path_nonconformity(observation: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    y, mu, sd = map(lambda x: np.asarray(x, dtype=np.float64), (observation, mean, scale))
    if y.shape != mu.shape or sd.shape != y.shape or y.ndim < 2 or np.any(sd <= 0):
        raise ValueError("invalid path calibration arrays")
    return np.max(np.abs(y - mu) / sd, axis=tuple(range(1, y.ndim)))


def conformal_quantile(scores: Sequence[float], coverage: float = .9) -> float:
    x = np.sort(np.asarray(scores, dtype=np.float64))
    if x.ndim != 1 or not len(x) or not 0 < coverage < 1:
        raise ValueError("invalid conformal request")
    index = min(len(x) - 1, int(np.ceil((len(x) + 1) * coverage)) - 1)
    return float(x[index])


def top_fraction_mass(values: Sequence[float], fraction: float = .2) -> float:
    x = np.maximum(np.asarray(values, dtype=np.float64), 0)
    if x.ndim != 1 or not len(x) or not 0 < fraction <= 1:
        raise ValueError("invalid consequence values")
    count = max(1, int(np.ceil(len(x) * fraction)))
    return float(np.sort(x)[-count:].sum() / max(x.sum(), 1e-15))


def rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = map(lambda x: np.asarray(x, dtype=np.float64), (left, right))
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("invalid ranking vectors")
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
    rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def phase_shift_chunk(actions: np.ndarray, start: int, length: int, shift: int) -> np.ndarray:
    """Deterministic one-step reference index edit used by I-C."""
    x = np.asarray(actions, dtype=np.float64)
    if x.ndim != 2 or shift not in (-1, 1) or length < 1 or not 0 <= start < len(x):
        raise ValueError("invalid phase edit")
    indexes = np.clip(np.arange(start, start + length) + shift, 0, len(x) - 1)
    return x[indexes].copy()


def shift_gripper_transition(commands: Sequence[float], transition: int, shift: int) -> np.ndarray:
    """Advance/delay a binary sign transition without interpolation."""
    x = np.sign(np.asarray(commands, dtype=np.float64))
    if x.ndim != 1 or shift not in (-1, 1) or not 1 <= transition < len(x):
        raise ValueError("invalid gripper transition edit")
    target = int(np.clip(transition + shift, 1, len(x) - 1))
    out = x.copy(); out[:] = x[0]; out[target:] = x[transition]
    return out


def object_centric_features(eef: np.ndarray, objects: np.ndarray) -> np.ndarray:
    e, o = np.asarray(eef, dtype=np.float64), np.asarray(objects, dtype=np.float64)
    if e.shape != (3,) or o.ndim != 2 or o.shape[1] != 3 or not len(o):
        raise ValueError("invalid object-centric geometry")
    relative = o - e
    pairwise = [o[j] - o[i] for i in range(len(o)) for j in range(i + 1, len(o))]
    return np.concatenate([relative.reshape(-1), np.asarray(pairwise).reshape(-1)])


def switching_regime_index(contact_mode: str, previous_mode: str | None = None) -> int:
    """Frozen coarse R0/R1/R2/R5-like index from exact identity strings."""
    mode = contact_mode.lower()
    has_gripper = "gripper" in mode
    active = mode not in ("", "[]", "()")
    if previous_mode and "gripper" in previous_mode.lower() and not has_gripper:
        return 6
    if not active: return 0
    if has_gripper and mode.count("|") > 1: return 5
    if has_gripper: return 2
    return 1


def complete_demo_fold(task: str, episode: str, folds: int = 5) -> int:
    if folds < 2: raise ValueError("need at least two folds")
    digest = hashlib.sha256(f"{task}|{episode}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % folds


def raw_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
