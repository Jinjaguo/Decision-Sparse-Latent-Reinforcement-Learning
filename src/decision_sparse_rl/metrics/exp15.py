"""Pure retrieval and recovery metrics for EXP15."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def standardized_distance(query: Sequence[float], library: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    q, x, s = np.asarray(query, float), np.asarray(library, float), np.asarray(scale, float)
    if x.ndim != 2 or q.shape != (x.shape[1],) or s.shape != q.shape:
        raise ValueError("incompatible retrieval arrays")
    safe = s.copy(); safe[safe < 1e-8] = 1.0
    return np.linalg.norm((x-q)/safe, axis=1)


def weighted_chunk(chunks: np.ndarray, distances: Sequence[float]) -> np.ndarray:
    values, distance = np.asarray(chunks, float), np.asarray(distances, float)
    if values.ndim != 3 or values.shape[2] != 7 or distance.shape != (len(values),):
        raise ValueError("invalid chunk aggregation")
    weights = 1.0 / np.maximum(distance, 1e-4); weights /= weights.sum()
    out = np.tensordot(weights, values, axes=(0,0))
    out[:,6] = np.where(np.tensordot(weights, np.sign(values[:,:,6]), axes=(0,0)) >= 0, 1.0, -1.0)
    return out


def monotone_window(episodes: Sequence[str], indexes: Sequence[int], episode: str, previous: int, width: int = 20) -> np.ndarray:
    ep, idx = np.asarray(episodes), np.asarray(indexes, int)
    return np.flatnonzero((ep == episode) & (idx >= previous) & (idx <= previous + width))


def recovery_metrics(rows: list[dict], default_route: str) -> dict:
    groups={}
    for row in rows:groups.setdefault(row["branch_id"],[]).append(row)
    availability=[];demand=[];recovered=[]
    for values in groups.values():
        default=next(x for x in values if x["route"]==default_route)
        safe=any(x["success"] and not x["safety_stop"] for x in values)
        need=not (default["success"] and not default["safety_stop"])
        availability.append(safe);demand.append(need);recovered.append((not need) or safe)
    demand_count=sum(demand)
    return {"group_count":len(groups),"safe_candidate_availability":float(np.mean(availability)),"decision_demand_rate":float(np.mean(demand)),"demand_recovery_rate":float(sum(a and b for a,b in zip(demand,availability))/demand_count) if demand_count else 0.0}

