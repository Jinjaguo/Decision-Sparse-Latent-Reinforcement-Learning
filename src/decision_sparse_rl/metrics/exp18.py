"""Selection metrics for same-state recovery candidate sets."""

from __future__ import annotations

import numpy as np


def selector_metrics(selected: list[dict], default: list[dict], oracle: list[dict]) -> dict:
    if not (len(selected)==len(default)==len(oracle)) or not selected:
        raise ValueError("selector rows must be aligned and non-empty")
    safe=lambda x:bool(x["success"] and not x["safety_stop"])
    s=np.asarray([safe(x) for x in selected]);d=np.asarray([safe(x) for x in default]);o=np.asarray([safe(x) for x in oracle]);demand=~d
    headroom=max(1,int(o.sum()-d.sum()))
    return {"safe_success_rate":float(s.mean()),"default_safe_success_rate":float(d.mean()),"improvement_points":float(s.mean()-d.mean()),"demand_recovery_rate":float(s[demand].mean()) if demand.any() else 0.0,"oracle_safe_success_rate":float(o.mean()),"oracle_headroom_capture":float((s.sum()-d.sum())/headroom),"safety_stop_rate":float(np.mean([x["safety_stop"] for x in selected])),"default_safety_stop_rate":float(np.mean([x["safety_stop"] for x in default]))}

