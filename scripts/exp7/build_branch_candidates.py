#!/usr/bin/env python
"""Choose twelve deterministic, reference-only balanced EXP7 branches per demo."""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def choose(rows:list[dict])->list[dict]:
    n=len(rows); targets=np.linspace(0,n-1,12); chosen=[]; used=set()
    strata={}
    for row in rows:
        key=(bool(json.loads(row["contact_mode_json"])),row["boundary_margin_class"],row["gripper_state"],row["predicate"]); strata.setdefault(key,[]).append(row)
    ordered=sorted(strata.items(),key=lambda item:(len(item[1]),str(item[0])))
    for _,members in ordered:
        if len(chosen)>=12: break
        candidate=min(members,key=lambda row:(min(abs(row["action_index"]-x) for x in targets),row["action_index"]))
        if candidate["action_index"] not in used: chosen.append(candidate); used.add(candidate["action_index"])
    for target in targets:
        if len(chosen)>=12: break
        candidate=min((x for x in rows if x["action_index"] not in used),key=lambda row:(abs(row["action_index"]-target),row["action_index"])); chosen.append(candidate); used.add(candidate["action_index"])
    chosen.sort(key=lambda x:x["action_index"]); return chosen


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--geometry",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); rows=pq.read_table(args.geometry).to_pylist(); trajectories=[]
    keys=sorted({(x["task"],x["episode"]) for x in rows})
    for task,episode in keys:
        all_rows=[x for x in rows if x["task"]==task and x["episode"]==episode]; selected=choose(all_rows)
        branches=[{"kind":"balanced_contact_margin_time_progress","action_index":x["action_index"],"branch_time":x["action_index"],"normalized_time":x["normalized_time"],"physical_progress_clipped":x["physical_progress_clipped"],"reference_contact_mode_json":x["contact_mode_json"],"reference_signed_gap_m":x["signed_gap_m"],"boundary_margin_class":x["boundary_margin_class"],"reference_gripper_state":x["gripper_state"],"reference_predicate_state":x["predicate"]} for x in selected]
        trajectories.append({"task":task,"episode":episode,"trajectory_length":all_rows[0]["trajectory_length"],"branches":branches})
    result={"schema_version":1,"selection_rule":"one seed per rare joint contact/margin/gripper/predicate stratum, then nearest unused to 12 uniform time quantiles","trajectory_count":len(trajectories),"branch_count":sum(len(x["branches"]) for x in trajectories),"trajectories":trajectories,"gate":{"passed":len(trajectories)==30 and all(len(x["branches"])==12 for x in trajectories)}}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"trajectories":len(trajectories),"branches":result["branch_count"],"gate":result["gate"]},indent=2)); return 0 if result["gate"]["passed"] else 2


if __name__=="__main__": raise SystemExit(main())
