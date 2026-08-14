#!/usr/bin/env python
"""Predeclared EXP7 exact-mode versus contact-count/contact-group ablation."""

from __future__ import annotations

import argparse, json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--raw-run",type=Path,required=True); p.add_argument("--analysis-run",type=Path,required=True); args=p.parse_args(); schema=json.loads((ROOT/"experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json").read_text()); target={task:set(x["pair"] for x in spec["pair_groups"].get("target_gripper",[])) for task,spec in schema["tasks"].items()}
    cols=["task","intervention_id","continuation_offset","zero_contact_pairs_json","perturbed_contact_pairs_json","zero_contact_mode_json","perturbed_contact_mode_json"]; steps=pq.read_table(args.raw_run/"artifacts/per_step_effects.parquet",columns=cols).to_pandas(); rows=[]
    for intervention,group in steps.groupby("intervention_id",sort=False):
        group=group.sort_values("continuation_offset"); task=group.task.iloc[0]; zraw=[set(json.loads(x)) for x in group.zero_contact_pairs_json]; praw=[set(json.loads(x)) for x in group.perturbed_contact_pairs_json]; zmode=[set(json.loads(x)) for x in group.zero_contact_mode_json]; pmode=[set(json.loads(x)) for x in group.perturbed_contact_mode_json]
        tests={"raw_contact_count":[len(a)==len(b) for a,b in zip(zraw,praw)],"raw_exact_pair_set":[a==b for a,b in zip(zraw,praw)],"frozen_all_groups":[a==b for a,b in zip(zmode,pmode)],"target_gripper_only":[(a&target[task])==(b&target[task]) for a,b in zip(zmode,pmode)]}
        for horizon in (1,3,5,len(group)):
            label="remaining" if horizon==len(group) else str(horizon)
            for definition,values in tests.items(): rows.append({"task":task,"intervention_id":intervention,"horizon":label,"mode_definition":definition,"preserved":all(values[:horizon])})
    frame=pd.DataFrame(rows); summary=frame.groupby(["task","horizon","mode_definition"],as_index=False).preserved.agg(["count","mean"]).reset_index(); out=args.analysis_run/"artifacts/exact_mode_ablation.parquet"; pq.write_table(pa.Table.from_pandas(summary,preserve_index=False),out,compression="zstd"); payload={"overall_preservation_rates":frame.groupby(["horizon","mode_definition"]).preserved.mean().unstack().to_dict("index"),"rows":len(frame),"artifact":out.name}; (args.analysis_run/"artifacts/exact_mode_ablation_summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
