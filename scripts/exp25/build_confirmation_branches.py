"""Freeze unique EXP25 confirmation branches from successful demos 0--9."""

from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from scripts.exp7.contact_geometry import load_schema
from scripts.exp11.run_replacement_stage import choose_branches,phases_from_boundaries


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--reference-runs",type=Path,nargs="+",required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts=out/"artifacts";artifacts.mkdir(parents=True);schema=load_schema(ROOT/"experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json");records=[];hashes={}
    for relative in args.reference_runs:
        reference=ROOT/relative;manifest=reference/"artifacts/reference_snapshots_manifest.json";hashes[reference.name]=sha(manifest)
        for row in json.loads(manifest.read_text())["episodes"]:
            index=int(row["episode"].split("_")[-1])
            if 0<=index<=9:records.append((reference,row))
    keys=[(x[1]["task"],x[1]["episode"]) for x in records]
    if len(records)!=30 or len(set(keys))!=30 or not all(x[1]["success"] and x[1]["all_snapshots_finite"] for x in records):raise RuntimeError("expected 30 unique successful references for demos 0--9")
    branches=[]
    for reference,record in records:
        directory=reference/record["relative_directory"];boundaries=json.loads((directory/"boundaries.json").read_text())
        with np.load(directory/"trajectory_states.npz",allow_pickle=False) as z:actions=np.asarray(z["actions"],float)
        for branch in choose_branches(phases_from_boundaries(boundaries,schema,record["task"]),actions,record["task"]):
            t=branch["branch_time"];branches.append({**branch,"branch_id":f"{record['task']}|{record['episode']}|{t}","task":record["task"],"episode":record["episode"],"trajectory_length":len(actions),"reference_run":str(reference.relative_to(ROOT)),"reference_directory":record["relative_directory"]})
    ids=[x["branch_id"] for x in branches]
    if len(branches)!=120 or len(set(ids))!=120:raise RuntimeError(f"confirmation branches must be 120/120 unique, got {len(branches)}/{len(set(ids))}")
    dump(artifacts/"branch_manifest.json",branches);metrics={"status":"completed","branch_count":120,"unique_branch_count":120,"demo_count":30,"episodes":list(range(10)),"reference_hashes":hashes,"selection":"reference-only EXP11 landmarks","frozen_before_candidate_outcomes":True,"target_future_access":False};dump(out/"metrics.json",metrics);print(json.dumps(metrics,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

