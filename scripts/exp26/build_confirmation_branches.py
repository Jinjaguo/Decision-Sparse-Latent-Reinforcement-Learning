"""Freeze the last untouched recovery demos 28--29 with unique branch IDs."""

from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from scripts.exp7.contact_geometry import load_schema
from scripts.exp11.run_replacement_stage import choose_branches,phases_from_boundaries


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--reference-run",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts=out/"artifacts";artifacts.mkdir(parents=True);reference=ROOT/args.reference_run;manifest=reference/"artifacts/reference_snapshots_manifest.json";records=[x for x in json.loads(manifest.read_text())["episodes"] if int(x["episode"].split("_")[-1]) in (28,29)];schema=load_schema(ROOT/"experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json");branches=[]
    if len(records)!=6 or not all(x["success"] and x["all_snapshots_finite"] for x in records):raise RuntimeError("expected six successful demos 28--29")
    for record in records:
        directory=reference/record["relative_directory"];boundaries=json.loads((directory/"boundaries.json").read_text())
        with np.load(directory/"trajectory_states.npz",allow_pickle=False) as z:actions=np.asarray(z["actions"],float)
        for branch in choose_branches(phases_from_boundaries(boundaries,schema,record["task"]),actions,record["task"]):
            t=branch["branch_time"];branches.append({**branch,"branch_id":f"{record['task']}|{record['episode']}|{t}","task":record["task"],"episode":record["episode"],"trajectory_length":len(actions),"reference_directory":record["relative_directory"]})
    ids=[x["branch_id"] for x in branches]
    if len(branches)!=24 or len(set(ids))!=24:raise RuntimeError(f"expected 24/24 unique branches, got {len(branches)}/{len(set(ids))}")
    dump(artifacts/"branch_manifest.json",branches);metrics={"status":"completed","branch_count":24,"unique_branch_count":24,"demo_count":6,"episodes":[28,29],"reference_run":reference.name,"reference_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"frozen_before_candidate_outcomes":True,"target_future_access":False};dump(out/"metrics.json",metrics);print(json.dumps(metrics,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

