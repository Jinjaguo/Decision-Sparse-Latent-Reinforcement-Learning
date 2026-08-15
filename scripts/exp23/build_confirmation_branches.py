"""Freeze four reference-only landmarks on the prospectively amended EXP23 cohort."""

from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from scripts.exp7.contact_geometry import load_schema
from scripts.exp11.run_replacement_stage import choose_branches,phases_from_boundaries

EPISODES=set(range(21,28))


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--reference-run",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts=out/"artifacts";artifacts.mkdir(parents=True);reference=ROOT/args.reference_run;manifest=reference/"artifacts/reference_snapshots_manifest.json";records=[x for x in json.loads(manifest.read_text())["episodes"] if int(x["episode"].split("_")[-1]) in EPISODES];schema=load_schema(ROOT/"experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json");branches=[]
    if len(records)!=21 or not all(x["success"] and x["all_snapshots_finite"] for x in records):raise RuntimeError("confirmation references are incomplete or invalid")
    for record in records:
        directory=reference/record["relative_directory"];boundaries=json.loads((directory/"boundaries.json").read_text())
        with np.load(directory/"trajectory_states.npz",allow_pickle=False) as z:actions=np.asarray(z["actions"],float)
        phases=phases_from_boundaries(boundaries,schema,record["task"])
        for branch in choose_branches(phases,actions,record["task"]):
            t=branch["branch_time"];branches.append({**branch,"branch_id":f"{record['task']}|{record['episode']}|{t}","task":record["task"],"episode":record["episode"],"trajectory_length":len(actions),"reference_directory":record["relative_directory"]})
    if len(branches)!=84:raise RuntimeError(f"expected 84 branches, got {len(branches)}")
    dump(artifacts/"branch_manifest.json",branches);metrics={"status":"completed","branch_count":len(branches),"demo_count":len(records),"episodes":sorted(EPISODES),"reference_run":reference.name,"reference_manifest_sha256":sha(manifest),"selection":"reference-only EXP11 landmarks","frozen_before_candidate_outcomes":True,"target_future_access":False};dump(out/"metrics.json",metrics);print(json.dumps(metrics,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

