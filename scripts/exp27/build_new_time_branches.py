"""Freeze two previously unexecuted reference-only times per demo 0--9."""

from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--reference-runs",type=Path,nargs="+",required=True);p.add_argument("--exclude-manifest",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts=out/"artifacts";artifacts.mkdir(parents=True);excluded={x["branch_id"] for x in json.loads((ROOT/args.exclude_manifest).read_text())};records=[];hashes={}
    for relative in args.reference_runs:
        reference=ROOT/relative;manifest=reference/"artifacts/reference_snapshots_manifest.json";hashes[reference.name]=hashlib.sha256(manifest.read_bytes()).hexdigest()
        for row in json.loads(manifest.read_text())["episodes"]:
            if 0<=int(row["episode"].split("_")[-1])<=9:records.append((reference,row))
    if len(records)!=30:raise RuntimeError(f"expected 30 references, got {len(records)}")
    branches=[]
    for reference,record in records:
        directory=reference/record["relative_directory"]
        with np.load(directory/"trajectory_states.npz",allow_pickle=False) as z:length=len(z["actions"])
        used=set()
        for q,label in ((.28,"new_q28"),(.68,"new_q68")):
            t=min(length-11,max(0,int(round(q*(length-1)))))
            while f"{record['task']}|{record['episode']}|{t}" in excluded or t in used:t+=1
            if t>length-11:raise RuntimeError("cannot place unique new branch")
            used.add(t);branches.append({"branch_time":t,"phase":"NEW","kind":label,"branch_id":f"{record['task']}|{record['episode']}|{t}","task":record["task"],"episode":record["episode"],"trajectory_length":length,"reference_run":str(reference.relative_to(ROOT)),"reference_directory":record["relative_directory"]})
    ids=[x["branch_id"] for x in branches]
    if len(branches)!=60 or len(set(ids))!=60 or set(ids)&excluded:raise RuntimeError("new confirmation branches are not 60/60 disjoint")
    dump(artifacts/"branch_manifest.json",branches);metrics={"status":"completed","branch_count":60,"unique_branch_count":60,"disjoint_from_exp25":True,"quantiles":[.28,.68],"reference_hashes":hashes,"frozen_before_candidate_outcomes":True,"target_future_access":False};dump(out/"metrics.json",metrics);print(json.dumps(metrics,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

