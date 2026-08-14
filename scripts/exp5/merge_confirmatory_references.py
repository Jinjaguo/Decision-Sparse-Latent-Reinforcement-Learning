#!/usr/bin/env python
"""Merge passing EXP5 reference shards under the amended cohort rule."""

from __future__ import annotations

import argparse, hashlib, json, shlex, shutil, sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402

def digest(path:Path)->str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run-id",required=True); p.add_argument("--source-runs",nargs="+",required=True); p.add_argument("--run-root",type=Path,default=ROOT/"runs"); args=p.parse_args()
    run=create_run_directory(args.run_root,args.run_id); records=[]; source_hashes={}; rejected=[]
    for name in args.source_runs:
        source=(args.run_root/name).resolve(); metrics=json.loads((source/"metrics.json").read_text()); source_hashes[name]=digest(source/"metrics.json")
        for record in metrics.get("episode_records",[]):
            if not record["success"]:
                rejected.append({"source_run":name,**record}); continue
            key=(record["task"],record["episode"])
            if any((x["task"],x["episode"])==key for x in records): raise RuntimeError(f"duplicate reference {key}")
            src=source/record["relative_directory"]; rel=Path("artifacts/references")/record["task"]/record["episode"]; dst=run/rel; shutil.copytree(src,dst)
            records.append({**record,"relative_directory":rel.as_posix(),"source_run":name})
    counts=Counter(x["task"] for x in records); expected={"open_the_middle_drawer_of_the_cabinet":10,"turn_on_the_stove":10,"put_the_bowl_on_the_plate":10}
    required_drawer={f"demo_{i}" for i in range(10,17)}|{"demo_18","demo_19","demo_20"}
    drawer={x["episode"] for x in records if x["task"]=="open_the_middle_drawer_of_the_cabinet"}
    criteria={"exactly_30_references":len(records)==30,"exactly_10_per_task":dict(counts)==expected,"all_success":all(x["success"] for x in records),"all_finite":all(x["all_snapshots_finite"] for x in records),"roundtrips_exact":all(x["maximum_roundtrip_state_l2"]["integration"]==0 and x["maximum_controller_roundtrip_error"]==0 for x in records),"amended_drawer_cohort_exact":drawer==required_drawer}
    gate={"passed":all(criteria.values()),"criteria":criteria}; manifest={"schema_version":1,"run_id":args.run_id,"cohort":"eligibility-conditioned amended EXP5 confirmation","source_run_metrics_sha256":source_hashes,"replacement":{"rejected":"drawer/demo_17","accepted":"drawer/demo_20","rule":"first passing unused demo in increasing order from demo20"},"episodes":records,"rejected_references":rejected,"gate":gate}
    write_json(run/"artifacts/reference_snapshots_manifest.json",manifest); write_json(run/"artifacts/failure_examples.json",rejected)
    metrics={"run_id":args.run_id,"status":"completed","gate":gate,"episode_count":len(records),"task_counts":dict(counts),"rejected_count":len(rejected),"maximum_integration_roundtrip_l2":max(x["maximum_roundtrip_state_l2"]["integration"] for x in records),"maximum_controller_roundtrip_error":max(x["maximum_controller_roundtrip_error"] for x in records)}
    write_run_record(run,config={"stage":"EXP5-4 amended cohort merge","sources":args.source_runs},command=shlex.join([sys.executable,*sys.argv]),environment={"python":sys.version},git_state={"project":git_record(ROOT)},stdout=json.dumps(metrics,sort_keys=True),stderr="",metrics=metrics); return 0 if gate["passed"] else 2

if __name__=="__main__": raise SystemExit(main())
