#!/usr/bin/env python
"""Merge the first ten passing independent EXP7 references per task."""

from __future__ import annotations

import argparse, json, shlex, shutil, sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record
from decision_sparse_rl.utils.environment_audit import git_record


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); p.add_argument("--source-runs",nargs="+",required=True); p.add_argument("--run-root",type=Path,default=ROOT/"runs"); args=p.parse_args()
    run=create_run_directory(args.run_root,args.run_id); by_task=defaultdict(list); rejected=[]
    for name in args.source_runs:
        source=(args.run_root/name).resolve(); metrics=json.loads((source/"metrics.json").read_text())
        for record in metrics.get("episode_records",[]):
            qualified=bool(record["success"] and record["all_snapshots_finite"] and record["maximum_roundtrip_state_l2"]["integration"] == 0 and record["maximum_controller_roundtrip_error"] == 0)
            (by_task[record["task"]] if qualified else rejected).append((name,record) if qualified else {"source_run":name,**record})
    records=[]
    for task, candidates in sorted(by_task.items()):
        for name, record in sorted(candidates,key=lambda x:int(x[1]["episode"].split("_")[-1]))[:10]:
            source=args.run_root/name; rel=Path("artifacts/references")/task/record["episode"]; shutil.copytree(source/record["relative_directory"],run/rel)
            records.append({**record,"relative_directory":rel.as_posix(),"source_run":name})
    counts={task:sum(x["task"]==task for x in records) for task in sorted(by_task)}
    criteria={"exactly_30":len(records)==30,"exactly_10_per_task":len(counts)==3 and all(v==10 for v in counts.values()),"all_qualified":all(x["success"] and x["all_snapshots_finite"] for x in records),"independent_episode_floor":all(int(x["episode"].split("_")[-1]) >= (21 if x["task"].startswith("open_") else 20) for x in records)}
    gate={"passed":all(criteria.values()),"criteria":criteria}; manifest={"schema_version":1,"run_id":args.run_id,"cohort":"independent EXP7 cohort","selection_rule":"first ten qualified unused demonstrations in ascending task-local episode order","episodes":records,"rejected_references":rejected,"gate":gate}
    write_json(run/"artifacts/reference_snapshots_manifest.json",manifest); write_json(run/"artifacts/failure_examples.json",rejected)
    metrics={"run_id":args.run_id,"status":"completed","gate":gate,"task_counts":counts,"episode_count":len(records),"rejected_count":len(rejected)}
    write_run_record(run,config={"stage":"EXP7 independent cohort merge","sources":args.source_runs},command=shlex.join([sys.executable,*sys.argv]),environment={"python":sys.version},git_state={"project":git_record(ROOT)},stdout=json.dumps(metrics),stderr="",metrics=metrics); print(json.dumps(metrics,indent=2)); return 0 if gate["passed"] else 2


if __name__=="__main__": raise SystemExit(main())
